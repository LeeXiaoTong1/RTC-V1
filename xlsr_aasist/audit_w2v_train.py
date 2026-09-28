"""CPU-only, read-only Train inventory. No torch, training, cache generation or audio export."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from urllib.parse import urlsplit
import zipfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
CUT_SECONDS = 64600 / 16000
LABELS = {'fake': 0, 'spoof': 0, 'real': 1, 'bonafide': 1, 'bona-fide': 1}
CLASS_NAMES = {0: 'fake', 1: 'real'}
SCHEMA = 'w2v_train_inventory_v1'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def within(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def relative_id(value):
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError('Empty or invalid source ID')
    value = value.strip().replace('\\', '/')
    parts = PurePosixPath(value).parts
    if value.startswith('/') or re.match(r'^[A-Za-z]:', value) or '..' in parts:
        raise ValueError('Source ID must stay relative to its data root: '+value)
    return str(PurePosixPath(value))


def tags(source):
    parts = PurePosixPath(source).parts[:-1]
    langs = set(parts) & {'en', 'zh'}
    domains = set(parts) & {'offline', 'online'}
    return (next(iter(langs)) if len(langs) == 1 else 'unknown',
            next(iter(domains)) if len(domains) == 1 else 'unknown',
            str(PurePosixPath(source).parent))


class Issues:
    def __init__(self):
        self.counts, self.examples = Counter(), []

    def add(self, code, source='', detail=''):
        self.counts[code] += 1
        if len(self.examples) < 5000:
            self.examples.append({'code': code, 'source': str(source), 'detail': str(detail)})

    def result(self):
        return {'counts': dict(self.counts), 'total': sum(self.counts.values()),
                'examples_saved': len(self.examples),
                'examples_truncated': sum(self.counts.values()) > len(self.examples)}


def protocol_rows(path, data_root, issues):
    rows, seen, line_count = [], {}, 0
    data_root, last = Path(data_root).resolve(), time.monotonic()
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            parts = line.split()
            if not parts:
                continue
            line_count += 1
            if len(parts) == 2:
                source, name = parts
            elif len(parts) >= 5:
                source, name = parts[1], parts[4]
            else:
                raise ValueError(f'Invalid labeled protocol line {number}')
            source = relative_id(source)
            if name.lower() not in LABELS:
                raise ValueError(f'Unsupported label on line {number}: {name}')
            label = LABELS[name.lower()]
            if source in seen:
                issues.add('duplicate_protocol_id' if seen[source] == label else 'conflicting_protocol_label', source, number)
                continue
            seen[source] = label
            lang, domain, directory = tags(source)
            audio = (data_root/source).resolve()
            if not audio.is_relative_to(data_root):
                raise ValueError('Audio/symlink escapes data root: '+source)
            if lang == 'unknown':
                issues.add('unknown_language_directory', source)
            rows.append({'source_id': source, 'label': label, 'class': CLASS_NAMES[label],
                         'language_group': lang, 'domain': domain, 'source_directory': directory,
                         'audio_path': str(audio), 'protocol_line': number})
            if time.monotonic()-last >= 10:
                print(f'Train protocol indexing: {len(rows)} unique IDs',flush=True)
                last=time.monotonic()
    if not rows:
        raise ValueError('Empty Train protocol')
    return rows, line_count


def segment_stats(audio, sr):
    mono = audio.mean(axis=1, dtype=np.float64)
    if not len(mono) or not np.isfinite(mono).all():
        raise ValueError('Empty or non-finite audio segment')
    frame = max(1, round(sr*.02))
    full = len(mono)//frame*frame
    values = np.sqrt(np.mean(mono[:full].reshape(-1,frame)**2,axis=1)) if full else np.empty(0)
    if full < len(mono):
        values = np.append(values,np.sqrt(np.mean(mono[full:]**2)))
    return {'rms_dbfs': float(20*np.log10(max(float(np.sqrt(np.mean(mono**2))), 1e-12))),
            'active_frame_fraction': float(np.mean(values > 10**(-50/20))),
            'near_full_scale_fraction': float(np.mean(np.abs(audio) >= .999)),
            'zero_fraction': float(np.mean(audio == 0))}


def inspect_audio(row):
    result = dict(row)
    try:
        path = Path(row['audio_path'])
        before = path.stat()
        result.update(file_bytes=before.st_size, file_mtime_ns=before.st_mtime_ns, file_sha256=sha256(path))
        with sf.SoundFile(path) as stream:
            sr, frames, channels = stream.samplerate, stream.frames, stream.channels
            if sr <= 0 or frames <= 0:
                raise ValueError('Audio has zero samples or invalid sample rate')
            take = min(frames, max(1, round(CUT_SECONDS*sr)))
            head = stream.read(take, dtype='float32', always_2d=True)
            if len(head) != take:
                raise ValueError('Audio header and decoded head length disagree')
            result.update(sample_rate=sr, channels=channels, frames=frames, subtype=stream.subtype,
                          seconds=frames/sr, used_head_seconds=take/sr,
                          head_unused_seconds=max(0., frames/sr-CUT_SECONDS),
                          head=segment_stats(head, sr))
            middle_start = max(0, (frames-take)//2)
            stream.seek(middle_start)
            middle = stream.read(take, dtype='float32', always_2d=True)
            if len(middle) != take:
                raise ValueError('Audio header and decoded middle length disagree')
            result.update(middle=segment_stats(middle, sr), middle_start_seconds=middle_start/sr)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source file changed while being read')
        result['status'] = 'ok'
    except Exception as exc:
        result['status'], result['error'] = 'error', f'{type(exc).__name__}: {exc}'
    return result


def scan_audio(rows, workers, out):
    scanned, pending, cursor, last = [], {}, 0, time.monotonic()
    started = last
    with ThreadPoolExecutor(max_workers=workers) as pool, (out/'audio_inventory.jsonl').open('w', encoding='utf-8') as stream:
        while cursor < len(rows) or pending:
            while cursor < len(rows) and len(pending) < workers*2:
                future = pool.submit(inspect_audio, rows[cursor])
                pending[future] = cursor
                cursor += 1
            done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                record = future.result()
                scanned.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            now = time.monotonic()
            if now-last >= 10 or len(scanned) == len(rows):
                stream.flush()
                print(f'Audio scan: {len(scanned)}/{len(rows)}; elapsed={now-started:.0f}s; '
                      f'rate={len(scanned)/max(1., now-started):.1f} files/s', flush=True)
                last = now
    return sorted(scanned, key=lambda row: row['protocol_line'])


class Components:
    def __init__(self, ids):
        self.parent = {key:key for key in ids}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def duplicates(rows, components, issues):
    mapping = defaultdict(list)
    for row in rows:
        if row.get('file_sha256'):
            mapping[row['file_sha256']].append(row)
    result = []
    for digest, members in mapping.items():
        if len(members) < 2:
            continue
        conflict = len({r['label'] for r in members}) > 1
        if conflict:
            issues.add('same_file_bytes_conflicting_labels', members[0]['source_id'], len(members))
        else:
            for row in members[1:]:
                components.join(members[0]['source_id'], row['source_id'])
        result.append({'sha256':digest, 'label_conflict':conflict,
                       'sources':[r['source_id'] for r in members], 'labels':[r['label'] for r in members]})
    return result


def pair_inventory(path, records, components, issues):
    if path is None:
        return {'available':False, 'reason':'rtc_pairs absent in run config'}
    counts, covered, edges, online_map, seen_edges = Counter(), set(), [], {}, set()
    with path.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                off, on, label = relative_id(row['offline']), relative_id(row['online']), row['label']
                if type(label) is not int or label not in (0,1):
                    raise ValueError('Pair label must be integer 0/1')
                if off not in records or on not in records or any(records[x]['label'] != label for x in (off,on)):
                    raise ValueError('Pair endpoint/label disagrees with Train protocol')
                if records[off]['domain'] != 'offline' or records[on]['domain'] != 'online':
                    raise ValueError('Pair domain is not explicit Offline/Online')
                if records[off]['language_group'] != records[on]['language_group']:
                    raise ValueError('Pair endpoints cross language directory groups')
                if (off,on) in seen_edges or on in online_map and online_map[on] != off:
                    raise ValueError('Duplicate pair or conflicting Online correspondence')
                seen_edges.add((off,on))
                online_map[on] = off
                components.join(off,on)
                covered.update((off,on))
                key = records[off]['language_group']+'-'+CLASS_NAMES[label]
                counts[key] += 1
                edges.append({'offline':off, 'online':on, 'group':key})
            except (ValueError, KeyError, TypeError) as exc:
                issues.add('invalid_rtc_pair', number, exc)
    return {'available':True, 'valid_pairs':len(edges), 'pairs_by_group':dict(counts),
            'covered_ids':covered, 'edges':edges}


def cache_inventory(path, records, protocol_hash, issues):
    path = Path(path).resolve()
    config_path, manifest_path = path/'config.json', path/'manifest.jsonl'
    result = {'path':str(path), 'available':config_path.is_file() and manifest_path.is_file()}
    if not result['available']:
        issues.add('missing_train_cache_metadata', path)
        return result
    config = read_json(config_path)
    result.update(role=config.get('role'), generation=config.get('generation'),
                  profile=config.get('processing',{}).get('profile','legacy'))
    if config.get('role') != 'train':
        issues.add('non_train_cache_role', path, config.get('role'))
        result['skipped'] = 'Not a Train cache; excluded from coverage'
        return result
    stored_protocol = config.get('protocol_sha256', config.get('train_protocol_sha256'))
    if stored_protocol and stored_protocol != protocol_hash:
        issues.add('cache_protocol_hash_mismatch', path)
    groups, coverage, seen, total, valid = defaultdict(lambda: {'sources':set(),'rows':0}), defaultdict(set), set(), 0, 0
    with manifest_path.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream,1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
                source, label, band = relative_id(row['source']), row['label'], row['band']
                if source not in records or type(label) is not int or label != records[source]['label']:
                    raise ValueError('Cache source/label disagrees with Train protocol')
                if records[source]['domain'] != 'offline':
                    raise ValueError('Noisy cache source is not explicit Offline')
                if type(band) is not int or band not in range(4) or (source,band) in seen:
                    raise ValueError('Invalid/duplicate cache band')
                seen.add((source,band))
                if row.get('role') != 'train' or row.get('generation') != config.get('generation'):
                    raise ValueError('Cache row role/generation disagrees with config')
                if not records[source].get('file_sha256') or row.get('source_sha256') != records[source]['file_sha256']:
                    raise ValueError('Cache source SHA256 differs from current audio')
                audio_id = relative_id(row['audio'])
                audio = (path/audio_id).resolve()
                if not audio.is_relative_to(path) or not audio.is_file():
                    raise ValueError('Cached audio missing or outside cache root')
                family = row.get('processing',{}).get('family','ffmpeg')
                allowed_families = config.get('processing',{}).get('families',['ffmpeg'])
                if not isinstance(family,str) or family not in allowed_families:
                    raise ValueError('Processing family disagrees with Train cache config')
                snr = float(row['snr_db'])
                if not math.isfinite(snr) or not (5+5*band <= snr <= 10+5*band):
                    raise ValueError('SNR does not match its recorded band')
                key = (records[source]['language_group'], CLASS_NAMES[label], family, band)
                groups[key]['sources'].add(source)
                groups[key]['rows'] += 1
                coverage[source].add(band)
                valid += 1
            except (ValueError, KeyError, TypeError) as exc:
                issues.add('invalid_noisy_cache_row', f'{path}:{number}', exc)
            if total % 10000 == 0:
                print(f'Cache metadata: {path.name}; rows={total}; valid={valid}', flush=True)
    expected = {source for source,r in records.items() if r['domain']=='offline'}
    missing = sorted(expected-set(coverage))
    incomplete = sorted(source for source,bands in coverage.items() if bands != set(range(4)))
    for source in incomplete:
        issues.add('incomplete_noisy_bands', source, str(path))
    # Missing source coverage is a finding, not necessarily a malformed file.
    result.update(rows=total, valid_rows=valid, covered_sources=len(coverage), expected_offline_sources=len(expected),
                  missing_source_ids=missing, incomplete_source_ids=incomplete,
                  coverage_by_group=[{'group':lang+'-'+cls, 'offline_protocol_sources':sum(records[x]['language_group']==lang and records[x]['class']==cls for x in expected),
                    'covered_sources':sum(records[x]['language_group']==lang and records[x]['class']==cls for x in coverage)}
                    for lang in ('en','zh','unknown') for cls in ('fake','real')],
                  conditions=[{'language_group':key[0],'class':key[1],'family':key[2],'band':key[3],
                               'views':value['rows'],'unique_sources':len(value['sources'])}
                              for key,value in sorted(groups.items())])
    return result


def summary_group(rows, components, paired):
    good = [r for r in rows if r['status']=='ok']
    durations = np.asarray([r['seconds'] for r in good])
    hashes = {r['file_sha256'] for r in rows if r.get('file_sha256')}
    return {'protocol_ids':len(rows), 'readable_audio':len(good), 'read_errors':len(rows)-len(good),
            'hashed_files':sum(bool(r.get('file_sha256')) for r in rows), 'unique_file_sha256':len(hashes),
            'known_source_components':len({components.find(r['source_id']) for r in good}),
            'explicit_pair_covered_ids':sum(r['source_id'] in paired for r in rows),
            'hours':float(durations.sum()/3600),
            'seconds_quantiles':dict(zip(('min','p10','median','p90','max'), np.quantile(durations,[0,.1,.5,.9,1]).tolist())) if len(good) else {},
            'shorter_than_crop':sum(r['seconds'] < CUT_SECONDS for r in good),
            'longer_than_crop':sum(r['seconds'] > CUT_SECONDS for r in good),
            'unused_tail_hours':sum(r['head_unused_seconds'] for r in good)/3600,
            'head_low_energy_files':sum(r['head']['active_frame_fraction'] < .1 for r in good),
            'low_head_active_middle_files':sum(r['head']['active_frame_fraction'] < .1 and r['middle']['active_frame_fraction'] >= .5 for r in good),
            'head_near_full_scale_files':sum(r['head']['near_full_scale_fraction'] > .01 for r in good),
            'sample_rates':dict(Counter(str(r['sample_rate']) for r in good)),
            'channels':dict(Counter(str(r['channels']) for r in good))}


def save_csv(path, rows):
    rows = list(rows)
    columns = list(dict.fromkeys(k for r in rows for k in r)) or ['note']
    def cell(value):
        if isinstance(value,(dict,list)):
            value = json.dumps(value,ensure_ascii=False)
        if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):
            return "'"+value
        return value
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:cell(v) for k,v in row.items()})


def report_markdown(summary):
    lines = ['# Train 四组数据核查', '',
             '本次只读取协议、录音和已有缓存元数据；未加载模型、未运行训练、未生成音频缓存。', '',
             '## 数据规模、去重和已有对应关系', '',
             '| 组 | 协议源 ID | 可读音频 | 文件字节去重 | 已知源组件 | 小时 | 中位时长/秒 | 短于4.0375秒 | 首段低能量 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['groups']:
        q=row['seconds_quantiles']
        median=f"{q['median']:.2f}" if q else '无'
        lines.append(f"| {row['group']} | {row['protocol_ids']} | {row['readable_audio']} | {row['unique_file_sha256']} | {row['known_source_components']} | {row['hours']:.2f} | {median} | {row['shorter_than_crop']} | {row['head_low_energy_files']} |")
    lines += ['', '“已知源组件”仅合并字节完全相同的文件及清单明确确认的 Offline/Online 对；不是说话人数，也不是已完全去重的独立录音数。不同编码/不同裁剪的相同录音未必能识别。组件在跨目录分组表中可能重复出现，不应把分组组件数相加当作全局唯一总数。', '',
              '## 首段与内容覆盖', '',
              '当前模型截取 4.0375 秒。此处读取原音频首段和中间片段：20ms 窗 RMS 高于 -50dBFS 定义为能量活跃；活跃窗少于10%标为首段低能量。这是音量启发式，不是语音活动检测或标签真值；噪声同样可能有能量。短音频不先重复填充。', '',
              'unused_tail_hours 是逐文件首段以外的总时长，可能包含静音及配对重复，不等于可新增的独立语音时长。`audio_flags.csv` 提供待核查文件；不要根据这些标记直接删除或改标签。', '',
              '## 现有训练缓存覆盖', '']
    for bank in summary['noisy_caches']:
        if 'rows' not in bank:
            lines.append(f"- {bank['path']}：无法统计或不是 Train 缓存。")
            continue
        lines.append(f"- {bank['path']}：有效视图 {bank['valid_rows']}/{bank['rows']}；覆盖 Offline 源 {bank['covered_sources']}/{bank['expected_offline_sources']}；缺失源 {len(bank['missing_source_ids'])}；缺档源 {len(bank['incomplete_source_ids'])}。")
    lines += ['', '分组×处理算法×SNR档位见 `cache_conditions.csv`。这些是库中可用视图，不等于训练时每轮实际抽到的次数；新库混合比例、配对轮转和类别采样还会影响实际暴露。仅检查处理后文件存在及源 hash/标签/角色/档位，没有解码或重建处理后缓存。', '',
              '## 完整性和解释边界', '',
              f"- 输入问题记录：{summary['issues']['total']}，按类型为 `{json.dumps(summary['issues']['counts'],ensure_ascii=False)}`；详情见 `issues.csv`。存在文件/标签/缓存异常时，应先处理异常再解读覆盖差异。",
              '- en/zh 来自路径中的独立目录名；没有可靠目录信息的记录归 unknown，不猜语种。来源目录见 `directories.csv`；本核查没有已核实的录音库/说话人/设备标注。',
              '- 完整文件 SHA256 用于字节级重复核查，不把相同文件误算为额外语音；异标签相同字节单独报告。没有跨 Train/Dev 的音频内容去重检查。',
              '- 每个源文件在读取前后检查大小和修改时间；配置/协议/清单在核查结束重新计算 hash。没有遍历 exp 或加载任何 checkpoint。',
              '- 当前日志的汇总 train recall 不能代表 en-real 等四组各自表现。本次四组 train recall 状态为未测量；不能从文件数推断“训练已经学会”。',
              '- 若 en-real 的独立来源和录音量确实少，应优先补来源覆盖；若量充足仍有低召回，应结合分组推理和具体错例检查内容/来源偏差，不能只重复增加同一批样本。', '',
              '文件：`summary.json` 完整汇总，`audio_inventory.jsonl` 每个源文件的时长/hash/能量统计，`duplicates.json` 重复与标签冲突，`rtc_pairs.json` 明确的配对覆盖，`cache_inventory.json` 缓存覆盖，`inputs.json` 输入指纹。ZIP 不包含音频或模型权重。', '']
    return '\n'.join(lines)


def resolve_config_path(value, project_root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root/path).resolve()


def run(args):
    started = time.monotonic()
    run_dir = Path(args.run_dir).expanduser().resolve()
    print('Starting CPU-only Train inventory; reading recorded paths and metadata.',flush=True)
    config_path = run_dir/'stage3'/'config.json'
    config = read_json(config_path)
    project = Path(args.project_root).expanduser().resolve()
    data_root = resolve_config_path(config['train_data_path'],project)
    protocol = resolve_config_path(config['train_protocol'],project)
    pair_path = resolve_config_path(config['rtc_pairs'],project) if config.get('rtc_pairs') else None
    bank_values = [config['train_noisy_cache']] if config.get('train_noisy_cache') else []
    extra = config.get('extra_train_noisy_cache') or []
    if isinstance(extra,str):
        extra=[extra]
    bank_paths = list(dict.fromkeys(resolve_config_path(value,project) for value in bank_values+extra))
    out = Path(args.out).expanduser().resolve()
    download = Path(args.download_dir).expanduser().resolve()
    protected = [data_root,run_dir]+bank_paths
    protected += [resolve_config_path(config[key],project) for key in ('dev_data_path','dev_noisy_cache','dev_heldout_cache','ssl_path','feature_cache') if config.get(key)]
    protected += [resolve_config_path(config[key],project).parent for key in ('baseline_path','finetune_from') if config.get(key)]
    if out == download or within(download,out) or any(within(out,p) or within(download,p) for p in protected):
        raise ValueError('Output/download directories must be separate from input data, runs, checkpoints and caches')
    if not data_root.is_dir() or not protocol.is_file():
        raise FileNotFoundError('Train data root or protocol is missing; check the chosen run config')
    if out.exists():
        raise FileExistsError('Use a NEW audit output directory: '+str(out))
    issues = Issues()
    metadata = [config_path,protocol]+([pair_path] if pair_path else [])
    metadata += [p/name for p in bank_paths for name in ('config.json','manifest.jsonl') if (p/name).is_file()]
    missing = [p for p in metadata if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing input metadata: '+str(missing))
    fingerprints = {}
    for p in metadata:
        print('Checking metadata: '+str(p),flush=True)
        fingerprints[str(p)]=sha256(p)
    for path,value in fingerprints.items():
        recorded = config.get('data_fingerprints',{}).get(path)
        if recorded and recorded != value:
            issues.add('recorded_run_fingerprint_mismatch',path)
    print('Indexing Train protocol; resolving source paths without loading a model.',flush=True)
    rows, protocol_count = protocol_rows(protocol,data_root,issues)
    print(f'TRAIN_ROOT={data_root}\nPROTOCOL={protocol}\nTrain IDs={len(rows)}; workers={args.workers}; GPU/model not used.',flush=True)
    print('Hashing original files and reading headers/head/middle segments. Progress prints every 10 seconds.',flush=True)
    out.mkdir(parents=True,exist_ok=False)
    download.mkdir(parents=True,exist_ok=True)
    inputs = {'schema':SCHEMA,'created_utc':datetime.now(timezone.utc).isoformat(),'source_run':str(run_dir),
              'metadata_sha256':fingerprints,'data_root':str(data_root),'cut_seconds':CUT_SECONDS,
              'noise_environment':config.get('noise_environment',{}),
              'training_settings':{key:config.get(key) for key in ('ordinary_sampling','class_counts','class_weights','real_ce_weight','noisy_extra_fraction','noisy_mix_warmup_epochs','noisy_bank_policy','algo')},
              'training_group_recall':'not_measured_no_model_inference'}
    write_json(out/'inputs.json',inputs)
    rows = scan_audio(rows,args.workers,out)
    records = {r['source_id']:r for r in rows}
    for row in rows:
        if row['status']!='ok':
            issues.add('source_audio_read_error',row['source_id'],row.get('error'))
    components = Components(records)
    dup = duplicates(rows,components,issues)
    pairs = pair_inventory(pair_path,records,components,issues)
    paired = pairs.pop('covered_ids',set())
    pairs['covered_ids']=sorted(paired)
    print('Reading explicit RTC pairs and existing TRAIN cache metadata; no cache generation.',flush=True)
    caches = [cache_inventory(path,records,fingerprints[str(protocol)],issues) for path in bank_paths]
    groups = [{'group':lang+'-'+cls, **summary_group([r for r in rows if r['language_group']==lang and r['class']==cls],components,paired)}
              for lang in ('en','zh','unknown') for cls in ('fake','real') if lang != 'unknown' or any(r['language_group']==lang and r['class']==cls for r in rows)]
    directory_groups = defaultdict(list)
    for row in rows:
        directory_groups[(row['source_directory'],row['class'])].append(row)
    directories = [{'directory':key[0],'class':key[1],**summary_group(value,components,paired)} for key,value in sorted(directory_groups.items())]
    for path,value in fingerprints.items():
        if sha256(path)!=value:
            issues.add('metadata_changed_during_audit',path)
    current_counts = [sum(r['label']==label for r in rows) for label in (0,1)]
    if config.get('class_counts') is not None and config['class_counts']!=current_counts:
        issues.add('recorded_class_counts_mismatch','',f"recorded={config['class_counts']}, current={current_counts}")
    summary = {'schema':SCHEMA,'status':'complete_with_input_issues' if issues.counts else 'complete',
               'protocol_lines':protocol_count,'unique_protocol_ids':len(rows),'class_counts':current_counts,
               'groups':groups,'all':summary_group(rows,components,paired),'noisy_caches':caches,
               'issues':issues.result(),'seconds':time.monotonic()-started,
               'training_group_recall':'not_measured_no_model_inference',
               'speaker_identity':'not_available_directory_groups_only'}
    write_json(out/'summary.json',summary)
    write_json(out/'duplicates.json',dup)
    write_json(out/'rtc_pairs.json',pairs)
    write_json(out/'cache_inventory.json',caches)
    save_csv(out/'groups.csv',groups)
    save_csv(out/'directories.csv',directories)
    save_csv(out/'issues.csv',issues.examples)
    flags = [dict(r) for r in rows if r['status']!='ok' or r['head']['active_frame_fraction'] < .1 or r['head']['near_full_scale_fraction'] > .01 or r['seconds'] < 1.]
    save_csv(out/'audio_flags.csv',flags)
    save_csv(out/'cache_conditions.csv',({'bank':bank['path'],**row} for bank in caches for row in bank.get('conditions',[])))
    (out/'report.md').write_text(report_markdown(summary),encoding='utf-8')
    files = sorted(p for p in out.iterdir() if p.is_file())
    write_json(out/'file_hashes.json',{p.name:sha256(p) for p in files})
    archive = download/(out.name+'.zip')
    print('Saving report ZIP (metadata only).',flush=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():
                z.write(p,out.name+'/'+p.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Report ZIP CRC check failed')
    completed = {'status':summary['status'],'archive':str(archive),'archive_sha256':sha256(archive),
                 'seconds':time.monotonic()-started,'input_issues':summary['issues']['total']}
    write_json(out/'completed.json',completed)
    print(f"STATUS={summary['status']}\nINPUT_ISSUES={summary['issues']['total']}\nREPORT={out/'report.md'}\nDOWNLOAD_ZIP={archive}\nAUDIT_COMPLETE=True",flush=True)
    if getattr(args,'upload_temp',False):
        print('Uploading the report ZIP to temp.sh (no audio or checkpoints).',flush=True)
        try:
            link=upload_report(archive)
            (out/'temp_download_url.txt').write_text(link+'\n',encoding='utf-8')
            completed['temp_download_url']=link
            print('TEMP_DOWNLOAD_URL='+link+'\nUPLOAD_COMPLETE=True',flush=True)
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            completed['upload_error']=str(exc)
            print('UPLOAD_COMPLETE=False\nReport ZIP is ready locally; upload may be retried.\n'+str(exc),flush=True)
        write_json(out/'completed.json',completed)
    return summary


def upload_report(archive):
    result=subprocess.run(['curl','--fail','--show-error','--silent','--max-time','180',
                           '-F','file=@'+str(archive),'https://temp.sh/upload'],
                          capture_output=True,text=True,timeout=200,check=True)
    link=result.stdout.strip()
    parsed=urlsplit(link)
    if parsed.scheme!='https' or parsed.netloc!='temp.sh' or not parsed.path or any(c.isspace() for c in link):
        raise ValueError('Unexpected temp.sh response; local report ZIP is preserved')
    return link


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--project-root',default=str(ROOT))
    parser.add_argument('--out',required=True)
    parser.add_argument('--download-dir',default=str(Path.home()/'LXT'/'temp'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--upload-temp',action='store_true',help='Upload report ZIP to temp.sh and print its download link')
    args=parser.parse_args()
    if not 1<=args.workers<=32:
        parser.error('--workers must be between 1 and 32')
    run(args)


if __name__=='__main__':
    main()
