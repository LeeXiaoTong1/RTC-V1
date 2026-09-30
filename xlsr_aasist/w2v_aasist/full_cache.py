"""Two full-duration Train views, with immutable recipes and resumable generation."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
import soundfile as sf

from rtc_noisy.common import noise_catalog
from rtc_noisy.simulator import LocalRTC, RTCSettings
from rtc_noisy_v2.plan import BANDS, PLAN_ID, SCHEMA, plan_definition, settings_for, stable_seed
from utils.env_noise import mix_at_snr
from .runtime import atomic_json, sha256
from .progress import progress

FORMAT = 'rtc_noisy_full2_cache_v1'


def assignments(records, seed):
    groups = defaultdict(list)
    for row in records:
        if row['domain'] == 'offline':
            groups[(row['language'], row['label'])].append(row['id'])
    allowed = settings_for('train')
    result = {}
    for group, ids in sorted(groups.items()):
        ids = sorted(ids)
        random.Random(stable_seed(seed, 'source-order', *group)).shuffle(ids)
        settings = list(allowed)
        random.Random(stable_seed(seed, 'setting-order', *group)).shuffle(settings)
        for rank, name in enumerate(ids):
            # Both severity ranges for every source; balanced sub-bands/settings within each group.
            result[name] = [{'version': v, 'band': 2*v + rank % 2,
                             'rtc': RTCSettings(*settings[(rank+7*v) % len(settings)]).as_dict()}
                            for v in (0, 1)]
    return result


def continuous_mix(augment, durations, waveform, rng, snr):
    # One contiguous recording covers the utterance: no speech/noise tiling or padded silent tail.
    eligible = [i for i, seconds in enumerate(durations) if seconds*16000 >= len(waveform)]
    if not eligible:
        raise ValueError(f'No existing Train noise recording covers {len(waveform)/16000:.2f}s')
    for _ in range(40):
        record = augment.records[eligible[int(rng.randint(len(eligible)))]]
        noise = augment._read_noise(record['path'], 16000)
        if len(noise) < len(waveform):
            continue
        start = int(rng.randint(len(noise)-len(waveform)+1))
        try:
            wave, info = mix_at_snr(waveform, noise[start:start+len(waveform)], 16000, snr)
        except ValueError as exc:
            if 'Noise is silent' in str(exc):
                continue
            raise
        if not info.get('applied'):
            raise ValueError('Source contains no active speech')
        return wave, dict(info, noise_path=record['path'], offset=start, contiguous_full_length=True)
    raise ValueError('No active full-duration noise excerpt after 40 draws')


def read_index(folder, role, protocol, official_records, verify_audio_hash=False):
    from .data import safe_audio_path
    from rtc_noisy_v2.cache import check_metadata
    folder = Path(folder).resolve()
    cfg = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    done = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    if (role != 'train' or cfg.get('format') != FORMAT or cfg.get('role') != 'train'
            or cfg.get('split') != 'train' or cfg.get('views_per_source') != 2
            or cfg.get('sr') != 16000 or cfg.get('subtype') != 'FLOAT' or cfg.get('cut') is not None
            or cfg.get('protocol_sha256') != sha256(protocol) or cfg.get('limit') != 0
            or done.get('config_sha256') != sha256(folder/'config.json')
            or done.get('manifest_sha256') != sha256(folder/'manifest.jsonl')):
        raise ValueError('Full noisy cache is incomplete or has an incompatible role/recipe')
    official = {r['id']: r for r in official_records if r['domain'] == 'offline'}
    expected = assignments(official.values(), cfg['seed'])
    rows, raw_rows, seen = [], [], set()
    with (folder/'manifest.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            source, version = r['source'], r['version']
            if source not in official or type(version) is not int or version not in (0,1) or (source,version) in seen:
                raise ValueError('Duplicate/unknown full noisy source or version')
            assignment = expected[source][version]
            if (r['label'] != official[source]['label'] or r['rtc'] != assignment['rtc']
                    or r['band'] != assignment['band'] or not BANDS[r['band']][0] <= r['snr_db'] <= BANDS[r['band']][1]
                    or r.get('output_samples') != r.get('source_samples') or r['source_samples'] <= 0
                    or not r.get('noise',{}).get('contiguous_full_length')):
                raise ValueError('Full noisy label, condition or duration differs from recipe')
            audio = safe_audio_path(folder, r['audio'])
            header = sf.info(audio)
            original = sf.info(official[source]['audio'])
            if (header.frames != r['output_samples'] or header.samplerate != 16000 or header.channels != 1
                    or header.subtype != 'FLOAT' or original.frames != r['source_samples']
                    or original.samplerate != 16000 or original.channels != 1):
                raise ValueError('Full noisy/source audio header differs from manifest')
            if verify_audio_hash and sha256(audio) != r['audio_sha256']:
                raise ValueError('Full noisy cache contents changed')
            rows.append(dict(official[source], audio=str(audio), source_audio=official[source]['audio'],
                             source_sha256=r['source_sha256'], noisy=True, full_length=True,
                             version=version, output_samples=r['output_samples'], band=r['band'],
                             bank=str(folder), processing_family='ffmpeg'))
            raw_rows.append(r)
            seen.add((source,version))
            if len(rows) % 20000 == 0:
                print(f'Checking full noisy cache: {len(rows)}/{2*len(official)}', flush=True)
    if (len(rows) != 2*len(official) or cfg['offline_count'] != len(official)
            or done.get('rows') != len(rows)):
        raise ValueError('Every Offline Train source must have exactly two complete views')
    check_metadata(cfg, raw_rows, 'train')
    return rows, (raw_rows, cfg)


def prepare(source, output, noise_manifest, ffmpeg, workers=4, seed=1234):
    from .data import read_protocol, read_wave
    from .launch import run_lock
    from rtc_noisy_v2.cache import check_suite
    output = Path(output).resolve()
    protocol = source['train_protocol']
    ordinary = read_protocol(protocol, source['train_data_path'])
    offline = [r for r in ordinary if r['domain'] == 'offline']
    if not offline:
        raise ValueError('No Offline Train sources')
    for key in ('train_data_path','dev_data_path','dev_noisy_cache','dev_heldout_cache'):
        protected = Path(source[key]).resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Cache output overlaps a protected input directory')
    choices = assignments(ordinary, seed)
    print('Checking existing Train noise recordings and hashes...',flush=True)
    augment, catalog = noise_catalog(noise_manifest, 'train')
    rtc = LocalRTC(ffmpeg)
    durations = [sf.info(r['path']).duration for r in augment.records]
    counts, frames = {}, 0
    for row in progress(offline,total=len(offline),label='Train source headers',every=2000):
        info = sf.info(row['audio'])
        if info.samplerate != 16000 or info.channels != 1 or info.frames < 1:
            raise ValueError('Full-cache recipe requires the audited 16 kHz mono Train sources')
        if info.duration > max(durations):
            raise ValueError('No Train noise recording covers source ' + row['id'])
        counts[row['id']] = info.frames
        frames += info.frames
    longest = max(counts.values()) / 16000
    common_pool = [(record,duration) for record,duration in zip(augment.records,durations) if duration >= longest]
    augment.records = [record for record,_ in common_pool]
    durations = [duration for _,duration in common_pool]
    print(f'CONTINUOUS_TRAIN_NOISE_POOL={len(common_pool)} LONGEST_SOURCE_SECONDS={longest:.3f}',flush=True)
    cfg = dict(format=FORMAT, optimization_schema=SCHEMA, role='train', split='train',
               seed=seed, generation=0, plan_id=PLAN_ID, plan=plan_definition(),
               allowed_settings=[list(x) for x in settings_for('train')],
               protocol_sha256=sha256(protocol), noise=catalog, cut=None, sr=16000, subtype='FLOAT',
               snr_bands=[list(x) for x in BANDS], limit=0, offline_count=len(offline), views_per_source=2,
               ffmpeg_version=rtc.version, engine='FFmpeg afftdn + dynaudnorm + libopus (NOT WebRTC)',
               order='full source -> contiguous background noise -> one fixed RTC setting -> full FLOAT WAV',
               assignment='v0: 5-15 dB; v1: 15-25 dB; balanced sub-bands/settings within language and class',
               noise_selection='shared Train recording pool long enough for the longest source; identical pool for all labels/languages',
               continuous_noise_recordings=len(common_pool), longest_source_seconds=longest,
               generator_sha256=sha256(__file__), simulator_sha256=sha256(Path(__file__).parents[1]/'rtc_noisy/simulator.py'),
               noise_code_sha256=sha256(Path(__file__).parents[1]/'utils/env_noise.py'))
    # Reject Train/Dev noise leakage or a different processing executable BEFORE allocating the cache.
    for key in ('dev_noisy_cache','dev_heldout_cache'):
        devcfg = json.loads((Path(source[key])/'config.json').read_text(encoding='utf-8'))
        if cfg['ffmpeg_version'] != devcfg['ffmpeg_version']:
            raise ValueError('Use the same FFmpeg executable/version as the fixed Dev caches')
        for field in ('recording_ids','file_sha256'):
            if set(catalog[field]) & set(devcfg['noise'][field]):
                raise ValueError('Train noise overlaps fixed Dev noise')
    probe = np.random.RandomState(3).normal(0,.05,70321).astype(np.float32)
    for settings in settings_for('train'):
        rtc(probe, RTCSettings(*settings))
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output/'.prepare.lock'):
        config_file = output/'config.json'
        if config_file.exists():
            if json.loads(config_file.read_text(encoding='utf-8')) != cfg:
                raise ValueError('Existing cache has a different recipe; use a new directory')
        elif any(p.name != '.prepare.lock' for p in output.iterdir()):
            raise ValueError('Nonempty unrecognized cache directory')
        else:
            atomic_json(config_file, cfg)
        estimate = frames*4*2 + len(offline)*2*4096
        existing = sum(p.stat().st_size for p in (output/'audio').rglob('*.wav')) if (output/'audio').exists() else 0
        required = max(0, estimate-existing) + 2*1024**3
        print(f'FULL_NOISY_SOURCES={len(offline)} VIEWS={2*len(offline)} ESTIMATED_GiB={estimate/1024**3:.2f}', flush=True)
        if shutil.disk_usage(output).free < required:
            raise OSError(f'Need {required/1024**3:.2f} GiB free to finish new cache; old caches were not deleted')
        def generate(row):
            name, path = row['id'], Path(row['audio'])
            before = path.stat()
            source_hash, wave = sha256(path), read_wave(path)
            if len(wave) != counts[name]:
                raise ValueError('Source length changed')
            key = hashlib.sha256(name.encode()).hexdigest()[:24]
            folder = output/'audio'/key[:2]
            folder.mkdir(parents=True, exist_ok=True)
            result = []
            for a in choices[name]:
                v, band = a['version'], a['band']
                audio = folder/f'{key}_v{v}.wav'
                sidecar = audio.with_suffix('.json')
                if audio.exists() and sidecar.exists():
                    saved = json.loads(sidecar.read_text(encoding='utf-8'))
                    if (saved.get('source_sha256') == source_hash and saved.get('source') == name
                            and saved.get('version') == v and saved.get('band') == band
                            and saved.get('rtc') == a['rtc'] and saved.get('label') == row['label']
                            and saved.get('output_samples') == len(wave) and saved.get('audio_sha256') == sha256(audio)):
                        result.append(saved)
                        continue
                rng = np.random.RandomState(stable_seed(seed,'full2',name,v) % 2**32)
                snr = float(rng.uniform(*BANDS[band]))
                mixed, info = continuous_mix(augment, durations, wave, rng, snr)
                processed = rtc(mixed, RTCSettings(**a['rtc']))
                if len(processed) != len(wave) or not np.isfinite(processed).all():
                    raise ValueError('Processing changed full source length or produced nonfinite output')
                if shutil.disk_usage(output).free < processed.nbytes + 1024**3:
                    raise OSError('Cache free-space floor reached; resume after freeing space')
                tmp = audio.with_suffix('.tmp.wav')
                try:
                    sf.write(tmp, processed, 16000, subtype='FLOAT')
                    os.replace(tmp, audio)
                finally:
                    if tmp.exists(): tmp.unlink()
                saved = dict(a, source=name, source_sha256=source_hash, label=row['label'],
                             role='train', generation=0, snr_db=snr, noise=info,
                             source_samples=len(wave), output_samples=len(processed),
                             mix_id=hashlib.sha256(mixed.astype('<f4').tobytes()).hexdigest(),
                             audio=audio.relative_to(output).as_posix(), audio_sha256=sha256(audio))
                atomic_json(sidecar, saved)
                result.append(saved)
            after = path.stat()
            if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                raise RuntimeError('Train source changed during generation')
            return result
        if (output/'complete.json').exists():
            (output/'complete.json').unlink()
        temporary = output/'manifest.jsonl.tmp'
        with ThreadPoolExecutor(max_workers=workers) as pool, temporary.open('w',encoding='utf-8',newline='\n') as stream:
            for records in progress(pool.map(generate,offline), total=len(offline), label='Full noisy sources', every=100):
                for row in records:
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        os.replace(temporary, output/'manifest.jsonl')
        atomic_json(output/'complete.json', dict(rows=2*len(offline), config_sha256=sha256(config_file),
                                               manifest_sha256=sha256(output/'manifest.jsonl')))
        try:
            read_index(output,'train',protocol,ordinary)
        except Exception:
            (output/'complete.json').unlink()
            raise
        print('FULL_NOISY_CACHE_COMPLETE=True', flush=True)
    return output
