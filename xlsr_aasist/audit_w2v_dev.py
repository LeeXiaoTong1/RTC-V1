"""Read-only, resumable paired Dev inference; never instantiate a training bundle."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from w2v_rebuild.core import Metrics, atomic_json, load_checkpoint, sha256
from w2v_rebuild.data import FeatureCollator, worker_init
from w2v_rebuild.feature_cache import FeatureCache
from w2v_rebuild.model import Detector, forward_chunks

ROOT = Path(__file__).resolve().parent
FORMAT = 'w2v_paired_dev_audit_v1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ReadOnlyFeatureCache(FeatureCache):
    def publish(self, path, value):
        # Missing features are recomputed in RAM using the unchanged extractor.
        # Crucially, neither the cache implementation nor its namespace changes.
        pass


class ReadOnlyCollator(FeatureCollator):
    def __call__(self, rows):
        if self.extractor is None:
            from transformers import AutoFeatureExtractor
            self.extractor = AutoFeatureExtractor.from_pretrained(self.directory, local_files_only=True)
            if self.extractor.sampling_rate != 16000:
                raise ValueError('Expected the original 16 kHz extractor')
        if self.cache_root and self.cache is None:
            self.cache = ReadOnlyFeatureCache(self.cache_root, self.directory, self.extractor)
        return super().__call__(rows)


def input_fingerprints(config):
    paths = [Path(config['dev_protocol']), Path(config['ssl_path'])/'preprocessor_config.json',
             Path(config['ssl_path'])/'config.json']
    for name in ('dev_noisy_cache', 'dev_heldout_cache'):
        paths.extend(Path(config[name])/f for f in ('config.json', 'manifest.jsonl'))
    expected = config['data_fingerprints']
    result = {}
    for path in paths:
        key = str(path.resolve())
        value = sha256(path)
        if expected.get(key) != value:
            raise ValueError('Dev input differs from the completed run: '+key)
        result[key] = value
    return result


def check_dev_pair(seen, held):
    a, ca = seen
    b, cb = held
    def mapping(rows):
        return {(r['source'], r['band']): (r['label'], r['source_sha256'], r['mix_id']) for r in rows}
    if mapping(a) != mapping(b) or ca['noise'] != cb['noise'] or ca['ffmpeg_version'] != cb['ffmpeg_version']:
        raise ValueError('Seen/heldout are not the same fixed sources/noise/SNR')
    hf = set(cb.get('processing', {}).get('families', []))
    sf = set(ca.get('processing', {}).get('families', ['ffmpeg']))
    if hf & sf:
        raise ValueError('Heldout processing families overlap seen')


def base_record(condition, source, label, path, band=-1):
    stat = Path(path).stat()
    return dict(condition=condition, source_id=source, label=int(label), band=band,
                audio_path=str(path), source_directory=str(Path(source).parent), snr_db='',
                processing_family='clean', processing_json='{}', rtc_json='{}', mix_id='',
                audio_size=stat.st_size, audio_mtime_ns=stat.st_mtime_ns)


def dev_streams(config):
    from utils.data_utils import build_dataset_from_protocol
    from rtc_noisy.common import audio_domain
    from rtc_noisy.data import NoisyDevDataset
    from rtc_noisy_v2.cache import load_v2_cache
    clean, ids, labels = build_dataset_from_protocol(config['dev_protocol'], config['dev_data_path'], mode='dev')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate Dev IDs')
    records = [base_record(audio_domain(i), i, labels[i], Path(config['dev_data_path'])/i) for i in ids]
    streams = [('clean', clean, 'ordinary', records)]
    banks = []
    for condition, key, role in [('seen', 'dev_noisy_cache', 'dev_seen'),
                                 ('heldout', 'dev_heldout_cache', 'dev_heldout')]:
        print('Checking existing '+condition+' Dev cache (no generation).', flush=True)
        bank = load_v2_cache(config[key], role, config['dev_protocol'], config['dev_data_path'])
        banks.append(bank)
        dataset = NoisyDevDataset(bank[0])
        metadata = []
        for r in dataset.rows:  # EXACT order used by the original validation loader.
            row = base_record(condition, r['source'], r['label'], r['audio'], r['band'])
            row.update(snr_db=r['snr_db'], processing_family=r.get('processing', {}).get('family', 'ffmpeg'),
                       processing_json=json.dumps(r.get('processing', {}), sort_keys=True),
                       rtc_json=json.dumps(r['rtc'], sort_keys=True), mix_id=r['mix_id'],
                       noise_json=json.dumps(r.get('noise', {}), sort_keys=True),
                       source_sha256=r['source_sha256'],
                       source_samples=r.get('source_samples', ''),
                       output_samples_before_crop=r.get('output_samples_before_crop', ''))
            metadata.append(row)
        streams.append((condition, dataset, 'noisy_dev', metadata))
    check_dev_pair(*banks)
    return streams


def check_audio_unchanged(records):
    for row in records:
        stat = Path(row['audio_path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (row['audio_size'], row['audio_mtime_ns']):
            raise ValueError('Dev waveform changed during audit: '+row['audio_path'])


def loader_for(dataset, kind, config, device, workers):
    options = dict(dataset=dataset, batch_size=config.get('eval_batch', 16), shuffle=False,
                   num_workers=workers, pin_memory=device.type == 'cuda',
                   collate_fn=ReadOnlyCollator(config['ssl_path'], kind, config.get('feature_cache')))
    if workers:
        options.update(multiprocessing_context='spawn', worker_init_fn=worker_init, prefetch_factor=2)
    return DataLoader(**options)


def row_key(row):
    return [row['condition'], row['source_id'], row['band'], row['label']]


def reuse_scores(path, records, identity):
    marker = path.with_suffix('.done.json')
    if not marker.is_file():
        return None
    meta = read_json(marker)
    if meta.get('identity') != identity or meta.get('sha256') != sha256(path):
        raise ValueError('Stored scores changed or belong to a different audit: '+str(path))
    scores = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    if len(scores) != len(records) or any(s.pop('key') != row_key(r) for s, r in zip(scores, records)):
        raise ValueError('Stored score IDs/order/labels differ')
    return scores


@torch.inference_mode()
def infer_stream(model, loader, records, path, identity, microbatch, device, description):
    cached = reuse_scores(path, records, identity)
    if cached is not None:
        print('Reusing completed '+description, flush=True)
        return cached
    result, offset = [], 0
    partial = path.with_suffix('.partial')
    with partial.open('w', encoding='utf-8') as stream:
        for batch in tqdm(loader, total=len(loader), desc=description, mininterval=5):
            take = len(batch['labels'])
            meta = records[offset:offset+take]
            if batch['labels'].tolist() != [r['label'] for r in meta]:
                raise ValueError('Data/metadata label order differs')
            if 'ids' in batch and batch['ids'] != [r['source_id'] for r in meta]:
                raise ValueError('Clean Dev ID order differs')
            if 'bands' in batch and batch['bands'] != [r['band'] for r in meta]:
                raise ValueError('Noisy Dev band order differs')
            # Match train.validate: FP32, eval mode, fixed padded microbatch.
            logits, _ = forward_chunks(model, batch['features'].to(device, non_blocking=True),
                                       batch['mask'].to(device, non_blocking=True), microbatch, pad_last=True)
            logits = logits.float().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite Dev logits')
            probabilities = logits.softmax(1)[:, 0].tolist()
            for r, z, probability in zip(meta, logits.tolist(), probabilities):
                score = dict(logit_fake=z[0], logit_real=z[1], pfake=probability, margin=z[0]-z[1])
                result.append(score)
                stream.write(json.dumps(dict(key=row_key(r), **score))+'\n')
            stream.flush()
            offset += take
    if offset != len(records):
        raise ValueError('Dev stream ended early')
    partial.replace(path)
    atomic_json(dict(identity=identity, count=offset, sha256=sha256(path)), path.with_suffix('.done.json'))
    return result


def replay_metrics(records, scores):
    report = {}
    for kind in ('online', 'offline', 'seen', 'heldout'):
        bands = range(4) if kind in ('seen', 'heldout') else [-1]
        parts = []
        for band in bands:
            indices = [i for i, r in enumerate(records) if r['condition'] == kind and r['band'] == band]
            meter = Metrics()
            meter.update(torch.tensor([[scores[i]['logit_fake'], scores[i]['logit_real']] for i in indices]),
                         torch.tensor([records[i]['label'] for i in indices]))
            parts.append(meter.result())
        report[kind] = parts[0] if len(parts) == 1 else dict(bands=parts,
            macro_f1=sum(x['macro_f1'] for x in parts)/4,
            balanced_ce=sum(x['balanced_ce'] for x in parts)/4)
    report['robust_f1'] = sum(w*report[k]['macro_f1'] for k, w in [('online', .3), ('seen', .35), ('heldout', .35)])
    report['robust_ce'] = sum(w*report[k]['balanced_ce'] for k, w in [('online', .3), ('seen', .35), ('heldout', .35)])
    return report


def metric_parity(actual, expected):
    mismatches = []
    for kind in ('online', 'offline', 'seen', 'heldout'):
        left = actual[kind].get('bands', [actual[kind]])
        right = expected[kind].get('bands', [expected[kind]])
        for i, (a, b) in enumerate(zip(left, right)):
            if a['confusion'] != b['confusion']:
                mismatches.append(f'{kind}/{i}: confusion differs')
            if abs(a['balanced_ce']-b['balanced_ce']) > 1e-5:
                mismatches.append(f'{kind}/{i}: balanced CE differs by more than 1e-5')
    return dict(matched=not mismatches, mismatches=mismatches,
                note='Mismatch requires checking environment/data before drawing a performance conclusion.')


@contextmanager
def audit_lock(out):
    with (out/'audit.lock').open('a+b') as stream:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def package_report(out, download_dir):
    if download_dir == out or out in download_dir.parents:
        raise ValueError('Download directory must be outside the audit report directory')
    download_dir.mkdir(parents=True, exist_ok=True)
    archive = download_dir/(out.name+'.zip')
    if archive.exists():
        raise FileExistsError('Download already exists; keep it or use another download directory: '+str(archive))
    partial = archive.with_suffix('.zip.partial')
    # Explicit allowlist: never archive audio, feature cache or checkpoint files.
    allowed = {'.json', '.jsonl', '.csv', '.md', '.log', '.txt'}
    files = [p for p in sorted(out.iterdir()) if p.is_file() and p.suffix in allowed and p.name != 'file_hashes.json']
    inventory = {p.name: sha256(p) for p in files}
    atomic_json(inventory, out/'file_hashes.json')
    files.append(out/'file_hashes.json')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, out.name+'/'+path.name)
    with zipfile.ZipFile(partial) as z:
        if z.testzip() is not None:
            raise RuntimeError('Download ZIP failed integrity check')
    partial.replace(archive)
    checksum = sha256(archive)
    archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n', encoding='ascii')
    return archive, checksum


def run(args):
    stage = Path(args.run_dir).expanduser().resolve()
    if stage.name != 'stage3':
        stage = stage/'stage3'
    config = read_json(stage/'config.json')
    completed = read_json(stage/'completed.json')
    if config.get('stage') != 3 or completed.get('candidate_epoch') != 1:
        raise ValueError('This audit expects the completed Stage3 epoch-1 candidate')
    baseline = Path(config['baseline_path']).expanduser().resolve()
    candidate = stage/'candidate_best.pt'
    if baseline == candidate or not baseline.is_file() or not candidate.is_file():
        raise ValueError('Need the original baseline and the separate epoch-1 candidate')
    out = Path(args.out).expanduser().resolve()
    protected = [stage.parent, baseline.parent, Path(config['dev_data_path']).resolve(),
                 Path(config['dev_noisy_cache']).resolve(), Path(config['dev_heldout_cache']).resolve()]
    if config.get('feature_cache'):
        protected.append(Path(config['feature_cache']).resolve())
    for directory in protected:
        if out == directory or directory in out.parents:
            raise ValueError('Audit output must be separate from input experiments/data/caches')
    if args.resume:
        if not (out/'manifest.json').is_file():
            raise FileNotFoundError('Resume requires an existing audit manifest')
    else:
        out.mkdir(parents=True, exist_ok=False)
    with audit_lock(out):
        run_locked(args, config, stage, baseline, candidate, out)


def run_locked(args, config, stage, baseline, candidate, out):
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no silent CPU fallback')
    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    print('Checking checkpoint hashes and fixed Dev inputs; this can take several minutes.', flush=True)
    hashes = dict(baseline=sha256(baseline), candidate=sha256(candidate))
    if hashes['baseline'] != config.get('init_sha256'):
        raise ValueError('Original best differs from the adaptation initialization SHA256')
    inputs = input_fingerprints(config)
    streams = dev_streams(config)
    records = [row for _, _, _, rows in streams for row in rows]
    microbatch = config.get('eval_microbatch', 4)
    manifest = dict(format=FORMAT, stage=str(stage), checkpoint_paths=dict(baseline=str(baseline), candidate=str(candidate)),
        checkpoint_sha256=hashes, input_sha256=inputs, records_sha256=digest(records), count=len(records),
        eval_microbatch=microbatch, eval_batch=config.get('eval_batch', 16), arithmetic='FP32 as train.validate',
        device=str(device), torch_version=str(torch.__version__),
        feature_cache=config.get('feature_cache'), cache_policy='read-only; misses recomputed in RAM',
        source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__),
            ROOT/'w2v_rebuild/model.py', ROOT/'w2v_rebuild/data.py',
            ROOT/'w2v_rebuild/core.py', ROOT/'w2v_rebuild/feature_cache.py']})
    if args.resume and read_json(out/'manifest.json') != manifest:
        raise ValueError('Resume inputs/code/checkpoints/runtime differ; use a new output directory')
    atomic_json(manifest, out/'manifest.json')
    atomic_json(records, out/'records.json')
    shutil.copyfile(stage/'baseline_dev.json', out/'baseline_recorded_dev.json')
    shutil.copyfile(stage/'epoch_001_evaluation.json', out/'candidate_recorded_evaluation.json')
    if (stage/'metrics.jsonl').is_file():
        shutil.copyfile(stage/'metrics.jsonl', out/'training_metrics.jsonl')
    all_scores, parity = {}, {}
    for tag, checkpoint in [('baseline', baseline), ('candidate', candidate)]:
        paths = [out/f'{tag}_{name}_scores.jsonl' for name, _, _, _ in streams]
        identities = [digest([manifest, tag, name]) for name, _, _, _ in streams]
        reused = [reuse_scores(p, rows, identity) for p, (_, _, _, rows), identity in zip(paths, streams, identities)]
        model = None
        if any(scores is None for scores in reused):
            print(f'Loading {tag}: {checkpoint}', flush=True)
            ckpt = load_checkpoint(checkpoint)
            if ckpt['stage'] != 3 or (tag == 'candidate' and (ckpt.get('epoch') != 1 or ckpt.get('kind') != 'weights')):
                raise ValueError('Unexpected checkpoint stage/kind/epoch')
            if ckpt['model_config'] != config['model_config']:
                raise ValueError('Checkpoint architecture differs from the recorded run')
            for path, value in inputs.items():
                # Baseline predates the current noisy Dev bank: require its original
                # Dev protocol/extractor, but use CURRENT fixed Dev for both models.
                check = tag == 'candidate' or path in {
                    str(Path(config['dev_protocol']).resolve()),
                    str((Path(config['ssl_path'])/'preprocessor_config.json').resolve()),
                    str((Path(config['ssl_path'])/'config.json').resolve())}
                if check and ckpt['data_fingerprints'].get(path) != value:
                    raise ValueError('Checkpoint/input fingerprint mismatch: '+path)
            model = Detector.load(config['ssl_path'], ckpt['model_config'], checkpointing=False)
            model.load_state_dict(ckpt['model'], strict=True)
            del ckpt
            gc.collect()
            model.requires_grad_(False).to(device).eval()
        scores = []
        for (name, dataset, kind, rows), path, identity, prior in zip(streams, paths, identities, reused):
            if prior is None:
                loader = loader_for(dataset, kind, config, device, args.workers)
                prior = infer_stream(model, loader, rows, path, identity, microbatch, device, tag+' '+name)
                del loader
            else:
                print('Reusing completed '+tag+' '+name, flush=True)
            scores.extend(prior)
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        all_scores[tag] = scores
        actual = replay_metrics(records, scores)
        expected = read_json(out/'baseline_recorded_dev.json') if tag == 'baseline' else read_json(out/'candidate_recorded_evaluation.json')['dev']
        parity[tag] = metric_parity(actual, expected)
        atomic_json(actual, out/f'{tag}_replayed_dev.json')
        print(tag+' recorded-metric parity: '+str(parity[tag]['matched']), flush=True)
    print('Writing per-sample comparisons and grouped reports.', flush=True)
    from audit_w2v_report import build_report
    build_report(records, all_scores['baseline'], all_scores['candidate'], out, bootstrap=args.bootstrap)
    atomic_json(parity, out/'metric_parity.json')
    with (out/'report.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\n## 原日志复核\n\n')
        for tag, result in parity.items():
            stream.write(f'- {tag}: '+('混淆矩阵及 CE 与原日志一致。' if result['matched'] else
                '存在差异，请先检查环境或数据：'+ '; '.join(result['mismatches']))+'\n')
    check_audio_unchanged(records)
    if input_fingerprints(config) != inputs or any(sha256(p) != hashes[tag] for tag, p in [('baseline', baseline), ('candidate', candidate)]):
        raise RuntimeError('An input changed during inference; do not use this report')
    atomic_json(dict(status='complete', seconds=time.perf_counter()-started, original_best_preserved=True,
                     candidate_preserved=True, metric_parity=parity), out/'completed.json')
    if args.log_file and Path(args.log_file).is_file():
        sys.stdout.flush(); sys.stderr.flush()
        shutil.copyfile(args.log_file, out/'run.log')
    archive, checksum = package_report(out, Path(args.download_dir).expanduser().resolve())
    print(f'REPORT_DIR={out}\nDOWNLOAD_ZIP={archive}\nSHA256={checksum}\nAUDIT_COMPLETE=True', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--out', default=str(ROOT/'exp'/('w2v_dev_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))))
    parser.add_argument('--download-dir', default=str(Path.home()/'LXT'/'temp'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=400)
    parser.add_argument('--resume', action='store_true', help='Reuse completed model/condition score files in --out')
    parser.add_argument('--log-file')
    args = parser.parse_args()
    if args.workers < 0 or args.bootstrap < 0:
        parser.error('workers/bootstrap must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
