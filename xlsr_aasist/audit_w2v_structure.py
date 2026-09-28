"""Read-only local-structure screening on official Train/Dev and existing caches.

No checkpoint update, waveform generation, Dev fitting, threshold search or
submission generation occurs. Intermediate representations remain in RAM only.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import time
import zipfile

import numpy as np
import torch

from audit_w2v_dev import ReadOnlyCollator, compare_model_configs
from audit_w2v_train import Issues, protocol_rows, relative_id, save_csv, upload_report
from start_w2v_en import guard_output, verified_baseline
from w2v_rebuild.core import atomic_json, load_checkpoint, sha256
from w2v_rebuild.model import Detector, forward_chunks
from w2v_rebuild.structure_probe import analyze, physical_descriptors


ROOT = Path(__file__).resolve().parent
FORMAT = 'w2v_structure_audit_v1'
EXPORT_FILES = ('manifest.json', 'source_inventory.csv', 'coverage.csv', 'summary.json',
                'metrics.csv', 'invariance.csv', 'paired_transitions.csv', 'scores.csv',
                'alignment.csv', 'report.md', 'completed.json')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def extra_banks(config):
    from w2v_rebuild.structure_recipe import existing_diverse_cache
    extra = config.get('extra_train_noisy_cache') or []
    if not isinstance(extra, list):
        raise ValueError('Recorded extra_train_noisy_cache must be a list')
    primary = Path(config['train_noisy_cache']).expanduser().resolve()
    diverse = existing_diverse_cache(config)
    if primary == diverse:
        raise ValueError('Original and diverse cache banks must differ')
    return [primary, diverse]


def metadata_paths(config, source):
    paths = [source/'stage3'/'config.json', Path(config['train_protocol']), Path(config['dev_protocol']),
             Path(config['ssl_path'])/'config.json', Path(config['ssl_path'])/'preprocessor_config.json']
    banks = extra_banks(config) + [Path(config['dev_noisy_cache']), Path(config['dev_heldout_cache'])]
    for bank in banks:
        paths.extend([bank/'config.json', bank/'manifest.jsonl'])
    return sorted({p.expanduser().resolve() for p in paths})


def fingerprints(config, source):
    result = {}
    recorded = config.get('data_fingerprints')
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError('Reference run has no recorded input fingerprints')
    for path in metadata_paths(config, source):
        print('Checking metadata: '+str(path), flush=True)
        value = sha256(path)
        if str(path) in recorded and recorded[str(path)] != value:
            raise ValueError('Input differs from recorded run: '+str(path))
        result[str(path)] = value
    return result


def index_bank(folder, role, protocol, audio_root, protocol_map):
    """Validate all metadata; decode only selected WAVs, not hundreds of thousands.

    No partial/smoke cache is accepted. Existing V2 role/family/noise-split checks
    still apply. Every selected source is independently hashed before inference.
    """
    from rtc_noisy_v2.cache import check_metadata
    from rtc_noisy.common import CACHE_FORMAT, CUT, SR, SNR_BANDS
    folder = Path(folder).resolve()
    cfg = read_json(folder/'config.json')
    if (cfg.get('format') != CACHE_FORMAT or cfg.get('split') != ('train' if role == 'train' else 'dev')
            or cfg.get('protocol_sha256') != sha256(protocol) or cfg.get('limit')
            or cfg.get('cut') != CUT or cfg.get('sr') != SR
            or cfg.get('snr_bands') != [list(x) for x in SNR_BANDS]):
        raise ValueError('Cache format, complete protocol, cut or sample rate differs: '+str(folder))
    offline = {key: row for key, row in protocol_map.items() if row['domain'] == 'offline'}
    rows, seen = [], set()
    last = time.monotonic()
    with (folder/'manifest.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            source = relative_id(row['source']); band = row['band']
            if source not in offline or type(row['label']) is not int or row['label'] != offline[source]['label']:
                raise ValueError('Cached source or class differs from official protocol')
            if type(band) is not int or band not in range(4) or (source, band) in seen:
                raise ValueError('Missing/duplicate SNR band in cache')
            low, high = SNR_BANDS[band]
            if not low <= row['snr_db'] <= high:
                raise ValueError('SNR is outside recorded band')
            path = (folder/row['audio']).resolve()
            if not path.is_relative_to(folder):
                raise ValueError('Cached waveform escapes bank')
            if not isinstance(row.get('source_sha256'), str) or len(row['source_sha256']) != 64:
                raise ValueError('Cache lacks original-source hash')
            row.update(source=source, audio=str(path))
            rows.append(row); seen.add((source, band))
            if time.monotonic()-last > 10:
                print(f'Indexed {len(rows)} cache rows: {folder.name}', flush=True)
                last = time.monotonic()
    if len(seen) != len(offline)*4 or cfg.get('offline_count') != len(offline):
        raise ValueError('Incomplete cache metadata: '+str(folder))
    check_metadata(cfg, rows, role)
    return rows, cfg


def select_sources(protocol_map, banks, count, seed, excluded_hashes=()):
    """Equal groups, without replacement by known original-file content hash."""
    if count < 8:
        raise ValueError('At least eight sources per language/class are needed even for screening')
    source_hashes = {}
    for rows, _ in banks:
        for row in rows:
            previous = source_hashes.setdefault(row['source'], row['source_sha256'])
            if previous != row['source_sha256']:
                raise ValueError('Source hash differs across banks or bands')
    candidates = defaultdict(list)
    for source, row in protocol_map.items():
        if row['domain'] == 'offline' and row['language_group'] in ('en', 'zh') and source in source_hashes:
            candidates[(row['language_group'], row['label'])].append(source)
    selected, used = [], set(excluded_hashes)
    rng = random.Random(seed)
    for key in [('en', 0), ('en', 1), ('zh', 0), ('zh', 1)]:
        pool = sorted(candidates[key]); rng.shuffle(pool)
        take = []
        for source in pool:
            digest = source_hashes[source]
            if digest in used:
                continue
            used.add(digest); take.append(source)
            if len(take) == count:
                break
        if len(take) != count:
            raise ValueError(f'Insufficient distinct official sources for {key}: requested {count}, found {len(take)}')
        selected.extend(take)
    return selected, source_hashes


def build_records(config, train_count, dev_count, seed):
    from rtc_noisy_v2.cache import check_suite
    issues = Issues(); protocols = {}
    for split in ('train', 'dev'):
        rows, _ = protocol_rows(config[split+'_protocol'], config[split+'_data_path'], issues)
        protocols[split] = {r['source_id']: r for r in rows}
    if issues.counts:
        raise ValueError('Official protocol has ambiguous/duplicate metadata: '+str(dict(issues.counts)))
    train_banks = [index_bank(bank, 'train', config['train_protocol'], config['train_data_path'], protocols['train'])
                   for bank in extra_banks(config)]
    dev_banks = [index_bank(config[key], role, config['dev_protocol'], config['dev_data_path'], protocols['dev'])
                 for key, role in [('dev_noisy_cache', 'dev_seen'), ('dev_heldout_cache', 'dev_heldout')]]
    check_suite(train_banks, *dev_banks)
    train_sources, train_hashes = select_sources(protocols['train'], train_banks, train_count, seed)
    # Exclude ALL known Train original hashes, not only sampled training sources.
    dev_sources, dev_hashes = select_sources(protocols['dev'], dev_banks, dev_count, seed+1, set(train_hashes.values()))
    records, inventory, coverage = [], [], Counter()
    audio_hashes = {}
    for split, sources, banks, hashes in [('train', train_sources, train_banks, train_hashes),
                                          ('dev', dev_sources, dev_banks, dev_hashes)]:
        for source in sorted(sources):
            proto = protocols[split][source]
            original = Path(proto['audio_path'])
            actual = sha256(original)
            if actual != hashes[source]:
                raise ValueError('Original audio differs from cache source hash: '+str(original))
            inventory.append({'split': split, 'source_id': source, 'language': proto['language_group'],
                              'label': proto['label'], 'source_sha256': actual})
            base = dict(split=split, source_id=source, language=proto['language_group'], label=proto['label'],
                        source_sha256=actual)
            records.append(dict(base, condition='clean', bank=-1, family='clean', band=-1,
                                audio_path=str(original), snr_db=None))
        chosen = set(sources)
        for bank_index, (rows, _) in enumerate(banks):
            condition = 'train_noisy' if split == 'train' else ['seen', 'heldout'][bank_index]
            for row in sorted(rows, key=lambda r: (r['source'], r['band'])):
                if row['source'] not in chosen:
                    continue
                proto = protocols[split][row['source']]
                family = row.get('processing', {}).get('family', 'ffmpeg')
                records.append(dict(split=split, source_id=row['source'], label=row['label'], language=proto['language_group'],
                                    source_sha256=row['source_sha256'], audio_path=row['audio'], condition=condition,
                                    bank=bank_index, family=family, band=row['band'], snr_db=row['snr_db']))
                coverage[(split, condition, bank_index, proto['language_group'], row['label'], family, row['band'])] += 1
    for i, row in enumerate(records):
        if i % 128 == 0:
            print(f'Checking selected audio identities: {i}/{len(records)}', flush=True)
        path = Path(row['audio_path']); stat = path.stat()
        row.update(audio_size=stat.st_size, audio_mtime_ns=stat.st_mtime_ns)
        audio_hashes[str(path)] = sha256(path)
    # Include empty expected cells explicitly; random source sampling is not a
    # guarantee that every algorithm appears in every group/band.
    for split, banks in [('train', train_banks), ('dev', dev_banks)]:
        for bank_index, (rows, _) in enumerate(banks):
            condition = 'train_noisy' if split == 'train' else ['seen', 'heldout'][bank_index]
            families = sorted({r.get('processing', {}).get('family', 'ffmpeg') for r in rows})
            for language in ('en', 'zh'):
                for label in (0, 1):
                    for family in families:
                        for band in range(4):
                            coverage.setdefault((split, condition, bank_index, language, label, family, band), 0)
    coverage_rows = [dict(zip(('split', 'condition', 'bank', 'language', 'label', 'family', 'band'), key), views=value)
                     for key, value in sorted(coverage.items())]
    return records, inventory, coverage_rows, audio_hashes


def read_wave(row):
    import soundfile as sf
    from rtc_noisy.common import CUT, SR, read_wave as read_original
    from utils.data_utils import pad_audio
    if row['band'] < 0:
        return torch.from_numpy(np.asarray(pad_audio(read_original(row['audio_path']), CUT), np.float32).copy())
    wave, sr = sf.read(row['audio_path'], dtype='float32')
    if sr != SR or wave.shape != (CUT,) or not np.isfinite(wave).all():
        raise ValueError('Invalid selected cached waveform: '+row['audio_path'])
    return torch.from_numpy(wave)


@torch.inference_mode()
def infer(model, records, collator, device, microbatch=4):
    from w2v_rebuild.local_structure import descriptors, local_structure_per_pair
    features = defaultdict(list); valid = defaultdict(list)
    probabilities, alignment = [], []
    # Only clean frame sequences need to remain alive for paired diagnostics.
    clean_frames = {}
    total = len(records)
    for start in range(0, total, microbatch):
        rows = records[start:start+microbatch]
        waves = [read_wave(r) for r in rows]
        batch = collator([(w, r['label'], r['source_id']) for w, r in zip(waves, rows)])
        logits, readout, frames = forward_chunks(model, batch['features'].to(device), batch['mask'].to(device),
                                                 microbatch, pad_last=True, return_frames=True)
        if not torch.isfinite(logits).all() or not torch.isfinite(frames).all():
            raise FloatingPointError('Non-finite frozen model output')
        local = descriptors(frames)
        pooled = torch.cat((frames.mean(1), frames.std(1, unbiased=False)), -1)
        spectral, spectral_pool = physical_descriptors(torch.stack(waves))
        values = dict(readout=readout.cpu().numpy(), pooled=pooled.cpu().numpy(),
                      structure=local['embedding'].cpu().numpy(),
                      spectral_dynamic=spectral, spectral_pool=spectral_pool)
        probabilities.extend(logits.softmax(1)[:, 0].cpu().tolist())
        for name, x in values.items():
            features[name].append(x)
            valid[name].extend(local['valid'].cpu().tolist() if name == 'structure' else [True]*len(rows))
        for i, row in enumerate(rows):
            key = (row['split'], row['source_id'])
            if row['band'] < 0:
                clean_frames[key] = frames[i:i+1].detach().cpu()
            else:
                if key not in clean_frames:
                    raise ValueError('Clean counterpart must precede its processed view')
                loss, usable, stats = local_structure_per_pair(clean_frames[key].to(device), frames[i:i+1])
                alignment.append({key: row[key] for key in ('split', 'source_id', 'language', 'label', 'condition', 'bank', 'family', 'band')}
                                 | {'usable': bool(usable[0]), 'local_loss': float(loss[0]),
                                    **{k: float(v) for k, v in stats.items() if torch.is_tensor(v) and v.numel() == 1}})
        if start == 0 or start+microbatch >= total or start % (microbatch*8) == 0:
            print(f'Frozen original-best inference: {min(total, start+microbatch)}/{total} views', flush=True)
    return {k: np.concatenate(v) for k, v in features.items()}, {k: np.asarray(v, bool) for k, v in valid.items()}, probabilities, alignment


def package_report(out, download_dir):
    out, download_dir = Path(out).resolve(), Path(download_dir).resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    archive = download_dir/(out.name+'.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for name in EXPORT_FILES:
            path = out/name
            if not path.is_file() or path.is_symlink():
                raise ValueError('Missing/unsafe report artifact: '+name)
            stream.write(path, arcname=name)
    return archive


def report_markdown(summary):
    lines = ['# Frozen original-best local-structure screening', '',
             '**Recommendation: '+summary['recommendation']+'**. This is a mechanism screen, not a new trained model.', '',
             'Only official Train fits probes and feature normalization. Dev is a held-out diagnostic; no threshold or hyperparameter search is performed. '+
             'All metrics use fake probability >= 0.5. Four language/class groups are sampled equally: these scores are NOT official full-Dev or platform scores.', '',
             'Local structure describes relationships between learned frame features; its dimensions are not literal frequency bins. '+
             'The spectral-dynamic candidate uses actual WAV frequency bands. Stability without discrimination is insufficient.', '',
             '| Descriptor | Dev condition | Macro-F1 | AUROC | Fake recall | Real recall |',
             '|---|---|---:|---:|---:|---:|']
    def number(value):
        return 'n/a' if value is None else f'{100*value:.3f}'
    for row in summary['metrics']:
        if row['group_kind'] == 'condition' and row['group'] in ('clean', 'all_noisy', 'heldout'):
            lines.append('| '+row['descriptor']+' | '+row['group']+' | '+ ' | '.join(number(row[key]) for key in
                          ('macro_f1', 'auc', 'recall_fake', 'recall_real'))+' |')
    lines.extend(['', '## Paired matched-control uncertainty', '',
                  'Intervals resample original sources within language/class and keep every noisy view together. '+
                  'Speaker/corpus identities are unavailable; exploratory comparisons are not corrected for multiple testing.'])
    for row in summary['paired_intervals']:
        lines.append(f"- {row['candidate']} minus {row['control']}: {100*row['delta_macro_f1']:+.3f} pp; 95% interval [{100*row['ci95'][0]:+.3f}, {100*row['ci95'][1]:+.3f}] pp.")
    lines.extend(['', '## Alignment feasibility', '',
                  '| Split | Language/class | Views | Usable pairs | Mean accepted-bin fraction |',
                  '|---|---|---:|---:|---:|'])
    for row in summary['alignment_groups']:
        lines.append(f"| {row['split']} | {row['group']} | {row['views']} | {row['usable_pairs']} | {row['mean_accepted_fraction']:.3f} |")
    lines.extend(['', '## Interpretation and limits', '', summary['recommendation_rule'], '',
                  'Alignment gate: '+summary['alignment_gate']+'. Empty expected cache cells: '+str(summary['empty_coverage_cells'])+'.', '',
                  '- Original best weights were frozen and are hash-verified unchanged. No classifier or probe weights are exported.',
                  '- Controls have equal projected dimensionality; combined probes are diagnostic complements, not output fusion or submissions.',
                  '- Source content hashes prevent known byte-identical Train/Dev leakage. This does not establish speaker or utterance independence for re-encoded copies.',
                  '- Selection uses one fixed seed and existing cache views. Missing family/group cells and alignment rejection are in coverage.csv and alignment.csv.',
                  '- No new noise cache was generated. All representations were computed in RAM; the ZIP contains metadata, scalar scores and reports only.',
                  '- Baseline history may already include simulated conditions. Heldout refers to the current cache split, not proven global novelty.',
                  '- High similarity alone is not a reason to train. Review separability, matched controls, class/language tradeoffs and heldout behavior together.',
                  '- A completed audit never starts GPU fine-tuning automatically. Inconclusive/reject reports require reconsideration before any training.', ''])
    return '\n'.join(lines)


def alignment_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['split'], row['language']+'/'+str(row['label']))].append(row)
    return [dict(split=split, group=group, views=len(values),
                 usable_pairs=sum(bool(r['usable']) for r in values),
                 usable_fraction=sum(bool(r['usable']) for r in values)/len(values),
                 mean_accepted_fraction=float(np.mean([r['accepted_fraction'] for r in values])))
            for (split, group), values in sorted(grouped.items())]


def validate_reviewed_audit(directory, baseline_sha256):
    root = Path(directory).expanduser().resolve()
    manifest = read_json(root/'manifest.json'); summary = read_json(root/'summary.json'); completed = read_json(root/'completed.json')
    if (manifest.get('format') != FORMAT or manifest.get('baseline_sha256') != baseline_sha256
            or completed.get('status') != 'complete' or summary.get('status') != 'complete'
            or not completed.get('original_best_preserved') or not summary.get('original_best_preserved')
            or completed.get('summary_sha256') != sha256(root/'summary.json')
            or completed.get('manifest_sha256') != sha256(root/'manifest.json')):
        raise ValueError('Audit is incomplete, changed, or uses a different original baseline')
    if summary.get('recommendation') != 'ready_for_review':
        raise ValueError('Audit did not produce positive matched-control evidence for review: '+str(summary.get('recommendation')))
    if sha256(manifest['baseline_path']) != baseline_sha256:
        raise ValueError('Original best changed after audit')
    for path, expected in {**manifest['input_sha256'], **manifest['selected_audio_sha256']}.items():
        if sha256(path) != expected:
            raise ValueError('Audited metadata changed: '+path)
    for relative, expected in manifest['source_hashes'].items():
        if sha256(ROOT/relative) != expected:
            raise ValueError('Audited implementation changed: '+relative)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', required=True)
    parser.add_argument('--out', default=str(ROOT/'exp'/('w2v_structure_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))))
    parser.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    parser.add_argument('--upload-temp', action='store_true')
    parser.add_argument('--train-per-group', type=int, default=64)
    parser.add_argument('--dev-per-group', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--bootstrap', type=int, default=300)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--microbatch', type=int, default=4)
    args = parser.parse_args(argv)
    if args.train_per_group < 8 or args.dev_per_group < 8 or args.microbatch < 1 or args.bootstrap < 50:
        parser.error('Use >=8 sources/group, positive microbatch and >=50 bootstrap replicates')
    source = Path(args.from_run).expanduser().resolve()
    config = read_json(source/'stage3'/'config.json')
    baseline, original_hash = verified_baseline(config, source)
    out = Path(args.out).expanduser().resolve()
    protected = [source, baseline.parent, config['ssl_path'], config['train_data_path'], config['dev_data_path'],
                 *extra_banks(config), config['dev_noisy_cache'], config['dev_heldout_cache']]
    if config.get('feature_cache'):
        protected.append(config['feature_cache'])
    guard_output(out, protected)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no implicit CPU fallback')
    inputs = fingerprints(config, source)
    print('Selecting fixed balanced official Train/Dev sources; no audio augmentation or cache generation.', flush=True)
    records, inventory, coverage, audio_hashes = build_records(config, args.train_per_group, args.dev_per_group, args.seed)
    from w2v_rebuild.structure_recipe import structure_config, structure_recipe_contract
    from w2v_rebuild.train import source_hashes
    code_paths = ['audit_w2v_structure.py', 'audit_w2v_dev.py', 'audit_w2v_train.py',
                  'start_w2v_structure.py', 'start_w2v_coverage.py', 'start_w2v_en.py',
                  'start_w2v_adapt.py', 'start_w2v_refine.py', 'recover_w2v_storage.py']
    implementation_hashes = source_hashes()
    implementation_hashes.update({path: sha256(ROOT/path) for path in code_paths})
    manifest = dict(format=FORMAT, source_run=str(source), baseline_path=str(baseline), baseline_sha256=original_hash,
                    input_sha256=inputs, selected_audio_sha256=audio_hashes,
                    source_hashes=implementation_hashes,
                    settings={key: getattr(args, key) for key in ('train_per_group', 'dev_per_group', 'seed', 'bootstrap', 'device', 'microbatch')},
                    training_recipe_contract=structure_recipe_contract(structure_config(config)),
                    arithmetic='FP32 eval; frozen checkpoint; read-only feature-cache misses computed in RAM',
                    selection='fixed seed; four groups equal; only Offline; byte-distinct sources; all selected cache bands',
                    counts={'sources': len(inventory), 'views': len(records)},
                    torch_version=str(torch.__version__))
    atomic_json(manifest, out/'manifest.json')
    save_csv(out/'source_inventory.csv', inventory); save_csv(out/'coverage.csv', coverage)
    print(f"Selected {len(inventory)} original sources and {len(records)} total views. Loading frozen original best.", flush=True)
    checkpoint = load_checkpoint(baseline)
    comparison = compare_model_configs(checkpoint['model_config'], config['model_config'])
    if checkpoint.get('stage') != 3 or not comparison['matched']:
        raise ValueError('Original checkpoint architecture/stage differs after JSON normalization')
    for path in [config['train_protocol'], config['dev_protocol'], str(Path(config['ssl_path'])/'config.json'),
                 str(Path(config['ssl_path'])/'preprocessor_config.json')]:
        key = str(Path(path).resolve())
        if checkpoint.get('data_fingerprints', {}).get(key) != inputs[key]:
            raise ValueError('Original best protocol/extractor fingerprint differs: '+key)
    model = Detector.load(config['ssl_path'], checkpoint['model_config'], checkpointing=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint; gc.collect()
    model.requires_grad_(False).to(device).eval()
    collator = ReadOnlyCollator(config['ssl_path'], 'ordinary', config.get('feature_cache'))
    features, valid, probabilities, alignment = infer(model, records, collator, device, args.microbatch)
    del model, collator; gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('Fitting fixed diagnostic probes on official Train only; Dev is never fitted.', flush=True)
    summary, probes = analyze(records, features, valid, probabilities, args.bootstrap)
    del features
    summary['alignment_groups'] = alignment_summary(alignment)
    weak_alignment = [r for r in summary['alignment_groups'] if r['usable_fraction'] < .1 or r['usable_pairs'] < 8]
    if len(summary['alignment_groups']) != 8 or weak_alignment:
        summary['alignment_gate'] = 'insufficient usable matches in one or more Train/Dev language/class groups'
        if summary['recommendation'] == 'ready_for_review':
            summary['recommendation'] = 'inconclusive'
    else:
        summary['alignment_gate'] = 'at least 10% and eight usable pairs in each language/class and split; inspect finer conditions manually'
    summary['empty_coverage_cells'] = sum(r['views'] == 0 for r in coverage)
    if summary['empty_coverage_cells'] and summary['recommendation'] == 'ready_for_review':
        summary['recommendation'] = 'inconclusive'
    for path, expected in {**inputs, **audio_hashes, str(baseline): original_hash}.items():
        if sha256(path) != expected:
            raise ValueError('Read-only audit input changed during execution: '+path)
    summary.update(status='complete', original_best_preserved=True, format=FORMAT, counts=manifest['counts'],
                   seconds=time.monotonic()-started)
    atomic_json(summary, out/'summary.json')
    save_csv(out/'metrics.csv', summary['metrics']); save_csv(out/'invariance.csv', summary['invariance'])
    save_csv(out/'paired_transitions.csv', summary['paired_transitions']); save_csv(out/'alignment.csv', alignment)
    score_rows = []
    for i, record in enumerate(records):
        row = {key: record[key] for key in ('split', 'source_id', 'language', 'label', 'condition', 'bank', 'family', 'band')}
        row.update({name+'_pfake': float(values[i]) for name, values in probes.items()})
        score_rows.append(row)
    save_csv(out/'scores.csv', score_rows)
    (out/'report.md').write_text(report_markdown(summary), encoding='utf-8')
    atomic_json(dict(status='complete', original_best_preserved=True, summary_sha256=sha256(out/'summary.json'),
                     manifest_sha256=sha256(out/'manifest.json')), out/'completed.json')
    archive = package_report(out, args.download_dir)
    print(f"REPORT_DIR={out}\nDOWNLOAD_ZIP={archive}\nRECOMMENDATION={summary['recommendation']}\nAUDIT_COMPLETE=True\nORIGINAL_BEST_PRESERVED=True", flush=True)
    if args.upload_temp:
        try:
            link = upload_report(archive)
            (out/'temp_download_url.txt').write_text(link+'\n', encoding='utf-8')
            print('TEMP_DOWNLOAD_URL='+link+'\nUPLOAD_COMPLETE=True', flush=True)
        except Exception as exc:
            print(f'UPLOAD_COMPLETE=False\nLocal report ZIP remains available. {type(exc).__name__}: {exc}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
