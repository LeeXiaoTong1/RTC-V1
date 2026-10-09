"""Re-evaluate a completed detector on the same labeled Dev used in training."""
import argparse
from pathlib import Path

import numpy as np
import torch

from w2v_v313.replay import rows_signature
from w2v_v313.train import infer
from w2v_v39.common import ROOT, atomic_json, digest, read_json, verify_files
from .config import verify_parent
from .data import bundles
from .metrics import measure, print_metrics
from .model import load_model, load_reference
from .state import KINDS, apply_candidate, load_selected


def validate(run, out, device='cuda:0', workers=2, checkpoint_kind='best_guarded'):
    """Read fixed inputs; write small results outside the immutable training run."""
    run, out = Path(run).expanduser().resolve(), Path(out).expanduser().resolve()
    if out == run or run in out.parents:
        raise ValueError('Dev validation output must be outside the training run')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Dev output already exists and is not empty; choose a new --out directory')
    if workers < 0:
        raise ValueError('workers must be nonnegative')
    if not (run/'completed.json').is_file():
        raise ValueError('Standalone Dev validation requires a completed V3.16 run. '
                         'Training already validates automatically; follow watch_w2v_v316.sh.')
    if device != 'cpu' and not device.startswith('cuda'):
        raise ValueError('Use cpu or cuda for --device')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu only for an intentional CPU evaluation')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint, done = load_selected(run, checkpoint_kind)
    cfg = dict(checkpoint['config'], device=device, workers=workers, feature_workers=workers)
    pinned = {}
    for key in ('code_fingerprints', 'source_fingerprints', 'data_fingerprints', 'pretrained_fingerprints'):
        pinned.update(cfg[key])
    for key in ('base_checkpoint', 'starting_checkpoint', 'parent_checkpoint'):
        pinned[cfg[key]] = cfg[key+'_sha256']
    weight_path = Path(done['checkpoint_path'])
    pinned[str(weight_path)] = done['checkpoint_sha256']
    for name in ('config.json', 'completed.json', 'dev_rows.json', 'execution_plan.json'):
        pinned[str(run/name)] = digest(run/name)
    verify_files(pinned)
    verify_parent(cfg)
    with bundles(cfg) as (_, dev):
        rows = dev['rows']
    if rows_signature(rows) != rows_signature(read_json(run/'dev_rows.json')):
        raise ValueError('Fixed Dev row identity/order differs from the training run')

    fallback = done['baseline_fallback']
    print('DEV_SELECTED='+done['selected']+'; kind='+checkpoint_kind, flush=True)
    print('DEV_BASELINE_FALLBACK='+str(fallback), flush=True)
    if fallback:
        print('DEV_ACTUAL_MODEL=historical V3.15 parent; selected='+cfg['parent_selected_tag'], flush=True)
    else:
        print('DEV_ACTUAL_MODEL=trained V3.16 detector; initialization='+cfg['init_mode'], flush=True)
    model = (load_reference(cfg, device, training=False) if fallback
             else load_model(cfg, device, training=False))
    apply_candidate(model, checkpoint['candidate'])
    model.requires_grad_(False).eval()
    with torch.no_grad():
        logits, _ = infer(model, rows, cfg, 'V3.16 standalone '+checkpoint_kind+' fixed Dev')
    if np.shape(logits) != (len(rows), 2) or not np.isfinite(logits).all():
        raise ValueError('Invalid Dev logits or incomplete inference coverage')
    metrics = measure(rows, logits, target=cfg['matched_fake_recall'])
    if not metrics['complete']:
        raise ValueError('Fixed Dev is incomplete; refusing a partial Weighted score')
    # A completed run and all its pinned inputs must remain unchanged during replay.
    verify_files(pinned)
    verify_parent(cfg)
    out.mkdir(parents=True, exist_ok=True)
    score_path = out/'dev_scores.npz'
    temporary = score_path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, logits=logits,
                            ids=np.asarray([r['id'] for r in rows]),
                            labels=np.asarray([r['label'] for r in rows], dtype=np.int64))
    temporary.replace(score_path)
    atomic_json(out/'metrics.json', metrics)
    atomic_json(out/'validation_meta.json', dict(version='3.16', variant='offline_reference_tfcl_v1',
        run=str(run), selected=done['selected'], checkpoint_kind=checkpoint_kind,
        checkpoint_path=str(weight_path), checkpoint_sha256=done['checkpoint_sha256'],
        baseline_fallback=fallback, training_initialization=cfg['init_mode'],
        actual_model='historical_parent' if fallback else 'trained_v316',
        parent_run=cfg['parent_run'], parent_selected_tag=cfg['parent_selected_tag'],
        parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],
        count=len(rows), rows_signature=rows_signature(rows),
        scores_sha256=digest(score_path), metrics_sha256=digest(out/'metrics.json'),
        validation_code_sha256=digest(Path(__file__)), threshold=.5, score='P(fake)',
        input_policy='full utterance', eval_amp='none', fixed_dev_only=True,
        weighted_formula='0.3 * Clean F1 + 0.7 * Noisy F1',
        parameter_updates=0, selection_updated=False, new_audio_cache_bytes=0))
    print_metrics('standalone '+checkpoint_kind+':'+done['selected'], metrics)
    print('DEV_METRICS='+str(out/'metrics.json'), flush=True)
    print('DEV_METADATA='+str(out/'validation_meta.json'), flush=True)
    return metrics


def main():
    from w2v_aasist.full_workflow import ensure_idle
    from w2v_aasist.launch import run_lock
    import os
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', help='Completed TFCL run; defaults to V316_TFCL_RUN or the latest run')
    p.add_argument('--checkpoint', choices=KINDS, default='best_guarded')
    p.add_argument('--out', help='New output directory; defaults to temp/<run>_dev_<checkpoint>')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    if args.workers < 0:
        p.error('--workers must be nonnegative')
    run = args.run or os.environ.get('V316_TFCL_RUN')
    if not run:
        pointer = ROOT/'exp'/'.latest_v316_tfcl_run'
        if not pointer.is_file() or not pointer.read_text(encoding='utf-8').strip():
            p.error('No V3.16 TFCL run found; specify --run')
        run = pointer.read_text(encoding='utf-8').strip()
    run = Path(run).expanduser().resolve()
    out = args.out or ROOT.parent.parent/'temp'/(run.name+'_dev_'+args.checkpoint)
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        validate(run, out, args.device, args.workers, args.checkpoint)


if __name__ == '__main__':
    main()
