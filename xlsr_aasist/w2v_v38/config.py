"""Predeclared Train-only search and fixed Dev acceptance rules."""
import argparse
from pathlib import Path

from .common import ROOT, digest, read_json, verify_files


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', help='Completed V3.7 baseline run; defaults to .latest_v37_run')
    p.add_argument('--resume', help='Reuse completed compact fitting stages of a V3.8 run')
    p.add_argument('--device', default='auto', help='auto, cpu, or cuda:0; cached fitting only')
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    return p


def code_fingerprints():
    paths = []
    for folder in ('w2v_v38', 'w2v_v37', 'w2v_v36'):
        paths += [p for p in (ROOT / folder).glob('*.py') if not p.name.startswith('test_')]
    for folder, names in (('w2v_v3', ('model.py', 'data.py', 'step.py')),
                          ('w2v_aasist', ('model.py', 'data.py', 'runtime.py')),
                          ('w2v_rebuild', ('model.py',))):
        paths += [ROOT / folder / name for name in names]
    return {str(p.resolve()): digest(p) for p in sorted(set(paths))}


def configuration(args):
    import torch
    from w2v_v37.patch import load_selected
    pointer = ROOT / 'exp' / '.latest_v37_run'
    source = args.source_run or (pointer.read_text(encoding='utf-8').strip() if pointer.is_file() else '')
    if not source:
        raise ValueError('Pass --source-run pointing to the completed V3.7 run')
    source = Path(source).expanduser().resolve()
    patch, done = load_selected(source)
    if done['selected'] != 'baseline' or patch.get('language_debias_applied'):
        raise ValueError('V3.8 requires the audited V3.7 baseline fallback; refusing to discard an existing correction')
    cfg = dict(patch['config'])
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device != 'cpu' and not device.startswith('cuda'):
        raise ValueError('Use cpu or a CUDA device')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was explicitly requested but is unavailable; use --device cpu')
    verify_files({cfg['base_checkpoint']: cfg['base_checkpoint_sha256']})
    cfg.update(version='3.8', v37_run=str(source),
        v37_completed_sha256=digest(source / 'completed.json'),
        v37_patch_sha256=digest(source / 'best_patch.pt'),
        data_fingerprints=read_json(source / 'data_fingerprints.json'),
        device=device, seed=3801, fit_threads=1, trainable_layers=0, threshold=.5,
        batch_rows=8192, epochs=24, student_epochs=30, learning_rate=.001,
        student_learning_rate=.002, holdout_fraction=.2, student_hidden=64,
        residual_hidden=64, residual_cap=2., lambda_grid=[.02, .1],
        fake_protection=2., real_protection=1., protection_slack=.1,
        min_student_r2=.02, min_student_language_accuracy=.65,
        min_teacher_language_accuracy=.70,
        min_gain=.001, max_clean_drop=.001, max_fake_drop=.002,
        max_real_drop=.005, max_auc_drop=.0005, min_en_real_gain=.005,
        min_language_control_gain=.0005, min_ranking_gain=.002,
        matched_fake_recall=.99, max_matched_real_drop=.005,
        code_fingerprints=code_fingerprints(),
        algorithm='zero-start bounded residual; centered real-only language student; fake-margin protection',
        optimization_policy='grouped Train validation and full Train refit; fixed Dev used once for acceptance',
        inference_policy='one frozen detector, optional internal residual; no teacher or language metadata',
        cache_policy='completed V3.7 detector and teacher vectors borrowed read-only; no new audio/features')
    return cfg
