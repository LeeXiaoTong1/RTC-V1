"""Predeclared Train-only search and fixed Dev acceptance rules."""
import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform

from .common import ROOT, digest, read_json, verify_files


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', help='Completed V3.8 or V3.7 baseline run; defaults to latest V3.8, then V3.7')
    p.add_argument('--resume', help='Reuse completed compact fitting stages of a V3.9 run')
    p.add_argument('--device', default='auto', help='auto, cpu, or cuda:0; cached fitting only')
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    return p


def code_fingerprints():
    paths = []
    # Include the V3.8 source loader and the reused model/data/runtime code.
    # Older modules remain immutable; their identity belongs to every fit stage.
    for folder in ('w2v_v39', 'w2v_v38', 'w2v_v37', 'w2v_v36', 'w2v_v35',
                   'w2v_v34', 'w2v_v33', 'w2v_v32', 'w2v_v31', 'w2v_v3',
                   'w2v_aasist', 'w2v_rebuild', 'rtc_noisy', 'rtc_noisy_v2'):
        paths += [p for p in (ROOT / folder).glob('*.py') if not p.name.startswith('test_')]
    paths += [ROOT / 'live_progress.py']
    return {str(p.resolve()): digest(p) for p in sorted(set(paths))}


def runtime_versions():
    values = {'python': platform.python_version()}
    for package in ('torch', 'numpy', 'scipy', 'transformers', 'soundfile', 'huggingface-hub'):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = 'not-installed'
    return values


def source_configuration(source):
    """Unwrap only verified baseline fallbacks; never silently drop a correction."""
    import torch
    from w2v_v37.patch import load_selected as load_v37
    source = Path(source).expanduser().resolve()
    version_id = read_json(source / 'completed.json').get('version')
    if version_id == '3.8':
        from w2v_v38.patch import load_selected as load_v38
        patch, done = load_v38(source)
        if done.get('status') != 'complete':
            raise ValueError('V3.8 source must have completed successfully')
        if done['selected'] != 'baseline' or patch.get('language_debias_applied'):
            raise ValueError('Refusing to discard an existing V3.8 correction; source must select baseline')
        recorded = patch['config']
        original = Path(recorded['v37_run']).expanduser().resolve()
        verify_files({str(original / 'completed.json'): recorded['v37_completed_sha256'],
                      str(original / 'best_patch.pt'): recorded['v37_patch_sha256']})
        old_patch, old_done = load_v37(original)
        if any(recorded[k] != old_patch['config'][k]
               for k in ('base_checkpoint', 'base_checkpoint_sha256', 'base_tag')):
            raise ValueError('V3.8 and original V3.7 baseline identity differs')
        if any(not torch.equal(patch['spec']['state'][k], old_patch[k]) for k in ('weight', 'bias')):
            raise ValueError('V3.8 baseline classifier differs from original V3.7')
        if recorded['data_fingerprints'] != read_json(original / 'data_fingerprints.json'):
            raise ValueError('V3.8 and original V3.7 data identity differs')
    elif version_id == '3.7':
        original = source
        old_patch, old_done = load_v37(source)
    else:
        raise ValueError('Pass a completed V3.8 or V3.7 baseline run')
    if old_done['selected'] != 'baseline' or old_patch.get('language_debias_applied'):
        raise ValueError('Refusing to discard an existing V3.7 correction; source must select baseline')
    cfg = dict(old_patch['config'])
    tracked = {source / 'completed.json', source / 'best_patch.pt',
               original / 'completed.json', original / 'best_patch.pt',
               original / 'data_fingerprints.json'}
    cfg.update(v37_run=str(original), v37_completed_sha256=digest(original / 'completed.json'),
               v37_patch_sha256=digest(original / 'best_patch.pt'),
               data_fingerprints=read_json(original / 'data_fingerprints.json'),
               source_run=str(source), source_version=version_id,
               source_fingerprints={str(p): digest(p) for p in sorted(tracked)})
    verify_files({cfg['base_checkpoint']: cfg['base_checkpoint_sha256']})
    return cfg


def configuration(args):
    import torch
    source = args.source_run
    if not source:
        for name in ('.latest_v38_run', '.latest_v37_run'):
            pointer = ROOT / 'exp' / name
            if pointer.is_file() and pointer.read_text(encoding='utf-8').strip():
                source = pointer.read_text(encoding='utf-8').strip()
                break
    if not source:
        raise ValueError('Pass --source-run pointing to the completed V3.8 or V3.7 baseline run')
    cfg = source_configuration(source)
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device != 'cpu' and not device.startswith('cuda'):
        raise ValueError('Use cpu or a CUDA device')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was explicitly requested but is unavailable; use --device cpu')
    cfg.update(version='3.9',
        device=device, seed=3901, fit_threads=1, trainable_layers=0, threshold=.5,
        batch_rows=8192, epochs=20, student_epochs=30, learning_rate=.0005,
        student_learning_rate=.0007, holdout_fraction=.2, student_hidden=64,
        residual_hidden=64, residual_cap=2., lambda_grid=[.005, .02],
        loss_temperature=2., hard_example_gain=3., hard_example_temperature=2.,
        safety_margin=1., train_max_recall_drop=.005, weight_decay=1e-4,
        fake_protection=2., real_protection=1., protection_slack=.1,
        min_student_r2=.02, min_student_language_accuracy=.65,
        min_teacher_language_accuracy=.70,
        min_gain=.001, max_clean_drop=.001, max_fake_drop=.002,
        max_real_drop=.005, max_auc_drop=.0005, min_en_real_gain=.005,
        min_language_control_gain=.0005, min_ranking_gain=.002,
        matched_fake_recall=.99, max_matched_real_drop=.005,
        code_fingerprints=code_fingerprints(), runtime_versions=runtime_versions(),
        algorithm='zero-start adaptive-margin repair; Train source-balanced hard examples; safe-margin protection',
        optimization_policy='grouped Train validation and full Train refit; fixed Dev used once for acceptance',
        inference_policy='one frozen detector, optional internal residual; no teacher or language metadata',
        cache_policy='completed V3.7 detector and teacher vectors borrowed read-only; no new audio/features')
    return cfg
