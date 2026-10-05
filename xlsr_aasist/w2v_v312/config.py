"""Pinned starting model, bounded compute and disk, explicit gradient adaptation."""
import argparse
from pathlib import Path

from w2v_v39.common import ROOT, digest, read_json, verify_files
from w2v_v39.config import runtime_versions, code_fingerprints as previous_code


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', help='Completed V3.11/3.10/3.9/3.8/3.7 baseline run')
    p.add_argument('--resume', help='Resume the latest atomically committed V3.12 validation boundary')
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--frame-budget', type=int, default=2400)
    p.add_argument('--adv-weight', type=float, default=.05, help='Maximum gradient reversal strength; zero is an ablation')
    p.add_argument('--retention-weight', type=float, default=.2, help='One-sided correct-margin preservation; zero disables it')
    p.add_argument('--no-gradient-protection', action='store_true', help='Explicit unprotected-gradient ablation')
    p.add_argument('--stability-weight', type=float, default=.1, help='Feature-norm and logit-scale penalty; watchdogs remain enabled')
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    return p


def source_configuration(source):
    from w2v_v39.config import source_configuration as earlier
    import torch
    source = Path(source).expanduser().resolve()
    if read_json(source / 'completed.json').get('version') == '3.11':
        from w2v_v311.state import load_selected as load_previous
        from w2v_v311.config import source_configuration as previous_source
        checkpoint, done = load_previous(source)
        if done['selected'] != 'baseline' or not done['baseline_fallback']:
            raise ValueError('V3.12 must not discard a selected V3.11 improvement')
        recorded = checkpoint['config']
        cfg = previous_source(recorded['source_run'])
        verify_files(recorded['source_fingerprints'])
        for key in ('base_checkpoint', 'base_checkpoint_sha256', 'base_tag', 'data_fingerprints'):
            if recorded[key] != cfg[key]:
                raise ValueError('V3.11 baseline provenance differs: '+key)
        cfg['source_fingerprints'].update(recorded['source_fingerprints'])
        cfg['source_fingerprints'].update({str(source/name):digest(source/name) for name in ('completed.json','best.pt')})
        cfg.update(source_run=str(source),source_version='3.11')
        return cfg
    if read_json(source / 'completed.json').get('version') == '3.10':
        from w2v_v310.state import load_selected as load_previous
        from w2v_v310.config import source_configuration as previous_source
        checkpoint, done = load_previous(source)
        if done['selected'] != 'baseline' or not done['baseline_fallback']:
            raise ValueError('V3.12 baseline restart must not discard a selected V3.10 improvement')
        recorded = checkpoint['config']
        cfg = previous_source(recorded['source_run'])
        verify_files(recorded['source_fingerprints'])
        for key in ('base_checkpoint', 'base_checkpoint_sha256', 'base_tag', 'data_fingerprints'):
            if recorded[key] != cfg[key]:
                raise ValueError('V3.10 baseline provenance differs: '+key)
        cfg['source_fingerprints'].update(recorded['source_fingerprints'])
        cfg['source_fingerprints'].update({str(source/name): digest(source/name)
                                          for name in ('completed.json', 'best.pt')})
        cfg.update(source_run=str(source), source_version='3.10')
        return cfg
    if read_json(source / 'completed.json').get('version') != '3.9':
        return earlier(source)
    from w2v_v39.patch import load_selected
    patch, done = load_selected(source)
    if done.get('status') != 'complete' or done['selected'] != 'baseline':
        raise ValueError('V3.12 requires the verified baseline; refusing to discard a selected correction')
    recorded = patch['config']
    original = Path(recorded['v37_run'])
    cfg = earlier(original)
    verify_files(recorded['source_fingerprints'])
    for key in ('base_checkpoint', 'base_checkpoint_sha256', 'base_tag', 'data_fingerprints'):
        if recorded[key] != cfg[key]:
            raise ValueError('V3.9 and original baseline identity differs: ' + key)
    old = torch.load(original / 'best_patch.pt', map_location='cpu', weights_only=True)
    if any(not torch.equal(patch['spec']['state'][k], old[k]) for k in ('weight', 'bias')):
        raise ValueError('V3.9 baseline classifier differs from original best')
    cfg['source_fingerprints'].update(recorded['source_fingerprints'])
    cfg['source_fingerprints'].update({str(source / name): digest(source / name)
                                      for name in ('completed.json', 'best_patch.pt')})
    cfg.update(source_run=str(source), source_version='3.9')
    return cfg


def code_fingerprints():
    result = previous_code()
    result.update({str(p.resolve()): digest(p) for p in (ROOT / 'w2v_v310').glob('*.py')
                   if not p.name.startswith('test_')})
    result.update({str(p.resolve()): digest(p) for p in (ROOT / 'w2v_v311').glob('*.py')
                   if not p.name.startswith('test_')})
    result.update({str(p.resolve()): digest(p) for p in (ROOT / 'w2v_v312').glob('*.py')
                   if not p.name.startswith('test_')})
    return result


def configuration(args):
    import torch
    source = args.source_run
    if not source:
        for name in ('v311', 'v310', 'v39', 'v38', 'v37'):
            pointer = ROOT / 'exp' / ('.latest_' + name + '_run')
            if pointer.is_file() and pointer.read_text(encoding='utf-8').strip():
                source = pointer.read_text(encoding='utf-8').strip()
                break
    if not source:
        raise ValueError('Pass --source-run with the completed baseline run')
    if (args.epochs < 1 or args.workers < 0 or args.microbatch < 1 or args.frame_budget < 1
            or not 0 <= args.adv_weight <= 1 or not 0 <= args.retention_weight <= 2
            or not 0 <= args.stability_weight <= 2):
        raise ValueError('Invalid training budget, workers, microbatch, or adversarial strength')
    cfg = source_configuration(source)
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device != 'cpu' and not device.startswith('cuda'):
        raise ValueError('Use cpu or a CUDA device')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    cfg.update(version='3.12', device=device, seed=31001, trainable_layers=8,
        epochs=args.epochs, workers=args.workers, feature_workers=args.workers,
        source_batch=16, microbatch=args.microbatch, frame_budget=args.frame_budget,
        eval_batch=16, feature_batch=16, checkpointing=True,
        adapter_hidden=128, adversary_hidden=128, encoder_lr=2e-7, head_lr=2e-6,
        adapter_lr=2e-5, adversary_lr=1e-4, layer_decay=.8, weight_decay=1e-4,
        max_grad_norm=1., adv_weight=args.adv_weight, adv_warmup_fraction=.1, adv_ramp_fraction=.9,
        gradient_protection=not args.no_gradient_protection, retention_weight=args.retention_weight,
        retention_min_margin=1., retention_cap=4., retention_slack=.5,
        adapter_max_ratio=.2, output_projection_frozen=True, stability_weight=args.stability_weight,
        feature_rms_floor=1e-4, feature_ratio_low=.5, feature_ratio_high=1.5,
        margin_soft_floor=8., margin_reference_factor=1.5, margin_reference_slack=2.,
        hard_feature_ratio=4., hard_margin_ratio=4., probe_max_variance_ratio=4.,
        validation_norm_ratio=2., validation_margin_ratio=3.,
        lr_warmup_fraction=.05, lr_floor=.1, amp='bf16' if device.startswith('cuda') else 'none',
        checks_per_epoch=2, patience=4, minimum_epochs=2, progress_min_delta=.0002, catastrophic_weighted_drop=.02,
        probe_sources_per_language=256, probe_steps=150, probe_hidden=64, probe_min_drop=.01,
        min_gain=.001, max_clean_drop=.001, max_fake_drop=.002, max_real_drop=.005,
        max_auc_drop=.0005, min_en_real_gain=.005, matched_fake_recall=.99, max_matched_real_drop=.005,
        threshold=.5, disk_margin_bytes=256*1024**2,
        metric_definition='fixed full-wave Online Clean macro-F1 .3 + mean(Seen,Heldout) macro-F1 .7; local proxy only',
        algorithm='bounded residual adaptation; decision-and-radial tangent language reversal; Train reference norm/score penalties and fail-closed scale watchdogs',
        code_fingerprints=code_fingerprints(), runtime_versions=runtime_versions())
    return cfg


def verify_inputs(cfg):
    if cfg.get('version') != '3.12':
        raise ValueError('Expected a V3.12 configuration')
    for key in ('code_fingerprints', 'source_fingerprints', 'data_fingerprints'):
        verify_files(cfg[key])
    verify_files({cfg['base_checkpoint']: cfg['base_checkpoint_sha256']})
    if cfg['runtime_versions'] != runtime_versions():
        raise ValueError('Resume requires the original runtime versions')
