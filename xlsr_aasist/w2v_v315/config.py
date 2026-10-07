"""The submitted V3.12 LAST, fresh matched RTC pairs, and bounded resources."""
import argparse

import torch

from w2v_v313.config import source_configuration
from w2v_v314.config import code_fingerprints as prior_code
from w2v_v39.common import ROOT, digest, verify_files
from w2v_v39.config import runtime_versions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run')
    p.add_argument('--resume')
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--microbatch', type=int, default=16, help='Maximum physical waveforms; logical batch is 16 sources/48 views')
    p.add_argument('--frame-budget', type=int, default=9600)
    p.add_argument('--cache-gib', type=float, default=6.)
    p.add_argument('--no-autotune', action='store_true')
    p.add_argument('--ce-only', action='store_true', help='Optional matched compute/data control; never launched automatically')
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    return p


def code_fingerprints():
    result = prior_code()
    folders = ('w2v_v315', 'rtc_noisy')
    for folder in folders:
        result.update({str(p.resolve()): digest(p) for p in (ROOT/folder).glob('*.py')
                       if not p.name.startswith('test_')})
    for path in (ROOT/'utils'/'env_noise.py', ROOT/'w2v_v33'/'cache.py'):
        result[str(path.resolve())] = digest(path)
    return result


def configuration(args):
    if not args.source_run:
        raise ValueError('Specify --source-run: the submitted, completed V3.12 LAST run')
    if not 1 <= args.epochs <= 4 or args.workers < 0 or args.microbatch < 2 or args.frame_budget < 1:
        raise ValueError('epochs=1..4, workers>=0, physical microbatch>=2, frame budget>0 required')
    if not 0 <= args.cache_gib <= 6:
        raise ValueError('Rolling augmentation cache must be between 0 and 6 GiB')
    cfg = source_configuration(args.source_run)
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device != 'cpu' and not device.startswith('cuda'):
        raise ValueError('Use cpu or cuda')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cfg.update(version='3.15', device=device, seed=31501, epochs=args.epochs,
        workers=args.workers, feature_workers=args.workers, feature_batch=16, eval_batch=16,
        microbatch=args.microbatch, frame_budget=args.frame_budget,
        source_batch=16, trainable_layers=8, checkpointing=True,
        encoder_lr=2e-7, head_lr=2e-6, adapter_lr=5e-6, output_lr=5e-6,
        tfcl_lr=1e-4, layer_decay=.8, lr_warmup_fraction=.05,
        weight_decay=.01, max_grad_norm=1., amp='bf16' if device.startswith('cuda') else 'none',
        eval_amp='none', tfcl_time_weight=0. if args.ce_only else .15,
        tfcl_structure_weight=0. if args.ce_only else .045, tfcl_bins=201,
        tfcl_heads=8, objective_ramp_epochs=.25, reference_ce_mass=.1,
        initial_noisy_ce_mass=.5, main_noisy_ce_mass=.6,
        adv_weight=0., pair_weight=0., ranking_weight=0., retention_weight=0., stability_weight=0.,
        min_gain=.0002, max_clean_drop=.003, max_noisy_drop=.0005,
        max_fake_drop=.005, max_real_drop=.015, max_auc_drop=.002,
        max_matched_real_drop=.02, matched_fake_recall=.99, max_panel_drop=.003,
        patience=2, minimum_epochs=1, progress_min_delta=.0002, checks_per_epoch=2,
        catastrophic_weighted_drop=.02, threshold=.5, disk_margin_bytes=128*1024**2,
        rolling_cache_bytes=int(args.cache_gib*1024**3), free_reserve_bytes=10*1024**3,
        maximum_new_peak_bytes=20*1024**3, autotune=not args.no_autotune,
        gpu_reserve_bytes=6*1024**3, panel_seed=3150901, noise_cache_mib=96,
        code_fingerprints=code_fingerprints(), runtime_versions=runtime_versions(),
        effective_objective='source/group-balanced full-wave CE + bidirectional temporal alignment + per-source channel CKA',
        effective_views='ordinary / matched RTC reference / input-noise-before-RTC; final CE mass 30/10/60',
        selection_policy='independent best_weighted, best_guarded, last; default best_guarded',
        effective_auxiliary_losses=[] if args.ce_only else ['temporal_cross_attention', 'channel_CKA'],
        external_teacher_at_inference=False)
    from .augment import bind_augmentation
    cfg.update(bind_augmentation(cfg))
    return cfg


def verify_inputs(cfg):
    if cfg.get('version') != '3.15' or cfg.get('starting_kind') != 'trained_v312_last':
        raise ValueError('Expected V3.15 bound to the trained V3.12 LAST')
    for key in ('code_fingerprints', 'source_fingerprints', 'data_fingerprints', 'augmentation_files'):
        verify_files(cfg[key])
    verify_files({cfg['base_checkpoint']: cfg['base_checkpoint_sha256'],
                  cfg['starting_checkpoint']: cfg['starting_checkpoint_sha256']})
    if cfg['runtime_versions'] != runtime_versions():
        raise ValueError('Runtime changed; resume requires the recorded environment')
    from .augment import runtime
    if cfg['augmentation_runtime'] != runtime(cfg['augmentation_runtime'].get('ffmpeg_path')):
        raise ValueError('FFmpeg/WebRTC runtime changed; no silent recipe change on resume')
