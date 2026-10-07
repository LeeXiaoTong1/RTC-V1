"""Pin the actual submitted V3.12 LAST and record the effective V3.14 recipe."""
import argparse

import torch

from w2v_v313.config import source_configuration, code_fingerprints as prior_code
from w2v_v39.common import ROOT, digest, verify_files
from w2v_v39.config import runtime_versions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', help='Completed V3.12 run; never substitutes best.pt for last.pt')
    p.add_argument('--resume')
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=2, help='Maximum joint epochs (1 or 2), after head adaptation')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--frame-budget', type=int, default=2400)
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    return p


def code_fingerprints():
    result = prior_code()
    result.update({str(p.resolve()): digest(p) for p in (ROOT/'w2v_v314').glob('*.py')
                   if not p.name.startswith('test_')})
    return result


def configuration(args):
    if not args.source_run:
        raise ValueError('Specify --source-run: the submitted V3.12 LAST run')
    if args.epochs not in (1, 2) or args.workers < 0 or min(args.microbatch, args.frame_budget) < 1:
        raise ValueError('Joint epochs must be 1..2; workers nonnegative and batch limits positive')
    cfg = source_configuration(args.source_run)
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device != 'cpu' and not device.startswith('cuda'):
        raise ValueError('Use cpu or cuda')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    # Inherited source metadata is retained for reconstruction. These are the
    # effective training fields; none of the earlier auxiliary losses is called.
    cfg.update(version='3.14', device=device, seed=31401, epochs=args.epochs,
        workers=args.workers, feature_workers=args.workers, microbatch=args.microbatch,
        frame_budget=args.frame_budget, feature_batch=16, eval_batch=16,
        feature_microbatch=args.microbatch, feature_frame_budget=args.frame_budget,
        feature_commit_rows=1024, feature_free_margin_bytes=128*1024**2,
        source_batch=16, trainable_layers=8, checkpointing=True,
        encoder_lr=2e-7, head_lr=2e-6, adapter_lr=1e-5, output_lr=2e-5,
        layer_decay=.8, lr_warmup_fraction=.05, weight_decay=.01, max_grad_norm=1.,
        amp='bf16' if device.startswith('cuda') else 'none', eval_amp='none',
        head_steps=80, head_anchor=.1, head_chunk=8192, head_grad_tolerance=1e-7,
        adv_weight=0., pair_weight=0., ranking_weight=0., retention_weight=0., stability_weight=0.,
        min_gain=.0002, max_clean_drop=.003, max_fake_drop=.005, max_real_drop=.015,
        max_auc_drop=.002, max_matched_real_drop=.02, min_en_real_gain=0., matched_fake_recall=.99,
        patience=2, minimum_epochs=0, progress_min_delta=.0002, checks_per_epoch=2,
        catastrophic_weighted_drop=.02, disk_margin_bytes=128*1024**2,
        threshold=.5, code_fingerprints=code_fingerprints(), runtime_versions=runtime_versions(),
        effective_objective='Stage A: source-balanced CE + anchored parameter L2; Stage B: CE + AdamW',
        effective_views='one ordinary + one noisy per source; 50% CE mass each; EN/ZH x fake/real each 25%',
        effective_auxiliary_losses=[], selection_policy='independent best_weighted, best_guarded, last; default best_guarded')
    return cfg


def verify_inputs(cfg):
    if cfg.get('version') != '3.14' or cfg.get('starting_kind') != 'trained_v312_last':
        raise ValueError('Expected V3.14 bound to the trained V3.12 LAST')
    for key in ('code_fingerprints', 'source_fingerprints', 'data_fingerprints'):
        verify_files(cfg[key])
    verify_files({cfg['base_checkpoint']: cfg['base_checkpoint_sha256'],
                  cfg['starting_checkpoint']: cfg['starting_checkpoint_sha256']})
    if cfg['runtime_versions'] != runtime_versions():
        raise ValueError('Runtime changed; resume requires the recorded environment')
