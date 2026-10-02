"""V3.4: verified submitted V3.3 weights, inherited data, and layer-wise adaptation."""
import argparse
import math
from pathlib import Path

from w2v_aasist.launch import ROOT, BASELINE_SHA256
from w2v_aasist.runtime import sha256
from .source import resolve_source


DEFAULT_SOURCE = ROOT/'exp'/'w2v_v33_20261002_004018_198a'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', default=str(DEFAULT_SOURCE))
    p.add_argument('--submission-meta', help='Original V3.3 submission_meta.json; required to match source SHA256/tag')
    p.add_argument('--resume', help='V3.4 run directory; restores its saved recipe exactly')
    p.add_argument('--epochs', type=int, choices=(1, 2), default=2)
    p.add_argument('--trainable-layers', type=int, choices=range(1, 25), default=24)
    p.add_argument('--encoder-lr', type=float, default=5e-7, help='Top encoder block learning rate before scheduling')
    p.add_argument('--head-lr', type=float, default=2e-6)
    p.add_argument('--layer-decay', type=float, default=.9, help='LR multiplier for each earlier encoder block')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', choices=('bf16', 'none'), default='bf16')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=0)
    return p


def configuration(args):
    if args.workers < 0 or args.smoke_steps < 0 or not 1 <= args.epochs <= 2:
        raise ValueError('Workers/smoke steps must be nonnegative; 1-2 epochs are supported')
    if not 1 <= args.trainable_layers <= 24:
        raise ValueError('trainable-layers must be between 1 and 24')
    if not all(math.isfinite(x) and x > 0 for x in (args.encoder_lr, args.head_lr)):
        raise ValueError('Learning rates must be finite and positive')
    if not math.isfinite(args.layer_decay) or not 0 < args.layer_decay <= 1:
        raise ValueError('layer-decay must be finite and in (0, 1]')
    source = resolve_source(args.source_run, args.submission_meta)
    cfg = source['config']
    if ('MultiConv' not in cfg.get('algorithm', '') or not cfg.get('full_noisy')
            or not cfg.get('train_noisy_cache_v33')):
        raise ValueError('Source must retain the V3.3 MultiConv full-wave paired cache recipe')
    if cfg.get('baseline_sha256') != BASELINE_SHA256 or sha256(cfg['baseline']) != BASELINE_SHA256:
        raise ValueError('Protected original 91.68 checkpoint differs')
    # A selected control/fallback legitimately inherits pair_weight=0. Never turn
    # a submitted control into a candidate objective while changing unfreezing.
    cfg.update(version='3.4', source_run=str(Path(args.source_run).expanduser().resolve()),
        source_v33_run=source['provenance']['source_v33_run'],
        source_provenance=source['provenance'],
        source_data_fingerprints=source['provenance']['source_data_fingerprints'],
        source_config_sha256=source['provenance']['file_fingerprints'][str(Path(args.source_run).expanduser().resolve()/source['selected_arm']/'config.json')],
        warm_checkpoint=source['checkpoint'], warm_checkpoint_sha256=source['checkpoint_sha256'],
        expected_warm_tag=source['checkpoint_tag'], source_arm=source['selected_arm'],
        arm=source['selected_arm'], arms=[source['selected_arm']], head_epochs=0, joint_epochs=args.epochs,
        trainable_layers=args.trainable_layers, encoder_lr=args.encoder_lr,
        head_lr=args.head_lr, joint_head_lr=args.head_lr, layer_decay=args.layer_decay,
        joint_warmup_steps=100, lr_warmup_steps=100, min_lr_scale=.2,
        device=args.device, amp=args.amp, workers=args.workers, eval_amp='none',
        evals_per_epoch=2, checkpointing=True,
        optimizer_initialization='Fresh AdamW with layer-wise encoder LR; exact paired weight/moment restores within V3.4',
        algorithm='w2v-BERT 2.0 + MultiConv; V3.4 submitted-winner layer-wise encoder adaptation',
        optimization_policy='One run; last N encoder blocks trainable with earlier-layer LR decay; non-encoder front-end projection frozen; inherited V3.3 data and objective',
        cache_policy='Reuse verified complete V3.3 cache only; no generation or deletion')
    # The transport fix is numerical-equivalent; limit worker lookahead while
    # preserving the logical source batch and every loss/sampling setting.
    cfg['prefetch_factor'] = 1
    cfg.setdefault('gpu_activation_gib', 18.)
    cfg.setdefault('gpu_reserve_gib', 8.)
    return cfg


build_config = configuration
