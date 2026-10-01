"""A bounded new objective; never resume a V3 run with a changed recipe."""
import argparse
import json
import math
from pathlib import Path
from w2v_aasist.launch import ROOT, BASELINE_SHA256
from w2v_aasist.runtime import sha256

DEFAULT_SOURCE = ROOT / 'exp' / 'w2v_v3_20260930_181111_5eca'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', default=str(DEFAULT_SOURCE))
    p.add_argument('--resume', help='V3.1 run directory; restores the saved recipe exactly')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--amp', choices=('bf16', 'none'), default='bf16')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--encoder-lr', type=float, default=5e-8)
    p.add_argument('--head-lr', type=float, default=2e-6)
    p.add_argument('--short-loss-weight', type=float, default=.3)
    p.add_argument('--short-min-seconds', type=float, default=3.)
    p.add_argument('--short-max-seconds', type=float, default=6.)
    p.add_argument('--activation-budget-gib', type=float, default=0.)
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    return p


def configuration(args):
    source = Path(args.source_run).expanduser().resolve()
    path = source / 'config.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if 'MultiConv' not in cfg.get('algorithm', '') or not cfg.get('full_noisy'):
        raise ValueError('Source must be the full-duration V3 MultiConv run')
    if args.workers < 0 or args.smoke_steps < 0 or not 1 <= args.epochs <= 2:
        raise ValueError('Workers/smoke steps must be nonnegative; adaptation budget is 1 or 2 epochs')
    for v in (args.encoder_lr, args.head_lr, args.short_min_seconds, args.short_max_seconds):
        if not math.isfinite(v) or v <= 0:
            raise ValueError('Learning rates and crop lengths must be finite and positive')
    if not .4 <= args.short_min_seconds <= args.short_max_seconds:
        raise ValueError('Short waveform lengths must satisfy .4 <= min <= max')
    if not math.isfinite(args.short_loss_weight) or not 0 < args.short_loss_weight < 1:
        raise ValueError('short_loss_weight must be strictly between 0 and 1')
    if not math.isfinite(args.activation_budget_gib) or args.activation_budget_gib < 0:
        raise ValueError('Invalid host activation memory budget')
    if cfg.get('baseline_sha256') != BASELINE_SHA256 or sha256(cfg['baseline']) != BASELINE_SHA256:
        raise ValueError('Protected original 91.68 checkpoint differs')
    warm = source / 'best_model.pt'
    cfg.update(version='3.1', source_run=str(source), source_config_sha256=sha256(path),
        warm_checkpoint=str(warm), warm_checkpoint_sha256=sha256(warm),
        device=args.device, workers=args.workers, amp=args.amp, eval_amp='none',
        head_epochs=0, joint_epochs=args.epochs, trainable_layers=4,
        head_lr=args.head_lr, joint_head_lr=args.head_lr, encoder_lr=args.encoder_lr,
        joint_warmup_steps=100, lr_warmup_steps=100, min_lr_scale=.2,
        noisy_weight_start=.5, noisy_weight=.5, noisy_ramp_epochs=0.,
        cka_weight=.01, cka_warmup_steps=0, evals_per_epoch=2,
        short_loss_weight=args.short_loss_weight, short_min_seconds=args.short_min_seconds,
        short_max_seconds=args.short_max_seconds, short_prefix_probability=.5,
        offload_activations=True, activation_budget_gib=args.activation_budget_gib,
        plateau_evals=2, reduced_lr_evals=2, max_lr_reductions=1, drift_rescues=1,
        lr_factor=.5, en_real_tolerance=.005, noisy_fake_tolerance=.003,
        clean_tolerance=.002, selection_min_delta=.0001, metric_epsilon=1e-12,
        input_policy='full utterance at inference; waveform-level full/short supervision in Train',
        optimizer_initialization='fresh AdamW for new objective; paired moments retained on all in-run restores',
        coverage_policy='every ordinary row and both full noisy versions once; short views share each row loss budget',
        algorithm='w2v-BERT 2.0 + MultiConv; V3.1 full/short supervised adaptation; full-view CKA')
    return cfg
