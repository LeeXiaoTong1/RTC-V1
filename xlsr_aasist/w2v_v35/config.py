"""Explicit V3.5 recipe; submitted weights are a reference, not initialization."""
import argparse
import math
from pathlib import Path

from w2v_aasist.launch import ROOT, BASELINE_SHA256
from w2v_aasist.runtime import sha256
from w2v_v34.source import resolve_source


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', default=str(ROOT/'exp'/'w2v_v33_20261002_004018_198a'))
    p.add_argument('--submission-meta')
    p.add_argument('--pretrained-path', help='Local generic facebook/w2v-bert-2.0; exact official weight SHA is required')
    p.add_argument('--resume', help='V3.5 run directory; restores its saved recipe')
    p.add_argument('--train-noise-manifest')
    p.add_argument('--dev-noise-manifest')
    p.add_argument('--cache-root', default=str(ROOT/'data'/'rtc_v35'))
    p.add_argument('--ffmpeg')
    p.add_argument('--head-epochs', type=int, default=1)
    p.add_argument('--joint-epochs', type=int, default=5)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--joint-head-lr', type=float, default=1e-5)
    p.add_argument('--encoder-lr', type=float, default=1e-6)
    p.add_argument('--layer-decay', type=float, default=.9)
    p.add_argument('--source-batch', type=int, default=16)
    p.add_argument('--source-chunk', type=int, default=4)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--frame-budget', type=int, default=2400)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--cache-workers', type=int, default=4)
    p.add_argument('--gpu-activation-gib', type=float, default=16.)
    p.add_argument('--gpu-reserve-gib', type=float, default=8.)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', choices=('bf16', 'none'), default='bf16')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    return p


def configuration(args):
    for key in ('source_batch', 'source_chunk', 'microbatch', 'frame_budget', 'cache_workers'):
        if getattr(args, key) < 1:
            raise ValueError(key+' must be positive')
    if args.workers < 0 or args.smoke_steps < 0 or args.head_epochs != 1 or not 1 <= args.joint_epochs <= 5:
        raise ValueError('One head epoch and 1-5 joint epochs; nonnegative workers/smoke steps required')
    for key in ('head_lr', 'joint_head_lr', 'encoder_lr'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError('Invalid '+key)
    if not math.isfinite(args.layer_decay) or not 0 < args.layer_decay <= 1:
        raise ValueError('layer-decay must be in (0,1]')
    if not math.isfinite(args.gpu_activation_gib) or args.gpu_activation_gib < 0 or not math.isfinite(args.gpu_reserve_gib) or args.gpu_reserve_gib < 4:
        raise ValueError('Activation budget >=0 and GPU reserve >=4 GiB required')
    source = resolve_source(args.source_run, args.submission_meta)
    cfg = dict(source['config'])
    if cfg.get('baseline_sha256') != BASELINE_SHA256 or sha256(cfg['baseline']) != BASELINE_SHA256:
        raise ValueError('Protected original 91.68 checkpoint differs')
    train_noise = Path(args.train_noise_manifest or cfg['train_noise_manifest']).expanduser().resolve()
    dev_noise = Path(args.dev_noise_manifest or train_noise.with_name('dev.jsonl')).expanduser().resolve()
    if train_noise == dev_noise or not train_noise.is_file() or not dev_noise.is_file():
        raise ValueError('Distinct existing Train/Dev noise manifests required; use --dev-noise-manifest')
    cache = Path(args.cache_root).expanduser().resolve()
    protected = [Path(cfg[k]).resolve() for k in ('train_data_path', 'dev_data_path', 'dev_noisy_cache', 'dev_heldout_cache')]
    protected += [Path(p).resolve() for p in cfg.get('train_caches', [])]
    if any(cache == p or cache in p.parents or p in cache.parents for p in protected):
        raise ValueError('V3.5 cache root must not overlap existing data or caches')
    previous_ssl = cfg['ssl_path']
    cfg.update(version='3.5', source_run=str(Path(args.source_run).resolve()),
        source_provenance=source['provenance'], reference_checkpoint=source['checkpoint'],
        reference_checkpoint_sha256=source['checkpoint_sha256'], reference_tag=source['checkpoint_tag'],
        reference_ssl_path=previous_ssl, warm_checkpoint=source['checkpoint'],
        warm_checkpoint_sha256=source['checkpoint_sha256'], expected_warm_tag=source['checkpoint_tag'],
        pretrained_path=str(Path(args.pretrained_path).expanduser().resolve()) if args.pretrained_path else None,
        pretrained_candidate_path=previous_ssl,
        train_noise_manifest=str(train_noise), dev_noise_manifest=str(dev_noise),
        cache_root=str(cache), train_cache_root=str(cache/'train'), full_dev_cache_root=str(cache/'dev'),
        ffmpeg=args.ffmpeg or cfg.get('ffmpeg'),
        head_epochs=args.head_epochs, joint_epochs=args.joint_epochs, trainable_layers=24,
        head_lr=args.head_lr, joint_head_lr=args.joint_head_lr, encoder_lr=args.encoder_lr,
        layer_decay=args.layer_decay, warmup_fraction=.05, min_lr_scale=.1,
        source_batch=args.source_batch, source_chunk=args.source_chunk, source_batch_size=args.source_batch,
        microbatch=args.microbatch, frame_budget=args.frame_budget, workers=args.workers,
        cache_workers=args.cache_workers, prefetch_factor=1,
        gpu_activation_gib=args.gpu_activation_gib, gpu_reserve_gib=args.gpu_reserve_gib,
        activation_budget_gib=0., offload_activations=True, checkpointing=True,
        device=args.device, amp=args.amp, eval_amp='none', seed=args.seed,
        evals_per_epoch=1, eval_batch=8, max_seconds=0., full_noisy=True,
        offline_weight=.1, online_weight=.3, noisy_weight=.6, worst_view_weight=.5,
        short_loss_weight=0., rawboost=0, cka_weight=0., pair_weight=0.,
        ema_decay=.999, ema_device='model', early_stop_patience=2,
        selection_min_delta=.0001, weight_decay=1e-4, grad_clip=1.,
        initialization_policy='Exact generic w2v-BERT 2.0 weights plus random MultiConv head; submitted checkpoint is evaluation-only reference',
        algorithm='w2v-BERT 2.0 + MultiConv; V3.5 four-group Online-focused worst-view classification',
        input_policy='full utterance', decision_threshold=.5, score_column='P(fake)',
        coverage_policy='Each unique official Train source once per epoch; four groups have equal cumulative CE coefficient budgets; two new full processed views per epoch',
        cache_policy='Lazy CPU-worker generation; at most current and preceding epoch caches; fixed full Dev is separate',
        production_layout=True)
    # Do not carry old controller/pair experiments into this recipe.
    cfg.pop('source_data_fingerprints', None)
    cfg.pop('arms', None)
    return cfg
