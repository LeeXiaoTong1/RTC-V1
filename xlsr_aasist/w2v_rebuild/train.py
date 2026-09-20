"""One audited training engine for all three stages. Run with -m w2v_rebuild.train."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from tqdm import tqdm
from . import SCHEMA
from .model import Detector, forward_chunks
from .data import DataBundle, combine_batches
from .core import (Metrics, Schedule, objective, build_optimizer, rng_state, restore_rng,
                   atomic_save, atomic_json, sha256, load_checkpoint, gradient_audit, finish_audit)

DEFAULTS = {1: (30, 1e-6, 1e-4), 2: (10, 5e-7, 1e-5), 3: (20, 5e-7, 1e-5)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train_data_path', 'dev_data_path', 'train_protocol', 'dev_protocol', 'rtc_pairs',
                 'train_noise_manifest', 'train_noisy_cache', 'dev_noisy_cache', 'dev_heldout_cache', 'ssl_path', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--stage', type=int, choices=[1, 2, 3], required=True)
    p.add_argument('--extra_train_noisy_cache', action='append', default=[])
    p.add_argument('--init', help='Previous-stage rebuild best_model.pt; NOT an old experiment')
    p.add_argument('--resume', help='Same-stage rebuild last.pt; restores optimizer/scheduler/RNG/rotation')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', choices=['bf16', 'none'], default='bf16')
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--eval_batch', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--epochs', type=int)
    p.add_argument('--encoder_lr', type=float)
    p.add_argument('--head_lr', type=float)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=float, default=1.)
    p.add_argument('--pair_warmup_epochs', type=float, default=2.)
    p.add_argument('--grad_clip', type=float, default=1.)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--earlystop', type=int, default=8)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--no_checkpointing', action='store_true')
    p.add_argument('--check_data', action='store_true')
    p.add_argument('--preflight', action='store_true', help='One real logical batch: backward/update and eval reproducibility; no training checkpoint')
    # The V2 waveform pipeline consumes these exact RawBoost parameters.
    raw = dict(algo=5, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000, minCoeff=10,
               maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20,
               N_f=5, P=10, g_sd=2, SNRmin=10, SNRmax=40)
    for name, value in raw.items():
        p.add_argument('--'+name, type=int, default=value)
    return p


def source_hashes():
    root = Path(__file__).resolve().parent.parent
    paths = list((root/'w2v_rebuild').glob('*.py'))
    paths += [root/p for p in ['utils/data_utils.py', 'utils/env_noise.py', 'utils/RawBoost.py',
                               'utils/rtc_data.py', 'utils/rtc_pairs.py', 'rtc_noisy/data.py',
                               'rtc_noisy/common.py', 'rtc_noisy_v2/plan.py', 'rtc_noisy_v2/cache.py',
                               'rtc_noisy_v2/sampling.py']]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


@torch.no_grad()
def validate(model, bundle, device, microbatch):
    model.eval()
    meters = {k: Metrics() for k in ['all', 'online', 'offline']}
    for b in tqdm(bundle.dev['clean'], desc='Dev clean', leave=False):
        logits, _ = forward_chunks(model, b['features'].to(device), b['mask'].to(device), microbatch, pad_last=True)
        z, y = logits.cpu(), b['labels']
        meters['all'].update(z, y)
        for kind in ('online', 'offline'):
            select = torch.tensor([kind in Path(x).parts for x in b['ids']])
            meters[kind].update(z[select], y[select])
    report = {k: m.result() for k, m in meters.items()}
    for kind in ('seen', 'heldout'):
        bands = [Metrics() for _ in range(4)]
        for b in tqdm(bundle.dev[kind], desc='Dev '+kind, leave=False):
            logits, _ = forward_chunks(model, b['features'].to(device), b['mask'].to(device), microbatch, pad_last=True)
            z, y, labels = logits.cpu(), b['labels'], torch.tensor(b['bands'])
            for i, m in enumerate(bands):
                pick = labels == i
                m.update(z[pick], y[pick])
        parts = [m.result() for m in bands]
        report[kind] = {'macro_f1': sum(x['macro_f1'] for x in parts)/4,
                        'balanced_ce': sum(x['balanced_ce'] for x in parts)/4, 'bands': parts}
    report['robust_f1'] = .3*report['online']['macro_f1'] + .35*report['seen']['macro_f1'] + .35*report['heldout']['macro_f1']
    report['robust_ce'] = .3*report['online']['balanced_ce'] + .35*report['seen']['balanced_ce'] + .35*report['heldout']['balanced_ce']
    return report


def print_validation(dev):
    print(f"Dev OnlineF1={100*dev['online']['macro_f1']:.3f} SeenF1={100*dev['seen']['macro_f1']:.3f} "
          f"HeldoutF1={100*dev['heldout']['macro_f1']:.3f} RobustF1={100*dev['robust_f1']:.3f} "
          f"RobustBalancedCE={dev['robust_ce']:.6f}", flush=True)
    print('Online: recall[fake,real]=', dev['online']['recall'], 'confusion=', dev['online']['confusion'],
          'predict_fake_fraction=', dev['online']['fake_prediction_fraction'], flush=True)


def train_step(model, opt, schedule, b, layout, weights, args, stage_step, pair_steps, audit=False):
    model.train()  # SAME rule in every stage; no hidden .eval() calls inside forward.
    schedule.before_step(stage_step)
    opt.zero_grad(set_to_none=True)
    device = torch.device(args.device)
    features, mask, labels = (b[k].to(device, non_blocking=True) for k in ('features', 'mask', 'labels'))
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and args.amp == 'bf16' else nullcontext()
    with ctx:
        logits, readout = forward_chunks(model, features, mask, args.microbatch)
    ramp = min(1., (stage_step+1)/max(1, pair_steps))
    rw = .1*ramp if args.stage == 2 else (.1 if args.stage == 3 else 0.)
    nw = .1*ramp if args.stage == 3 else 0.
    loss, stats = objective(logits, readout, labels, *layout, weights, rw, nw)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite loss: stop instead of silently accepting corrupt updates')
    loss.backward()
    audits = gradient_audit(model) if audit else None
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
    opt.step()
    if audit:
        stats['gradient_audit'] = finish_audit(*audits)
    stats.update(loss=float(loss.detach()), grad_norm=float(gnorm), rtc_weight=rw, noisy_weight=nw)
    return stats, logits.detach(), labels.detach()


def main():
    args = parser().parse_args()
    defaults = DEFAULTS[args.stage]
    args.epochs = args.epochs if args.epochs is not None else defaults[0]
    args.encoder_lr = args.encoder_lr if args.encoder_lr is not None else defaults[1]
    args.head_lr = args.head_lr if args.head_lr is not None else defaults[2]
    for name in ('encoder_lr', 'head_lr', 'grad_clip'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(name+' must be positive and finite')
    if min(args.epochs, args.microbatch, args.eval_batch, args.patience, args.earlystop) < 1 or args.num_workers < 0:
        raise ValueError('Invalid counts')
    if args.warmup_epochs < 0 or args.pair_warmup_epochs < 0:
        raise ValueError('Warmup must be nonnegative')
    if args.init and args.resume or args.preflight and args.resume:
        raise ValueError('--init/--resume and --preflight/--resume are mutually exclusive')
    if args.stage == 1 and args.init:
        raise ValueError('Stage 1 starts from official generic-speech pretraining, not a detector checkpoint')
    if args.stage > 1 and not (args.init or args.resume or args.check_data):
        raise ValueError('Stage 2/3 need the preceding REBUILD checkpoint')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent fallback')
    if args.amp == 'bf16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support BF16')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = Path(args.out).resolve()
    if not args.preflight and not args.check_data and out.exists() and (out/'config.json').exists() and not args.resume:
        raise FileExistsError('Experiment already exists. Use --resume or choose a new run directory.')
    bundle = DataBundle(args)
    if args.check_data:
        print('PASS: V2 protocols, pair manifests, noise split and all cache roles checked; no model loaded')
        return
    codes = source_hashes()
    ckpt = load_checkpoint(args.resume or args.init) if (args.resume or args.init) else None
    if ckpt and ckpt['stage'] != (args.stage if args.resume else args.stage - 1):
        raise ValueError('Checkpoint stage does not match requested progression')
    if args.resume and ckpt.get('kind') != 'training':
        raise ValueError('--resume requires last.pt, not best_model.pt')
    if ckpt and ckpt['data_fingerprints'] != bundle.fingerprints:
        raise ValueError('Dataset/cache/protocol changed since prior checkpoint')
    if args.resume and ckpt['source_hashes'] != codes:
        raise ValueError('Code differs from resume state. Do not claim exact resume under changed code.')
    model = Detector.load(args.ssl_path, ckpt['model_config'] if ckpt else None,
                          checkpointing=not args.no_checkpointing).to(device)
    if ckpt:
        model.load_state_dict(ckpt['model'], strict=True)
    if len(model.backbone.encoder.layers) != 24 or not all(p.requires_grad for layer in model.backbone.encoder.layers for p in layer.parameters()):
        raise RuntimeError('All 24 pretrained encoder layers must participate in every stage')
    opt = build_optimizer(model, args.encoder_lr, args.head_lr, args.weight_decay)
    schedule = Schedule(opt, round(bundle.steps*args.warmup_epochs), patience=args.patience)
    config = vars(args).copy()
    config.update(model_config=model.backbone.config.to_dict(), source_hashes=codes,
                  data_fingerprints=bundle.fingerprints, logical_batch=40,
                  class_counts=bundle.counts.tolist(), class_weights=bundle.weights.tolist(),
                  torch_version=str(torch.__version__), pretrained_origin='facebook/w2v-bert-2.0',
                  init_sha256=sha256(args.init) if args.init else None)
    start_epoch, global_step, best_key, stale = 1, 0, (-math.inf, -math.inf), 0
    if args.resume:
        for name in ('stage', 'epochs', 'microbatch', 'encoder_lr', 'head_lr', 'weight_decay', 'seed', 'amp',
                     'warmup_epochs', 'pair_warmup_epochs', 'grad_clip', 'patience', 'earlystop', 'no_checkpointing', 'torch_version'):
            if ckpt['config'][name] != config[name]:
                raise ValueError(f'Resume setting changed: {name}')
        if Path(args.resume).resolve().parent != out:
            raise ValueError('Resume into the SAME experiment directory so best_model.pt remains available')
        opt.load_state_dict(ckpt['optimizer'])
        schedule.load_state_dict(ckpt['schedule'])
        bundle.load_sampler_state(ckpt['sampler'])
        start_epoch, global_step = ckpt['epoch'] + 1, ckpt['global_step']
        best_key, stale = tuple(ckpt['best_key']), ckpt['stale']
        restore_rng(ckpt['rng'])
    del ckpt
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        atomic_json(config, out/'config.json') if not args.preflight else None
    print(f'REBUILD Stage {args.stage}: 24/24 layers + feature projection TRAINABLE; no BatchNorm; '
          f'logical=40 micro={args.microbatch}; encoder={args.encoder_lr:.2e} head={args.head_lr:.2e}', flush=True)

    def snapshot(epoch, dev, kind):
        state = {'schema': SCHEMA, 'kind': kind, 'stage': args.stage, 'epoch': epoch,
                 'model': model.state_dict(), 'model_config': model.backbone.config.to_dict(),
                 'config': config, 'data_fingerprints': bundle.fingerprints, 'source_hashes': codes, 'dev': dev}
        if kind == 'training':
            state.update(optimizer=opt.state_dict(), schedule=schedule.state_dict(), rng=rng_state(),
                         sampler=bundle.sampler_state(), global_step=global_step, best_key=best_key, stale=stale)
        return state

    if args.preflight:
        bundle.begin(1)
        b, layout = combine_batches([next(iter(loader)) for loader in bundle.train])  # materialized below
        stats, _, _ = train_step(model, opt, schedule, b, layout, bundle.weights, args, 0,
                                 round(bundle.steps*args.pair_warmup_epochs), audit=True)
        model.eval()
        with torch.no_grad():
            # Evaluate the same first sample with a fixed kernel batch shape:
            # once alongside real companions and once padded with copies of itself.
            # This detects cross-sample normalization without requiring different
            # CUDA batch shapes to be numerically identical through GraphPool top-k.
            take = min(args.microbatch, len(b['features']))
            f = b['features'][:take].to(device)
            m = b['mask'][:take].to(device)
            group, _ = forward_chunks(model, f, m, args.microbatch, pad_last=True)
            single, _ = forward_chunks(model, f[:1], m[:1], args.microbatch, pad_last=True)
        gap = float((group[:1]-single).abs().max())
        if not torch.allclose(group[:1], single, rtol=1e-4, atol=1e-5):
            raise RuntimeError('Evaluation has cross-sample dependence despite fixed batch shape')
        stats['eval_fixed_shape_companion_gap'] = gap
        if device.type == 'cuda':
            stats['peak_cuda_GiB'] = torch.cuda.max_memory_allocated()/1024**3
        atomic_json(stats, out/'preflight.json')
        print('PASS: real Train batch, feature projection + all 24 layers updated; fixed-shape evaluation independence checked.')
        return

    if not args.resume and args.stage > 1:
        dev = validate(model, bundle, device, args.microbatch)
        best_key = (dev['robust_f1'], -dev['robust_ce'])
        atomic_save(snapshot(0, dev, 'weights'), out/'best_model.pt')
        print_validation(dev)
    with (out/'metrics.jsonl').open('a', encoding='utf-8') as log:
        for epoch in range(start_epoch, args.epochs + 1):
            bundle.begin(epoch)
            meter, losses, last_stats = Metrics(), [], {}
            begin = time.perf_counter()
            done = 0
            for batches in tqdm(zip(*bundle.train), total=bundle.steps, desc=f'S{args.stage} epoch {epoch}'):
                b, layout = combine_batches(batches)
                do_audit = done == 0
                stats, logits, labels = train_step(model, opt, schedule, b, layout, bundle.weights, args,
                                                   global_step, round(bundle.steps*args.pair_warmup_epochs), do_audit)
                if do_audit:
                    atomic_json(stats['gradient_audit'], out/f'gradient_epoch_{epoch:03d}.json')
                meter.update(logits, labels)
                losses.append(stats['loss'])
                last_stats = {k: v for k, v in stats.items() if k != 'gradient_audit'}
                global_step += 1
                done += 1
            if done != bundle.steps:
                raise RuntimeError('A stream ended early; do not commit sampler history')
            bundle.end(done)
            dev = validate(model, bundle, device, args.microbatch)
            key = (dev['robust_f1'], -dev['robust_ce'])
            improved = key > best_key
            if improved:
                best_key, stale = key, 0
                atomic_save(snapshot(epoch, dev, 'weights'), out/'best_model.pt')
            else:
                stale += 1
            reduced = schedule.validate(dev['robust_ce'], global_step)
            row = {'epoch': epoch, 'steps': global_step, 'train': meter.result(),
                   'mean_train_loss': float(np.mean(losses)), 'last_batch': last_stats, 'dev': dev,
                   'best': improved, 'lr_scale_next': schedule.scale, 'seconds': time.perf_counter()-begin,
                   'used_lr': {g['name']: g['lr'] for g in opt.param_groups}}
            log.write(json.dumps(row) + '\n')
            log.flush()
            atomic_save(snapshot(epoch, dev, 'training'), out/'last.pt')
            print_validation(dev)
            print(f'epoch={epoch} best={improved} plateau_reduced={reduced}; Adam history retained; next_scale={schedule.scale}', flush=True)
            if stale >= args.earlystop and global_step > schedule.warmup_steps:
                print('Early stopping: no robust Dev improvement for configured patience')
                break
    atomic_json({'stage': args.stage, 'best_key': best_key, 'best_model': str(out/'best_model.pt')}, out/'completed.json')
    print('Best model:', out/'best_model.pt')


if __name__ == '__main__':
    main()
