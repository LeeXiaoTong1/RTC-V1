"""One audited training engine for all three stages. Run with -m w2v_rebuild.train."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import re
import time
import numpy as np
import torch
from tqdm import tqdm
from . import SCHEMA
from .model import Detector, forward_chunks
from .data import DataBundle, combine_batches
from .storage import checkpoint_headroom, protect_checkpoint_space, require_space
from .selection import candidate_decision, guard_metrics, noisy_metrics, selection_key
from .control import AdaptationControl, NoisyAdaptationControl
from .group_metrics import GroupMetrics, DevGroupRecorder
from .core import (Metrics, Schedule, objective, build_optimizer, rng_state, restore_rng,
                   atomic_save, atomic_json, sha256, load_checkpoint, gradient_audit, finish_audit, materialize_stats)

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
    p.add_argument('--finetune_from', help='Stage3 weights for a NEW Stage3 experiment; resets optimizer/scheduler')
    p.add_argument('--trainable_encoder_layers', type=int, default=24,
                   help='New Stage3 adaptation only: train the last N layers; freeze the prefix and its dropout')
    p.add_argument('--real_ce_weight', type=float, default=1., help='Additional real-class error cost; default preserves legacy CE')
    p.add_argument('--language_weighting', action='store_true', help='Class-conditional en/zh CE budgets; complete ordinary traversal')
    p.add_argument('--en_real_budget', type=float, default=.35)
    p.add_argument('--en_fake_budget', type=float, default=.40)
    p.add_argument('--coverage_training', action='store_true',
                   help='Opt-in ordinary mixed crops and condition-stratified cached noisy sampling')
    p.add_argument('--coverage_prefix_probability', type=float, default=.5)
    p.add_argument('--guard_baseline', action='store_true', help='Keep the starting best unless real recall and condition F1 floors pass')
    p.add_argument('--adaptation_control', action='store_true',
                   help='Separate score candidate from targeted best; early stop on actual progress')
    p.add_argument('--noisy_extra_fraction', type=float, default=0.,
                   help='Scheduled fraction of noisy pair batches drawn from extra banks; 0 preserves old rotation')
    p.add_argument('--noisy_mix_warmup_epochs', type=float, default=1.)
    p.add_argument('--ordinary_sampling', choices=['legacy', 'balanced'], default='legacy')
    p.add_argument('--noisy_bank_policy', choices=['cycle', 'mixed'], default='cycle')
    p.add_argument('--consistency_weight', type=float, default=0.)
    p.add_argument('--consistency_confidence', type=float, default=.8)
    p.add_argument('--local_structure_weight', type=float, default=0.,
                   help='Opt-in local relation loss replacing noisy global InfoNCE; same-source pairs only')
    p.add_argument('--local_structure_warmup_steps', type=int, default=100,
                   help='Local objective ramp independent of the legacy multi-epoch pair ramp')
    p.add_argument('--noisy_selection', action='store_true',
                   help='Rank noisy F1 first; promote only with better noisy/weighted F1 and protected clean F1')
    p.add_argument('--feature_cache', help='Optional lossless fixed-waveform feature cache directory')
    p.add_argument('--no_feature_cache', action='store_true', help='Recompute fixed inputs; do not read/write feature caches')
    p.add_argument('--noise_cache_mb', type=int, default=128, help='Bounded decoded-noise cache per worker; 0 disables')
    p.add_argument('--profile_steps', type=int, default=0, help='Synchronize/time this many training steps (diagnostic only)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', choices=['bf16', 'none'], default='bf16')
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--eval_batch', type=int, default=16)
    p.add_argument('--eval_microbatch', type=int, default=4, help='Reference inference kernel batch size; default unchanged')
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
    paths += [root/p for p in ['utils/data_utils.py', 'utils/coverage_crop.py', 'utils/env_noise.py', 'utils/RawBoost.py',
                               'utils/rtc_data.py', 'utils/rtc_pairs.py', 'rtc_noisy/data.py',
                               'rtc_noisy/common.py', 'rtc_noisy_v2/plan.py', 'rtc_noisy_v2/cache.py',
                               'rtc_noisy_v2/sampling.py', 'rtc_noisy/diverse.py', 'rtc_noisy/simulator.py']]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def archive_uncommitted_diagnostics(out, start_epoch):
    """Keep a failed attempt's reports before replaying its uncommitted epoch."""
    paths = []
    for path in out.iterdir():
        match = re.fullmatch(r'epoch_(\d+)_(scores\.jsonl|evaluation\.json)', path.name)
        if match and int(match.group(1)) >= start_epoch:
            if path.is_symlink() or not path.is_file():
                raise ValueError('Unexpected uncommitted diagnostic path: '+str(path))
            paths.append(path)
    if paths:
        folder = out/'interrupted_diagnostics'
        if folder.is_symlink():
            raise ValueError('Interrupted diagnostic directory must not be a symlink')
        destination = folder/str(time.time_ns())
        destination.mkdir(parents=True, exist_ok=False)
        for path in paths:
            path.rename(destination/path.name)
        print('Preserved uncommitted evaluation reports: '+str(destination), flush=True)


@torch.no_grad()
def validate(model, bundle, device, microbatch, score_path=None):
    if score_path is None:
        return _validate(model, bundle, device, microbatch)
    with DevGroupRecorder(score_path) as recorder:
        report = _validate(model, bundle, device, microbatch, recorder)
        report['language_groups'] = recorder.result()
    return report


def _validate(model, bundle, device, microbatch, recorder=None):
    model.eval()
    begin = time.perf_counter()
    times = {}
    meters = {k: Metrics() for k in ['all', 'online', 'offline']}
    for b in tqdm(bundle.dev['clean'], desc='Dev clean', leave=False):
        if not bool(b['mask'].bool().all()):
            raise ValueError('Invalid CPU validation mask')
        logits, _ = forward_chunks(model, b['features'].to(device, non_blocking=True),
                                  b['mask'].to(device, non_blocking=True), microbatch,
                                  pad_last=True, validated_mask=True)
        z, y = logits.cpu(), b['labels']
        meters['all'].update(z, y)
        for kind in ('online', 'offline'):
            select = torch.tensor([kind in Path(x).parts for x in b['ids']])
            meters[kind].update(z[select], y[select])
            if recorder is not None:
                ids = [source for source, chosen in zip(b['ids'], select.tolist()) if chosen]
                recorder.update(kind, z[select], y[select], ids)
    report = {k: m.result() for k, m in meters.items()}
    times['clean'] = time.perf_counter() - begin
    for kind in ('seen', 'heldout'):
        begin = time.perf_counter()
        bands = [Metrics() for _ in range(4)]
        for b in tqdm(bundle.dev[kind], desc='Dev '+kind, leave=False):
            if not bool(b['mask'].bool().all()):
                raise ValueError('Invalid CPU validation mask')
            logits, _ = forward_chunks(model, b['features'].to(device, non_blocking=True),
                                      b['mask'].to(device, non_blocking=True), microbatch,
                                      pad_last=True, validated_mask=True)
            z, y, labels = logits.cpu(), b['labels'], torch.tensor(b['bands'])
            if recorder is not None:
                recorder.update(kind, z, y, b['source_ids'], b['bands'])
            for i, m in enumerate(bands):
                pick = labels == i
                m.update(z[pick], y[pick])
        parts = [m.result() for m in bands]
        report[kind] = {'macro_f1': sum(x['macro_f1'] for x in parts)/4,
                        'balanced_ce': sum(x['balanced_ce'] for x in parts)/4, 'bands': parts}
        times[kind] = time.perf_counter() - begin
    report['robust_f1'] = .3*report['online']['macro_f1'] + .35*report['seen']['macro_f1'] + .35*report['heldout']['macro_f1']
    report['robust_ce'] = .3*report['online']['balanced_ce'] + .35*report['seen']['balanced_ce'] + .35*report['heldout']['balanced_ce']
    report['seconds_by_condition'] = times
    return report


def print_validation(dev):
    print(f"Dev OnlineF1={100*dev['online']['macro_f1']:.3f} SeenF1={100*dev['seen']['macro_f1']:.3f} "
          f"HeldoutF1={100*dev['heldout']['macro_f1']:.3f} RobustF1={100*dev['robust_f1']:.3f} "
          f"RobustBalancedCE={dev['robust_ce']:.6f}", flush=True)
    print('Online: recall[fake,real]=', dev['online']['recall'], 'confusion=', dev['online']['confusion'],
          'predict_fake_fraction=', dev['online']['fake_prediction_fraction'], flush=True)
    for kind in ('seen', 'heldout'):
        print(f"{kind}: mean_recall[fake,real]=" + str([
            round(100*sum(b['recall'][c] for b in dev[kind]['bands'])/4, 3) for c in (0, 1)]), flush=True)
        print(f"{kind}: real_recall_by_band=" + str([round(100*b['recall'][1], 3) for b in dev[kind]['bands']]), flush=True)


def train_step(model, opt, schedule, b, layout, weights, args, stage_step, pair_steps, audit=False):
    # from_pretrained may leave the backbone in eval while the outer Detector
    # already reports training=True; always restore every child's training mode.
    model.train()
    # Validate pair/mask invariants once on CPU, not separately in ten GPU chunks.
    y = b['labels']
    n, r, s = layout
    if y.device.type != 'cpu' or b['mask'].device.type != 'cpu':
        raise ValueError('Training batches must be validated on CPU before transfer')
    if len(y) != n + 2*r + 2*s or not bool(((y == 0) | (y == 1)).all()) or not bool(b['mask'].bool().all()):
        raise ValueError('Invalid CPU labels/layout/mask')
    for offset, pairs in ((n, r), (n + 2*r, s)):
        if pairs and (not torch.equal(y[offset:offset+pairs], y[offset+pairs:offset+2*pairs])
                      or y[offset:offset+pairs].unique().numel() != 2):
            raise ValueError('Each pair group needs matching labels and both classes')
    language_weights = None
    if getattr(args, 'language_weighting', False):
        if 'languages' not in b or 'language_weights' not in b:
            raise ValueError('Language-weighted training requires source metadata')
        languages, language_weights = b['languages'], b['language_weights']
        if (languages.device.type != 'cpu' or language_weights.device.type != 'cpu'
                or languages.shape != y.shape or language_weights.shape != y.shape
                or not bool(((languages == 0) | (languages == 1)).all())
                or not bool((torch.isfinite(language_weights) & (language_weights > 0)).all())):
            raise ValueError('Invalid CPU language metadata')
        for offset, pairs in ((n, r), (n + 2*r, s)):
            if pairs and any(not torch.equal(v[offset:offset+pairs], v[offset+pairs:offset+2*pairs])
                             for v in (languages, language_weights)):
                raise ValueError('Paired language metadata differs between views')
    schedule.before_step(stage_step)
    opt.zero_grad(set_to_none=True)
    device = torch.device(args.device)
    profile = device.type == 'cuda' and stage_step < getattr(args, 'profile_steps', 0)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if profile else None
    stream = torch.cuda.current_stream(device) if profile else None
    if events:
        events[0].record(stream)
    features, mask, labels = (b[k].to(device, non_blocking=True) for k in ('features', 'mask', 'labels'))
    if events:
        events[1].record(stream)
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and args.amp == 'bf16' else nullcontext()
    local_weight = getattr(args, 'local_structure_weight', 0.)
    with ctx:
        if local_weight:
            if not s:
                raise ValueError('Local structure requires same-source noisy pairs')
            logits, readout, frames = forward_chunks(model, features, mask, args.microbatch,
                                                     validated_mask=True, return_frames=True)
        else:
            logits, readout = forward_chunks(model, features, mask, args.microbatch, validated_mask=True)
    ramp = min(1., (stage_step+1)/max(1, pair_steps))
    rw = .1*ramp if args.stage == 2 else (.1 if args.stage == 3 else 0.)
    nw = .1*ramp if args.stage == 3 and not local_weight else 0.
    cw = getattr(args, 'consistency_weight', 0.) * ramp
    loss, stats = objective(logits, readout, labels, *layout, weights, rw, nw,
                            consistency_weight=cw,
                            consistency_confidence=getattr(args, 'consistency_confidence', .8),
                            tensor_stats=True, check_labels=False,
                            real_ce_weight=getattr(args, 'real_ce_weight', 1.),
                            language_weights=language_weights.to(device, non_blocking=True) if language_weights is not None else None)
    if local_weight:
        from .local_structure import local_structure_loss
        first = n + 2*r
        local, local_stats = local_structure_loss(frames[first:first+s], frames[first+s:first+2*s])
        accepted = local_stats.pop('accept_by_pair', None)
        usable_pairs = local_stats.pop('valid_by_pair', None)
        stats.update({'structure_'+key: value for key, value in local_stats.items()})
        stats['structure_loss'] = local.detach()
        local_ramp = min(1., (stage_step+1)/max(1, getattr(args, 'local_structure_warmup_steps', 100)))
        stats['structure_weight'] = logits.new_tensor(local_weight*local_ramp)
        if accepted is not None:
            # Aggregate per-group eligibility without rerunning the backbone/loss.
            for label, label_name in ((0, 'fake'), (1, 'real')):
                for language, language_name in ((0, 'en'), (1, 'zh')):
                    if 'languages' not in b:
                        continue
                    pick = ((b['labels'][first:first+s] == label) &
                            (b['languages'][first:first+s] == language)).to(device)
                    stats[f'structure_{language_name}_{label_name}_accepted_sum'] = (accepted*pick).sum().detach()
                    stats[f'structure_{language_name}_{label_name}_pairs'] = pick.sum().float().detach()
                    if usable_pairs is not None:
                        stats[f'structure_{language_name}_{label_name}_usable_sum'] = (usable_pairs*pick).sum().float().detach()
        loss = loss + local_weight*local_ramp*local
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite loss: stop instead of silently accepting corrupt updates')
    if events:
        events[2].record(stream)
    loss.backward()
    audits = gradient_audit(model) if audit else None
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
    if events:
        events[3].record(stream)
    opt.step()
    if events:
        events[4].record(stream)
        events[4].synchronize()
        stats['profile_ms'] = {k: events[i].elapsed_time(events[i+1]) for i, k in
                               enumerate(('h2d', 'forward_loss', 'backward_clip', 'optimizer'))}
    if audit:
        stats['gradient_audit'] = finish_audit(*audits)
    stats.update(loss=loss.detach(), grad_norm=gnorm.detach(), rtc_weight=rw, noisy_weight=nw, consistency_weight=cw)
    return stats, logits.detach(), labels.detach()


def main():
    args = parser().parse_args()
    if args.no_feature_cache:
        args.feature_cache = None
    defaults = DEFAULTS[args.stage]
    args.epochs = args.epochs if args.epochs is not None else defaults[0]
    args.encoder_lr = args.encoder_lr if args.encoder_lr is not None else defaults[1]
    args.head_lr = args.head_lr if args.head_lr is not None else defaults[2]
    for name in ('encoder_lr', 'head_lr', 'grad_clip'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(name+' must be positive and finite')
    if min(args.epochs, args.microbatch, args.eval_batch, args.eval_microbatch, args.patience, args.earlystop) < 1 or args.num_workers < 0:
        raise ValueError('Invalid counts')
    if args.warmup_epochs < 0 or args.pair_warmup_epochs < 0:
        raise ValueError('Warmup must be nonnegative')
    if sum(bool(x) for x in (args.init, args.resume, args.finetune_from)) > 1 or args.preflight and args.resume:
        raise ValueError('--init/--resume/--finetune_from are exclusive; preflight cannot resume')
    if args.finetune_from and args.stage != 3:
        raise ValueError('--finetune_from is only for a new Stage3 experiment')
    if not 0 <= args.trainable_encoder_layers <= 24:
        raise ValueError('trainable_encoder_layers must be between 0 and 24')
    if not math.isfinite(args.real_ce_weight) or args.real_ce_weight <= 0:
        raise ValueError('real_ce_weight must be positive and finite')
    if any(not math.isfinite(q) or not 0 < q < 1 for q in (args.en_real_budget, args.en_fake_budget)):
        raise ValueError('English class-conditional budgets must lie strictly between 0 and 1')
    if args.language_weighting and (args.stage != 3 or not (args.finetune_from or args.resume or args.check_data)
                                   or args.ordinary_sampling != 'legacy'):
        raise ValueError('Language weighting requires Stage3 adaptation and complete ordinary traversal')
    if not math.isfinite(args.coverage_prefix_probability) or not 0 <= args.coverage_prefix_probability <= 1:
        raise ValueError('coverage_prefix_probability must lie in [0, 1]')
    if args.coverage_training and (not args.language_weighting or args.stage != 3
                                  or args.ordinary_sampling != 'legacy' or args.noisy_bank_policy != 'cycle'
                                  or not args.noisy_extra_fraction):
        raise ValueError('Coverage requires Stage3 language metadata, legacy ordinary traversal and scheduled noisy mixture')
    if (args.trainable_encoder_layers != 24 or args.guard_baseline or args.adaptation_control) and not (
            args.stage == 3 and (args.finetune_from or args.resume)):
        raise ValueError('Partial freezing/baseline guard require Stage3 adaptation or its exact resume')
    if (not math.isfinite(args.noisy_extra_fraction) or not 0 <= args.noisy_extra_fraction <= 1
            or not math.isfinite(args.noisy_mix_warmup_epochs) or args.noisy_mix_warmup_epochs < 0):
        raise ValueError('Invalid noisy mixture settings')
    if args.noisy_extra_fraction and (args.stage != 3 or not args.extra_train_noisy_cache
                                     or args.noisy_bank_policy != 'cycle'):
        raise ValueError('Scheduled noisy mixing needs Stage3, extra caches and cycle policy')
    if (not math.isfinite(args.consistency_weight) or args.consistency_weight < 0
            or not .5 <= args.consistency_confidence <= 1 or args.noise_cache_mb < 0 or args.profile_steps < 0):
        raise ValueError('Invalid consistency/cache/profiling settings')
    if args.consistency_weight and args.stage != 3:
        raise ValueError('Consistency requires Stage3 noisy pairs')
    if (not math.isfinite(args.local_structure_weight) or args.local_structure_weight < 0
            or args.local_structure_warmup_steps < 0):
        raise ValueError('local_structure_weight must be finite and nonnegative')
    if args.local_structure_weight and (args.stage != 3 or args.consistency_weight):
        raise ValueError('Local structure requires Stage3; do not combine with score consistency')
    if args.noisy_selection and not (args.adaptation_control and args.guard_baseline):
        raise ValueError('noisy_selection requires adaptation_control and guard_baseline')
    if args.stage == 1 and args.init:
        raise ValueError('Stage 1 starts from official generic-speech pretraining, not a detector checkpoint')
    if args.stage > 1 and not (args.init or args.resume or args.finetune_from or args.check_data):
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
    if args.finetune_from:
        source_dir = Path(args.finetune_from).resolve().parent
        if out == source_dir or source_dir in out.parents:
            raise ValueError('Use a NEW experiment outside the baseline checkpoint directory')
    if not args.preflight and not args.check_data and out.exists() and (out/'config.json').exists() and not args.resume:
        raise FileExistsError('Experiment already exists. Use --resume or choose a new run directory.')
    os.environ['RTC_NOISE_CACHE_MB'] = str(args.noise_cache_mb)
    print('Loading protocols and checking cached audio; this may take several minutes.', flush=True)
    bundle = DataBundle(args)
    if args.noisy_extra_fraction:
        bundle.rotation.configure_mixture(args.noisy_extra_fraction, round(bundle.steps*args.noisy_mix_warmup_epochs))
    if args.check_data:
        print('PASS: V2 protocols, pair manifests, noise split and all cache roles checked; no model loaded')
        return
    codes = source_hashes()
    initial_path = args.resume or args.init or args.finetune_from
    ckpt = load_checkpoint(initial_path) if initial_path else None
    if ckpt and ckpt['stage'] != (args.stage if args.resume or args.finetune_from else args.stage - 1):
        raise ValueError('Checkpoint stage does not match requested progression')
    if args.resume and ckpt.get('kind') != 'training':
        raise ValueError('--resume requires last.pt, not best_model.pt')
    if args.finetune_from:
        for path, digest in bundle.base_fingerprints.items():
            if ckpt['data_fingerprints'].get(path) != digest:
                raise ValueError('Fine-tune changed base protocol/pairs/noise/preprocessor: '+path)
    elif ckpt and ckpt['data_fingerprints'] != bundle.fingerprints:
        raise ValueError('Dataset/cache/protocol changed since prior checkpoint')
    if args.resume and ckpt['source_hashes'] != codes:
        raise ValueError('Code differs from resume state. Do not claim exact resume under changed code.')
    print('Loading detector checkpoint and configuring trainable layers.', flush=True)
    model = Detector.load(args.ssl_path, ckpt['model_config'] if ckpt else None,
                          checkpointing=not args.no_checkpointing).to(device)
    if ckpt:
        model.load_state_dict(ckpt['model'], strict=True)
    if len(model.backbone.encoder.layers) != 24:
        raise RuntimeError('Expected the original 24-layer encoder')
    model.configure_trainable_layers(args.trainable_encoder_layers)
    opt = build_optimizer(model, args.encoder_lr, args.head_lr, args.weight_decay)
    extra_weight_files = ('candidate_best.pt',) if args.adaptation_control else ()
    best_bytes, full_bytes = protect_checkpoint_space(out, model, args.feature_cache, extra_weight_files)
    device_weights = bundle.weights.to(device)
    schedule = Schedule(opt, round(bundle.steps*args.warmup_epochs), patience=args.patience)
    config = vars(args).copy()
    config.update(model_config=model.backbone.config.to_dict(), source_hashes=codes,
                  data_fingerprints=bundle.fingerprints, logical_batch=40,
                  class_counts=bundle.counts.tolist(), class_weights=bundle.weights.tolist(),
                  torch_version=str(torch.__version__), pretrained_origin='facebook/w2v-bert-2.0',
                  init_sha256=sha256(args.init or args.finetune_from) if args.init or args.finetune_from else None,
                  adaptation=bool(args.finetune_from), baseline_path=args.finetune_from,
                  noise_environment={k: os.environ.get(k, v) for k, v in
                                     [('RTC_B_NOISE_PROB', '0.5'), ('RTC_B_SNR_MIN', '10'), ('RTC_B_SNR_MAX', '30')]})
    if args.language_weighting:
        config['language_budgets'] = bundle.language_budgets
    if args.local_structure_weight:
        from dataclasses import asdict
        from .local_structure import LocalStructureConfig
        config['local_structure_config'] = asdict(LocalStructureConfig())
    control_class = NoisyAdaptationControl if args.noisy_selection else AdaptationControl
    start_epoch, global_step, best_key, stale = 1, 0, (-math.inf, -math.inf), 0
    baseline_dev, best_epoch, control = None, 0, None
    if args.resume:
        for name in ('stage', 'epochs', 'microbatch', 'encoder_lr', 'head_lr', 'weight_decay', 'seed', 'amp',
                     'warmup_epochs', 'pair_warmup_epochs', 'grad_clip', 'patience', 'earlystop', 'no_checkpointing', 'torch_version',
                     'ordinary_sampling', 'noisy_bank_policy', 'consistency_weight', 'consistency_confidence', 'eval_microbatch', 'eval_batch', 'noise_environment'):
            if ckpt['config'][name] != config[name]:
                raise ValueError(f'Resume setting changed: {name}')
        for name in ('trainable_encoder_layers', 'real_ce_weight', 'guard_baseline'):
            if ckpt['config'].get(name) != config[name]:
                raise ValueError(f'Resume setting changed: {name}')
        for name, default in (('adaptation_control', False), ('noisy_extra_fraction', 0.),
                              ('noisy_mix_warmup_epochs', 1.), ('language_weighting', False),
                              ('en_real_budget', .35), ('en_fake_budget', .40),
                              ('coverage_training', False), ('coverage_prefix_probability', .5),
                              ('local_structure_weight', 0.), ('local_structure_warmup_steps', 100), ('noisy_selection', False)):
            if ckpt['config'].get(name, default) != config[name]:
                raise ValueError(f'Resume setting changed: {name}')
        if args.language_weighting and ckpt['config'].get('language_budgets') != config['language_budgets']:
            raise ValueError('Language source pools or coefficients changed on resume')
        baseline_dev = ckpt.get('baseline_dev')
        best_epoch = ckpt.get('best_epoch', 0)
        if (args.guard_baseline or args.adaptation_control) and baseline_dev is None:
            raise ValueError('Guarded resume requires the original baseline metrics')
        if args.adaptation_control:
            control = control_class(baseline_dev)
            control.load_state_dict(ckpt['adaptation_control_state'])
            if control.candidate_epoch and not (out/'candidate_best.pt').is_file():
                raise FileNotFoundError('Candidate checkpoint missing; restore it before exact resume')
        if Path(args.resume).resolve().parent != out:
            raise ValueError('Resume into the SAME experiment directory so best_model.pt remains available')
        opt.load_state_dict(ckpt['optimizer'])
        schedule.load_state_dict(ckpt['schedule'])
        bundle.load_sampler_state(ckpt['sampler'])
        start_epoch, global_step = ckpt['epoch'] + 1, ckpt['global_step']
        best_key, stale = tuple(ckpt['best_key']), ckpt['stale']
        restore_rng(ckpt['rng'])
        for key in ('init_sha256', 'adaptation', 'baseline_path'):
            config[key] = ckpt['config'].get(key)
    del ckpt
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and args.language_weighting:
        archive_uncommitted_diagnostics(out, start_epoch)
    if not args.resume:
        atomic_json(config, out/'config.json') if not args.preflight else None
    if args.language_weighting and not args.preflight:
        atomic_json(bundle.language_budgets, out/'language_budget.json')
    print(f'REBUILD Stage {args.stage}: last {args.trainable_encoder_layers}/24 encoder layers trainable; '
          f'feature projection trainable={args.trainable_encoder_layers == 24}; no BatchNorm; '
          f'logical=40 micro={args.microbatch}; encoder={args.encoder_lr:.2e} head={args.head_lr:.2e}', flush=True)
    print(f'Real CE cost={args.real_ce_weight}; baseline guard={args.guard_baseline}', flush=True)
    if args.noisy_selection:
        print('Selection: noisy F1 first; promotion requires higher noisy and weighted F1, '
              'with no clean Online F1 decrease. Class/language recall changes remain diagnostic.', flush=True)
    elif args.adaptation_control:
        print('Selection: independent score candidate; best requires higher RobustF1 and no decrease '
              'in Online real recall, mean noisy real recall and mean noisy F1. Offline is diagnostic.', flush=True)

    def snapshot(epoch, dev, kind):
        state = {'schema': SCHEMA, 'kind': kind, 'stage': args.stage, 'epoch': epoch,
                 'model': model.state_dict(), 'model_config': model.backbone.config.to_dict(),
                 'config': config, 'data_fingerprints': bundle.fingerprints, 'source_hashes': codes, 'dev': dev}
        if kind == 'training':
            state.update(optimizer=opt.state_dict(), schedule=schedule.state_dict(), rng=rng_state(),
                         sampler=bundle.sampler_state(), global_step=global_step, best_key=best_key, stale=stale,
                         baseline_dev=baseline_dev, best_epoch=best_epoch)
            if control is not None:
                state['adaptation_control_state'] = control.state_dict()
        return state

    if args.preflight:
        bundle.begin(1)
        b, layout = combine_batches([next(iter(loader)) for loader in bundle.train])  # materialized below
        stats, _, _ = train_step(model, opt, schedule, b, layout, device_weights, args, 0,
                                 round(bundle.steps*args.pair_warmup_epochs), audit=True)
        stats = materialize_stats(stats)
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
        print('PASS: real Train batch; configured trainable modules updated, frozen modules have no gradients; '
              'fixed-shape evaluation independence checked.')
        return

    if not args.resume and args.stage > 1:
        print('Evaluating the starting checkpoint on fixed Dev conditions before any update.', flush=True)
        dev = (validate(model, bundle, device, args.eval_microbatch, out/'baseline_scores.jsonl')
               if args.language_weighting else validate(model, bundle, device, args.eval_microbatch))
        baseline_dev = dev
        schedule.anchor(dev['robust_ce'])
        best_key = selection_key(dev, args.noisy_selection)
        if args.adaptation_control:
            control = control_class(dev)
        atomic_save(snapshot(0, dev, 'weights'), out/'best_model.pt')
        atomic_json(dev, out/'baseline_dev.json')
        print_validation(dev)
        if args.noisy_selection:
            print('Target floors:', {'online_f1': dev['online']['macro_f1'],
                                     'robust_f1': dev['robust_f1'], 'noisy_f1': noisy_metrics(dev)['noisy_f1']}, flush=True)
        elif args.adaptation_control:
            print('Target floors:', {'online_real_recall': baseline_dev['online']['recall'][1],
                                     **noisy_metrics(baseline_dev)}, flush=True)
        elif args.guard_baseline:
            print('Baseline floors:', guard_metrics(baseline_dev), flush=True)
    with (out/'metrics.jsonl').open('a', encoding='utf-8') as log:
        for epoch in range(start_epoch, args.epochs + 1):
            require_space(out, checkpoint_headroom(out, best_bytes, full_bytes, extra_weight_files), 'Before training epoch')
            bundle.begin(epoch)
            mixture_report = None
            if args.noisy_extra_fraction:
                mixture_report = bundle.rotation.plan_summary()
                print('Noisy mixture: '+json.dumps({k: v for k, v in mixture_report.items() if k != 'cells'}), flush=True)
            coverage_meter = None
            if args.coverage_training:
                from .coverage import CoverageMeter
                coverage_meter = CoverageMeter()
                atomic_json(mixture_report, out/f'coverage_plan_epoch_{epoch:03d}.json')
                print('Coverage: mixed ordinary crops; language/family/SNR quotas; no noisy language CE multiplier.', flush=True)
            meter, loss_total, last_stats = Metrics(device=device, check_finite=False), torch.zeros((), dtype=torch.float64, device=device), {}
            group_meters = ({name: GroupMetrics(device=device) for name in
                             ('all', 'ordinary', 'rtc_pair', 'noisy_reference', 'noisy_processed')}
                            if args.language_weighting else {})
            begin = time.perf_counter()
            ready, data_wait, profiles = begin, 0., []
            done = 0
            totals = {}
            for batches in tqdm(zip(*bundle.train), total=bundle.steps, desc=f'S{args.stage} epoch {epoch}'):
                data_wait += time.perf_counter() - ready
                b, layout = combine_batches(batches)
                do_audit = done == 0
                stats, logits, labels = train_step(model, opt, schedule, b, layout, device_weights, args,
                                                   global_step, round(bundle.steps*args.pair_warmup_epochs), do_audit)
                if coverage_meter is not None:
                    coverage_meter.update(b['coverage_rows'])
                    if (done+1) % 256 == 0 or done == 0:
                        atomic_json(coverage_meter.result(), out/f'coverage_actual_epoch_{epoch:03d}.json')
                if do_audit:
                    atomic_json(stats['gradient_audit'], out/f'gradient_epoch_{epoch:03d}.json')
                meter.update(logits, labels)
                if group_meters:
                    languages = b['languages'].to(device, non_blocking=True)
                    n, r, s = layout
                    ranges = {'all': (0, len(labels)), 'ordinary': (0, n), 'rtc_pair': (n, n+2*r),
                              'noisy_reference': (n+2*r, n+2*r+s), 'noisy_processed': (n+2*r+s, len(labels))}
                    for name, (first, last) in ranges.items():
                        group_meters[name].update(logits[first:last], labels[first:last], languages[first:last])
                loss_total += stats['loss'].double()
                for name in ('ce', 'rtc', 'noisy_pair', 'consistency', 'consistency_accepted',
                             'ce_ordinary', 'ce_real_pair', 'ce_noisy_reference', 'ce_noisy_processed'):
                    if name in stats:
                        totals[name] = totals.get(name, 0.) + stats[name].detach().double()
                for name, value in stats.items():
                    if name.startswith('structure_'):
                        totals[name] = totals.get(name, 0.) + value.detach().double()
                if 'profile_ms' in stats:
                    profiles.append(stats['profile_ms'])
                last_stats = {k: v for k, v in stats.items() if k != 'gradient_audit'}
                global_step += 1
                done += 1
                if (args.local_structure_weight and done == 100
                        and float(totals.get('structure_usable_fraction', 0.)) <= 0):
                    raise RuntimeError('No usable local-structure pair in the first 100 training steps. '
                                       'Stopped this attempt; original best remains protected. '
                                       'Inspect alignment/dropout instead of completing an ineffective epoch.')
                ready = time.perf_counter()
            if done != bundle.steps:
                raise RuntimeError('A stream ended early; do not commit sampler history')
            coverage_report = None
            if coverage_meter is not None:
                coverage_meter.verify(mixture_report, int(bundle.counts.sum()))
                coverage_report = coverage_meter.result()
                atomic_json(coverage_report, out/f'coverage_actual_epoch_{epoch:03d}.json')
                print(f"Coverage completed: ordinary={coverage_report['ordinary_views']} "
                      f"unique_files={coverage_report['ordinary_unique_files']} "
                      f"noisy_processed={coverage_report['noisy_processed_views']}", flush=True)
            bundle.end(done)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - begin
            dev_begin = time.perf_counter()
            dev = (validate(model, bundle, device, args.eval_microbatch, out/f'epoch_{epoch:03d}_scores.jsonl')
                   if args.language_weighting else validate(model, bundle, device, args.eval_microbatch))
            dev_seconds = time.perf_counter() - dev_begin
            print_validation(dev)
            # Preserve the completed evaluation even if a later checkpoint write fails.
            evaluation = {'epoch': epoch, 'steps': global_step, 'train': meter.result(), 'dev': dev,
                          'checkpoint_saved': False}
            train_groups = {name: value.result() for name, value in group_meters.items()}
            if train_groups:
                evaluation['train_groups'] = train_groups
                print('Train language groups (augmented online predictions): '+json.dumps(train_groups['all']), flush=True)
            if coverage_report is not None:
                evaluation['coverage'] = coverage_report
            atomic_json(evaluation, out/f'epoch_{epoch:03d}_evaluation.json')
            key = selection_key(dev, args.noisy_selection)
            improved, selection = candidate_decision(
                dev, best_key, baseline_dev if args.guard_baseline or args.adaptation_control else None,
                policy='noisy' if args.noisy_selection else ('targeted' if args.adaptation_control else 'strict'))
            save_begin = time.perf_counter()
            if improved:
                best_key, stale, best_epoch = key, 0, epoch
                atomic_save(snapshot(epoch, dev, 'weights'), out/'best_model.pt')
            else:
                stale += 1
            best_save_seconds = time.perf_counter() - save_begin
            reduced = schedule.validate(dev['robust_ce'], global_step)
            candidate_saved, progress = False, None
            candidate_save_seconds = 0.
            if control is not None:
                candidate_saved, progress = control.observe(dev, epoch, reduced)
                stale = control.stale
                if candidate_saved:
                    save_begin = time.perf_counter()
                    atomic_save(snapshot(epoch, dev, 'weights'), out/'candidate_best.pt')
                    candidate_save_seconds = time.perf_counter() - save_begin
            row = {'epoch': epoch, 'steps': global_step, 'train': meter.result(),
                   'mean_train_loss': float(loss_total / done), 'last_batch': materialize_stats(last_stats), 'dev': dev,
                   'mean_batch': materialize_stats({name: value/done for name, value in totals.items()}),
                   'selection': selection,
                   'best': improved, 'lr_scale_next': schedule.scale, 'seconds': time.perf_counter()-begin,
                   'used_lr': {g['name']: g['lr'] for g in opt.param_groups}}
            if train_groups:
                row['train_groups'] = train_groups
            if coverage_report is not None:
                row['coverage'] = coverage_report
            if control is not None:
                row.update(candidate_saved=candidate_saved, candidate_epoch=control.candidate_epoch,
                           candidate_key=control.candidate_key, progress=progress)
            if mixture_report is not None:
                row['noisy_mixture'] = mixture_report
            save_begin = time.perf_counter()
            atomic_save(snapshot(epoch, dev, 'training'), out/'last.pt')
            evaluation['checkpoint_saved'] = True
            atomic_json(evaluation, out/f'epoch_{epoch:03d}_evaluation.json')
            row['timing'] = {'train_seconds': train_seconds, 'data_wait_host_seconds': data_wait,
                             'dev_seconds': dev_seconds, 'best_save_seconds': best_save_seconds,
                             'last_save_seconds': time.perf_counter() - save_begin,
                             'profile_steps_ms': profiles}
            row['timing']['candidate_save_seconds'] = candidate_save_seconds
            row['seconds'] = time.perf_counter() - begin
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(f'epoch={epoch} best={improved} plateau_reduced={reduced}; Adam history retained; next_scale={schedule.scale}', flush=True)
            if (args.guard_baseline or args.adaptation_control) and not improved:
                print('Best kept; rejected candidate:', ', '.join(selection['reasons']), flush=True)
            if control is not None:
                print(f'Candidate: saved={candidate_saved}, best_epoch={control.candidate_epoch}; '
                      f'progress={progress}; promotion_warnings={selection["warnings"]}', flush=True)
                if reduced:
                    print('Reduced LR will be used next epoch.' if epoch < args.epochs else
                          'Epoch budget exhausted; the scheduled reduced LR was not used.', flush=True)
            stop = control.should_stop(epoch, args.earlystop) if control is not None else stale >= args.earlystop
            if stop and global_step > schedule.warmup_steps:
                print('Early stopping: no score, noisy F1 or CE progress after the reduced-rate opportunity.'
                      if control is not None else 'Early stopping: no candidate passed all selection conditions')
                break
    completed = {'stage': args.stage, 'best_key': best_key, 'best_model': str(out/'best_model.pt'),
                 'best_epoch': best_epoch, 'status': 'improved' if best_epoch else 'no_eligible_improvement'}
    if control is not None:
        completed.update(candidate_epoch=control.candidate_epoch, candidate_key=control.candidate_key,
                         candidate_model=str(out/'candidate_best.pt') if control.candidate_epoch else None)
        if not best_epoch and control.candidate_epoch:
            completed['status'] = 'candidate_only'
    atomic_json(completed, out/'completed.json')
    if best_epoch == 0:
        print('No eligible improvement: best_model.pt still contains the starting checkpoint weights.', flush=True)
    print('Best model:', out/'best_model.pt')
    if control is not None and control.candidate_epoch:
        print('Score candidate (review real/noisy tradeoffs):', out/'candidate_best.pt', flush=True)


if __name__ == '__main__':
    main()
