"""Warm full/short adaptation with a measured baseline fallback and exact resume."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from w2v_aasist.progress import phase, training_bar
from w2v_aasist.runtime import (Metrics, atomic_json, atomic_save, seed_all,
                                sha256, storage_size)
from .control import Controller, lr_scale
from .data import build_data, loader
from w2v_v3.model import Detector
from .step import supervised_step
from w2v_v3.validation import validate

from . import SCHEMA
from w2v_v3.train import (read_state, optimizer_for, rng_state, restore_rng,
                          initialize as initialize_v3, source_fingerprints as v3_fingerprints)


def source_fingerprints():
    root = Path(__file__).resolve().parent.parent
    return {**v3_fingerprints(), **{str(p.relative_to(root)): sha256(p)
            for p in sorted((root/'w2v_v31').glob('*.py'))}}


def initialize(cfg):
    warm = cfg.get('warm_checkpoint')
    if not warm or sha256(warm) != cfg.get('warm_checkpoint_sha256'):
        raise ValueError('V3.1 requires the recorded, unchanged MultiConv best checkpoint')
    state = read_state(warm)
    if state.get('schema') not in ('rtc_w2v_multiconv_v3', SCHEMA) or state.get('kind') != 'weights':
        raise ValueError('Warm start must be a validated V3/V3.1 weight checkpoint, not last.pt')
    del state
    return initialize_v3(cfg)


def restore_selected(model, optimizer, run, expected_tag=None):
    """Restore a paired weight/Adam snapshot; keep controller budget and data cursor."""
    selected = read_state(Path(run) / 'control_best.pt')
    if selected.get('schema') != SCHEMA or 'optimizer' not in selected:
        raise ValueError('Missing matching optimizer state for selected checkpoint')
    if expected_tag is not None and selected['tag'] != expected_tag:
        raise ValueError('Selected optimizer checkpoint does not match the controller validation point')
    model.load_state_dict(selected['model'], strict=True)
    optimizer.load_state_dict(selected['optimizer'])
    print('Restored selected weights AND matching Adam moments: ' + selected['tag'], flush=True)
    return selected


WINNERS = {'best_weighted': 'best_weighted.pt', 'best_noisy': 'best_noisy.pt',
           'best_safe': 'best_model.pt'}


def _checkpoint_tag(path):
    if not Path(path).is_file():
        return None
    state = read_state(path)
    if state.get('schema') != SCHEMA:
        raise ValueError('Foreign checkpoint at a V3 winner path')
    return state['tag']


def recover_winners(run, controller):
    """Reconcile an interrupted promotion with the last committed resume state."""
    expected = {filename: controller.state[key]['tag'] for key, filename in WINNERS.items()}
    expected['control_best.pt'] = controller.state['best_safe']['tag']
    for filename, tag in expected.items():
        path = Path(run)/filename
        previous = path.with_name(path.name+'.previous')
        if _checkpoint_tag(path) != tag:
            if _checkpoint_tag(previous) != tag:
                raise ValueError(f'Cannot resume: {filename} does not match saved validation {tag}')
            os.replace(previous, path)
        elif previous.is_file():
            previous.unlink()


def backup_winners(run, names):
    for name in names:
        path = Path(run)/name
        previous = path.with_name(path.name+'.previous')
        if previous.exists():
            raise RuntimeError('Unreconciled checkpoint promotion: '+str(previous))
        if path.exists():
            os.replace(path, previous)


def commit_winners(run, names):
    for name in names:
        previous = (Path(run)/name).with_name(name+'.previous')
        if previous.exists():
            previous.unlink()


def write_report(run, history, controller, status):
    chosen = controller.state['best_safe']['tag']
    lines = ['# V3.1: full/short MultiConv adaptation', '', f'Status: {status}',
             f'Promoted checkpoint: {chosen}',
             'Fixed Dev and threshold 0.5; these are proxies, not platform scores.',
             'The baseline row is evaluated again before any optimizer update.',
             'Every Train row keeps its full waveform; an auxiliary waveform crop shares its CE budget.', '',
             '| Checkpoint | Clean | Seen | Heldout | Noisy | Weighted | Offline EN real | Online EN real | Seen EN real | Heldout EN real | Noisy EN fake | Decision |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in history:
        d, decision = r['dev'], r['decision']
        values = [f'{100*d[k]:.3f}' for k in ('clean_f1','seen_f1','heldout_f1','noisy_f1','weighted_f1')]
        values += [f'{100*decision["quality"][k]:.3f}' for k in
                   ('offline_en_real','online_en_real','seen_en_real','heldout_en_real','noisy_en_fake')]
        note = decision['action'] + ('; ' + ', '.join(decision['warnings']) if decision['warnings'] else '')
        lines.append('| ' + ' | '.join([r['tag'], *values, note]) + ' |')
    lines += ['', 'best_model.pt remains the validated starting checkpoint until an eligible improvement occurs.',
              'Promotion: Weighted gain >= configured minimum, Noisy nondecreasing, independent per-condition recall floors and Clean floor.',
              'best_weighted.pt and best_noisy.pt retain unconditional winners for review only.',
              'Fresh AdamW starts the changed objective; subsequent restores retain matching moments.',
              'last.pt resumes the last saved validation boundary; unsaved steps replay.',
              'Existing V3, original 91.68 best, and all original/Dev/Train cache files are unchanged.']
    target = Path(run)/'report.md'
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    os.replace(temporary, target)


def _meters_dump(meters):
    return {k: {'cm': v.cm, 'ce': v.ce} for k, v in meters.items()}


def _meters_load(values):
    result = defaultdict(Metrics)
    for k, v in values.items():
        result[k].cm, result[k].ce = v['cm'].clone(), v['ce'].clone()
    return result


def train(cfg, run, resume=None, smoke_steps=0):
    phase('V3.1 checking protected weights and complete Train caches')
    run = Path(run); run.mkdir(parents=True, exist_ok=True)
    if not resume and any((run/name).exists() for name in (*WINNERS.values(), 'control_best.pt', 'last.pt')):
        raise ValueError('Output already contains checkpoints; resume last.pt or choose a new run directory')
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    if device.type == 'cuda' and cfg['amp'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported; explicitly select AMP none')
    if sha256(cfg['baseline']) != cfg['baseline_sha256']:
        raise RuntimeError('Original checkpoint SHA256 changed')
    if sha256(cfg['warm_checkpoint']) != cfg['warm_checkpoint_sha256']:
        raise RuntimeError('Starting V3 best SHA256 changed')
    if cfg['evals_per_epoch'] != 2 or cfg['head_epochs'] != 0 or not 1 <= cfg['joint_epochs'] <= 2:
        raise ValueError('V3.1 is joint adaptation only, 1-2 epochs, two validations per epoch')
    seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    plan, validation, weights, counts, fingerprints = build_data(cfg)
    codes = source_fingerprints()
    state = read_state(resume) if resume else None
    if state and (state.get('schema') != SCHEMA or state.get('kind') != 'training' or
                  state['config'] != cfg or state['data_fingerprints'] != fingerprints or state['source_hashes'] != codes):
        raise ValueError('Exact resume requires a V3 training checkpoint and identical config, metadata, code, versions')
    if state:
        model = Detector.from_checkpoint(state, checkpointing=cfg['checkpointing'])
        historical = state['baseline_dev']
    else:
        model, historical = initialize(cfg)
    controller = Controller(cfg, state['controller'] if state else None)
    if state:
        recover_winners(run, controller)
    model.configure_trainable_layers(0 if controller.state['phase'] == 'head' else cfg['trainable_layers'])
    model.to(device)
    optimizer = optimizer_for(model, cfg)
    if state:
        optimizer.load_state_dict(state['optimizer'])
    weights, noisy_weights = weights.to(device), plan.noisy_weights.to(device)
    model_bytes = storage_size(model.state_dict())
    potential_trainable = sum(p.numel()*p.element_size() for p in model.head.parameters())
    potential_trainable += sum(p.numel()*p.element_size() for layer in
                              model.backbone.encoder.layers[-cfg['trainable_layers']:] for p in layer.parameters())
    # Up to four old winner files coexist with their replacements until last.pt
    # commits the decision. This is bounded crash recovery, not epoch retention.
    needed = 10*model_bytes + 8*potential_trainable + 2*1024**3
    if shutil.disk_usage(run).free < needed:
        raise OSError(f'V3 checkpoint headroom requires {needed/1024**3:.1f} GiB free')
    print(f'V3 Train fake/real={counts.tolist()}; ordinary weights={weights.tolist()}; '
          f'noisy weights={noisy_weights.tolist()}; full traversal, threshold=0.5.', flush=True)
    atomic_json(run / 'inputs.json', {'data_fingerprints': fingerprints, 'source_hashes': codes,
                                     'baseline_dev': historical, 'baseline_sha256': cfg['baseline_sha256'],
                                     'warm_checkpoint': cfg['warm_checkpoint'],
                                     'warm_checkpoint_sha256': cfg['warm_checkpoint_sha256']})
    if hasattr(plan, 'coverage'):
        atomic_json(run/'planned_coverage.json', plan.coverage())
    history = state['history'] if state else []
    epoch, cursor = (state['epoch'], state['cursor']) if state else (1, 0)
    global_steps = state['global_steps'] if state else 0
    epoch_steps = state['epoch_steps'] if state else 0
    aggregate = defaultdict(float, state['aggregate'] if state else {})
    modes = Counter(state['composition_counts'] if state else {})
    meters = _meters_load(state['meters']) if state else defaultdict(Metrics)
    # A bootstrap checkpoint precedes the deterministic first-epoch reseed.
    resumed_rng = state['rng'] if state and cursor else None
    completed = bool(state and state.get('complete'))

    def snapshot(tag, dev, kind='weights'):
        return {'schema': SCHEMA, 'kind': kind, 'tag': tag, 'epoch': epoch,
                'phase': controller.state['phase'], 'model': model.state_dict(), **model.architecture(),
                'config': cfg, 'dev': dev, 'baseline_dev': historical,
                'baseline_sha256': cfg['baseline_sha256'], 'data_fingerprints': fingerprints,
                'source_hashes': codes}

    if not state and not smoke_steps:
        phase('V3.1 validating starting best before any update')
        initial_dev = validate(model, validation, cfg, device, run/'baseline_scores.jsonl')
        decision = controller.initialize(initial_dev, 'baseline')
        record = {'tag': 'baseline', 'epoch': 0, 'cursor': 0, 'phase': 'joint',
                  'global_steps': 0, 'dev': initial_dev, 'decision': decision,
                  'train': {}, 'train_groups': {}, 'learning_rates': {}}
        history.append(record)
        atomic_json(run/'baseline.json', record)
        # Separate files make each metric winner independently replaceable.
        for filename in WINNERS.values():
            atomic_save(run/filename, snapshot('baseline', initial_dev))
        atomic_save(run/'control_best.pt', {**snapshot('baseline', initial_dev, 'control'),
                                          'optimizer': optimizer.state_dict()})
        atomic_save(run/'last.pt', {**snapshot('baseline', initial_dev, 'training'),
            'optimizer': optimizer.state_dict(), 'controller': controller.dump(),
            'history': history, 'cursor': 0, 'global_steps': 0, 'epoch_steps': 0,
            'aggregate': {}, 'composition_counts': {}, 'meters': {}, 'rng': rng_state(),
            'complete': False, 'completion_reason': 'phase_budget'})
        write_report(run, history, controller, 'training')
        print('BASELINE_SAVED=True; original V3 best is the fallback, before training.', flush=True)

    reason = state.get('completion_reason', 'phase_budget') if state else 'phase_budget'
    while not completed:
        batches = plan.batches(epoch)
        if len(batches) < cfg['evals_per_epoch']:
            raise ValueError('Not enough training steps for requested validation frequency')
        boundaries = {math.ceil(i*len(batches)/cfg['evals_per_epoch'])
                      for i in range(1, cfg['evals_per_epoch']+1)}
        if cursor == len(batches):
            epoch += 1; cursor = 0; epoch_steps = 0
            aggregate, modes, meters = defaultdict(float), Counter(), defaultdict(Metrics)
            # The uninterrupted run seeds the next epoch at this boundary too.
            resumed_rng = None
            batches = plan.batches(epoch)
            boundaries = {math.ceil(i*len(batches)/cfg['evals_per_epoch'])
                          for i in range(1, cfg['evals_per_epoch']+1)}
        if cursor == 0 and resumed_rng is None:
            seed_all(cfg['seed'] + epoch*100003)
        if resumed_rng is not None:
            restore_rng(resumed_rng); resumed_rng = None
        end = min(x for x in boundaries if x > cursor)
        selected_batches = batches[cursor:end]
        if smoke_steps:
            selected_batches = selected_batches[:smoke_steps]
        stage = controller.state['phase']
        model.train()
        started, segment_start = time.perf_counter(), cursor
        progress = training_bar(loader(plan.records, cfg, training=True, epoch=epoch, batches=selected_batches),
                                len(selected_batches), f'V3.1 {stage} epoch {epoch} steps {cursor+1}-{cursor+len(selected_batches)}')
        for examples in progress:
            phase_steps = cfg[stage+'_epochs'] * plan.steps
            schedule_cfg = {**cfg, 'lr_warmup_steps': cfg.get('head_warmup_steps', 200) if stage == 'head'
                            else cfg.get('joint_warmup_steps', 100)}
            scale = lr_scale(schedule_cfg, controller.state['phase_steps'], phase_steps,
                             controller.state['lr_scale'])
            for group in optimizer.param_groups:
                if group['name'] == 'encoder':
                    base_lr = 0. if stage == 'head' else cfg['encoder_lr']
                else:
                    base_lr = cfg['head_lr'] if stage == 'head' else cfg['joint_head_lr']
                    if stage == 'head' and cfg.get('warm_checkpoint'):
                        base_lr = cfg['joint_head_lr']
                group['lr'] = base_lr * scale
            ramp = min(1., (global_steps+1)/max(1, cfg.get('noisy_ramp_epochs', 1.)*plan.steps))
            noisy_weight = cfg.get('noisy_weight_start', .3) + ramp*(cfg['noisy_weight']-cfg.get('noisy_weight_start', .3))
            cka_weight = cfg['cka_weight'] * min(1., (global_steps+1)/max(1, cfg.get('cka_warmup_steps', 200)))
            stats, logits = supervised_step(model, examples, optimizer, weights, device, cfg['amp'],
                                           noisy_weight=noisy_weight, cka_weight=cka_weight,
                                           grad_clip=cfg['grad_clip'], microbatch=cfg['microbatch'],
                                           frame_budget=cfg['frame_budget'], noisy_class_weights=noisy_weights,
                                           offload_activations=cfg.get('offload_activations', cfg.get('activation_offload', True)),
                                           activation_budget_gib=cfg.get('activation_budget_gib', 0.))
            for k, value in stats.items(): aggregate[k] += value
            for i, ex in enumerate(examples):
                group = ('noisy' if ex['noisy'] else 'ordinary') + '/' + ex['language'] + '/' + ex['view']
                meters[group].update(logits[i:i+1], [ex['label']])
                modes[f'{ex["composition"]}/{ex["language"]}/{ex["label"]}/{ex["view"]}'] += 1
                modes[f'augmentation/{ex.get("augmentation", "unknown")}/{ex["language"]}/{ex["label"]}/{ex["view"]}'] += 1
                if ex['noisy']: modes[f'band{ex["band"]}/{ex["language"]}/{ex["label"]}/{ex["view"]}'] += 1
                aggregate['audio_seconds'] += ex['audio_seconds']
            cursor += 1; epoch_steps += 1; global_steps += 1; controller.state['phase_steps'] += 1
            progress.set_postfix(loss=f'{stats["loss"]:.5f}', refresh=False)
            if cursor == segment_start+1 or cursor % 100 == 0 or cursor == end:
                per = (time.perf_counter()-started)/(cursor-segment_start)
                print(f'STEP {cursor}/{len(batches)} LOSS={stats["loss"]:.6f} '
                      f'grad={stats["grad_norm"]:.3f} noisy_weight={noisy_weight:.3f} '
                      f'cka_weight={cka_weight:.5f} seconds/step={per:.3f} '
                      f'ETA_min={per*(len(batches)-cursor)/60:.1f} '
                      f'offload_GiB={stats.get("activation_offload_gib", 0.):.3f}/'
                      f'{stats.get("activation_budget_gib", 0.):.3f}', flush=True)
        if smoke_steps:
            atomic_json(run/'smoke.json', {'passed': True, 'steps': len(selected_batches), 'losses': dict(aggregate)})
            print('GPU_SMOKE_PASSED=True; no checkpoint saved.', flush=True)
            return
        tag = f'epoch_{epoch}' if cursor == len(batches) else f'epoch_{epoch}_step_{cursor}'
        dev = validate(model, validation, cfg, device, run/(tag+'_scores.jsonl'))
        decision = controller.observe(dev, tag)
        record = {'tag': tag, 'epoch': epoch, 'cursor': cursor, 'phase': stage,
                  'global_steps': global_steps, 'train': {k:v/max(1,epoch_steps) for k,v in aggregate.items()},
                  'train_groups': {k:v.result() for k,v in meters.items()},
                  'composition_counts': dict(modes), 'dev': dev, 'decision': decision,
                  'learning_rates': {g['name']:g['lr'] for g in optimizer.param_groups}}
        history.append(record); atomic_json(run/(tag+'.json'), record)
        phase('V3.1 saving validation winners and resume state')
        replacing = [WINNERS[w] for w in decision['save']]
        if 'best_safe' in decision['save']:
            replacing.append('control_best.pt')
        backup_winners(run, replacing)
        for winner, filename in WINNERS.items():
            if winner in decision['save']:
                atomic_save(run/filename, snapshot(tag, dev))
        if 'best_safe' in decision['save']:
            atomic_save(run/'control_best.pt', {**snapshot(tag, dev, 'control'), 'optimizer': optimizer.state_dict()})
        print(f'Dev {tag} Clean={100*dev["clean_f1"]:.3f} Seen={100*dev["seen_f1"]:.3f} '
              f'Heldout={100*dev["heldout_f1"]:.3f} Weighted={100*dev["weighted_f1"]:.3f} '
              f'promoted={"best_safe" in decision["save"]} action={decision["action"]} '
              f'LR={record["learning_rates"]}', flush=True)
        for group in ('offline/en','online/en','seen/en','heldout/en'):
            print(group+' recall [fake,real]='+str(dev['groups'][group]['recall']), flush=True)
        if decision['warnings']: print('Selection warnings: '+', '.join(decision['warnings']), flush=True)
        resume_tag, resume_dev = tag, dev
        if decision['action'] == 'restore_reduce':
            restored = restore_selected(model, optimizer, run, controller.state['best_safe']['tag'])
            resume_tag, resume_dev = restored['tag'], restored['dev']
        elif decision['action'] == 'phase_complete':
            completed = True
            reason = 'joint_budget' if decision['remaining_evaluations'] == 0 else 'joint_plateau_after_reduced_lr_trials'
        atomic_save(run/'last.pt', {**snapshot(resume_tag, resume_dev, 'training'), 'optimizer': optimizer.state_dict(),
                                   'controller': controller.dump(), 'history': history, 'cursor': cursor,
                                   'global_steps': global_steps, 'epoch_steps': epoch_steps,
                                   'aggregate': dict(aggregate), 'composition_counts': dict(modes),
                                   'meters': _meters_dump(meters), 'rng': rng_state(), 'complete': completed,
                                   'completion_reason': reason})
        commit_winners(run, replacing)
        write_report(run, history, controller, 'complete' if completed else 'training')
    phase('V3.1 verifying protected original and input metadata')
    if sha256(cfg['baseline']) != cfg['baseline_sha256']:
        raise RuntimeError('Original checkpoint changed externally')
    if sha256(cfg['warm_checkpoint']) != cfg['warm_checkpoint_sha256']:
        raise RuntimeError('Starting V3 best changed externally')
    if any(sha256(p) != digest for p,digest in fingerprints.items()):
        raise RuntimeError('Input metadata changed during training')
    atomic_json(run/'completed.json', {'reason': reason, 'selection': controller.dump(),
                'original_best_preserved': True, 'starting_v3_best_preserved': True,
                'eligible_improvement': controller.state['best_safe']['tag'] != 'baseline',
                'cache_metadata_unchanged': True})
    print('V31_TRAINING_COMPLETE=True ORIGINAL_BEST_PRESERVED=True', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p.add_argument('--resume'); p.add_argument('--smoke-steps', type=int, default=0)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    from w2v_aasist.launch import run_lock
    with run_lock(Path(args.out)/'.lock'):
        train(cfg, args.out, args.resume, args.smoke_steps)


if __name__ == '__main__':
    main()
