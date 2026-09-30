"""Full-wave V3 training with bounded, recall-aware adaptation and exact resume."""
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
from .model import Detector
from .step import supervised_step
from .validation import validate

SCHEMA = 'rtc_w2v_multiconv_v3'


def source_fingerprints():
    root = Path(__file__).resolve().parent.parent
    files = list((root / 'w2v_v3').glob('*.py')) + list((root / 'w2v_aasist').glob('*.py'))
    files += [root / p for p in ('w2v_rebuild/model.py', 'utils/data_utils.py',
              'utils/RawBoost.py', 'utils/env_noise.py', 'rtc_noisy_v2/cache.py',
              'rtc_noisy_v2/plan.py', 'rtc_noisy/diverse.py', 'live_progress.py')]
    import transformers
    return {**{str(p.relative_to(root)): sha256(p) for p in files},
            'torch': str(torch.__version__), 'transformers': transformers.__version__}


def read_state(path):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise ValueError('Checkpoint must be a dictionary')
    return state


def initialize(cfg):
    old = read_state(cfg['baseline'])
    if old.get('schema') != 'rtc_w2v_rebuild_v1':
        raise ValueError('Cold initialization requires the original 91.68 AASIST checkpoint')
    mc = old['model_config']
    if cfg.get('production_layout', True) and (mc['hidden_size'], mc['num_hidden_layers'],
                                             mc['feature_projection_input_dim']) != (1024, 24, 160):
        raise ValueError('Expected original w2v-BERT 2.0 architecture')
    prep = [v for k, v in old.get('data_fingerprints', {}).items()
            if Path(k).name == 'preprocessor_config.json']
    if len(prep) != 1 or prep[0] != sha256(Path(cfg['ssl_path']) / 'preprocessor_config.json'):
        raise ValueError('Feature extractor differs from original best')
    warm = cfg.get('warm_checkpoint')
    if warm:
        if not cfg.get('warm_checkpoint_sha256') or sha256(warm) != cfg['warm_checkpoint_sha256']:
            raise ValueError('Explicit MultiConv warm start needs a matching recorded SHA256')
        print('Explicit warm start: loading compatible MultiConv weights.', flush=True)
        warm_state = read_state(warm)
        warm_preprocessor = [v for k,v in warm_state.get('data_fingerprints', {}).items()
                             if Path(k).name == 'preprocessor_config.json']
        if warm_preprocessor != prep:
            raise ValueError('Warm-start feature extractor fingerprint differs from original best')
        warm_config = warm_state['model_config']
        for key in ('hidden_size', 'num_hidden_layers', 'feature_projection_input_dim'):
            if warm_config[key] != mc[key]:
                raise ValueError('Warm-start encoder architecture differs from original w2v-BERT')
        model = Detector.from_checkpoint(warm_state, checkpointing=cfg['checkpointing'])
    else:
        print('Cold V3: original 91.68 encoder + newly initialized MultiConv head.', flush=True)
        model = Detector.from_original(old, head_config=cfg.get('head_config'), checkpointing=cfg['checkpointing'])
    return model, old.get('dev', {})


def optimizer_for(model, cfg):
    """Stable groups include frozen parameters so head moments survive unfreezing."""
    groups = []
    for name, module in [('encoder', model.backbone), ('head', model.head)]:
        for decay in (True, False):
            parameters = [p for n, p in module.named_parameters()
                          if (p.ndim > 1 and not n.endswith('bias')) == decay]
            if parameters:
                groups.append({'params': parameters, 'name': name, 'lr': 0.,
                               'weight_decay': cfg['weight_decay'] if decay else 0.})
    return torch.optim.AdamW(groups, eps=1e-8)


def rng_state():
    n = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (n[0], n[1].tolist(), n[2], n[3], n[4]),
            'torch': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    random.setstate(value['python'])
    n = value['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.random.set_rng_state(value['torch'])
    if value['cuda']:
        if not torch.cuda.is_available() or len(value['cuda']) != torch.cuda.device_count():
            raise ValueError('Exact resume requires the same CUDA device count')
        torch.cuda.set_rng_state_all(value['cuda'])


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
    s = controller.state
    chosen = s['best_safe']['tag'] if s['best_safe'] else 'none'
    lines = ['# V3: w2v-BERT 2.0 + MultiConv', '', f'Status: {status}',
             f'Promoted checkpoint: {chosen}', '',
             'All figures are fixed Dev proxies at threshold 0.5, not platform scores.',
             'Clean/ordinary inputs are full recordings; noisy Train has two full versions per source.',
             'Noisy Dev remains the original fixed short cache, preserving comparison conditions.', '',
             '| Checkpoint | Phase | Clean | Seen | Heldout | Noisy | Weighted | EN real mean | Noisy EN fake mean | Decision |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in history:
        d, decision = r['dev'], r['decision']
        values = [f'{100*d[k]:.3f}' for k in ('clean_f1','seen_f1','heldout_f1','noisy_f1','weighted_f1')]
        values += [f'{100*decision["quality"][k]:.3f}' for k in ('en_real','noisy_en_fake')]
        note = decision['action'] + ('; ' + ', '.join(decision['warnings']) if decision['warnings'] else '')
        lines.append('| ' + ' | '.join([r['tag'], r['phase'], *values, note]) + ' |')
    lines += ['', 'best_model.pt: weighted winner within the two recall guardrails.',
              'best_weighted.pt / best_noisy.pt: unconditional metric winners for review.',
              'A declined candidate is not proof that its architecture is worse; all scores are retained.',
              'Control restores matching Adam moments when reducing LR or entering joint adaptation.',
              'last.pt resumes at its saved validation boundary, including RNG and partial-epoch metrics.',
              'The original 91.68 best and all source/Dev caches remain protected.']
    (Path(run) / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _meters_dump(meters):
    return {k: {'cm': v.cm, 'ce': v.ce} for k, v in meters.items()}


def _meters_load(values):
    result = defaultdict(Metrics)
    for k, v in values.items():
        result[k].cm, result[k].ce = v['cm'].clone(), v['ce'].clone()
    return result


def train(cfg, run, resume=None, smoke_steps=0):
    phase('V3 checking protected weights and complete Train caches')
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
    if cfg['evals_per_epoch'] < 1 or cfg['head_epochs'] < 1 or cfg['joint_epochs'] < 1:
        raise ValueError('Positive bounded phase budgets and validation frequency required')
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
                                     'baseline_dev': historical, 'baseline_sha256': cfg['baseline_sha256']})
    if hasattr(plan, 'coverage'):
        atomic_json(run/'planned_coverage.json', plan.coverage())
    history = state['history'] if state else []
    epoch, cursor = (state['epoch'], state['cursor']) if state else (1, 0)
    global_steps = state['global_steps'] if state else 0
    epoch_steps = state['epoch_steps'] if state else 0
    aggregate = defaultdict(float, state['aggregate'] if state else {})
    modes = Counter(state['composition_counts'] if state else {})
    meters = _meters_load(state['meters']) if state else defaultdict(Metrics)
    resumed_rng = state['rng'] if state else None
    completed = bool(state and state.get('complete'))

    def snapshot(tag, dev, kind='weights'):
        return {'schema': SCHEMA, 'kind': kind, 'tag': tag, 'epoch': epoch,
                'phase': controller.state['phase'], 'model': model.state_dict(), **model.architecture(),
                'config': cfg, 'dev': dev, 'baseline_dev': historical,
                'baseline_sha256': cfg['baseline_sha256'], 'data_fingerprints': fingerprints,
                'source_hashes': codes}

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
                                len(selected_batches), f'V3 {stage} epoch {epoch} steps {cursor+1}-{cursor+len(selected_batches)}')
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
                group = ('noisy' if ex['noisy'] else 'ordinary') + '/' + ex['language']
                meters[group].update(logits[i:i+1], [ex['label']])
                modes[f'{ex["composition"]}/{ex["language"]}/{ex["label"]}'] += 1
                modes[f'augmentation/{ex.get("augmentation", "unknown")}/{ex["language"]}/{ex["label"]}'] += 1
                if ex['noisy']: modes[f'band{ex["band"]}/{ex["language"]}/{ex["label"]}'] += 1
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
        phase('V3 saving validation winners and resume state')
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
            if stage == 'head':
                restored = restore_selected(model, optimizer, run, controller.state['best_safe']['tag'])
                resume_tag, resume_dev = restored['tag'], restored['dev']
                controller.begin_joint()
                model.configure_trainable_layers(cfg['trainable_layers'])
                print('Phase joint: selected head checkpoint; matching Adam retained; last encoder layers unfrozen.', flush=True)
            else:
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
    phase('V3 verifying protected original and input metadata')
    if sha256(cfg['baseline']) != cfg['baseline_sha256']:
        raise RuntimeError('Original checkpoint changed externally')
    if any(sha256(p) != digest for p,digest in fingerprints.items()):
        raise RuntimeError('Input metadata changed during training')
    atomic_json(run/'completed.json', {'reason': reason, 'selection': controller.dump(),
                'original_best_preserved': True, 'cache_metadata_unchanged': True})
    print('V3_TRAINING_COMPLETE=True ORIGINAL_BEST_PRESERVED=True', flush=True)


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
