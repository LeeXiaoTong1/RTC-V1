"""V3.4 layer-wise encoder adaptation from an identified V3.3 submission model."""
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
from w2v_aasist.progress import phase as publish_phase, training_bar
from w2v_aasist.runtime import (Metrics, atomic_json, atomic_save, seed_all,
                                sha256, storage_size)
from w2v_v33.control import Controller, lr_scale, acceptance, pair_weight_for_exposure
from w2v_v33.data import build_data, loader
from w2v_v33.model import Detector
from w2v_v33.step import supervised_step
from w2v_v33.model import install_runtime
from w2v_v32.performance import TimedLoader
from w2v_v3.validation import validate

from . import SCHEMA
from w2v_v3.train import (read_state, rng_state, restore_rng,
                          source_fingerprints as v3_fingerprints)
from .optim import configure_model, optimizer_for, apply_learning_rates, UpdateDiagnostics


def phase(label):
    publish_phase(label)
    print('V32_EVENT='+json.dumps({'kind':'phase','label':label}),flush=True)


def print_validation(record):
    """Print only after report and paired checkpoint transaction are committed."""
    tag,dev,decision=record['tag'],record['dev'],record['decision']
    print('V32_EVENT='+json.dumps({'kind':'validation','tag':tag}),flush=True)
    scores=' '.join(f'{label}={100*dev[key]:.3f}' for label,key in
        (('Clean','clean_f1'),('Seen','seen_f1'),('Heldout','heldout_f1'),
         ('Noisy','noisy_f1'),('Weighted','weighted_f1')))
    print(f'Dev {tag} {scores} promoted={"best_safe" in decision["save"]} '
          f'action={decision["action"]} LR={record["learning_rates"]}',flush=True)
    for group in ('offline/en','online/en','seen/en','heldout/en'):
        print(group+' recall [fake,real]='+str(dev['groups'][group]['recall']),flush=True)
    if decision['warnings']:
        print('Selection warnings: '+', '.join(decision['warnings']),flush=True)


def source_fingerprints():
    import scipy
    root = Path(__file__).resolve().parent.parent
    return {**v3_fingerprints(), 'numpy':np.__version__, 'scipy':scipy.__version__, **{str(p.relative_to(root)): sha256(p)
            for folder in ('w2v_v31','w2v_v32','w2v_v33','w2v_v34') for p in sorted((root/folder).glob('*.py'))}}


def initialize(cfg):
    """Load exactly the weights whose submission provenance was resolved upstream."""
    warm = cfg.get('warm_checkpoint')
    if not warm or sha256(warm) != cfg.get('warm_checkpoint_sha256'):
        raise ValueError('V3.4 requires the recorded, unchanged V3.3 weight checkpoint')
    state = read_state(warm)
    if state.get('schema') != 'rtc_w2v_multiconv_v33' or state.get('kind') != 'weights':
        raise ValueError('Warm start must be a V3.3 weight checkpoint, not last.pt')
    expected = cfg.get('expected_warm_tag')
    if not expected or state.get('tag') != expected:
        raise ValueError('V3.3 warm checkpoint tag differs from expected_warm_tag')
    architecture = state['model_config']
    if cfg.get('production_layout', True) and (
            architecture['hidden_size'], architecture['num_hidden_layers'],
            architecture['feature_projection_input_dim']) != (1024, 24, 160):
        raise ValueError('Expected w2v-BERT 2.0 with 24 encoder blocks')
    prep = [digest for name, digest in state.get('data_fingerprints', {}).items()
            if Path(name).name == 'preprocessor_config.json']
    if len(prep) != 1 or prep[0] != sha256(Path(cfg['ssl_path'])/'preprocessor_config.json'):
        raise ValueError('Feature extractor differs from the submitted V3.3 checkpoint')
    model = Detector.from_checkpoint(state, checkpointing=cfg['checkpointing'])
    return model, state.get('dev', {})


def validate_resume(state, cfg, fingerprints, codes):
    """Only an identical V3.4 validation boundary is a resumable training state."""
    if (state.get('schema') != SCHEMA or state.get('kind') != 'training'
            or state.get('config') != cfg or state.get('data_fingerprints') != fingerprints
            or state.get('source_hashes') != codes or 'update_diagnostics' not in state):
        raise ValueError('Exact V3.4 resume requires identical config, metadata, code, versions and diagnostics')


def provenance(cfg):
    return {'warm_checkpoint': cfg['warm_checkpoint'],
            'warm_checkpoint_sha256': cfg['warm_checkpoint_sha256'],
            'warm_checkpoint_tag': cfg['expected_warm_tag'],
            'source_arm': cfg.get('arm', 'candidate'),
            'source_provenance': cfg.get('source_provenance', {}),
            'platform_reference': cfg.get('platform_reference', {}),
            'note': 'Platform results are user-reported provenance only, never a Dev target or selection score.'}


def verify_source_files(cfg, fingerprints=None):
    """Keep the submitted model, its identity records and source data immutable."""
    protected = cfg.get('source_provenance', {}).get('file_fingerprints', {})
    expected = cfg.get('source_data_fingerprints')
    if cfg.get('production_layout', True) and (not protected or not expected):
        raise ValueError('V3.4 production requires resolved source provenance and input fingerprints')
    for name, digest in protected.items():
        if not Path(name).is_file() or sha256(name) != digest:
            raise ValueError('V3.3 source provenance changed: ' + str(name))
    if expected is not None and fingerprints is not None and expected != fingerprints:
        raise ValueError('V3.4 data fingerprints differ from the submitted V3.3 source run')


def learning_rate_record(optimizer):
    detailed = {group['name']: group['lr'] for group in optimizer.param_groups}
    encoder = [rate for name, rate in detailed.items() if name.startswith('encoder.')]
    head = [rate for name, rate in detailed.items() if name.startswith('head.')]
    summary = {'encoder_min': min(encoder), 'encoder_max': max(encoder), 'head': max(head)}
    return summary, detailed


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
        raise ValueError('Foreign checkpoint at a V3.4 winner path')
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
    verdict = acceptance(controller.state['best_safe'], controller.state['anchor'], controller.cfg)
    atomic_json(Path(run)/'acceptance.json', {'arm': controller.cfg.get('arm', 'candidate'), **verdict})
    lines = ['# V3.4: all encoder blocks with layer-wise learning rates', '', f'Status: {status}',
             f'Promoted checkpoint: {chosen}',
             f'Source arm: {controller.cfg.get("arm", "candidate")}; local improvement target reached: {verdict["success"]}',
             'One adaptation run. Source CE, full/short conditions, CKA and source-arm pair policy are inherited.',
             'Fixed Dev and threshold 0.5; these are local proxies, not official platform scores.',
             'The user-reported platform Weighted 93.3941 is provenance only. The platform goal of 97 is not tested by this report.',
             'The baseline is reevaluated from the exact V3.3 warm checkpoint before any optimizer update.',
             'All configured encoder blocks and MultiConv adapt; feature projection outside those blocks remains frozen.',
             'Each canonical source retains one CE budget across conditions and full/short views.',
             'Pair ramp uses canonical source exposure. No cache regeneration or additional encoder passes are introduced.',
             'Parameter update diagnostics measure sampled deltas from the warm weights, not proof of generalization.',
             'Runtime measurements: performance.jsonl.', '',
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
              'Fresh AdamW starts the changed trainable parameter set; subsequent restores retain matching moments.',
              'Success requires Noisy +0.3 percentage points AND noisy English real recall +2 points; small safe gains can still be saved.',
              'last.pt resumes the last saved validation boundary; unsaved steps replay.',
              'Original checkpoints, submitted V3.3 weights, official Train audio and all existing caches are preserved.']
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


def write_baseline_fingerprint(run, cfg, observed, cached, fingerprints):
    """Show warm-checkpoint reproducibility without treating its Dev as a score target."""
    keys = ('clean_f1', 'seen_f1', 'heldout_f1', 'noisy_f1', 'weighted_f1')
    deltas = {key: observed[key] - cached[key] for key in keys if key in cached}
    atomic_json(Path(run)/'baseline_fingerprint.json', {
        'warm_checkpoint_sha256': cfg['warm_checkpoint_sha256'],
        'warm_checkpoint_tag': cfg['expected_warm_tag'],
        'source_provenance': provenance(cfg),
        'eval_amp': cfg.get('eval_amp', 'none'), 'threshold': .5,
        'input_metadata_hashes': fingerprints, 'cached_source_dev': cached,
        'observed_starting_dev': observed, 'metric_deltas': deltas,
        'matches_cached_metrics': all(abs(v) <= 1e-8 for v in deltas.values()) if deltas else None,
        'note': 'The recomputed starting Dev is the fixed anchor. Neither record is a platform score.'})


def record_performance(run, step, epoch, examples, wait, elapsed, stats, *, source_exposures, pair_weight):
    """Timing must count recordings, not the several views of each recording."""
    sources = stats['source_count']
    row = dict(global_step=step, epoch=epoch, source_exposures=source_exposures,
               sources=sources, encoded_views=len(examples), pair_weight=pair_weight,
               data_wait_seconds=wait, compute_seconds=elapsed, wall_seconds=wait+elapsed,
               sources_per_second=sources/max(1e-9, wait+elapsed),
               encoder_calls=stats['encoder_forward_microbatches'],
               mean_physical_batch=len(examples)/max(1, stats['encoder_forward_microbatches']),
               audio_seconds=sum(e['audio_seconds'] for e in examples),
               **{k:v for k,v in stats.items() if k != 'pair_weight' and (
                   k.startswith(('gpu_', 'activation_', 'pair_', 'weighted_pair_'))
                   or k in ('ce', 'cka', 'loss', 'time_loss', 'structure_loss', 'ce_shared_feature_grad_norm'))})
    with (Path(run)/'performance.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')
    atomic_json(Path(run)/'performance_latest.json', row)
    if 'ce_shared_feature_grad_norm' in row:
        with (Path(run)/'pair_gradient_diagnostics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
    return row


def train(cfg, run, resume=None, smoke_steps=0):
    phase('V3.4 checking protected weights and complete Train caches')
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
        raise RuntimeError('Starting submitted V3.3 weights SHA256 changed')
    if cfg['evals_per_epoch'] != 2 or cfg['head_epochs'] != 0 or not 1 <= cfg['joint_epochs'] <= 2:
        raise ValueError('V3.4 is joint adaptation only, 1-2 epochs, two validations per epoch')
    if cfg.get('eval_amp', 'none') != 'none':
        raise ValueError('V3.4 requires fixed FP32 Dev inference')
    verify_source_files(cfg)
    pair_weight_for_exposure(cfg, 0, 1)
    seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    plan, validation, weights, counts, fingerprints = build_data(cfg)
    verify_source_files(cfg, fingerprints)
    codes = source_fingerprints()
    state = read_state(resume) if resume else None
    if state:
        validate_resume(state, cfg, fingerprints, codes)
    if state:
        model = Detector.from_checkpoint(state, checkpointing=cfg['checkpointing'])
        historical = state['baseline_dev']
    else:
        model, historical = initialize(cfg)
    controller = Controller(cfg, state['controller'] if state else None)
    if state:
        recover_winners(run, controller)
    configure_model(model, cfg)
    model.to(device)
    model = install_runtime(model, chunk_layers=cfg.get('fusion_chunk_layers',5))
    optimizer = optimizer_for(model, cfg)
    updates = UpdateDiagnostics(model)
    if state:
        updates.load_state_dict(state['update_diagnostics'])
    if state:
        optimizer.load_state_dict(state['optimizer'])
    atomic_json(run/'optimizer_groups.json', [{'name': g['name'], 'base_lr': g['base_lr'],
        'weight_decay': g['weight_decay'], 'parameters': sum(p.numel() for p in g['params'])}
        for g in optimizer.param_groups])
    weights = weights.to(device)
    noisy_weights = weights  # The same canonical-source weights; no second balancing.
    model_bytes = storage_size(model.state_dict())
    potential_trainable = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    # Up to four old winner files coexist with their replacements until last.pt
    # commits the decision. This is bounded crash recovery, not epoch retention.
    needed = 10*model_bytes + 8*potential_trainable + 2*1024**3
    if shutil.disk_usage(run).free < needed:
        raise OSError(f'V3 checkpoint headroom requires {needed/1024**3:.1f} GiB free')
    print(f'V3.4 canonical Train fake/real={counts.tolist()}; source weights={weights.tolist()}; '
          f'source_arm={cfg.get("arm", "candidate")}; one source budget across conditions; threshold=0.5.', flush=True)
    atomic_json(run / 'inputs.json', {'data_fingerprints': fingerprints, 'source_hashes': codes,
                                     'baseline_dev': historical, 'baseline_sha256': cfg['baseline_sha256'],
                                     'warm_checkpoint': cfg['warm_checkpoint'],
                                     'warm_checkpoint_sha256': cfg['warm_checkpoint_sha256'],
                                     'source_provenance': provenance(cfg)})
    if hasattr(plan, 'coverage'):
        atomic_json(run/'planned_coverage.json', plan.coverage())
    history = state['history'] if state else []
    epoch, cursor = (state['epoch'], state['cursor']) if state else (1, 0)
    global_steps = state['global_steps'] if state else 0
    epoch_steps = state['epoch_steps'] if state else 0
    source_exposures = state['source_exposures'] if state else 0
    epoch_sources = state['epoch_sources'] if state else 0
    aggregate = defaultdict(float, state['aggregate'] if state else {})
    modes = Counter(state['composition_counts'] if state else {})
    meters = _meters_load(state['meters']) if state else defaultdict(Metrics)
    # A bootstrap checkpoint precedes the deterministic first-epoch reseed.
    resumed_rng = state['rng'] if state and cursor else None
    completed = bool(state and state.get('complete'))
    if state:
        # last.pt is the transaction commit. Heal a report interrupted just after
        # that commit, before a new update or a completed-run export.
        write_report(run, history, controller, 'complete' if completed else 'training')

    def snapshot(tag, dev, kind='weights'):
        return {'schema': SCHEMA, 'kind': kind, 'tag': tag, 'epoch': epoch,
                'phase': controller.state['phase'], 'model': model.state_dict(), **model.architecture(),
                'config': cfg, 'dev': dev, 'baseline_dev': historical,
                'baseline_sha256': cfg['baseline_sha256'], 'data_fingerprints': fingerprints,
                'source_hashes': codes, 'source_provenance': provenance(cfg)}

    if not state and not smoke_steps:
        phase('V3.4 validating starting best before any update')
        initial_dev = validate(model, validation, cfg, device, run/'baseline_scores.jsonl')
        decision = controller.initialize(initial_dev, 'baseline')
        record = {'tag': 'baseline', 'arm': 'V3.4', 'source_arm': cfg.get('arm', 'candidate'), 'epoch': 0, 'cursor': 0, 'phase': 'joint',
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
            'source_exposures': 0, 'epoch_sources': 0,
            'aggregate': {}, 'composition_counts': {}, 'meters': {}, 'rng': rng_state(),
            'update_diagnostics': updates.state_dict(),
            'complete': False, 'completion_reason': 'phase_budget'})
        write_baseline_fingerprint(run, cfg, initial_dev, historical, fingerprints)
        write_report(run, history, controller, 'training')
        print_validation(record)
        print('BASELINE_SAVED=True; submitted V3.3 weights are the fallback, before training.', flush=True)

    reason = state.get('completion_reason', 'phase_budget') if state else 'phase_budget'
    while not completed:
        batches = plan.batches(epoch)
        if len(batches) < cfg['evals_per_epoch']:
            raise ValueError('Not enough training steps for requested validation frequency')
        boundaries = {math.ceil(i*len(batches)/cfg['evals_per_epoch'])
                      for i in range(1, cfg['evals_per_epoch']+1)}
        if cursor == len(batches):
            epoch += 1; cursor = 0; epoch_steps = 0; epoch_sources = 0
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
        sources_per_epoch = sum(len(batch) for batch in batches)
        if smoke_steps:
            selected_batches = selected_batches[:smoke_steps]
        stage = controller.state['phase']
        model.train()
        started, segment_start = time.perf_counter(), cursor
        timed_loader = TimedLoader(loader(plan.records,cfg,training=True,epoch=epoch,batches=selected_batches))
        progress = training_bar(timed_loader,
                                len(selected_batches), f'V3.4 {stage} epoch {epoch} steps {cursor+1}-{cursor+len(selected_batches)}')
        for examples in progress:
            step_started = time.perf_counter()
            phase_steps = cfg[stage+'_epochs'] * plan.steps
            schedule_cfg = {**cfg, 'lr_warmup_steps': cfg.get('head_warmup_steps', 200) if stage == 'head'
                            else cfg.get('joint_warmup_steps', 100)}
            scale = lr_scale(schedule_cfg, controller.state['phase_steps'], phase_steps,
                             controller.state['lr_scale'])
            apply_learning_rates(optimizer, scale)
            step_sources = len({ex['source_id'] for ex in examples})
            if step_sources != len(batches[cursor]):
                raise RuntimeError('Prepared view expansion changed the canonical source exposure budget')
            noisy_weight, cka_weight = .5, cfg['cka_weight']
            pair_weight = pair_weight_for_exposure(cfg, source_exposures + step_sources, sources_per_epoch)
            ramp_sources = cfg.get('pair_warmup_fraction', cfg.get('pair_ramp_fraction', .1)) * sources_per_epoch
            reaches_full_pair_weight = (cfg.get('arm', 'candidate') == 'candidate' and pair_weight > 0
                                       and source_exposures < ramp_sources <= source_exposures + step_sources)
            stats, logits = supervised_step(model, examples, optimizer, weights, device, cfg['amp'],
                                           noisy_weight=noisy_weight, cka_weight=cka_weight,
                                           pair_weight=pair_weight, aux_max_tokens=cfg.get('aux_max_tokens', 256),
                                           pair_temperature=cfg.get('pair_temperature', .1),
                                           pair_time_prior=cfg.get('pair_time_prior', .25),
                                           diagnose_aux_grad=(global_steps in cfg.get('diagnose_aux_steps', [])
                                                              or reaches_full_pair_weight),
                                           grad_clip=cfg['grad_clip'], microbatch=cfg['microbatch'],
                                           frame_budget=cfg['frame_budget'], noisy_class_weights=noisy_weights,
                                           offload_activations=cfg.get('offload_activations', cfg.get('activation_offload', True)),
                                           activation_budget_gib=cfg.get('activation_budget_gib', 0.),
                                           gpu_activation_gib=cfg.get('gpu_activation_gib',18.),
                                           gpu_reserve_gib=cfg.get('gpu_reserve_gib',8.))
            if global_steps == 0 or cursor + 1 == end:
                updates.record_gradients(model)
            step_elapsed = time.perf_counter()-step_started
            if stats['source_count'] != step_sources:
                raise RuntimeError('Prepared view expansion changed the canonical source exposure budget')
            source_exposures += step_sources; epoch_sources += step_sources
            perf = record_performance(run,global_steps+1,epoch,examples,timed_loader.last_wait,step_elapsed,stats,
                                      source_exposures=source_exposures, pair_weight=pair_weight)
            for k, value in stats.items():
                if not k.startswith(('gpu_', 'activation_')): aggregate[k] += value
            for i, ex in enumerate(examples):
                group = ex['condition'] + '/' + ex['language'] + '/' + ex['view']
                meters[group].update(logits[i:i+1], [ex['label']])
                modes[f'{ex["condition"]}/{ex["language"]}/{ex["label"]}/{ex["view"]}'] += 1
                modes[f'augmentation/{ex.get("augmentation", "unknown")}/{ex["language"]}/{ex["label"]}/{ex["view"]}'] += 1
                if ex['view']=='full':
                    modes[f'processing/{ex.get("processing_condition","unknown")}/{ex["language"]}/{ex["label"]}'] += 1
                    if 'silence_applied' in ex:
                        status='applied' if ex['silence_applied'] else ex['silence_skip_reason']
                        modes[f'local_silence/{status}/{ex["language"]}/{ex["label"]}'] += 1
                if ex.get('noisy'): modes[f'band{ex.get("band", -1)}/{ex["language"]}/{ex["label"]}/{ex["view"]}'] += 1
                aggregate['audio_seconds'] += ex['audio_seconds']
            cursor += 1; epoch_steps += 1; global_steps += 1; controller.state['phase_steps'] += 1
            progress.set_postfix(loss=f'{stats["loss"]:.5f}', refresh=False)
            if cursor == segment_start+1 or cursor % 100 == 0 or cursor == end:
                per = (time.perf_counter()-started)/(cursor-segment_start)
                print(f'STEP {cursor}/{len(batches)} LOSS={stats["loss"]:.6f} '
                      f'grad={stats["grad_norm"]:.3f} noisy_weight={noisy_weight:.3f} '
                      f'cka_weight={cka_weight:.5f} pair_weight={pair_weight:.5f} '
                      f'source_exposures={source_exposures} seconds/step={per:.3f} '
                      f'ETA_min={per*(len(batches)-cursor)/60:.1f} '
                      f'loader_wait={timed_loader.last_wait:.3f}s compute={step_elapsed:.3f}s '
                      f'mean_microbatch={perf["mean_physical_batch"]:.2f} '
                      f'GPU_peak_GiB={stats.get("gpu_peak_allocated_gib",0.):.2f} '
                      f'GPU_saved_GiB={stats.get("gpu_saved_activations_gib",0.):.2f} '
                      f'offload_GiB={stats.get("activation_offload_gib", 0.):.3f}/'
                      f'{stats.get("activation_budget_gib", 0.):.3f}', flush=True)
        if smoke_steps:
            atomic_json(run/'smoke.json', {'passed': True, 'steps': len(selected_batches), 'losses': dict(aggregate)})
            print('GPU_SMOKE_PASSED=True; no checkpoint saved.', flush=True)
            return
        tag = f'epoch_{epoch}' if cursor == len(batches) else f'epoch_{epoch}_step_{cursor}'
        phase('V3.4 validating '+tag)
        dev = validate(model, validation, cfg, device, run/(tag+'_scores.jsonl'))
        decision = controller.observe(dev, tag)
        rate_summary, layer_rates = learning_rate_record(optimizer)
        update_report = updates.report(model)
        atomic_json(run/(tag+'_parameter_updates.json'), update_report)
        record = {'tag': tag, 'arm': 'V3.4', 'source_arm': cfg.get('arm', 'candidate'), 'epoch': epoch, 'cursor': cursor, 'phase': stage,
                  'global_steps': global_steps, 'source_exposures': source_exposures, 'epoch_sources': epoch_sources,
                  'pair_weight': pair_weight, 'train': {k:v/max(1,epoch_steps) for k,v in aggregate.items()},
                  'train_groups': {k:v.result() for k,v in meters.items()},
                  'composition_counts': dict(modes), 'dev': dev, 'decision': decision,
                  'learning_rates': rate_summary, 'per_layer_learning_rates': layer_rates,
                  'parameter_updates': update_report}
        history.append(record); atomic_json(run/(tag+'.json'), record)
        phase('V3.4 saving validation winners and resume state')
        replacing = [WINNERS[w] for w in decision['save']]
        if 'best_safe' in decision['save']:
            replacing.append('control_best.pt')
        backup_winners(run, replacing)
        for winner, filename in WINNERS.items():
            if winner in decision['save']:
                atomic_save(run/filename, snapshot(tag, dev))
        if 'best_safe' in decision['save']:
            atomic_save(run/'control_best.pt', {**snapshot(tag, dev, 'control'), 'optimizer': optimizer.state_dict()})
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
                                   'source_exposures': source_exposures, 'epoch_sources': epoch_sources,
                                   'aggregate': dict(aggregate), 'composition_counts': dict(modes),
                                   'meters': _meters_dump(meters), 'rng': rng_state(), 'complete': completed,
                                   'completion_reason': reason,
                                   'update_diagnostics': updates.state_dict()})
        commit_winners(run, replacing)
        write_report(run, history, controller, 'complete' if completed else 'training')
        print_validation(record)
    phase('V3.4 verifying protected original and input metadata')
    verify_source_files(cfg, fingerprints)
    if sha256(cfg['baseline']) != cfg['baseline_sha256']:
        raise RuntimeError('Original checkpoint changed externally')
    if sha256(cfg['warm_checkpoint']) != cfg['warm_checkpoint_sha256']:
        raise RuntimeError('Starting submitted V3.3 weights changed externally')
    if any(sha256(p) != digest for p,digest in fingerprints.items()):
        raise RuntimeError('Input metadata changed during training')
    atomic_json(run/'completed.json', {'reason': reason, 'selection': controller.dump(),
                'original_best_preserved': True, 'starting_v33_best_preserved': True,
                'source_provenance': provenance(cfg),
                'eligible_improvement': controller.state['best_safe']['tag'] != 'baseline',
                'success': acceptance(controller.state['best_safe'], controller.state['anchor'], cfg)['success'],
                'cache_metadata_unchanged': True})
    print('V34_TRAINING_COMPLETE=True ORIGINAL_BEST_PRESERVED=True', flush=True)


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
