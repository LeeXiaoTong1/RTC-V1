"""Two independent experiments from a completed last.pt; original files read-only."""
from collections import Counter
import copy
from itertools import islice
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch import nn
from w2v_aasist.launch import run_lock
from w2v_v318.common import (ROOT, read_json, atomic_json, digest, verify_files, fingerprint,
    schema_for, seed_all, apply_partial, partial_state, restore_rng, append_json)
from w2v_v318.config import verify_inputs
from w2v_v318.model import load_model, optimizer_for
from w2v_v318.inference import infer
from w2v_v318.data import Waves, loader, close, microbatches
from w2v_v318.sampling import Plan, validate_units
from w2v_v318.step import train_step
from w2v_v318.monitor import measure, summarize
from .measurements import (StepObserver, recalibration_mode, bn_buffers, tensor_digest, check_bn_only)


def emit(phase, count=None, total=None, start=None):
    suffix = ''
    if count is not None:
        seconds = time.monotonic()-start
        eta = seconds/max(1, count)*(total-count)
        suffix = f' {count}/{total} ({100*count/total:.1f}%) elapsed={seconds:.0f}s ETA={eta:.0f}s'
    print('AUDIT_PROGRESS=' + phase + suffix, flush=True)


class ProgressModel(nn.Module):
    def __init__(self, model, phase, total):
        super().__init__()
        self.model, self.phase, self.total = model, phase, total
        self.count = 0; self.start = self.last = time.monotonic()
    def forward(self, waves):
        result = self.model(waves)
        self.count += len(waves)
        if time.monotonic()-self.last >= 10 or self.count == self.total:
            emit(self.phase, self.count, self.total, self.start); self.last = time.monotonic()
        return result


def evaluate(model, dev, cfg, phase, partition):
    emit(phase)
    logits, _ = infer(ProgressModel(model, phase, len(dev)), dev, cfg)
    scores = dict(full=measure(dev, logits), groups=summarize(dev, logits)['groups'])
    for part in ('select', 'calibration'):
        ix = partition[part]
        scores[part] = measure([dev[i] for i in ix], logits[ix])
    print(f'AUDIT_DEV={phase} Weighted={100*scores["full"]["weighted_f1"]:.3f} '
          f'Selection={100*scores["select"]["weighted_f1"]:.3f}', flush=True)
    return logits, scores


def checkpoint(run):
    cfg = read_json(run/'config.json'); done = read_json(run/'completed.json')
    if done.get('version') != '3.18' or done.get('status') != 'complete':
        raise ValueError('A completed V3.18 run is required; wait for training to finish')
    if digest(run/'last.pt') != done['checkpoint_sha256']:
        raise ValueError('Original checkpoint checksum differs from completed.json')
    state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
    if state.get('schema') != schema_for(cfg) or state.get('identity') != fingerprint(cfg):
        raise ValueError('Checkpoint/config identity mismatch')
    if state.get('last_tag') != done['last_tag'] or state.get('epoch') != done['completed_epochs']:
        raise ValueError('Completed epoch metadata mismatch')
    if state['epoch'] <= cfg['warm_epochs']:
        raise ValueError('A completed joint-phase checkpoint is required')
    if not any(e['tag'] == state['last_tag'] for e in state['history']):
        raise ValueError('Selected checkpoint is absent from history')
    if not state.get('optimizer') or not state.get('rng'):
        raise ValueError('Original optimizer and RNG are required for a faithful update diagnostic')
    verify_files({str(run/n): cfg['saved_manifests'][n] for n in ('train_rows.json', 'dev_rows.json')})
    return cfg, done, state


def reset_copy(model, weights, state):
    # BN changes and short-update weights are never carried into another branch.
    apply_partial(model, weights)
    model.set_phase(True); model.eval()
    restore_rng(state['rng'])
    if tensor_digest(partial_state(model)) != tensor_digest(weights):
        raise RuntimeError('Independent checkpoint copy could not be restored exactly')


def train_batches(train, cfg, epoch, count):
    plan = Plan(train, cfg['stream_sources'], cfg['seed'])
    if count > plan.steps:
        raise ValueError(f'Requested {count} audit batches exceeds one logical epoch ({plan.steps})')
    tickets = list(islice(plan.batches(epoch), count))
    coverage = Counter()
    unique = set()
    for batch in tickets:
        for t in batch:
            row = train[t['index']]
            coverage[f'{t["kind"]}/{row["language"]}/{row["label"]}'] += 1
            unique.add((t['kind'], row['group_id']))
    metadata = dict(epoch_index=epoch, batches=count, views=count*cfg['stream_sources']*3,
                    draw_counts=dict(coverage), unique_sources=len(unique),
                    ticket_sha256=fingerprint(tickets), sampler='unchanged balanced Plan; joint augmentation')
    return loader(Waves(train, cfg), cfg, batch_sampler=tickets), metadata


def recalibrate(model, train, cfg, epoch, count, out):
    emit('bn_train_only')
    before = partial_state(model); allowed = bn_buffers(model)
    batches, metadata = train_batches(train, cfg, epoch, count)
    atomic_json(out/'bn_sampling.json', metadata)
    start = time.monotonic(); observed = 0; processing = Counter()
    try:
        with recalibration_mode(model):
            for step, units in enumerate(batches, 1):
                validate_units(units, cfg['stream_sources'])
                for unit in units:
                    for row in unit:
                        rec = row.get('augmentation', {}).get('recipe', {})
                        if rec: processing[f'{rec["family"]}/{rec["codec"]}'] += 1
                for micro in microbatches(units, cfg['microbatch'], cfg['frame_budget']):
                    rows = [r for unit in micro for r in unit]
                    model([r['wave'] for r in rows]); observed += len(rows)
                if step == 1 or step % 8 == 0 or step == count:
                    emit('bn_train_only', step, count, start)
    finally:
        close(batches)
    if observed != metadata['views']:
        raise RuntimeError('Incomplete BN sample coverage')
    model.eval()
    result = check_bn_only(before, partial_state(model), allowed)
    result.update(views=observed, processing_counts=dict(processing),
                  method='reset running stats; momentum=None; dropout off; only BN in train mode',
                  limitations='Batch-averaged estimate; variable window counts; no guarantee of improvement')
    # Small local BN buffer overlay only. Excluded from the report ZIP.
    torch.save(dict(source_partial_sha256=tensor_digest(before), bn_buffers=bn_buffers(model)), out/'bn_buffers.pt')
    atomic_json(out/'bn_changes.json', result)
    return result


def update_audit(model, train, cfg, state, count, out):
    emit('short_updates_original_checkpoint')
    optimizer = optimizer_for(model, cfg)
    # deepcopy prevents optimizer state tensors from aliasing the reference checkpoint on CPU.
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    rates = [dict(name=g['name'], lr=float(g['lr']), peak_lr=float(g['initial_lr']),
                  weight_decay=g['weight_decay']) for g in optimizer.param_groups]
    if any(not np.isfinite(g['lr']) or g['lr'] <= 0 for g in rates):
        raise ValueError('Expected positive finite joint-phase checkpoint learning rates')
    before_bn = tensor_digest(bn_buffers(model))
    before = partial_state(model)
    batches, sampling = train_batches(train, cfg, state['epoch'], count)
    atomic_json(out/'update_setup.json', dict(rates=rates, sampling=sampling,
        lr_policy='constant checkpoint terminal LR; original AdamW moments restored; no schedule extrapolation',
        risk_coefficient=cfg['risk_coefficient'], max_grad_norm=cfg['max_grad_norm'],
        weights_sha256=tensor_digest(before),
        limitation='This measures terminal-state updates, not historical epochs or an optimal LR. No LR change is tested.'))
    start = time.monotonic(); records = []
    try:
        for step, units in enumerate(batches, 1):
            observer = StepObserver(model, optimizer)
            with observer.active():
                stats = train_step(model, optimizer, units, cfg, cfg['risk_coefficient'])
            observed = observer.result(include_effective=step == 1 or step % 10 == 0 or step == count)
            row = dict(step=step, loss=stats['loss'], raw_ce=stats['raw_ce_sum']/stats['views'],
                       global_grad_pre=stats['grad_norm'], lrs=stats['lrs'],
                       gradients=observed['gradients'], updates=observed['updates'],
                       cells=stats['cells'], processing_counts=stats['processing_counts'])
            records.append(row); append_json(out/'updates.jsonl', row)
            append_json(out/'parameters.jsonl', dict(step=step, parameters=observed['parameters']))
            if observed['adapters']:
                append_json(out/'adapters.jsonl', dict(step=step, adapters=observed['adapters']))
            del observer
            if step == 1 or step % 10 == 0 or step == count:
                emit('short_updates_original_checkpoint', step, count, start)
    finally:
        close(batches)
    if len(records) != count:
        raise RuntimeError('Incomplete short-update diagnostic')
    if tensor_digest(bn_buffers(model)) != before_bn:
        raise RuntimeError('Joint update unexpectedly modified BatchNorm statistics')
    after = partial_state(model)
    changed = [n for n in before if not torch.equal(before[n], after[n])]
    result = dict(steps=count, bn_buffers_unchanged=True, changed_parameters=len(changed),
                  initial_partial_sha256=tensor_digest(before), final_partial_sha256=tensor_digest(after),
                  clipping_fraction=sum(r['global_grad_pre'] > cfg['max_grad_norm'] for r in records)/count,
                  groups={})
    for group in records[0]['gradients']:
        relative = [r['updates'][group]['relative'] for r in records if r['updates'][group]['relative'] is not None]
        result['groups'][group] = dict(
            lr=records[0]['lrs'][group],
            median_grad_pre=float(np.median([r['gradients'][group]['pre'] for r in records])),
            median_grad_post=float(np.median([r['gradients'][group]['post'] for r in records])),
            median_update_norm=float(np.median([r['updates'][group]['norm'] for r in records])),
            median_relative_update=float(np.median(relative)) if relative else None,
            zero_update_steps=sum(r['updates'][group]['changed_elements'] == 0 for r in records),
            missing_gradient_names=sorted({n for r in records for n in r['gradients'][group]['missing']}))
    atomic_json(out/'update_summary.json', result)
    return result


def execute(request, out, model_factory=load_model, verify=True):
    out = Path(out).resolve(); run = Path(request['run']).resolve()
    if out == run or run in out.parents:
        raise ValueError('Audit output must be outside the training run')
    result = dict(status='running', source_run=str(run), phases={}, errors=[], original_unchanged=False)
    def save(): atomic_json(out/'result.json', result)
    save()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        emit('verify_original_checkpoint_and_environment')
        original, done, state = checkpoint(run)
        if verify: verify_inputs(original)
        protected = {str(run/n): digest(run/n) for n in
                     ('config.json', 'last.pt', 'completed.json', 'train_rows.json', 'dev_rows.json', 'dev_partition.json')}
        try:
            cfg = dict(original, workers=request['workers'], device=request['device'])
            train, dev = read_json(run/'train_rows.json'), read_json(run/'dev_rows.json')
            partition = read_json(run/'dev_partition.json')
            select, cal = partition['select'], partition['calibration']
            if len(set(select)) != len(select) or len(set(cal)) != len(cal) or set(select) & set(cal) or set(select+cal) != set(range(len(dev))):
                raise ValueError('Dev partition does not cover the fixed manifest exactly once')
            result.update(source_tag=state['last_tag'], checkpoint_sha256=done['checkpoint_sha256'],
                          source_partial_sha256=tensor_digest(state['model']), frontend=cfg.get('omni_arch'),
                          variant=cfg['variant'], baseline_policy='full Dev raw logits; fixed threshold; no calibration fit',
                          selection_note='Selection Dev has already been used for checkpoint choice; this is diagnosis, not a fresh test set.')
            atomic_json(out/'audit_config.json', dict(request=request, source_config_sha256=fingerprint(original),
                source_tag=state['last_tag'], settings={k: cfg[k] for k in
                ('seed','workers','device','stream_sources','microbatch','frame_budget','eval_batch',
                 'lora_lr','head_lr','evidence_lr','warm_epochs','epochs','max_grad_norm','risk_coefficient')}))
            atomic_json(out/'dev_index.json', [{k: r[k] for k in ('id','condition','language','label')} for r in dev])
            seed_all(cfg['seed']); emit('load_omni_and_checkpoint')
            model = model_factory(cfg); reset_copy(model, state['model'], state)
            baseline, metric = evaluate(model, dev, cfg, 'baseline_full_dev', partition)
            np.savez_compressed(out/'baseline_scores.npz', logits=baseline)
            atomic_json(out/'baseline_metrics.json', metric)
            cached = state['last_logits'].numpy()
            if cached.shape != baseline.shape:
                raise ValueError('Saved Dev scores have a different shape')
            result['baseline_reproduction'] = dict(max_logit_difference=float(np.max(np.abs(cached-baseline))),
                cached_weighted=measure(dev, cached)['weighted_f1'], reproduced_weighted=metric['full']['weighted_f1'])
            result['phases']['baseline'] = 'passed'; save()
            try:
                reset_copy(model, state['model'], state)
                recalibrate(model, train, cfg, state['epoch'], request['bn_batches'], out)
                bn_logits, bn_metric = evaluate(model, dev, cfg, 'bn_reestimated_full_dev', partition)
                np.savez_compressed(out/'bn_scores.npz', logits=bn_logits)
                atomic_json(out/'bn_metrics.json', bn_metric)
                result['bn_delta_pp'] = {part: {key: 100*(bn_metric[part][key]-metric[part][key])
                    for key in ('weighted_f1','clean_f1','noisy_f1')} for part in ('full','select','calibration')}
                result['phases']['bn'] = 'passed'
            except Exception:
                result['phases']['bn'] = 'failed'; result['errors'].append(traceback.format_exc()); traceback.print_exc()
            save()
            # Independent state AND RNG restoration, even if BN failed halfway.
            reset_copy(model, state['model'], state)
            if cfg['device'].startswith('cuda'): torch.cuda.empty_cache()
            try:
                update_audit(model, train, cfg, state, request['update_steps'], out)
                result['phases']['updates'] = 'passed'
            except Exception:
                result['phases']['updates'] = 'failed'; result['errors'].append(traceback.format_exc()); traceback.print_exc()
            result['status'] = 'complete' if not result['errors'] else 'failed'
        finally:
            try:
                verify_files(protected)
                if verify: verify_files(original['code_fingerprints'])
                result['original_unchanged'] = True
            finally: save()
    return result
