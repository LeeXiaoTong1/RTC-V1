"""Adapt the final decision, then train CE with one ordinary and one noisy view."""
from contextlib import nullcontext
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from live_progress import Phase
from w2v_v32.model import microbatches
from w2v_v313.train import infer
from w2v_v313.replay import rows_signature
from w2v_v312.replay import delta
from w2v_v39.common import announce, atomic_json, digest, read_json, verify_files
from w2v_v39.metrics import measure, selection, change_audit
from .config import verify_inputs
from .data import bundles, SourcePlan, loader, loss_weights
from .features import extract_cache
from .head import fit as fit_head, predict as predict_head
from .model import load_model, joint_mode, optimizer_for
from .diagnostics import offline_rows, offline_metrics, source_replay
from .state import (SCHEMA, identity, partial_state, apply_partial, to_cpu, atomic_save,
    capture_rng, restore_rng, storage_budget, load_resume, prune_candidates, apply_candidate)


def acceptance(baseline, metrics, cfg):
    item = dict(name='supervised_detector', status='fitted', metrics=metrics)
    selection(baseline, [item], cfg)
    return item['eligible'], item['guardrails']


def print_metrics(tag, metrics, eligible, reasons):
    print(f'\n[Dev] V3.14 {tag} Clean={100*metrics["clean_f1"]:.3f} '
          f'Noisy={100*metrics["noisy_f1"]:.3f} Weighted={100*metrics["weighted_f1"]:.3f}', flush=True)
    for condition in ('online', 'seen', 'heldout'):
        for language in ('en', 'zh'):
            key = condition+'/'+language
            value = metrics['groups'][key]
            matched = metrics['matched'][key]['real_recall']
            print(f'  {key} fake={100*value["recall"][0]:.3f}% real={100*value["recall"][1]:.3f}% '
                  f'AUC={100*value["auc"]:.3f}% real@99%fake={100*matched:.3f}%', flush=True)
    print('  guarded_eligible='+str(eligible)+'; reasons='+','.join(reasons), flush=True)


def schedule(optimizer, step, total, cfg):
    warm = max(1, int(total*cfg['lr_warmup_fraction']))
    if step < warm:
        scale = (step+1)/warm
    else:
        fraction = min(1., (step-warm)/max(1, total-warm))
        scale = .1 + .9*.5*(1+math.cos(math.pi*fraction))
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr']*scale


def train_step(model, optimizer, examples, cfg):
    model.train()
    weights = loss_weights(examples)
    optimizer.zero_grad(set_to_none=True)
    total, maximum, visited, maximum_frames = 0., 0., [], 0
    for indices, x, mask in microbatches(examples, cfg['microbatch'], cfg['frame_budget']):
        maximum_frames = max(maximum_frames, x.shape[0]*x.shape[1])
        context = torch.autocast('cuda', dtype=torch.bfloat16) if cfg['amp'] == 'bf16' else nullcontext()
        with context:
            z, _ = model(x.to(cfg['device']), mask.to(cfg['device']))
        labels = torch.tensor([examples[i]['label'] for i in indices], device=cfg['device'])
        ce = F.cross_entropy(z.float(), labels, reduction='none')
        loss = (ce * z.new_tensor(weights[indices], dtype=torch.float32)).sum()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite supervised CE')
        loss.backward()
        total += float(loss.detach())
        maximum = max(maximum, float(ce.detach().max()))
        visited.extend(indices)
        # Release each microbatch immediately. No paired/teacher/probe graph.
        model.head.classifier.features = None
        model.head.classifier.monitor = {}
        del z, ce, loss
    if sorted(visited) != list(range(len(examples))):
        raise ValueError('Incomplete training microbatch coverage')
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['max_grad_norm'])
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite supervised gradient; update not applied')
    optimizer.step()
    return dict(classification_loss=total, maximum_example_ce=maximum, gradient_norm=float(norm),
                maximum_padded_frames=maximum_frames, noisy_ce_mass=.5)


def update_selections(selections, scores, candidates, tag, metrics, eligible, candidate):
    promoted = []
    for kind in ('best_weighted', 'best_guarded'):
        if (kind == 'best_weighted' or eligible) and metrics['weighted_f1'] > scores[kind]:
            selections[kind], scores[kind] = tag, metrics['weighted_f1']
            candidates[tag] = candidate
            promoted.append(kind)
    return prune_candidates(candidates, selections), promoted


def _close(bundle):
    for key in ('x', 'logits'):
        bundle[key]._mmap.close()


def _cache(model, rows, split, cfg, run):
    return extract_cache(model, rows, cfg, run/'features'/split,
        dict(version='3.14', split=split, starting_checkpoint_sha256=cfg['starting_checkpoint_sha256'],
             base_checkpoint_sha256=cfg['base_checkpoint_sha256'], config_identity=identity(cfg),
             data_fingerprints=cfg['data_fingerprints'], code_fingerprints=cfg['code_fingerprints']))


def baseline(model, dev, off, cfg, run):
    signature = dict(identity=identity(cfg), rows=rows_signature(dev['rows']), offline=rows_signature(off))
    marker = run/'baseline_complete.json'
    if marker.is_file():
        saved = read_json(marker)
        if saved['signature'] != signature:
            raise ValueError('Committed baseline inventory/configuration changed')
        verify_files({str(run/name):h for name, h in saved['files'].items()})
        with np.load(run/'dev_scores_baseline.npz', allow_pickle=False) as data:
            logits = data['logits'].copy()
        return logits, read_json(run/'baseline_metrics.json')
    announce('V3.14 verifying actual V3.12 LAST and extracting its Dev classifier inputs')
    cache = _cache(model, dev['rows'], 'dev', cfg, run)
    try:
        logits = np.array(cache['logits'], copy=True)
    finally:
        _close(cache)
    joint_mode(model, cfg)
    value = measure(dev['rows'], logits, target=cfg['matched_fake_recall'])
    atomic_json(run/'startup_replay.json', source_replay(cfg, dev['rows'], logits))
    atomic_json(run/'baseline_metrics.json', value)
    np.savez_compressed(run/'dev_scores_baseline.npz', logits=logits)
    if off:
        scores, _ = infer(model, off, cfg, 'V3.14 original Offline diagnostic')
        atomic_json(run/'offline_baseline.json', offline_metrics(off, scores))
    names = ['startup_replay.json', 'baseline_metrics.json', 'dev_scores_baseline.npz']
    if off:
        names.append('offline_baseline.json')
    atomic_json(marker, dict(signature=signature, files={name:digest(run/name) for name in names}))
    print_metrics('starting_last', value, True, [])
    return logits, value


def stage_a(model, train, dev, original_logits, original_metrics, cfg, run):
    announce('V3.14 Stage A: freeze encoder, MultiConv and adapter; fit final Linear on Train only')
    original = to_cpu(model.head.classifier[-1].state_dict())
    cache = _cache(model, train['rows'], 'train', cfg, run)
    try:
        head, report = fit_head(cache, original, cfg)
    finally:
        _close(cache)
    dev_cache = _cache(model, dev['rows'], 'dev', cfg, run)
    try:
        cached_scores = predict_head(dev_cache['x'], head, cfg['device'], cfg['head_chunk'])
    finally:
        _close(dev_cache)
    joint_mode(model, cfg)
    model.head.classifier[-1].load_state_dict(head)
    # Re-evaluate the actual deployment path before admitting any candidate.
    logits, _ = infer(model, dev['rows'], cfg, 'V3.14 Stage A actual detector replay')
    replay = delta(cached_scores, logits)
    if not replay['allclose'] or replay['decision_changes']:
        raise ValueError('Adapted output does not replay through the deployed detector')
    value = measure(dev['rows'], logits, target=cfg['matched_fake_recall'])
    eligible, reasons = acceptance(original_metrics, value, cfg)
    report.update(metrics=value, guarded_eligible=eligible, reasons=reasons, deployment_replay=replay,
                  joint_start='stage_a_head' if eligible else 'starting_last')
    atomic_json(run/'stage_a_report.json', report)
    np.savez_compressed(run/'dev_scores_stage_a_head.npz', logits=logits)
    print_metrics('stage_a_head', value, eligible, reasons)
    print('V314_JOINT_START='+report['joint_start'], flush=True)
    if not eligible:
        model.head.classifier[-1].load_state_dict(original)
    entry = dict(tag='stage_a_head', phase='head', cursor=0, metrics=value, eligible=eligible,
                 reasons=reasons, changed=change_audit(dev['rows'], original_logits, logits), committed=True)
    return entry, dict(kind='head', state=head)


def finish(cfg, run, total, safety=None):
    state = load_resume(run/'last.pt', cfg)
    atomic_json(run/'training_history.json', state['history'])
    pending = run/'validation_pending.json'
    if pending.is_file() and read_json(pending)['cursor'] <= state['cursor']:
        pending.unlink()
    done = dict(version='3.14', status='complete', state_file='last.pt',
        checkpoint_sha256=digest(run/'last.pt'), base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
        starting_checkpoint_sha256=cfg['starting_checkpoint_sha256'], starting_tag=cfg['starting_tag'],
        selections=state['selections'], scores=state['scores'], last_tag=state['last_tag'],
        committed_updates=state['cursor'], planned_updates=total,
        early_stopped=state['cursor'] < total, stop_reason=safety or state.get('stop_reason'),
        fallback_target='trained_v312_last', external_teacher_at_inference=False)
    atomic_json(run/'completed.json', done)
    for kind in ('best_weighted', 'best_guarded', 'last'):
        atomic_json(run/(kind+'.json'), dict(selector=kind, checkpoint='last.pt',
            tag=state['last_tag'] if kind == 'last' else state['selections'][kind],
            note='Alias into one checkpoint; identical selected states share tensors'))
    print('V314_SELECTED='+str(state['selections'])+'; last='+state['last_tag'], flush=True)
    return done


def run_experiment(cfg, run):
    verify_inputs(cfg)
    run = Path(run)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cfg['device'].startswith('cuda'):
        torch.cuda.set_device(torch.device(cfg['device']))
        if cfg['amp'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unavailable; no silent precision change')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    model = load_model(cfg)
    with bundles(cfg) as (train, dev):
        off, off_note = offline_rows(cfg, dev['rows'], train['rows'])
        atomic_json(run/'offline_policy.json', dict(count=len(off), note=off_note))
        atomic_json(run/'dev_rows.json', dev['rows'])
        plan = SourcePlan(train['rows'], cfg['source_batch'], cfg['seed'])
        total = plan.steps * cfg['epochs']
        budget = storage_budget(model, cfg, len(train['rows'])+len(dev['rows']))
        atomic_json(run/'storage_budget.json', budget)
        print(f'V314_STORAGE checkpoint_atomic_peak_estimate_GiB={budget["maximum_atomic_peak_bytes"]/1024**3:.2f}; '
              f'feature_tensors_MiB={budget["feature_tensor_bytes"]/1024**2:.1f}; new_audio_GiB=0', flush=True)
        original_logits, original_metrics = baseline(model, dev, off, cfg, run)
        joint_mode(model, cfg)
        optimizer = optimizer_for(model, cfg)
        if (run/'last.pt').is_file():
            state = load_resume(run/'last.pt', cfg)
            apply_partial(model, state['model'])
            optimizer.load_state_dict(state['optimizer'])
            restore_rng(state['rng'])
            print('V314_RESUME committed_joint_updates='+str(state['cursor'])+'; head fit already committed', flush=True)
        else:
            entry, candidate = stage_a(model, train, dev, original_logits, original_metrics, cfg, run)
            selections = dict(best_weighted='starting_last', best_guarded='starting_last')
            scores = {k:original_metrics['weighted_f1'] for k in selections}
            candidates, promoted = update_selections(selections, scores, {}, entry['tag'], entry['metrics'], entry['eligible'], candidate)
            entry['promoted'] = promoted
            state = dict(schema=SCHEMA, identity=identity(cfg), model=partial_state(model),
                optimizer=to_cpu(optimizer.state_dict()), rng=capture_rng(), cursor=0, history=[entry],
                candidates=candidates, selections=selections, scores=scores,
                last_tag=entry['tag'] if entry['eligible'] else 'starting_last', stale=0,
                progress_best=entry['metrics']['weighted_f1'] if entry['eligible'] else original_metrics['weighted_f1'],
                stop=False, stop_reason=None)
            atomic_save(run/'last.pt', state, cfg['disk_margin_bytes'])
        cursor, stale, progress_best = state['cursor'], state['stale'], state['progress_best']
        history, candidates, selections, scores = state['history'], state['candidates'], state['selections'], state['scores']
        stop = state['stop']
        del state
        atomic_json(run/'training_history.json', history)
        atomic_json(run/'training_design.json', dict(starting_checkpoint_sha256=cfg['starting_checkpoint_sha256'],
            trainable_parameters={name:p.numel() for name,p in model.named_parameters() if p.requires_grad},
            optimizer_groups=[dict(name=g['name'], lr=g['initial_lr'], parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups],
            final_linear_trainable=True, noisy_ce_mass=.5, group_ce_mass=.25, auxiliary_losses=[],
            max_joint_epochs=cfg['epochs'], patience_checks=cfg['patience'], teacher_passes_per_update=0))
        while cursor < total and not stop:
            epoch, within = divmod(cursor, plan.steps)
            atomic_json(run/f'sampling_epoch_{epoch+1}.json', plan.coverage(epoch))
            boundaries = sorted({math.ceil(plan.steps*i/cfg['checks_per_epoch']) for i in range(1, cfg['checks_per_epoch']+1)})
            boundary = next(b for b in boundaries if b > within)
            announce(f'V3.14 joint epoch {epoch+1} steps {within+1}-{boundary}; CE only; Noisy=50%')
            phase = Phase(f'V3.14 joint epoch {epoch+1} steps {within+1}-{boundary}', boundary-within)
            began, start, stats_sum, peak_ce = time.monotonic(), within, 0., 0.
            for batch in loader(plan, epoch, within, cfg):
                examples = [dict(r, features=torch.from_numpy(r['features']), mask=torch.from_numpy(r['mask'])) for r in batch]
                schedule(optimizer, cursor, total, cfg)
                try:
                    stats = train_step(model, optimizer, examples, cfg)
                except FloatingPointError as exc:
                    safety = dict(stage='training', reason=str(exc), attempted_update=cursor+1,
                                  note='Uncommitted segment discarded; last successful validation preserved')
                    atomic_json(run/'safety_stop.json', safety)
                    return finish(cfg, run, total, safety)
                cursor += 1; within += 1
                stats_sum += stats['classification_loss']; peak_ce = max(peak_ce, stats['maximum_example_ce'])
                phase.update(within-start, loss=stats['classification_loss'])
                if within == 1 or within % 100 == 0 or within == boundary:
                    gpu = f' peak_GPU_GiB={torch.cuda.max_memory_allocated()/1024**3:.2f}' if cfg['device'].startswith('cuda') else ''
                    print(f'  STEP {within}/{plan.steps} CE={stats["classification_loss"]:.5f} '
                          f'max_example_CE={stats["maximum_example_ce"]:.3f} grad={stats["gradient_norm"]:.3f} '
                          f'seconds/update={(time.monotonic()-began)/(within-start):.2f}{gpu}', flush=True)
                if within != boundary:
                    continue
                seconds = time.monotonic()-began
                tag = f'epoch_{epoch+1}_step_{within}'
                announce('V3.14 fixed Online/Noisy Dev: '+tag)
                try:
                    logits, _ = infer(model, dev['rows'], cfg, 'V3.14 '+tag+' Dev')
                    value = measure(dev['rows'], logits, target=cfg['matched_fake_recall'])
                    offline = None
                    if off and within == plan.steps:
                        oz, _ = infer(model, off, cfg, 'V3.14 '+tag+' Offline diagnostic only')
                        offline = offline_metrics(off, oz)
                except FloatingPointError as exc:
                    safety = dict(stage='validation', reason=str(exc), attempted_update=cursor)
                    atomic_json(run/'safety_stop.json', safety)
                    return finish(cfg, run, total, safety)
                eligible, reasons = acceptance(original_metrics, value, cfg)
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters() if p.requires_grad):
                    safety = dict(stage='validation', reason='Nonfinite parameter; previous checkpoint retained')
                    atomic_json(run/'safety_stop.json', safety)
                    return finish(cfg, run, total, safety)
                current = partial_state(model)
                candidates, promoted = update_selections(selections, scores, candidates, tag, value, eligible,
                                                         dict(kind='partial', state=current))
                if value['weighted_f1'] >= progress_best + cfg['progress_min_delta']:
                    stale, progress_best = 0, value['weighted_f1']
                else:
                    stale += 1
                catastrophic = value['weighted_f1'] < original_metrics['weighted_f1'] - cfg['catastrophic_weighted_drop']
                stop = catastrophic or stale >= cfg['patience']
                stop_reason = 'catastrophic_regression' if catastrophic else ('two_checks_without_progress' if stop else None)
                entry = dict(tag=tag, phase='joint', cursor=cursor, metrics=value, eligible=eligible,
                    reasons=reasons, promoted=promoted, committed=True, offline=offline,
                    changed=change_audit(dev['rows'], original_logits, logits),
                    segment_training=dict(updates=within-start, mean_ce=stats_sum/(within-start),
                        maximum_example_ce=peak_ce, training_seconds=seconds, seconds_per_update=seconds/(within-start)),
                    last_training_step=stats, checks_without_progress=stale, stop=stop, stop_reason=stop_reason)
                history.append(entry)
                np.savez_compressed(run/('dev_scores_'+tag+'.npz'), logits=logits)
                atomic_json(run/'validation_pending.json', entry)
                print_metrics(tag, value, eligible, reasons)
                print(f'  best_weighted={selections["best_weighted"]}; best_guarded={selections["best_guarded"]}; '
                      f'no_progress={stale}/{cfg["patience"]}; stop={stop}', flush=True)
                snapshot = dict(schema=SCHEMA, identity=identity(cfg), model=current,
                    optimizer=to_cpu(optimizer.state_dict()), rng=capture_rng(), cursor=cursor, history=history,
                    candidates=candidates, selections=selections, scores=scores, last_tag=tag,
                    stale=stale, progress_best=progress_best, stop=stop, stop_reason=stop_reason)
                announce('V3.14 committing current + selected partial states; shared states saved once')
                atomic_save(run/'last.pt', snapshot, cfg['disk_margin_bytes'])
                del snapshot, current
                atomic_json(run/'training_history.json', history)
                (run/'validation_pending.json').unlink(missing_ok=True)
                # Reconstruct a deterministic loader from this committed cursor.
                break
        verify_inputs(cfg)
        return finish(cfg, run, total)
