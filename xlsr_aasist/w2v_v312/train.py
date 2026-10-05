"""Full-wave fine-tuning with real-only conditional language adversaries."""
from contextlib import nullcontext
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from live_progress import Phase
from w2v_aasist.progress import progress
from w2v_v3.model import microbatches as exact_microbatches
from w2v_v32.model import microbatches as padded_microbatches
from w2v_v36.features import inference_batches
from w2v_v39.common import announce, atomic_json, digest, read_json
from w2v_v39.data import base_weights
from w2v_v39.metrics import measure, selection, change_audit
from .config import verify_inputs
from .data import bundles, probe_split, SourcePlan, loader, loss_weights
from .model import load_model, LanguageAdversary, optimizer_for
from .objectives import reference_rows, retention_loss
from .control import stopping
from .stability import StabilityStop, scale_loss, summarize, compare_diagnostics
from .probes import run_probes, compare
from .replay import fp32_inference, baseline as prepare_baseline
from .state import (SCHEMA, identity, partial_state, apply_partial, to_cpu, atomic_save,
                    capture_rng, restore_rng, storage_budget, load_resume)


def infer(model, rows, cfg, label, capture=False, diagnostics=None):
    model.eval()
    all_logits, all_features = [], []
    width = model.head.classifier[-1].in_features
    captured = []
    input_stats = []
    scalar_features, scalar_stats, scalar_adapter = [], [], []
    handle = model.head.classifier[-1].register_forward_pre_hook(lambda _module,args: captured.append(args[0]))
    stats_handle = model.head.classifier.register_forward_pre_hook(lambda _module,args: input_stats.append(args[0]))
    try:
        with fp32_inference(cfg['device']):
            for examples in progress(inference_batches(rows, cfg),
                                     total=math.ceil(len(rows)/cfg['feature_batch']), label=label, every=200):
                logits = np.empty((len(examples),2), dtype=np.float32)
                features = np.empty((len(examples),width), dtype=np.float32) if capture else None
                batch_rms = np.empty(len(examples),dtype=np.float32)
                batch_stats = np.empty(len(examples),dtype=np.float32)
                batch_adapter = np.zeros(len(examples),dtype=np.float32)
                visited = []
                for indices, x, mask in exact_microbatches(examples, cfg['microbatch'], cfg['frame_budget']):
                    captured.clear()
                    input_stats.clear()
                    z, _ = model(x.to(cfg['device']), mask.to(cfg['device']))
                    if len(captured)!=1:
                        raise ValueError('Expected one final classifier call per inference microbatch')
                    h = captured[0]
                    if not bool(torch.isfinite(z).all() and torch.isfinite(h).all()):
                        raise FloatingPointError('Nonfinite validation representation')
                    logits[indices] = z.float().cpu().numpy()
                    if capture:
                        features[indices] = h.float().cpu().numpy()
                    if diagnostics is not None:
                        batch_rms[indices] = (torch.linalg.vector_norm(h.float(),dim=-1)/math.sqrt(h.shape[-1])).cpu().numpy()
                        pooled = input_stats[0]
                        batch_stats[indices] = (torch.linalg.vector_norm(pooled.float(),dim=-1)/math.sqrt(pooled.shape[-1])).cpu().numpy()
                        monitor = getattr(model.head.classifier,'monitor',{})
                        if monitor:
                            batch_adapter[indices] = monitor['adapter_ratio'].float().cpu().numpy()
                    if hasattr(model.head.classifier,'features'):
                        model.head.classifier.features = None
                    visited.extend(indices)
                if sorted(visited) != list(range(len(examples))):
                    raise ValueError('Validation microbatch coverage changed')
                start = sum(len(z) for z in all_logits)
                if [(r['id'],r.get('condition')) for r in examples] != [(r['id'],r.get('condition')) for r in rows[start:start+len(examples)]]:
                    raise ValueError('Validation row order changed')
                all_logits.append(logits)
                if diagnostics is not None:
                    scalar_features.extend(batch_rms)
                    scalar_stats.extend(batch_stats)
                    scalar_adapter.extend(batch_adapter)
                if capture:
                    all_features.append(features)
    finally:
        handle.remove()
        stats_handle.remove()
        captured.clear()
        input_stats.clear()
        if hasattr(model.head.classifier,'features'):
            model.head.classifier.features = None
            model.head.classifier.monitor = {}
    if diagnostics is not None:
        diagnostics.update(summarize(rows,scalar_features,scalar_stats,np.concatenate(all_logits),scalar_adapter))
    return np.concatenate(all_logits), np.concatenate(all_features) if capture else None


def schedule(optimizer, step, total, epoch_steps, cfg):
    warm = max(1, int(total * cfg['lr_warmup_fraction']))
    if step < warm:
        scale = (step+1)/warm
    else:
        fraction = min(1., (step-warm)/max(1,total-warm))
        scale = cfg['lr_floor'] + (1-cfg['lr_floor'])*.5*(1+math.cos(math.pi*fraction))
    for group in optimizer.param_groups:
        # The reader learns from step one; reversal into the detector warms up separately.
        group['lr'] = group['initial_lr'] * (1. if group['name']=='language_adversary' else scale)
    # First let the reader learn. Then increase pressure over the rest of epoch one.
    fraction = (step-epoch_steps*cfg['adv_warmup_fraction'])/max(1,epoch_steps*cfg['adv_ramp_fraction'])
    return cfg['adv_weight'] * min(1., max(0., fraction))


def train_step(model, adversary, optimizer, examples, cfg, strength):
    model.train()
    adversary.train()
    optimizer.zero_grad(set_to_none=True)
    ce_weights, adversary_weights = loss_weights(examples)
    total_ce = total_adv = total_retention = total_stability = 0.
    maximum_feature_ratio = maximum_margin_ratio = maximum_adapter_ratio = maximum_example_ce = 0.
    retention_breaches = 0
    gradient_audit = []
    maximum_frames = 0
    for indices, x, mask in padded_microbatches(examples, cfg['microbatch'], cfg['frame_budget']):
        maximum_frames = max(maximum_frames, x.shape[0]*x.shape[1])
        if cfg['device'].startswith('cuda'):
            x, mask = x.pin_memory(), mask.pin_memory()
        rows = [examples[i] for i in indices]
        device = torch.device(cfg['device'])
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if cfg['amp']=='bf16' else nullcontext()
        with autocast:
            logits, _ = model(x.to(device, non_blocking=True), mask.to(device, non_blocking=True))
            h = model.head.classifier.features
            labels = torch.tensor([r['label'] for r in rows], device=device)
            weights = torch.tensor([ce_weights[i] for i in indices], device=device)
            example_ce = F.cross_entropy(logits.float(), labels, reduction='none')
            ce = (example_ce * weights).sum()
            maximum_example_ce = max(maximum_example_ce,float(example_ce.detach().max()))
            # Both safety losses act on already-computed vectors: one SSL forward/backward.
            targets = torch.tensor([r['retention_target'] for r in rows], device=device)
            retention, breaches = retention_loss(logits, labels, weights, targets)
            try:
                stability, scales = scale_loss(h,logits,rows,weights,cfg)
            except StabilityStop:
                optimizer.zero_grad(set_to_none=True)
                model.head.classifier.features = None
                model.head.classifier.monitor = {}
                raise
            maximum_feature_ratio = max(maximum_feature_ratio,scales['maximum_feature_ratio'])
            maximum_margin_ratio = max(maximum_margin_ratio,scales['maximum_margin_ratio'])
            maximum_adapter_ratio = max(maximum_adapter_ratio,float(model.head.classifier.monitor['adapter_ratio'].max()))
            with torch.autocast(device_type=device.type, enabled=False):
                classifier = model.head.classifier[-1]
                direction = classifier.weight.detach()[0]-classifier.weight.detach()[1]
                adv = adversary.loss(h.float(), rows, [adversary_weights[i] for i in indices], strength,
                    direction=direction, protect=cfg['gradient_protection'], audit=gradient_audit)
            loss = ce + cfg['retention_weight'] * retention + cfg['stability_weight'] * stability + adv
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite loss; previous committed checkpoint is preserved')
        # Backpropagate now; never retain an epoch or a whole source batch of SSL graphs.
        loss.backward()
        model.head.classifier.features = None
        model.head.classifier.monitor = {}
        total_ce += float(ce.detach())
        total_adv += float(adv.detach())
        total_retention += float(retention.detach())
        total_stability += float(stability.detach())
        retention_breaches += breaches
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['max_grad_norm'], error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(adversary.parameters(), cfg['max_grad_norm'], error_if_nonfinite=True)
    optimizer.step()
    audit = torch.stack(gradient_audit).sum(0).cpu().tolist() if gradient_audit else [0.,0.,0.,0.]
    return dict(classification_loss=total_ce, language_loss=total_adv, reversal_strength=strength,
                retention_loss=total_retention, retention_breaches=retention_breaches,
                stability_loss=total_stability, maximum_feature_ratio=maximum_feature_ratio,
                maximum_margin_ratio=maximum_margin_ratio, maximum_adapter_ratio=maximum_adapter_ratio,
                maximum_example_ce=maximum_example_ce,
                language_gradient_rows=int(audit[0]), conflicting_language_rows=int(audit[1]),
                language_gradient_energy=audit[2], removed_language_gradient_energy=audit[3],
                gradient_norm=float(norm), maximum_padded_frames=maximum_frames)


def print_metrics(tag, value, eligible, reasons):
    print(f'\n[Dev] V3.12 {tag} Clean={100*value["clean_f1"]:.3f} Noisy={100*value["noisy_f1"]:.3f} '
          f'Weighted={100*value["weighted_f1"]:.3f}', flush=True)
    for condition in ('online','seen','heldout'):
        for language in ('en','zh'):
            group = value['groups'][condition+'/'+language]
            print(f'  {condition}/{language} fake={100*group["recall"][0]:.3f}% '
                  f'real={100*group["recall"][1]:.3f}% AUC={100*group["auc"]:.3f}%', flush=True)
    print('  eligible=' + str(eligible) + '; reasons=' + ','.join(reasons), flush=True)


def acceptance(baseline, value, cfg):
    item = dict(name='feature_debias', status='fitted', metrics=value)
    selection(baseline,[item],cfg)
    gain = float(np.mean([value['groups'][c+'/en']['recall'][1]-baseline['groups'][c+'/en']['recall'][1]
                          for c in ('online','seen','heldout')]))
    if gain < cfg['min_en_real_gain']:
        item['guardrails'].append('en_real_gain_below_minimum')
    item['eligible'] = not item['guardrails']
    return item['eligible'], item['guardrails'], gain


def run_experiment(cfg, run):
    verify_inputs(cfg)
    run = Path(run)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cfg['device'].startswith('cuda'):
        torch.cuda.set_device(torch.device(cfg['device']))
        if cfg['amp']=='bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 training requires a supported GPU; no silent precision change')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    model = load_model(cfg)
    adversary = LanguageAdversary(model.head.classifier[-1].in_features, cfg['adversary_hidden']).to(cfg['device'])
    optimizer = optimizer_for(model, adversary, cfg)
    budget = storage_budget(model, adversary, cfg)
    atomic_json(run/'storage_budget.json', budget)
    free = shutil.disk_usage(run).free
    # On resume last.pt already occupies disk; only one further atomic generation is needed.
    required = budget['estimated_last_bytes'] if (run/'last.pt').is_file() else budget['peak_new_run_bytes']
    print(f'V312_STORAGE free_GiB={free/1024**3:.2f} required_free_GiB={required/1024**3:.2f}; new audio cache=0', flush=True)
    if free < required:
        raise OSError('Insufficient space for atomic partial resume state; no existing models or audio deleted')
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'V312_TRAINABLE encoder_last_layers={cfg["trainable_layers"]}; all MultiConv; feature adapter; parameters={trained:,}', flush=True)
    print(f'V312_OBJECTIVE gradient_protection={cfg["gradient_protection"]}; retention_weight={cfg["retention_weight"]}; '
          f'max_epochs={cfg["epochs"]}; extra_SSL_passes_per_update=0',flush=True)
    with bundles(cfg) as (train, dev):
        base_weights(cfg)  # Verify the submitted classifier and source checkpoint binding.
        referenced, retention_inventory = reference_rows(train['rows'], train['logits'], cfg, train['x'])
        training, probe_rows, _, split = probe_split(referenced, cfg)
        retention_inventory['adaptation_rows'] = len(training)
        retention_inventory['adaptation_protected'] = sum(r['retention_target'] > 0 for r in training)
        atomic_json(run/'retention_inventory.json', retention_inventory)
        atomic_json(run/'probe_split.json', split)
        baseline_logits, baseline, baseline_probe = prepare_baseline(model,dev,probe_rows,split,cfg,run,infer)
        baseline_diagnostics = read_json(run/'baseline_diagnostics.json')
        plan = SourcePlan(training, cfg['source_batch'], cfg['seed'])
        total = plan.steps * cfg['epochs']
        cursor, history, best_state, selected, best_score, stale = 0, [], None, 'baseline', baseline['weighted_f1'], 0
        if (run/'last.pt').is_file():
            saved = load_resume(run/'last.pt',cfg)
            apply_partial(model,saved['model'])
            adversary.load_state_dict(saved['adversary'],strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            cursor, history, best_state = saved['cursor'], saved['history'], saved['best_model']
            selected, best_score, stale = saved['selected'], saved['best_score'], saved['stale']
            if baseline_probe != saved['baseline_probe']:
                raise ValueError('Resume probe baseline differs from the committed original baseline')
            restore_rng(saved['rng'])
            print(f'V312_RESUME committed_updates={cursor}/{total}; selected={selected}',flush=True)
            del saved
        else:
            print_metrics('protected baseline', baseline, True, [])
        atomic_json(run/'dev_rows.json', [{k:r[k] for k in ('id','source_id','group_id','condition','language','label')}
                                         for r in dev['rows']])
        atomic_json(run/'data_reuse.json', dict(train_rows=len(training), dev_rows=len(dev['rows']),
            probe_rows=len(probe_rows), source_groups={str(k):len(v) for k,v in plan.pools.items()},
            steps_per_epoch=plan.steps, source_budget_per_epoch=plan.steps*cfg['source_batch'],
            balanced_sampling='cyclic shuffled group pools; epoch counts draws, not all distinct sources',
            new_audio_bytes=0, frozen_prefix_duplicated=False, teacher_encoder_used=False,
            retention_targets='existing verified Train logits, correct confident cases only',
            extra_ssl_passes_per_update=0, gradient_protection=cfg['gradient_protection']))
        stop = bool(history and history[-1]['stop'])
        committed_cursor, safety_stop = cursor, None
        while cursor < total and not stop:
            epoch, within = divmod(cursor,plan.steps)
            boundaries = sorted({math.ceil(plan.steps*i/cfg['checks_per_epoch']) for i in range(1,cfg['checks_per_epoch']+1)})
            next_check = next(b for b in boundaries if b>within)
            phase = Phase(f'V3.12 feature debias epoch {epoch+1} steps {within+1}-{next_check}', next_check-within)
            segment_start = within
            began = time.monotonic()
            unique_sources = set()
            segment_stats = dict(updates=0, classification_loss=0., language_loss=0., retention_loss=0., stability_loss=0.,
                retention_breaches=0, language_gradient_rows=0, conflicting_language_rows=0,
                language_gradient_energy=0., removed_language_gradient_energy=0.,
                maximum_feature_ratio=0.,maximum_margin_ratio=0.,maximum_adapter_ratio=0.,maximum_example_ce=0.)
            for batch in loader(plan,epoch,within,cfg):
                examples = [dict(r,features=torch.from_numpy(r['features']),mask=torch.from_numpy(r['mask'])) for r in batch]
                unique_sources.update(r['source_id'] for r in examples)
                strength = schedule(optimizer,cursor,total,plan.steps,cfg)
                try:
                    stats = train_step(model,adversary,optimizer,examples,cfg,strength)
                except StabilityStop as exc:
                    safety_stop = dict(exc.details,attempted_update=cursor+1,committed_updates=committed_cursor)
                    atomic_json(run/'stability_stop.json',safety_stop)
                    print('V312_STABILITY_STOP='+str(safety_stop)+'; no unsafe state committed',flush=True)
                    stop = True
                    break
                segment_stats['updates'] += 1
                for key in segment_stats:
                    if key.startswith('maximum_'):
                        segment_stats[key] = max(segment_stats[key],stats[key])
                    elif key != 'updates':
                        segment_stats[key] += stats[key]
                cursor += 1; within += 1
                phase.update(within-segment_start, loss=stats['classification_loss'])
                if within == 1 or within % 100 == 0:
                    gpu = f'; peak_GPU_GiB={torch.cuda.max_memory_allocated()/1024**3:.2f}' if cfg['device'].startswith('cuda') else ''
                    print(f'  STEP {within}/{plan.steps} CE={stats["classification_loss"]:.5f} '
                          f'language_CE={stats["language_loss"]:.5f} retention={stats["retention_loss"]:.5f} '
                          f'scale_loss={stats["stability_loss"]:.5f} norm_ratio={stats["maximum_feature_ratio"]:.2f} '
                          f'margin_ratio={stats["maximum_margin_ratio"]:.2f} '
                          f'GRL={strength:.4f} conflict_rows={stats["conflicting_language_rows"]}/{stats["language_gradient_rows"]} '
                          f'grad={stats["gradient_norm"]:.3f}'
                          f' seconds/update={(time.monotonic()-began)/max(1,within-segment_start):.2f}{gpu}',flush=True)
                if within != next_check:
                    continue
                segment_stats['training_seconds'] = time.monotonic()-began
                segment_stats['seconds_per_update'] = segment_stats['training_seconds']/segment_stats['updates']
                tag = f'epoch_{epoch+1}_step_{within}'
                announce('V3.12 fixed Dev and independent language probes: '+tag)
                diagnostics = {}
                try:
                    logits, _ = infer(model,dev['rows'],cfg,'V3.12 '+tag+' Dev',diagnostics=diagnostics)
                    _, probe_x = infer(model,probe_rows,cfg,'V3.12 held-out real language probes',capture=True)
                except (StabilityStop,FloatingPointError) as exc:
                    safety_stop = dict(stage='validation',reason=str(exc),cursor=cursor,committed_updates=committed_cursor)
                    atomic_json(run/'stability_stop.json',safety_stop)
                    print('V312_STABILITY_STOP='+str(safety_stop)+'; no unsafe state committed',flush=True)
                    stop = True
                    break
                value = measure(dev['rows'],logits,target=cfg['matched_fake_recall'])
                current_probe = run_probes(probe_x,probe_rows,split,cfg)
                evidence = compare(baseline_probe,current_probe,cfg)
                stability = compare_diagnostics(baseline_diagnostics,diagnostics,cfg)
                stability['reasons'].extend(evidence['instability_reasons'])
                stability['stable'] = not stability['reasons']
                # A probe is not evidence of useful debiasing if the representation
                # or scores have become unstable, even with improved weighted F1.
                if not stability['stable']:
                    evidence['evidence'] = False
                eligible, reasons, en_gain = acceptance(baseline,value,cfg)
                reasons.extend(stability['reasons'])
                eligible = eligible and stability['stable']
                promoted = eligible and value['weighted_f1'] > best_score
                if promoted:
                    best_state, selected, best_score = partial_state(model), tag, value['weighted_f1']
                progress_state = stopping(history,value,cursor,plan.steps,baseline,cfg)
                if not stability['stable']:
                    progress_state.update(stop=True,reason='representation_scale_exceeded')
                    safety_stop = dict(stage='validation',reason='representation_scale_exceeded',
                        reasons=stability['reasons'],cursor=cursor,committed_updates=committed_cursor)
                    atomic_json(run/'stability_stop.json',safety_stop)
                stale, stop = progress_state['checks_without_progress'], progress_state['stop']
                entry = dict(tag=tag, cursor=cursor, metrics=value, probes=current_probe, language_evidence=evidence,
                             eligible=eligible, promoted=promoted, reasons=reasons, en_real_mean_gain=en_gain,
                             changed=change_audit(dev['rows'],baseline_logits,logits), stop=stop,
                             unique_sources_in_segment=len(unique_sources), last_training_step=stats,
                             progress_control=progress_state, segment_training=segment_stats)
                entry.update(stability=stability,diagnostics=diagnostics,committed=stability['stable'])
                history.append(entry)
                atomic_json(run/'validation_pending.json',dict(committed=False,validation=entry))
                print_metrics(tag,value,eligible,reasons)
                print(f'  Selected={selected}; probe readability={baseline_probe["readability"]:.3f}->{current_probe["readability"]:.3f}; '
                      f'language evidence={evidence["evidence"]}; stop={stop}',flush=True)
                print(f'  Progress: {progress_state["reason"]}; checks_without_progress={stale}/{cfg["patience"]}; '
                      f'classification guards unchanged; max_epochs={cfg["epochs"]}',flush=True)
                np.savez_compressed(run/('dev_scores_'+tag+'.npz'),logits=logits)
                atomic_json(run/'training_history.json',history)
                if not stability['stable']:
                    print('V312_STABILITY_STOP=validation scale; previous last.pt and selected best preserved',flush=True)
                    del probe_x
                    break
                # The cursor is committed ONLY after successful validation AND state write.
                snapshot = dict(schema=SCHEMA,identity=identity(cfg),model=partial_state(model),
                    adversary=to_cpu(adversary.state_dict()),optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng(),
                    cursor=cursor,history=history,best_model=best_state,selected=selected,best_score=best_score,
                    stale=stale,baseline_probe=baseline_probe)
                announce('V3.12 committing partial weights and optimizer; frozen prefix is referenced')
                atomic_save(run/'last.pt',snapshot,cfg['disk_margin_bytes'])
                committed_cursor = cursor
                del snapshot, probe_x
                atomic_json(run/'training_history.json',history)
                (run/'validation_pending.json').unlink(missing_ok=True)
                if stop or within==plan.steps:
                    break
                next_check = next(b for b in boundaries if b>within)
                phase = Phase(f'V3.12 feature debias epoch {epoch+1} steps {within+1}-{next_check}',next_check-within)
                segment_start, began, unique_sources = within, time.monotonic(), set()
                segment_stats = {key:0 for key in segment_stats if key not in ('training_seconds','seconds_per_update')}
        verify_inputs(cfg)
        # Export one selected partial model. Adversaries and optimizer never enter deployment.
        if best_state is not None:
            apply_partial(model,best_state)
            model.eval()
            chosen = next(e for e in history if e['tag']==selected)
            evidence = chosen['language_evidence']['evidence']
        else:
            evidence = False
        checkpoint = dict(schema=SCHEMA,identity=identity(cfg),config=cfg,model=best_state,selected=selected,
                          language_probe_evidence=evidence,score='P(fake)',threshold=.5,input_policy='full utterance')
        atomic_save(run/'best.pt',checkpoint,cfg['disk_margin_bytes'])
        done = dict(version='3.12',status='complete',selected=selected,baseline_fallback=best_state is None,
                    checkpoint_sha256=digest(run/'best.pt'),base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
                    completed_updates=cursor,committed_updates=committed_cursor,planned_updates=total,early_stopped=stop,
                    stability_stop=safety_stop,discarded_uncommitted_updates=cursor-committed_cursor,
                    language_probe_evidence=evidence,external_teacher_at_inference=False)
        atomic_json(run/'completed.json',done)
        # Promoting a detector and establishing language-mechanism evidence are separate claims.
        atomic_json(run/'fit_report.json',dict(baseline=baseline,history=history,completed=done,
            note='Local proxy, not official Progress/Eval; held-out probes are not an adversary accuracy or a causal ablation'))
        print('V312_SELECTED='+selected+'; language_probe_evidence='+str(evidence),flush=True)
        return done
