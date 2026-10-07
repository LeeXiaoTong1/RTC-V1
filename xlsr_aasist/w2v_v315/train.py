"""Matched full-wave RTC/noisy adaptation with fixed Dev and exact resume."""
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from live_progress import Phase
from w2v_v313.train import infer
from w2v_v313.replay import rows_signature
from w2v_v314.train import schedule, update_selections
from w2v_v39.common import announce, atomic_json, digest, read_json, verify_files
from w2v_v39.metrics import measure, change_audit, GROUPS
from .config import verify_inputs
from .data import bundles, SourcePlan, loader, tensors
from .model import load_model, optimizer_for
from .objectives import TFCL
from .step import train_step
from .performance import select_execution, checkpointing
from .diagnostics import (offline_rows, offline_metrics, source_replay, panel_plan,
    panel_metrics, infer_panel, treatment_errors)
from .state import (SCHEMA, identity, partial_state, apply_partial, to_cpu, atomic_save,
    capture_rng, restore_rng, storage_budget, load_resume)


def acceptance(baseline, metrics, cfg, base_panel=None, panel=None):
    reasons=[]
    if not metrics.get('complete'):
        return False,['incomplete_fixed_dev']
    if metrics['weighted_f1'] < baseline['weighted_f1']+cfg['min_gain']:
        reasons.append('weighted_gain_below_minimum')
    if metrics['noisy_f1'] < baseline['noisy_f1']-cfg['max_noisy_drop']:
        reasons.append('noisy_below_guard')
    if metrics['clean_f1'] < baseline['clean_f1']-cfg['max_clean_drop']:
        reasons.append('clean_below_guard')
    for name in GROUPS:
        a,b=baseline['groups'][name],metrics['groups'][name]
        for label in (0,1):
            if b['recall'][label] < a['recall'][label]-cfg['max_fake_drop' if label==0 else 'max_real_drop']:
                reasons.append(name+('_fake_recall_drop' if label==0 else '_real_recall_drop'))
        if b['auc'] < a['auc']-cfg['max_auc_drop']:
            reasons.append(name+'_auc_drop')
        if metrics['matched'][name]['real_recall'] < baseline['matched'][name]['real_recall']-cfg['max_matched_real_drop']:
            reasons.append(name+'_matched_recall_drop')
    if base_panel is not None:
        if panel is None:
            reasons.append('robustness_panel_not_measured')
        elif panel['f1'] < base_panel['f1']-cfg['max_panel_drop']:
            reasons.append('robustness_panel_regression')
    return not reasons,reasons


def print_metrics(tag, metrics, eligible, reasons, panel=None):
    print(f'\n[Dev] V3.15 {tag} Clean={100*metrics["clean_f1"]:.3f} '
          f'Noisy={100*metrics["noisy_f1"]:.3f} Weighted={100*metrics["weighted_f1"]:.3f}',flush=True)
    for name in GROUPS:
        v=metrics['groups'][name]
        print(f'  {name} fake={100*v["recall"][0]:.3f}% real={100*v["recall"][1]:.3f}% '
              f'AUC={100*v["auc"]:.3f}% real@99%fake={100*metrics["matched"][name]["real_recall"]:.3f}%',flush=True)
    if panel:
        print(f'  Additional fixed robustness panel F1={100*panel["f1"]:.3f}; excluded from Weighted',flush=True)
    print('  guarded_eligible='+str(eligible)+'; reasons='+','.join(reasons),flush=True)


def _scores(run, name, compute, signature):
    path,marker=run/(name+'.npz'),run/(name+'.json')
    if marker.exists():
        saved=read_json(marker)
        if saved['signature']!=signature or saved['sha256']!=digest(path):
            raise ValueError('Committed scores changed: '+name)
        with np.load(path,allow_pickle=False) as z:return z['logits'].copy()
    value=compute()
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,logits=value)
    tmp.replace(path)
    atomic_json(marker,dict(signature=signature,sha256=digest(path)))
    return value


def _processed_audit(off, off_logits, dev_rows, dev_logits, panel_logits):
    indices=[i for i,r in enumerate(dev_rows) if r['condition'] in ('seen','heldout')]
    rows=[dev_rows[i] for i in indices]+[dict(r,condition='new_panel') for r in off]
    logits=np.concatenate((dev_logits[indices],panel_logits))
    return treatment_errors(off,off_logits,rows,logits)


def baseline(model, dev, off, recipes, cfg, run):
    signature=dict(identity=identity(cfg),dev_rows=rows_signature(dev),offline=rows_signature(off))
    announce('V3.15 replaying the submitted V3.12 LAST on unchanged fixed Dev')
    z=_scores(run,'dev_scores_baseline',lambda:infer(model,dev,cfg,'V3.15 baseline fixed Dev')[0],signature)
    atomic_json(run/'startup_replay.json',source_replay(cfg,dev,z))
    oz=_scores(run,'dev_scores_offline_baseline',lambda:infer(model,off,cfg,'V3.15 baseline Offline')[0],signature)
    pz=_scores(run,'dev_scores_panel_baseline',lambda:infer_panel(model,off,recipes,cfg,run,'V3.15 baseline new RTC panel'),signature)
    value=measure(dev,z,target=cfg['matched_fake_recall'])
    panel=panel_metrics(off,recipes,pz)
    atomic_json(run/'baseline_metrics.json',value)
    atomic_json(run/'offline_baseline.json',offline_metrics(off,oz))
    atomic_json(run/'panel_baseline.json',panel)
    atomic_json(run/'treatment_baseline.json',_processed_audit(off,oz,dev,z,pz))
    print_metrics('starting_last',value,True,[],panel)
    return z,value,panel


def finish(cfg, run, total, safety=None):
    state=load_resume(run/'last.pt',cfg)
    atomic_json(run/'training_history.json',state['history'])
    done=dict(version='3.15',status='complete',state_file='last.pt',checkpoint_sha256=digest(run/'last.pt'),
        base_checkpoint_sha256=cfg['base_checkpoint_sha256'],starting_checkpoint_sha256=cfg['starting_checkpoint_sha256'],
        starting_tag=cfg['starting_tag'],selections=state['selections'],scores=state['scores'],last_tag=state['last_tag'],
        committed_updates=state['cursor'],planned_updates=total,early_stopped=state['cursor']<total,
        stop_reason=safety or state.get('stop_reason'),fallback_target='trained_v312_last',
        execution_plan_sha256=state['execution_plan_sha256'],external_teacher_at_inference=False,
        training_only_tfcl_removed_at_inference=True)
    atomic_json(run/'completed.json',done)
    for kind in ('best_weighted','best_guarded','last'):
        atomic_json(run/(kind+'.json'),dict(selector=kind,checkpoint='last.pt',
            tag=state['last_tag'] if kind=='last' else state['selections'][kind]))
    print('V315_SELECTED='+str(state['selections'])+'; last='+state['last_tag'],flush=True)
    return done


def run_experiment(cfg, run):
    verify_inputs(cfg)
    run=Path(run)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if cfg['device'].startswith('cuda'):
        torch.cuda.set_device(torch.device(cfg['device']))
        if cfg['amp']=='bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unavailable; no silent precision change')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    model=load_model(cfg)
    auxiliary=TFCL(model.head.config.projection,cfg['tfcl_heads'],cfg['tfcl_bins']).to(cfg['device'])
    optimizer=optimizer_for(model,cfg,auxiliary)
    budget=storage_budget(model,cfg,auxiliary)
    free=shutil.disk_usage(run).free
    existing=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    needed=max(0,budget['required_start_free_bytes']-existing)
    atomic_json(run/'storage_budget.json',dict(budget,free_bytes=free,required_remaining_bytes=needed))
    print(f'V315_STORAGE free_GiB={free/1024**3:.2f} required_remaining_GiB={needed/1024**3:.2f} '
          f'new_peak_GiB={budget["maximum_new_peak_bytes"]/1024**3:.2f} rolling_cache_cap_GiB={cfg["rolling_cache_bytes"]/1024**3:.2f}',flush=True)
    if free<needed:
        raise OSError('Insufficient disk for checkpoint atomic save, bounded cache and 10 GiB reserve; source preserved')
    with bundles(cfg) as (train,dev):
        off,note=offline_rows(cfg,dev['rows'],train['rows'])
        if not off:
            raise ValueError('V3.15 requires verified original Dev sources for treatment diagnostics')
        recipes=panel_plan(off,cfg)
        atomic_json(run/'dev_rows.json',dev['rows']); atomic_json(run/'offline_rows.json',off)
        atomic_json(run/'panel_recipes.json',recipes)
        atomic_json(run/'offline_policy.json',dict(count=len(off),note=note))
        plan=SourcePlan(train['rows'],cfg['source_batch'],cfg['seed'])
        total=plan.steps*cfg['epochs']
        original_logits,original_metrics,original_panel=baseline(model,dev['rows'],off,recipes,cfg,run)
        execution=select_execution(model,auxiliary,optimizer,plan,cfg,run)
        exec_sha=digest(run/'execution_plan.json')
        effective=dict(cfg,**execution['selected'])
        checkpointing(model,effective['checkpointing'])
        print('V315_EXECUTION '+str(execution['selected'])+'; logical_sources=16 full_views=48',flush=True)
        if (run/'last.pt').exists():
            state=load_resume(run/'last.pt',cfg)
            if state['execution_plan_sha256']!=exec_sha:
                raise ValueError('Resume execution plan changed')
            apply_partial(model,state['model']); auxiliary.load_state_dict(state['auxiliary'],strict=True)
            optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng'])
            print('V315_RESUME committed_updates='+str(state['cursor']),flush=True)
        else:
            state=dict(schema=SCHEMA,identity=identity(cfg),model=partial_state(model),
                auxiliary=to_cpu(auxiliary.state_dict()),optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng(),
                cursor=0,history=[],candidates={},selections=dict(best_weighted='starting_last',best_guarded='starting_last'),
                scores={k:original_metrics['weighted_f1'] for k in ('best_weighted','best_guarded')},
                last_tag='starting_last',stale=0,progress_best=original_metrics['weighted_f1'],
                stop=False,stop_reason=None,execution_plan_sha256=exec_sha)
            atomic_save(run/'last.pt',state,cfg['disk_margin_bytes']+cfg['free_reserve_bytes'])
        cursor,stale,progress_best=state['cursor'],state['stale'],state['progress_best']
        history,candidates,selections,scores=state['history'],state['candidates'],state['selections'],state['scores']
        stop=state['stop']; del state
        atomic_json(run/'training_history.json',history)
        atomic_json(run/'training_design.json',dict(starting_checkpoint_sha256=cfg['starting_checkpoint_sha256'],
            trainable_parameters={n:p.numel() for n,p in model.named_parameters() if p.requires_grad},
            optimizer_groups=[dict(name=g['name'],lr=g['initial_lr'],parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups],
            logical_source_batch=cfg['source_batch'],views_per_source=3,main_ce_mass=[.3,.1,.6],group_ce_mass=.25,
            tfcl_time_weight=cfg['tfcl_time_weight'],tfcl_structure_weight=cfg['tfcl_structure_weight'],
            cka_scope='per source, not pooled across batch',detach_reference=False,
            full_wave_classification=True,temporal_all_valid_frames=True,structure_bins=cfg['tfcl_bins'],
            max_epochs=cfg['epochs'],minimum_epochs=cfg['minimum_epochs'],execution_plan=execution['selected']))
        while cursor<total and not stop:
            epoch,within=divmod(cursor,plan.steps)
            atomic_json(run/f'sampling_epoch_{epoch+1}.json',plan.coverage(epoch))
            boundaries=sorted({math.ceil(plan.steps*i/cfg['checks_per_epoch']) for i in range(1,cfg['checks_per_epoch']+1)})
            boundary=next(b for b in boundaries if b>within)
            announce(f'V3.15 epoch {epoch+1}/{cfg["epochs"]}, steps {within+1}-{boundary}; matched RTC + TFCL')
            phase=Phase(f'V3.15 epoch {epoch+1}',boundary-within)
            if cfg['device'].startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            start=within; began=time.monotonic(); ready=began
            sums=dict(ce=0.,time=0.,structure=0.,data_wait=0.,compute=0.); peak_ce=0.
            for batch in loader(plan,epoch,within,cfg,run,stop=boundary):
                now=time.monotonic(); sums['data_wait']+=now-ready
                examples=tensors(batch)
                schedule(optimizer,cursor,total,cfg)
                warm=min(1.,(cursor+1)/max(1,plan.steps*cfg['objective_ramp_epochs']))
                if cfg['device'].startswith('cuda'):torch.cuda.synchronize()
                started=time.monotonic()
                try:
                    stats=train_step(model,auxiliary,optimizer,examples,effective,warm,audit=(within==start))
                except FloatingPointError as exc:
                    safety=dict(stage='training',reason=str(exc),attempted_update=cursor+1,
                        note='Uncommitted segment discarded; previous validation state remains exportable')
                    atomic_json(run/'safety_stop.json',safety)
                    return finish(cfg,run,total,safety)
                if cfg['device'].startswith('cuda'):torch.cuda.synchronize()
                sums['compute']+=time.monotonic()-started
                cursor+=1; within+=1
                sums['ce']+=stats['classification_loss']; sums['time']+=stats['time_loss']; sums['structure']+=stats['structure_loss']
                peak_ce=max(peak_ce,stats['maximum_example_ce'])
                phase.update(within-start,loss=stats['classification_loss'])
                if 'aux_gradient_audit' in stats:
                    atomic_json(run/f'aux_gradient_{cursor}.json',stats['aux_gradient_audit'])
                    print('  TFCL_GRADIENT_REACH='+str(stats['aux_gradient_audit']),flush=True)
                if within==start+1 or within%100==0 or within==boundary:
                    peak=torch.cuda.max_memory_allocated()/1024**3 if cfg['device'].startswith('cuda') else 0.
                    count=within-start
                    print(f'  STEP {within}/{plan.steps} CE={stats["classification_loss"]:.5f} '
                        f'time={stats["time_loss"]:.5f} structure={stats["structure_loss"]:.5f} ramp={warm:.3f} '
                        f'pair_valid={stats["eligible_pairs"]}/{cfg["source_batch"]} max_CE={stats["maximum_example_ce"]:.3f} '
                        f'compute_s/update={sums["compute"]/count:.2f} wait_s/update={sums["data_wait"]/count:.2f} peak_GPU_GiB={peak:.2f}',flush=True)
                ready=time.monotonic()
            if within!=boundary:
                raise RuntimeError('Train loader did not complete the planned validation segment')
            elapsed=time.monotonic()-began; tag=f'epoch_{epoch+1}_step_{within}'
            announce('V3.15 unchanged fixed Online/Noisy Dev: '+tag)
            logits,_=infer(model,dev['rows'],cfg,'V3.15 '+tag+' fixed Dev')
            value=measure(dev['rows'],logits,target=cfg['matched_fake_recall'])
            eligible,reasons=acceptance(original_metrics,value,cfg)
            panel=offline=errors=None
            # Whole epochs always diagnose source failures. A promising half-epoch
            # also completes the panel before becoming best_guarded.
            if within==plan.steps or (eligible and value['weighted_f1']>scores['best_guarded']):
                oz,_=infer(model,off,cfg,'V3.15 '+tag+' Offline')
                pz=infer_panel(model,off,recipes,cfg,run,'V3.15 '+tag+' new RTC panel')
                offline=offline_metrics(off,oz); panel=panel_metrics(off,recipes,pz)
                errors=_processed_audit(off,oz,dev['rows'],logits,pz)
                np.savez_compressed(run/('dev_scores_offline_'+tag+'.npz'),logits=oz)
                np.savez_compressed(run/('dev_scores_panel_'+tag+'.npz'),logits=pz)
                atomic_json(run/('treatment_'+tag+'.json'),errors)
            eligible,reasons=acceptance(original_metrics,value,cfg,original_panel,panel)
            current=partial_state(model)
            if any(not bool(torch.isfinite(v).all()) for v in current.values()):
                raise FloatingPointError('Nonfinite detector parameters; prior checkpoint remains intact')
            candidates,promoted=update_selections(selections,scores,candidates,tag,value,eligible,dict(kind='partial',state=current))
            if value['weighted_f1']>=progress_best+cfg['progress_min_delta']:
                stale,progress_best=0,value['weighted_f1']
            else:stale+=1
            catastrophic=value['weighted_f1']<original_metrics['weighted_f1']-cfg['catastrophic_weighted_drop']
            enough=cursor>=cfg['minimum_epochs']*plan.steps
            stop=catastrophic or (enough and stale>=cfg['patience'])
            reason='catastrophic_regression' if catastrophic else ('no_fixed_dev_progress' if stop else None)
            count=within-start
            entry=dict(tag=tag,phase='matched_rtc_tfcl',cursor=cursor,metrics=value,panel=panel,offline=offline,
                eligible=eligible,reasons=reasons,promoted=promoted,committed=True,
                changed=change_audit(dev['rows'],original_logits,logits),checks_without_progress=stale,stop=stop,stop_reason=reason,
                last_training_step=stats,segment_training=dict(updates=count,mean_ce=sums['ce']/count,
                    mean_time=sums['time']/count,mean_structure=sums['structure']/count,maximum_example_ce=peak_ce,
                    elapsed_seconds=elapsed,compute_seconds=sums['compute'],data_wait_seconds=sums['data_wait']))
            history.append(entry)
            np.savez_compressed(run/('dev_scores_'+tag+'.npz'),logits=logits)
            atomic_json(run/'validation_pending.json',entry)
            print_metrics(tag,value,eligible,reasons,panel)
            print(f'  best_weighted={selections["best_weighted"]}; best_guarded={selections["best_guarded"]}; '
                  f'no_progress={stale}/{cfg["patience"]}; stop={stop}',flush=True)
            snapshot=dict(schema=SCHEMA,identity=identity(cfg),model=current,auxiliary=to_cpu(auxiliary.state_dict()),
                optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng(),cursor=cursor,history=history,candidates=candidates,
                selections=selections,scores=scores,last_tag=tag,stale=stale,progress_best=progress_best,stop=stop,
                stop_reason=reason,execution_plan_sha256=exec_sha)
            announce('V3.15 saving current, optimizer and at most two distinct selected detectors')
            atomic_save(run/'last.pt',snapshot,cfg['disk_margin_bytes']+cfg['free_reserve_bytes']); del snapshot,current
            atomic_json(run/'training_history.json',history)
            (run/'validation_pending.json').unlink(missing_ok=True)
            from .workflow import report
            report(run)
        verify_inputs(cfg)
        return finish(cfg,run,total)
