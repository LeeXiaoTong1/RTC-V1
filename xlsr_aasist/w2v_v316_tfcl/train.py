"""Retrain or explicitly warm-start; trustworthy Offline references and bounded I/O."""
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
from w2v_v314.train import schedule
from w2v_v312.replay import delta
from w2v_v315.train import _scores
from w2v_v39.common import announce,atomic_json,digest,read_json
from w2v_v39.metrics import change_audit
from .config import verify_inputs
from .data import bundles,SourcePlan,TrainingLoader,tensors
from .model import load_model,optimizer_for,load_reference,stage
from .objectives import TFCL
from .step import train_step
from .performance import select_execution,checkpointing
from .metrics import measure,print_metrics
from .selection import acceptance,guarded_score,update_selections
from .diagnostics import (offline_rows,build_panel,panel_partition,infer_panel,panel_metrics,
                          paired_prediction_changes,treatment_errors)
from .state import (SCHEMA,identity,partial_state,apply_partial,to_cpu,atomic_save,
                    capture_rng,restore_rng,storage_budget,load_resume,load_selected,apply_candidate)


def replay_parent(cfg,rows,logits):
    parent=Path(cfg['parent_run']);tag=cfg['parent_selected_tag']
    name='dev_scores_baseline' if tag=='starting_last' else 'dev_scores_'+tag
    source=parent/(name+'.npz')
    if not source.is_file() or not (parent/'dev_rows.json').is_file():
        return dict(status='historical_scores_unavailable',weights_sha256=cfg['parent_checkpoint_sha256'])
    old=read_json(parent/'dev_rows.json')
    keys=('id','source_id','group_id','condition','language','label')
    if [{k:r[k] for k in keys} for r in rows]!=[{k:r[k] for k in keys} for r in old]:
        raise ValueError('V3.15 parent replay Dev row identity/order differs')
    with np.load(source,allow_pickle=False) as data:
        result=delta(data['logits'],logits)
    if not result['allclose'] or result['decision_changes']:
        raise ValueError('V3.16 starting detector does not reproduce the selected V3.15 parent')
    return dict(result,status='passed',parent_selected=tag)


def baseline(model,dev,tune,recipes,audit,audit_recipes,cfg,run):
    signature=dict(identity=identity(cfg),dev_rows=rows_signature(dev),panel_rows=rows_signature(tune))
    z=_scores(run,'dev_scores_baseline',lambda:infer(model,dev,cfg,'V3.16 parent fixed Dev')[0],signature)
    atomic_json(run/'startup_replay.json',replay_parent(cfg,dev,z))
    pz=_scores(run,'panel_scores_baseline_tune',
        lambda:infer_panel(model,tune,recipes,cfg,run,'V3.16 parent selection panel'),signature)
    # These views are never shown to the selector, scheduler or stopping logic.
    az=_scores(run,'panel_scores_baseline_audit',
        lambda:infer_panel(model,audit,audit_recipes,cfg,run,'V3.16 parent final-only audit'),signature)
    value=measure(dev,z,target=cfg['matched_fake_recall'])
    panel=panel_metrics(tune,recipes,pz,target=cfg['matched_fake_recall'])
    atomic_json(run/'baseline_metrics.json',value)
    atomic_json(run/'panel_baseline_tune.json',panel)
    atomic_json(run/'panel_baseline_audit.json',panel_metrics(audit,audit_recipes,az,target=cfg['matched_fake_recall']))
    oz=_scores(run,'offline_scores_baseline_tune',
        lambda:infer(model,tune,cfg,'V3.16 parent bounded Offline diagnosis')[0],signature)
    atomic_json(run/'treatment_baseline.json',paired_treatment(tune,oz,dev,z,recipes,pz))
    print_metrics('starting_parent:'+cfg['parent_selected_tag'],value,panel=panel)
    return z,value,panel,pz


def paired_treatment(originals,original_logits,dev,dev_logits,recipes,panel_logits):
    hashes={r['audio_sha256'] for r in originals}
    indices=[i for i,r in enumerate(dev) if r['source_sha256'] in hashes]
    official=treatment_errors(originals,original_logits,[dev[i] for i in indices],dev_logits[indices])
    processed=[dict(r,condition=p['family']) for r,p in zip(originals,recipes)]
    return dict(official=official,simulated=treatment_errors(originals,original_logits,processed,panel_logits),
        original_sources=len(originals),used_for_selection=False)


def print_treatment(value):
    for scope in ('official','simulated'):
        groups=value[scope]['groups']
        for name,cell in groups.items():
            if '/en/' not in name:continue
            correct=cell['both_correct']+cell['original_correct_processed_wrong']
            print(f'  TREATMENT {scope}/{name} new_errors={cell["original_correct_processed_wrong"]}/{correct} previously_correct; '
                f'corrected={cell["original_wrong_processed_correct"]}; both_wrong={cell["both_wrong"]}; diagnostic_only',flush=True)


def finish(cfg,run,total,safety=None):
    state=load_resume(run/'last.pt',cfg)
    if state['cursor']==0 and cfg['init_mode']=='pretrained':
        raise RuntimeError('Fresh detector has no committed validation checkpoint; initialization cannot be marked complete or exported')
    atomic_json(run/'training_history.json',state['history'])
    done=dict(version='3.16',variant='offline_reference_tfcl_v1',init_mode=cfg['init_mode'],status='complete',state_file='last.pt',checkpoint_sha256=digest(run/'last.pt'),
        base_checkpoint_sha256=cfg['base_checkpoint_sha256'],starting_checkpoint_sha256=cfg['starting_checkpoint_sha256'],
        parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],parent_selected_tag=cfg['parent_selected_tag'],
        parent_selector=cfg['parent_selector'],starting_tag=cfg['starting_tag'],
        selections=state['selections'],scores=state['scores'],last_tag=state['last_tag'],
        committed_updates=state['cursor'],planned_updates=total,early_stopped=state['cursor']<total,
        stop_reason=safety or state.get('stop_reason'),fallback_target='pinned_v315_parent',
        execution_plan_sha256=state['execution_plan_sha256'],external_teacher_at_inference=False,
        training_only_tfcl_removed_at_inference=True)
    atomic_json(run/'completed.json',done)
    for kind in ('best_weighted','best_guarded','last'):
        tag=state['last_tag'] if kind=='last' else state['selections'][kind]
        atomic_json(run/(kind+'.json'),dict(checkpoint='last.pt',selected=tag,kind=kind))
    print('V316_TFCL_SELECTED='+str(done['selections'])+'; last='+done['last_tag'],flush=True)
    return done


def audit_selected(cfg,run,kind):
    """Post-selection audit only; no selector or stopping logic reads the result."""
    checkpoint,done=load_selected(run,kind)
    signature=dict(checkpoint_sha256=done['checkpoint_sha256'],selected=done['selected'])
    path=run/('post_selection_audit.json' if kind=='best_guarded' else 'post_selection_audit_best_weighted.json')
    if path.is_file():
        result=read_json(path)
        if result['signature']!=signature:
            raise ValueError('Final-only audit belongs to another selection')
        return result
    rows=read_json(run/'panel_rows.json');recipes=read_json(run/'panel_recipes.json')
    rows,recipes=panel_partition(rows,recipes,'audit')
    baseline_path=run/'panel_scores_baseline_audit.npz'
    if digest(baseline_path)!=read_json(run/'panel_scores_baseline_audit.json')['sha256']:
        raise ValueError('Saved parent audit predictions changed')
    with np.load(baseline_path,allow_pickle=False) as data:
        before=data['logits'].copy()
    if done['selected']=='starting_parent':
        logits=before.copy()
        original_logits=infer(load_reference(cfg),rows,cfg,'V3.16 audit Offline reference')[0]
    else:
        model=load_model(cfg)
        apply_candidate(model,checkpoint['candidate'])
        logits=infer_panel(model,rows,recipes,cfg,run,'V3.16 final-only audit '+done['selected'])
        original_logits=infer(model,rows,cfg,'V3.16 audit Offline '+done['selected'])[0]
    del checkpoint
    metrics=panel_metrics(rows,recipes,logits,target=cfg['matched_fake_recall'])
    changes=paired_prediction_changes(rows,before,logits)
    np.savez_compressed(run/('panel_scores_'+kind+'_audit.npz'),logits=logits,offline_logits=original_logits)
    result=dict(signature=signature,selected=done['selected'],metrics=metrics,
        baseline=read_json(run/'panel_baseline_audit.json'),changes=changes,
        treatment=treatment_errors(rows,original_logits,[dict(r,condition=p['family']) for r,p in zip(rows,recipes)],logits),
        used_for_model_selection=False,used_for_early_stopping=False,
        limitation='These processed views are final-only; their original sources also occur in historical fixed Dev.')
    atomic_json(path,result)
    print(f'V316_TFCL_FINAL_AUDIT selected={done["selected"]} macro_family_F1={100*metrics["macro_f1"]:.3f}; selection unchanged',flush=True)
    return result


def final_audit(cfg,run):
    # Always audit the trained weighted winner, even if guarded fell back to the
    # historical detector. Otherwise a fresh retraining failure would be hidden.
    run=Path(run);weighted=audit_selected(cfg,run,'best_weighted')
    done=read_json(run/'completed.json')
    if done['selections']['best_guarded']==done['selections']['best_weighted']:
        atomic_json(run/'post_selection_audit.json',weighted)
        return weighted
    return audit_selected(cfg,run,'best_guarded')


def run_experiment(cfg,run):
    verify_inputs(cfg);run=Path(run)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if cfg['device'].startswith('cuda'):
        torch.cuda.set_device(torch.device(cfg['device']))
        if cfg['amp']=='bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unavailable')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    model=load_reference(cfg)
    initialization=cfg.get('initialization_provenance',dict(mode=cfg['init_mode']))
    atomic_json(run/'initialization.json',initialization)
    print('V316_TFCL_INITIALIZATION='+str(initialization),flush=True)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(cfg['seed']+101)
        auxiliary=TFCL(model.head.config.projection,cfg['tfcl_heads'],cfg['tfcl_bins'],mode=cfg['tfcl_mode']).to(cfg['device'])
    optimizer=optimizer_for(model,cfg,auxiliary)
    budget=storage_budget(model,cfg,auxiliary)
    free=shutil.disk_usage(run).free
    existing=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    needed=max(0,budget['required_start_free_bytes']-existing)
    atomic_json(run/'storage_budget.json',dict(budget,free_bytes=free,required_remaining_bytes=needed))
    print(f'V316_TFCL_STORAGE free_GiB={free/1024**3:.2f} required_remaining_GiB={needed/1024**3:.2f}; generated_audio_cache_GiB=0',flush=True)
    if free<needed:
        raise OSError('Insufficient space for atomic checkpoint and 10 GiB reserve; parent retained')
    with bundles(cfg) as (train,dev):
        off,note=offline_rows(cfg,dev['rows'],train['rows'])
        if not off:
            raise ValueError('Verified original Dev sources required for broader diagnostics')
        panel_rows,recipes,manifest=build_panel(off,cfg)
        tune,tune_recipes=panel_partition(panel_rows,recipes,'tune')
        audit,audit_recipes=panel_partition(panel_rows,recipes,'audit')
        atomic_json(run/'dev_rows.json',dev['rows']);atomic_json(run/'panel_rows.json',panel_rows)
        atomic_json(run/'panel_recipes.json',recipes);atomic_json(run/'panel_manifest.json',manifest)
        plan=SourcePlan(train['rows'],cfg['source_batch'],cfg['seed'],cfg)
        total=plan.steps*cfg['epochs']
        original_logits,original_metrics,original_panel,original_panel_logits=baseline(
            model,dev['rows'],tune,tune_recipes,audit,audit_recipes,cfg,run)
        del model,optimizer
        if cfg['device'].startswith('cuda'):torch.cuda.empty_cache()
        model=load_model(cfg)
        optimizer=optimizer_for(model,cfg,auxiliary)
        execution=select_execution(model,auxiliary,optimizer,plan,cfg,run)
        exec_sha=digest(run/'execution_plan.json');effective=dict(cfg,**execution['selected'])
        checkpointing(model,effective['checkpointing'])
        print('V316_TFCL_EXECUTION '+str(execution['selected'])+'; logical_sources=16 full_views<=48; persistent workers; zero generated disk cache',flush=True)
        if (run/'last.pt').exists():
            state=load_resume(run/'last.pt',cfg)
            apply_partial(model,state['model']);auxiliary.load_state_dict(state['auxiliary'],strict=True)
            optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
            print('V316_TFCL_RESUME committed_updates='+str(state['cursor']),flush=True)
        else:
            state=dict(schema=SCHEMA,identity=identity(cfg),model=partial_state(model),
                auxiliary=to_cpu(auxiliary.state_dict()),optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng(),
                cursor=0,history=[],candidates={},selections=dict(best_weighted='initialization' if cfg['init_mode']=='pretrained' else 'starting_parent',best_guarded='starting_parent'),
                scores=dict(best_weighted=-1. if cfg['init_mode']=='pretrained' else original_metrics['weighted_f1'],best_guarded=guarded_score(original_metrics,original_panel)),
                last_tag='initialization' if cfg['init_mode']=='pretrained' else 'starting_parent',stale=0,progress_best=-1. if cfg['init_mode']=='pretrained' else original_metrics['weighted_f1'],
                progress_guarded=guarded_score(original_metrics,original_panel),
                stop=False,stop_reason=None,execution_plan_sha256=exec_sha)
            atomic_save(run/'last.pt',state,cfg['disk_margin_bytes']+cfg['free_reserve_bytes'])
        cursor,stale,progress_best=state['cursor'],state['stale'],state['progress_best']
        progress_guarded=state['progress_guarded']
        history,candidates,selections,scores=state['history'],state['candidates'],state['selections'],state['scores']
        stop=state['stop'];del state
        atomic_json(run/'training_history.json',history)
        atomic_json(run/'training_design.json',dict(parent_selected=cfg['parent_selected_tag'],
            parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],
            optimizer_groups=[dict(name=g['name'],lr=g['initial_lr'],parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups],
            logical_source_batch=cfg['source_batch'],views_per_source='3; 2 if official Online absent',init_mode=cfg['init_mode'],tfcl_mode=cfg['tfcl_mode'],
            main_ce_mass=dict(online=.5,offline=.1,noisy=.4),group_ce_mass=.25,
            auxiliary_edges=['offline_to_online','offline_to_noisy'],
            shared_forward=True,full_wave_classification=True,auxiliary_only_reliable_matched_frames=True,
            structure_window_frames=cfg['structure_window_frames'],learned_projection=cfg['tfcl_mode']=='original',
            max_epochs=cfg['epochs'],execution_plan=execution['selected'],
            final_only_processed_panel_not_used_for_selection=True))
        stream=TrainingLoader(plan,cfg,run)
        try:
            while cursor<total and not stop:
                epoch,within=divmod(cursor,plan.steps)
                coverage=plan.coverage(epoch)
                atomic_json(run/f'sampling_epoch_{epoch+1}.json',coverage)
                print('V316_TFCL_SOURCE_COVERAGE '+str({k:v for k,v in coverage.items()
                    if k in ('available_sources','missing_online_sources','excluded_source_groups','steps','unique_sources')}),flush=True)
                boundaries=sorted({math.ceil(plan.steps*i/cfg['checks_per_epoch']) for i in range(1,cfg['checks_per_epoch']+1)})
                boundary=next(b for b in boundaries if b>within)
                announce(f'V3.16 epoch {epoch+1}/{cfg["epochs"]}, steps {within+1}-{boundary}; reliable Offline reference TFCL')
                phase=Phase(f'V3.16 epoch {epoch+1}',boundary-within)
                if cfg['device'].startswith('cuda'):torch.cuda.reset_peak_memory_stats()
                start=within;began=time.monotonic();ready=began
                sums=dict(ce=0.,time=0.,structure=0.,data_wait=0.,compute=0.,total=0.);peak_ce=0.
                for batch in stream.segment(epoch,within,boundary):
                    now=time.monotonic();sums['data_wait']+=now-ready
                    examples=tensors(batch);schedule(optimizer,cursor,total,cfg)
                    warm,structure_strength=stage(model,cfg,cursor,plan.steps)
                    effective['structure_strength']=structure_strength
                    if cfg['device'].startswith('cuda'):torch.cuda.synchronize()
                    started=time.monotonic()
                    try:
                        stats=train_step(model,auxiliary,optimizer,examples,effective,warm,audit=(within==start))
                    except FloatingPointError as exc:
                        safety=dict(stage='training',reason=str(exc),attempted_update=cursor+1,
                            note='Uncommitted segment discarded; previous validated state retained')
                        atomic_json(run/'safety_stop.json',safety)
                        return finish(cfg,run,total,safety)
                    if cfg['device'].startswith('cuda'):torch.cuda.synchronize()
                    sums['compute']+=time.monotonic()-started;cursor+=1;within+=1
                    sums['ce']+=stats['classification_loss'];sums['time']+=stats['time_loss']
                    sums['structure']+=stats['structure_loss'];sums['total']+=stats['total_loss']
                    peak_ce=max(peak_ce,stats['maximum_example_ce']);phase.update(within-start,loss=stats['total_loss'])
                    if stats.get('aux_gradient_audit'):
                        atomic_json(run/f'aux_gradient_{cursor}.json',stats['aux_gradient_audit'])
                    if within==start+1 or within%100==0 or within==boundary:
                        peak=torch.cuda.max_memory_allocated()/1024**3 if cfg['device'].startswith('cuda') else 0.
                        reserved=torch.cuda.max_memory_reserved()/1024**3 if cfg['device'].startswith('cuda') else 0.
                        count=within-start
                        print(f'  STEP {within}/{plan.steps} TOTAL={stats["total_loss"]:.6f} CE={stats["classification_loss"]:.6f} '
                            f'time={stats["time_loss"]:.5f} weighted_time={stats["weighted_time_loss"]:.6f} '
                            f'CKA={stats["structure_loss"]:.5f} weighted_CKA={stats["weighted_structure_loss"]:.6f} '
                            f'offline_to_online={stats["bridge_pairs"]}/{cfg["source_batch"]} offline_to_noisy={stats["matched_pairs"]}/{cfg["source_batch"]} '
                            f'max_CE={stats["maximum_example_ce"]:.3f} compute_s/update={sums["compute"]/count:.2f} '
                            f'wait_s/update={sums["data_wait"]/count:.2f} peak_alloc/reserved_GiB={peak:.2f}/{reserved:.2f}',flush=True)
                                        
                    if within==start+1 or within%100==0 or within==boundary:
                        print('  REFERENCE_GROUPS='+str(stats['reference_groups'])+' ALIGN_COVERAGE='+str(stats['alignment_coverage_sum']/max(1,stats['alignment_attempts']))+' SALIENCY_S='+str(stats['saliency_seconds'])+' ALIGN_LOSS_S='+str(stats['alignment_seconds'])+' FEATURE_GRAD='+str(stats['aux_gradient_audit']),flush=True)
                    ready=time.monotonic()
                if within!=boundary:
                    raise RuntimeError('Train loader did not reach the committed validation boundary')
                tag=f'epoch_{epoch+1}_step_{within}';elapsed=time.monotonic()-began
                logits,_=infer(model,dev['rows'],effective,'V3.16 '+tag+' fixed Dev')
                value=measure(dev['rows'],logits,target=cfg['matched_fake_recall'])
                pz=infer_panel(model,tune,tune_recipes,effective,run,'V3.16 '+tag+' selection panel')
                panel=panel_metrics(tune,tune_recipes,pz,target=cfg['matched_fake_recall'])
                offline_logits,_=infer(model,tune,effective,'V3.16 '+tag+' bounded Offline diagnosis')
                treatment=paired_treatment(tune,offline_logits,dev['rows'],logits,tune_recipes,pz)
                np.savez_compressed(run/('offline_scores_'+tag+'.npz'),logits=offline_logits)
                eligible,reasons=acceptance(original_metrics,value,cfg,original_panel,panel)
                current=partial_state(model)
                if any(not bool(torch.isfinite(v).all()) for v in current.values()):
                    raise FloatingPointError('Nonfinite detector; previous checkpoint retained')
                candidates,promoted=update_selections(selections,scores,candidates,tag,value,eligible,
                    dict(kind='partial',state=current),panel)
                broader=guarded_score(value,panel)
                progress=(value['weighted_f1']>=progress_best+cfg['progress_min_delta'] or
                          eligible and broader>=progress_guarded+cfg['progress_min_delta'])
                stale=0 if progress else stale+1
                progress_best=max(progress_best,value['weighted_f1'])
                if eligible:progress_guarded=max(progress_guarded,broader)
                catastrophic=(not cfg.get('fixed_budget',False) and cfg['init_mode']=='parent'
                    and value['weighted_f1']<original_metrics['weighted_f1']-cfg['catastrophic_weighted_drop'])
                stop=not cfg.get('fixed_budget',False) and (catastrophic or (cursor>=cfg['minimum_epochs']*plan.steps and stale>=cfg['patience']))
                reason='catastrophic_regression' if catastrophic else ('no_fixed_or_guarded_progress' if stop else None)
                count=within-start
                entry=dict(tag=tag,cursor=cursor,phase='offline_reference_tfcl',metrics=value,panel=panel,treatment=treatment,
                    eligible=eligible,reasons=reasons,promoted=promoted,committed=True,
                    changed=change_audit(dev['rows'],original_logits,logits),
                    panel_changes=paired_prediction_changes(tune,original_panel_logits,pz),
                    checks_without_progress=stale,stop=stop,stop_reason=reason,last_training_step=stats,
                    segment_training=dict(updates=count,mean_ce=sums['ce']/count,mean_time=sums['time']/count,
                        mean_structure=sums['structure']/count,mean_total=sums['total']/count,
                        maximum_example_ce=peak_ce,elapsed_seconds=elapsed,
                        compute_seconds=sums['compute'],data_wait_seconds=sums['data_wait']))
                history.append(entry)
                np.savez_compressed(run/('dev_scores_'+tag+'.npz'),logits=logits)
                np.savez_compressed(run/('panel_scores_'+tag+'.npz'),logits=pz)
                atomic_json(run/'validation_pending.json',entry)
                print_metrics(tag,value,eligible,reasons,panel)
                print_treatment(treatment)
                print(f'  best_weighted={selections["best_weighted"]}; best_guarded={selections["best_guarded"]}; no_progress={stale}/{cfg["patience"]}; stop={stop}',flush=True)
                snapshot=dict(schema=SCHEMA,identity=identity(cfg),model=current,auxiliary=to_cpu(auxiliary.state_dict()),
                    optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng(),cursor=cursor,history=history,
                    candidates=candidates,selections=selections,scores=scores,last_tag=tag,stale=stale,
                    progress_best=progress_best,progress_guarded=progress_guarded,stop=stop,stop_reason=reason,
                    execution_plan_sha256=exec_sha)
                atomic_save(run/'last.pt',snapshot,cfg['disk_margin_bytes']+cfg['free_reserve_bytes'])
                del snapshot,current
                atomic_json(run/'training_history.json',history);(run/'validation_pending.json').unlink(missing_ok=True)
                from .workflow import report
                report(run)
        finally:
            stream.close()
        verify_inputs(cfg)
        return finish(cfg,run,total)
