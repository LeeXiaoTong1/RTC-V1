"""Four additional complete epochs, exact continuation state, concise Dev reports."""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import json
import shutil
import time
import numpy as np
import torch
from live_progress import Phase, publish
from w2v_v39.common import atomic_json, digest, read_json
from w2v_v313.replay import rows_signature
from w2v_v314.train import schedule
from w2v_v316_tfcl.data import bundles, SourcePlan, TrainingLoader, tensors
from w2v_v316_tfcl.metrics import measure
from .config import verify_inputs
from .model import load_model, optimizer_for, restore_detector_optimizer
from .objectives import TFCL
from .step import train_step
from .performance import select_execution, checkpointing
from .output import infer, print_metrics
from .cursor import source_cursor, segments, completed_tag
from .state import (SCHEMA, identity, partial_state, apply_partial, to_cpu, atomic_save,
                    capture_rng, restore_rng, source_state, storage_budget, promote,
                    load_resume, load_selected)


def save(run, cfg, state, model, auxiliary, optimizer):
    state.update(model=partial_state(model), auxiliary=to_cpu(auxiliary.state_dict()),
                 optimizer=to_cpu(optimizer.state_dict()), rng=capture_rng())
    # Reuse the same tensors when current is best; torch.save then writes them once.
    if state['best_tag']==state['last_tag']: state['best_model']=state['model']
    atomic_save(run/'last.pt', state, cfg['disk_margin_bytes']+cfg['free_reserve_bytes'])
    atomic_json(run/'training_history.json', state['history'])


def finish(run, cfg, state, total):
    done = dict(version='3.16.1', status='complete', checkpoint_sha256=digest(run/'last.pt'),
        best_tag=state['best_tag'], best_origin=state['best_origin'], last_tag=state['last_tag'],
        committed_updates=state['cursor'], planned_additional_updates=total,
        additional_epochs=cfg['epochs'], source_committed_updates=cfg['source_committed_updates'],
        source_run=cfg['source_run'], source_last_tag=cfg['source_last_tag'],
        source_checkpoint_sha256=cfg['source_checkpoint_sha256'], baseline_fallback=False,
        selection='maximum complete fixed Dev Weighted; no legacy fallback')
    atomic_json(run/'completed.json', done)
    for kind in ('best','best_weighted','last'):
        atomic_json(run/(kind+'.json'), dict(checkpoint='last.pt', kind=kind,
            selected=state['last_tag'] if kind=='last' else state['best_tag']))
    print(f'[Complete] best={state["best_tag"]} Weighted={100*state["best_metrics"]["weighted_f1"]:.3f}; '
          f'last={state["last_tag"]}; additional_epochs={cfg["epochs"]}', flush=True)
    return done


def run_experiment(cfg, run):
    run = Path(run); verify_inputs(cfg)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if cfg['device'].startswith('cuda'):
        torch.cuda.set_device(torch.device(cfg['device']))
        if cfg['amp']=='bf16' and not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 unavailable')
    model=load_model(cfg)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(cfg['seed']+3161)
        auxiliary=TFCL(model.backbone.config.hidden_size,cfg['tfcl_heads'],cfg['tfcl_bins']).to(cfg['device'])
    optimizer=optimizer_for(model,cfg,auxiliary)
    budget=storage_budget(model,auxiliary,cfg)
    free=shutil.disk_usage(run).free
    existing=(run/'last.pt').stat().st_size if (run/'last.pt').exists() else 0
    needed=max(0,budget['required_start_free_bytes']-existing)
    atomic_json(run/'storage_budget.json',dict(budget,free_bytes=free,required_remaining_bytes=needed))
    if free<needed: raise OSError(f'Need {needed/1024**3:.2f} GiB free including atomic save and reserve')
    print(f'[Storage] free={free/1024**3:.2f} GiB; required={needed/1024**3:.2f} GiB; new_audio_cache=0',flush=True)
    with bundles(cfg) as (train,dev):
        # The metric, rows and threshold must stay identical to source V3.16.
        old_rows=read_json(Path(cfg['source_run'])/'dev_rows.json')
        if rows_signature(dev['rows'])!=rows_signature(old_rows): raise ValueError('Fixed Dev changed')
        atomic_json(run/'dev_rows.json',dev['rows'])
        plan=SourcePlan(train['rows'],cfg['source_batch'],cfg['seed'],cfg)
        start_cursor=source_cursor(cfg,plan.steps)
        total=plan.steps*cfg['epochs']
        if (run/'last.pt').exists():
            state=load_resume(run,cfg)
            apply_partial(model,state['model']); auxiliary.load_state_dict(state['auxiliary'],strict=True)
            optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng'])
        else:
            source,best=source_state(cfg)
            apply_partial(model,source['model'])
            provenance=restore_detector_optimizer(optimizer,source['optimizer'])
            restore_rng(source['rng'])
            history_by_tag={e['tag']:e for e in source['history']}
            state=dict(schema=SCHEMA,identity=identity(cfg),cursor=0,history=[],
                last_tag='source:'+source['last_tag'],best_tag=best['tag'],best_model=best['model'],
                best_metrics=best['metrics'],best_origin=best['origin'])
            atomic_json(run/'initialization.json',dict(provenance,source_last=source['last_tag'],
                source_run=cfg['source_run'],source_checkpoint_sha256=cfg['source_checkpoint_sha256'],
                fresh_detector=False,detector_weights='exact source LAST',tfcl_branch='new full SSL attention/projection',
                maximum_additional_epochs=cfg['epochs'],source_sampling_epoch_offset=cfg['sampling_epoch_offset']))
            print('[Start] V3.16 LAST='+source['last_tag']+f'; continue {cfg["epochs"]} full epochs; Adam restored',flush=True)
            print_metrics('starting V3.16 LAST (committed source metrics)',history_by_tag[source['last_tag']]['metrics'])
            del source,best
        publish('Profiling full SSL TFCL; changes rolled back',force=True)
        with (run/'details.log').open('a',encoding='utf-8',buffering=1) as stream:
            with redirect_stdout(stream),redirect_stderr(stream):
                execution=select_execution(model,auxiliary,optimizer,plan,cfg,run)
        state['execution_plan_sha256']=digest(run/'execution_plan.json')
        effective=dict(cfg,**execution['selected']); checkpointing(model,effective['checkpointing'])
        print('[Execution] '+str(execution['selected'])+'; full SSL TFCL; Dev once per epoch',flush=True)
        if not (run/'last.pt').exists(): save(run,cfg,state,model,auxiliary,optimizer)
        loader=TrainingLoader(plan,effective,run)
        try:
            while state['cursor']<total:
                epoch,within=divmod(state['cursor'],plan.steps)
                if within: raise ValueError('Continuation commits complete epochs only')
                portions=list(segments(start_cursor+state['cursor'],plan.steps,plan.steps))
                tag=completed_tag(start_cursor+state['cursor']+plan.steps,plan.steps)
                sampling=[]
                for source_epoch,first,stop in portions:
                    sampling.append(dict(epoch=source_epoch+1,start_step=first+1,stop_step=stop,
                        plan_coverage=plan.coverage(source_epoch)))
                atomic_json(run/f'sampling_additional_epoch_{epoch+1}.json',sampling)
                phase=Phase(f'V3.16.1 {tag} (+{epoch+1}/{cfg["epochs"]})',plan.steps)
                totals=dict(updates=0,compute_seconds=0.,data_wait_seconds=0.,groups={})
                ready=time.monotonic()
                with (run/'training_steps.jsonl').open('a',encoding='utf-8',buffering=1) as log:
                    def source_batches():
                        for source_epoch,first,stop in portions:
                            for step,batch in enumerate(loader.segment(source_epoch,first,stop),first+1):
                                yield source_epoch,step,batch
                    for source_epoch,source_step,batch in source_batches():
                        waiting=time.monotonic()-ready
                        batch=tensors(batch)
                        schedule(optimizer,state['cursor'],total,cfg)
                        warm=min(1.,(state['cursor']+1)/max(1,plan.steps*cfg['objective_ramp_epochs']))
                        began=time.monotonic()
                        stats=train_step(model,auxiliary,optimizer,batch,effective,warm,audit=within==0)
                        elapsed=time.monotonic()-began
                        state['cursor']+=1; within+=1; totals['updates']+=1
                        totals['compute_seconds']+=elapsed; totals['data_wait_seconds']+=waiting
                        for k,v in stats.items():
                            if isinstance(v,(float,int)):
                                totals[k]=max(totals.get(k,0.),v) if k=='maximum_example_ce' else totals.get(k,0.)+v
                        for k,v in stats['groups'].items():
                            cell=totals['groups'].setdefault(k,dict(attempted=0,valid=0))
                            for name,count in v.items(): cell[name]+=count
                        log.write(json.dumps(dict(cursor=state['cursor'],epoch=source_epoch+1,step=source_step,
                            additional_epoch=epoch+1,additional_epoch_step=within,
                            warm=warm,compute_seconds=elapsed,data_wait_seconds=waiting,**stats),allow_nan=False)+'\n')
                        phase.update(within,stats['total_loss']); ready=time.monotonic()
                if within!=plan.steps: raise RuntimeError('Incomplete source coverage')
                logits=infer(model,dev['rows'],effective,'V3.16.1 '+tag+' Dev',run)
                value=measure(dev['rows'],logits,target=cfg['matched_fake_recall'])
                current=partial_state(model)
                if any(not bool(torch.isfinite(x).all()) for x in current.values()): raise FloatingPointError('Nonfinite weights')
                promoted=promote(state,tag,current,value,dict(run=str(run.resolve()),version='3.16.1',tag=tag))
                entry=dict(tag=tag,cursor=state['cursor'],metrics=value,promoted=promoted,committed=True,
                    segment_training=totals,last_training_step=stats,additional_epoch=epoch+1)
                state['last_tag']=tag; state['history'].append(entry)
                np.savez_compressed(run/('dev_scores_'+tag+'.npz'),logits=logits)
                atomic_json(run/'validation_pending.json',entry)
                publish('Saving validated LAST and selected best',force=True)
                save(run,cfg,state,model,auxiliary,optimizer)
                (run/'validation_pending.json').unlink(missing_ok=True)
                print_metrics(tag,value)
                n=totals['updates']
                print(f'  Mean loss={totals["total_loss"]/n:.6f} CE={totals["classification_loss"]/n:.6f} '
                    f'T={totals["weighted_time_loss"]/n:.6f} CKA={totals["weighted_structure_loss"]/n:.6f}; '
                    f'compute/wait={totals["compute_seconds"]/n:.2f}/{totals["data_wait_seconds"]/n:.2f}s',flush=True)
                print(f'  BEST={state["best_tag"]} Weighted={100*state["best_metrics"]["weighted_f1"]:.3f} '
                      f'updated={promoted}; completed +{epoch+1}/{cfg["epochs"]}',flush=True)
        finally:
            loader.close()
        verify_inputs(cfg)
        return finish(run,cfg,state,total)


def final_diagnostics(cfg,run):
    """One post-selection broader check; no repeated per-epoch panel inference."""
    from w2v_v316_tfcl.diagnostics import infer_panel,panel_metrics,panel_partition
    from .diagnostics import paired_treatment
    run=Path(run)
    if (run/'post_selection_audit.json').exists(): return
    source=Path(cfg['source_run'])
    if not (source/'panel_rows.json').exists():
        atomic_json(run/'post_selection_audit.json',dict(status='unavailable',reason='Source panel missing',used_for_selection=False))
        return
    checkpoint,meta=load_selected(run,'best')
    model=load_model(cfg,training=False); apply_partial(model,checkpoint['candidate']['state']); del checkpoint
    rows,recipes=read_json(source/'panel_rows.json'),read_json(source/'panel_recipes.json')
    rows,recipes=panel_partition(rows,recipes,'audit')
    with (run/'details.log').open('a',encoding='utf-8',buffering=1) as stream:
        with redirect_stdout(stream),redirect_stderr(stream):
            logits=infer_panel(model,rows,recipes,cfg,run,'V3.16.1 final-only broader diagnosis')
    off=infer(model,rows,cfg,'V3.16.1 final-only Offline diagnosis',run)
    dev=read_json(run/'dev_rows.json')
    tag=meta['selected']
    path=(source/('dev_scores_'+tag.removeprefix('source:')+'.npz') if tag.startswith('source:')
          else run/('dev_scores_'+tag+'.npz'))
    treatment=None
    if path.is_file():
        with np.load(path,allow_pickle=False) as scores:
            treatment=paired_treatment(rows,off,dev,scores['logits'])
    metrics=panel_metrics(rows,recipes,logits,target=cfg['matched_fake_recall'])
    atomic_json(run/'post_selection_audit.json',dict(status='complete',selected=tag,metrics=metrics,
        treatment=treatment,used_for_selection=False,
        limitation='Local processing panel; source utterances overlap historical Dev; not a guarantee for official Noisy'))
    print(f'[Audit] selected={tag}; broader local panel F1={100*metrics["macro_f1"]:.3f}; details saved, selection unchanged',flush=True)
