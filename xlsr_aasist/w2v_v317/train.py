"""Fresh bounded adaptation, atomic epoch resume and explicit Train/Dev gaps."""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import json
import random
import shutil
import time
import numpy as np
import torch
from live_progress import Phase,publish
from w2v_v39.common import atomic_json,digest,read_json
from w2v_v313.replay import rows_signature
from w2v_v314.train import schedule
from w2v_v316_tfcl.metrics import measure
from .config import verify_inputs
from .data import bundles,SourcePlan,TrainingLoader,tensors
from .model import load_model,optimizer_for,inventory
from .objectives import TFCL
from .step import train_step
from .performance import select_execution,checkpointing
from .output import infer,print_metrics
from .monitor import probe_rows,diagnostics
from .state import (SCHEMA,identity,partial_state,apply_partial,to_cpu,atomic_save,capture_rng,
                    restore_rng,storage_budget,promote,load_resume)


def save(run,cfg,state,model,auxiliary,optimizer):
    state.update(model=partial_state(model),auxiliary=to_cpu(auxiliary.state_dict()),
                 optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng())
    if state['best_tag']==state['last_tag']:state['best_model']=state['model']
    atomic_save(run/'last.pt',state,cfg['disk_margin_bytes']+cfg['free_reserve_bytes'])
    atomic_json(run/'training_history.json',state['history'])


def finish(run,cfg,state,total):
    if state['best_tag'] is None:raise ValueError('No trained V3.17 checkpoint to select')
    done=dict(version='3.17',status='complete',checkpoint_sha256=digest(run/'last.pt'),
        best_tag=state['best_tag'],best_origin=state['best_origin'],last_tag=state['last_tag'],
        committed_updates=state['cursor'],planned_updates=total,epochs=cfg['epochs'],
        initialization=cfg['initialization_provenance'],baseline_fallback=False,
        selection='maximum complete fixed Dev weighted within V3.17 only')
    atomic_json(run/'completed.json',done)
    for kind in ('best','best_weighted','last'):
        atomic_json(run/(kind+'.json'),dict(checkpoint='last.pt',kind=kind,
            selected=state['last_tag'] if kind=='last' else state['best_tag']))
    print(f'[Complete] V3.17 best={state["best_tag"]} Weighted={100*state["best_metrics"]["weighted_f1"]:.3f}; '
          f'last={state["last_tag"]}; epochs={cfg["epochs"]}',flush=True)
    return done


def run_experiment(cfg,run):
    run=Path(run);verify_inputs(cfg)
    torch.set_num_threads(1)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if cfg['device'].startswith('cuda'):
        torch.cuda.set_device(torch.device(cfg['device']))
        if cfg['amp']=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unavailable')
    model=load_model(cfg)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(cfg['seed']+317)
        auxiliary=TFCL(cfg['feature_dim'],cfg['tfcl_heads'],cfg['tfcl_bins']).to(cfg['device'])
    optimizer=optimizer_for(model,cfg,auxiliary)
    counts=inventory(model,auxiliary);atomic_json(run/'parameter_inventory.json',counts)
    print(f'[Model] V3.17 LoRA={counts["lora"]:,}; head={counts["head"]:,}; '
          f'TFCL={counts["auxiliary"]:,}; original encoder frozen',flush=True)
    budget=storage_budget(model,auxiliary,cfg);free=shutil.disk_usage(run).free
    existing=(run/'last.pt').stat().st_size if (run/'last.pt').exists() else 0
    needed=max(0,budget['required_start_free_bytes']-existing)
    atomic_json(run/'storage_budget.json',dict(budget,free_bytes=free,required_remaining_bytes=needed))
    if free<needed:raise OSError(f'Need {needed/1024**3:.2f} GiB including reserve and atomic save')
    print(f'[Storage] free={free/1024**3:.2f} GiB; required={needed/1024**3:.2f} GiB; new audio/frame cache=0',flush=True)
    with bundles(cfg) as (train,dev):
        old_rows=read_json(Path(cfg['data_run'])/'dev_rows.json')
        if rows_signature(dev['rows'])!=rows_signature(old_rows):raise ValueError('Fixed Dev identities changed')
        atomic_json(run/'dev_rows.json',dev['rows'])
        probe=probe_rows(train['rows'],cfg);atomic_json(run/'train_probe_rows.json',probe)
        plan=SourcePlan(train['rows'],cfg['source_batch'],cfg['seed'],cfg);total=plan.steps*cfg['epochs']
        if (run/'last.pt').exists():
            state=load_resume(run,cfg);apply_partial(model,state['model'])
            auxiliary.load_state_dict(state['auxiliary'],strict=True)
            optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        else:
            state=dict(schema=SCHEMA,identity=identity(cfg),cursor=0,history=[],last_tag='initialization',
                       best_tag=None,best_model=None,best_metrics=None,best_origin=None)
            atomic_json(run/'initialization.json',dict(cfg['initialization_provenance'],
                pretrained_fingerprints=cfg['pretrained_fingerprints'],seed=cfg['seed'],parameters=counts))
            print('[Start] public pretrained encoder + fresh LoRA/head; no historical detector or Adam loaded',flush=True)
        publish('Profiling LoRA/forensic training; all trial updates will be rolled back',force=True)
        with (run/'details.log').open('a',encoding='utf-8',buffering=1) as stream:
            with redirect_stdout(stream),redirect_stderr(stream):execution=select_execution(model,auxiliary,optimizer,plan,cfg,run)
        state['execution_plan_sha256']=digest(run/'execution_plan.json')
        effective=dict(cfg,**execution['selected']);checkpointing(model,effective['checkpointing'])
        print('[Execution] '+str(execution['selected'])+'; fixed Train/Dev replay once per epoch',flush=True)
        if not (run/'last.pt').exists():save(run,cfg,state,model,auxiliary,optimizer)
        loader=TrainingLoader(plan,effective,run)
        try:
            while state['cursor']<total:
                epoch,within=divmod(state['cursor'],plan.steps)
                if within:raise ValueError('Only complete epoch states may be resumed')
                coverage=plan.coverage(epoch);atomic_json(run/f'sampling_epoch_{epoch+1}.json',coverage)
                if epoch==0:print('[Sampling] '+str(coverage['source_groups'])+'; each group keeps 25% of CE/TFCL budget',flush=True)
                phase=Phase(f'V3.17 epoch {epoch+1}/{cfg["epochs"]}',plan.steps)
                totals=dict(updates=0,compute_seconds=0.,data_wait_seconds=0.,groups={});ready=time.monotonic()
                with (run/'training_steps.jsonl').open('a',encoding='utf-8',buffering=1) as log:
                    for step,batch in enumerate(loader.segment(epoch,0,plan.steps),1):
                        waiting=time.monotonic()-ready;batch=tensors(batch)
                        schedule(optimizer,state['cursor'],total,cfg)
                        warm=min(1.,(state['cursor']+1)/max(1,plan.steps*cfg['objective_ramp_epochs']))
                        began=time.monotonic()
                        audit=step in (1,max(1,plan.steps//2))
                        stats=train_step(model,auxiliary,optimizer,batch,effective,warm,audit=audit)
                        elapsed=time.monotonic()-began;state['cursor']+=1;totals['updates']+=1
                        totals['compute_seconds']+=elapsed;totals['data_wait_seconds']+=waiting
                        for k,v in stats.items():
                            if isinstance(v,(float,int)):
                                totals[k]=max(totals.get(k,0.),v) if k=='maximum_example_ce' else totals.get(k,0.)+v
                        for k,v in stats['groups'].items():
                            cell=totals['groups'].setdefault(k,dict(attempted=0,valid=0))
                            for name,count in v.items():cell[name]+=count
                        log.write(json.dumps(dict(cursor=state['cursor'],epoch=epoch+1,step=step,warm=warm,
                            learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
                            compute_seconds=elapsed,data_wait_seconds=waiting,**stats),allow_nan=False)+'\n')
                        phase.update(step,stats['total_loss']);ready=time.monotonic()
                if totals['updates']!=plan.steps:raise RuntimeError('Incomplete epoch source budget')
                tag=f'epoch_{epoch+1}_step_{plan.steps}'
                logits=infer(model,dev['rows'],effective,'V3.17 '+tag+' Dev',run)
                value=measure(dev['rows'],logits,target=cfg['matched_fake_recall'])
                # Eval loaders get their own RNG save/restore: diagnostic work cannot
                # change the following epoch's dropout/random update sequence.
                rng=capture_rng()
                try:train_logits=infer(model,probe,effective,'V3.17 fixed Train replay',run)
                finally:restore_rng(rng)
                gap=dict(train=diagnostics(probe,train_logits),dev=diagnostics(dev['rows'],logits))
                atomic_json(run/('generalization_'+tag+'.json'),gap)
                current=partial_state(model)
                if any(not bool(torch.isfinite(x).all()) for x in current.values()):raise FloatingPointError('Nonfinite trained parameters')
                promoted=promote(state,tag,current,value,dict(run=str(run.resolve()),version='3.17',tag=tag))
                state['last_tag']=tag
                state['history'].append(dict(tag=tag,cursor=state['cursor'],metrics=value,promoted=promoted,
                    committed=True,segment_training=totals,generalization=gap))
                np.savez_compressed(run/('dev_scores_'+tag+'.npz'),logits=logits)
                np.savez_compressed(run/('train_probe_scores_'+tag+'.npz'),logits=train_logits)
                publish('Saving LoRA/head + optimizer + best in one atomic checkpoint',force=True)
                save(run,cfg,state,model,auxiliary,optimizer)
                print_metrics(tag,value)
                a,b=gap['train']['conditions']['online'],gap['dev']['conditions']['online']
                print(f'  Fixed Online replay: Train balanced recall={100*a["balanced_recall"]:.2f}% CE={a["balanced_ce"]:.4f}; '
                      f'Dev balanced recall={100*b["balanced_recall"]:.2f}% CE={b["balanced_ce"]:.4f}',flush=True)
                n=totals['updates']
                print(f'  Mean CE={totals["classification_loss"]/n:.6f} TFCL-time={totals["weighted_time_loss"]/n:.6f} '
                      f'TFCL-CKA={totals["weighted_structure_loss"]/n:.6f}; compute/wait='
                      f'{totals["compute_seconds"]/n:.2f}/{totals["data_wait_seconds"]/n:.2f}s',flush=True)
                print(f'  BEST={state["best_tag"]} Weighted={100*state["best_metrics"]["weighted_f1"]:.3f}; updated={promoted}',flush=True)
        finally:loader.close()
        verify_inputs(cfg)
        return finish(run,cfg,state,total)
