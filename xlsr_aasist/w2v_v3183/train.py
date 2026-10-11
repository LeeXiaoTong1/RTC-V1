"""Verified parent continuation, six additional epochs and transactional resume."""
from collections import defaultdict,Counter
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .common import (schema_for,atomic_json,read_json,digest,fingerprint,seed_all,partial_state,apply_partial,
    atomic_save,to_cpu,capture_rng,restore_rng,append_json)
from .model import load_model,optimizer_for
from .sampling import Plan,probe_tickets
from .data import Waves,TicketSampler,loader,close
from .records import dev_partition
from .step import train_step,schedule
from .inference import infer
from .monitor import panel_rows,summarize,measure,fit_state,print_epoch,render
from .calibration import choose as choose_calibration
from .parent import initialize,baseline_check
from .recovery import trim_step_log


def select_key(metrics):
    return (metrics['weighted_f1'],metrics['noisy_f1'],metrics['clean_f1'])


def promote(state,tag,current,metrics,logits):
    improved=state['best_metrics'] is None or select_key(metrics)>select_key(state['best_metrics'])
    if improved:state.update(best_tag=tag,best_model=current,best_metrics=metrics,best_logits=logits)
    return improved


def save(run,cfg,state,model,optimizer):
    state.update(model=partial_state(model),optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng())
    if state['best_tag']==state['last_tag']:state['best_model']=state['model']
    atomic_save(Path(run)/'last.pt',state,cfg['free_reserve_bytes']+cfg['disk_margin_bytes'])
    atomic_json(Path(run)/'training_history.json',state['history'])


def run_experiment(cfg,run,train,dev,model_factory=load_model,baseline_only=False):
    if model_factory is load_model:
        from .interface_smoke import main as native_check
        native_check(cfg['device'])
    run=Path(run);seed_all(cfg['seed']);model=model_factory(cfg)
    optimizer=optimizer_for(model,cfg)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    head=sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    fusion_count=sum(p.numel() for p in model.fusion.parameters())
    inventory=dict(total=trainable,head=head,lora=trainable-head-fusion_count,fusion=fusion_count,lora_projections=model.lora_inventory,
        encoder_base_frozen=True,tfcl_parameters=0,head_buffers_saved=True,
        omni_arch=cfg.get('omni_arch','1b'),encoder_dim=cfg['encoder_dim'],encoder_layers=cfg.get('encoder_layers'))
    atomic_json(run/'parameter_inventory.json',inventory)
    if shutil.disk_usage(run).free<cfg['free_reserve_bytes']+1024**3:raise OSError('Need 1 GiB for partial checkpoints plus free-space reserve')
    print(f'[Model] V3.18.3 {cfg["variant"]} Omni W2V {cfg.get("omni_arch","1b").upper()}: trainable={trainable:,}; AASIST/aggregation={head:,}; LoRA={trainable-head-fusion_count:,}; fusion={fusion_count}; TFCL/InfoNCE=off',flush=True)
    plan=Plan(train,cfg['stream_sources'],cfg['seed'],cfg['group_quotas'])
    partition=dev_partition(dev,cfg['dev_pairs'],cfg['seed'],cfg['calibration_fraction'])
    parent=Path(cfg['initialization']['run'])
    if partition!=read_json(parent/'dev_partition.json'):raise ValueError('Parent Dev partition changed')
    atomic_json(run/'dev_partition.json',partition)
    selection=partition['select'];selection_rows=[dev[i] for i in selection]
    probes=read_json(parent/'train_probe_tickets.json')
    if probes!=probe_tickets(plan,cfg['train_probe_per_group']):raise ValueError('Parent train probes changed')
    atomic_json(run/'train_probe_tickets.json',probes)
    panel=read_json(parent/'independent_panel.json');panels,panel_tickets=panel['rows'],panel['tickets']
    atomic_json(run/'independent_panel.json',dict(rows=panels,tickets=panel_tickets))
    state=dict(schema=schema_for(cfg),identity=fingerprint(cfg),cursor=0,epoch=0,last_tag=None,best_tag=None,
        best_metrics=None,best_model=None,best_logits=None,last_logits=None,history=[],steps=[],patience_anchor=None,no_progress=0)
    if (run/'last.pt').is_file():
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if state.get('schema')!=schema_for(cfg) or state.get('identity')!=fingerprint(cfg):raise ValueError('Resume configuration changed')
        apply_partial(model,state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        if not state.get('baseline',{}).get('verified'):raise ValueError('Resume lacks verified parent baseline')
        print(f'[Resume] committed epochs={state["epoch"]}; optimizer and BN restored',flush=True)
    else:
        initialize(model,cfg)
        print('[Parent] reproducing full fixed Dev before any update; this takes one Dev inference pass',flush=True)
        state['baseline']=baseline_check(model,cfg,run,dev,partition,infer,measure)
        save(run,cfg,state,model,optimizer)
    trim_step_log(run/'steps.jsonl',state['cursor'])
    atomic_json(run/'training_history.json',state['history'])
    if baseline_only:
        print('BASELINE_VERIFIED_RUN='+str(run)+'; use --resume to train this exact configuration',flush=True)
        return dict(status='baseline_verified',version='3.18.3')
    sampler=TicketSampler();batches=loader(Waves(train,cfg),cfg,batch_sampler=sampler)
    try:
        for epoch in range(state['epoch'],cfg['epochs']):
            if state.get('stopped_early'):break
            joint=epoch>=cfg['warm_epochs'];model.set_phase(joint);model.train()
            coverage=plan.coverage(epoch);atomic_json(run/f'coverage_epoch_{epoch+1}.json',coverage)
            sampler.batches=list(plan.batches(epoch))
            print(f'\n[Train] V3.18.3 epoch {epoch+1}/{cfg["epochs"]} phase=joint updates={plan.steps}; quotas={cfg["group_quotas"]}; Online/Noisy loss=50/50%; group loss=25% each',flush=True)
            totals=defaultdict(float);cells=defaultdict(lambda:defaultdict(float));processing=Counter();recent=[];start=time.monotonic()
            for step,units in enumerate(batches):
                wait=time.monotonic()-start;risk=schedule(optimizer,cfg,epoch,step,plan.steps);compute=time.monotonic()
                stats=train_step(model,optimizer,units,cfg,risk,audit=(step==0 or (step+1)%100==0))
                if cfg['device'].startswith('cuda'):torch.cuda.synchronize()
                seconds=time.monotonic()-compute
                totals['loss']+=stats['loss'];totals['raw_ce_sum']+=stats['raw_ce_sum'];totals['views']+=stats['views']
                totals['compute_seconds']+=seconds;totals['wait_seconds']+=wait;totals['updates']+=1
                for key,values in stats['cells'].items():
                    for name,value in values.items():cells[key][name]+=value
                processing.update(stats['processing_counts']);totals['unique_sources_per_update']+=stats['unique_original_sources']
                state['cursor']=(epoch*plan.steps)+step+1
                recent.append(dict(loss=stats['loss'],raw_ce=stats['raw_ce_sum']/stats['views']))
                # All raw detail remains in a file; terminal shows only concise milestones.
                append_json(run/'steps.jsonl',dict(update=state['cursor'],epoch=epoch+1,step=step+1,**stats,
                    compute_seconds=seconds,wait_seconds=wait))
                if (step+1)%100==0 or step+1==plan.steps:
                    item=dict(update=state['cursor'],loss=float(np.mean([s['loss'] for s in recent])),raw_ce=float(np.mean([s['raw_ce'] for s in recent])))
                    state['steps'].append(item);recent=[]
                if (step+1)%500==0 or step+1==plan.steps:
                    print(f'  step {step+1}/{plan.steps} objective={stats["loss"]:.5f} CE={stats["raw_ce_sum"]/stats["views"]:.5f} risk={risk:.3f} compute/wait={seconds:.2f}/{wait:.2f}s',flush=True)
                start=time.monotonic()
            if int(totals['updates'])!=plan.steps:raise RuntimeError('Incomplete logical epoch')
            tag=f'epoch_{epoch+1}_step_{plan.steps}'
            print('[Eval] full fixed Dev, fixed Train probe, independent mechanisms; batch details suppressed',flush=True)
            logits,_=infer(model,dev,cfg)
            train_logits,probe_rows=infer(model,train,cfg,probes)
            panel_logits,panel_meta=infer(model,panels,cfg,panel_tickets,'dev')
            metrics=measure(dev,logits);selected_metrics=measure(selection_rows,logits[selection])
            current=partial_state(model)
            updated=promote(state,tag,current,selected_metrics,torch.from_numpy(logits.copy()))
            training=dict(loss=totals['loss']/plan.steps,raw_ce=totals['raw_ce_sum']/totals['views'],
                compute_seconds=totals['compute_seconds']/plan.steps,wait_seconds=totals['wait_seconds']/plan.steps,
                cells={k:dict(v) for k,v in cells.items()},views=int(totals['views']),processing_counts=dict(processing),
                mean_unique_sources_per_update=totals['unique_sources_per_update']/plan.steps)
            entry=dict(epoch=epoch+1,tag=tag,phase='joint' if joint else 'head_warmup',metrics=metrics,
                selection_metrics=selected_metrics,train=summarize(probe_rows,train_logits),dev=summarize(dev,logits),
                panel=summarize(panel_meta,panel_logits),training=training,promoted=updated)
            entry['fusion']=model.fusion.describe()
            entry['parent_selection_delta']=selected_metrics['weighted_f1']-state['baseline']['selection_metrics']['weighted_f1']
            entry['fit']=fit_state(state['history']+[entry])
            if joint:
                score=selected_metrics['weighted_f1']
                if state['patience_anchor'] is None or score>state['patience_anchor']+cfg['min_delta']:
                    state['patience_anchor']=score;state['no_progress']=0
                else:state['no_progress']+=1
            state['history'].append(entry);state['epoch']=epoch+1;state['last_tag']=tag
            state['last_logits']=torch.from_numpy(logits.copy())
            state['stopped_early']=bool(joint and cfg['patience'] and epoch+1-cfg['warm_epochs']>=cfg['min_joint_epochs'] and state['no_progress']>=cfg['patience'])
            np.savez_compressed(run/f'dev_scores_{tag}.npz',logits=logits)
            atomic_json(run/f'sampling_and_loss_{tag}.json',dict(coverage=coverage,actual=training))
            save(run,cfg,state,model,optimizer)
            print_epoch(entry,state);render(run,state['history'],state['steps'])
            if state['stopped_early']:
                print(f'[Early stop] no significant selection improvement for {state["no_progress"]} joint epochs; best/last both retained',flush=True);break
    finally:close(batches)
    if state['best_tag'] is None:raise RuntimeError('No completed V3.18.3 epoch')
    checkpoint_hash=digest(run/'last.pt')
    for kind in ('best','last'):
        z=state['best_logits' if kind=='best' else 'last_logits'].numpy()
        cal=choose_calibration(dev,z,partition)
        atomic_json(run/f'calibration_{kind}.json',dict(cal,checkpoint_sha256=checkpoint_hash,selected=state['best_tag' if kind=='best' else 'last_tag']))
    done=dict(version='3.18.3',status='complete',best_tag=state['best_tag'],last_tag=state['last_tag'],
        completed_epochs=state['epoch'],planned_epochs=cfg['epochs'],stopped_early=state.get('stopped_early',False),
        checkpoint_sha256=checkpoint_hash,baseline_fallback=False,selection=cfg['selection'],initialization=cfg['initialization'],
        parent_baseline=state['baseline'],best_selection_delta=state['best_metrics']['weighted_f1']-state['baseline']['selection_metrics']['weighted_f1'])
    atomic_json(run/'completed.json',done)
    print(f'[Complete] V3.18.3 epochs={state["epoch"]}/{cfg["epochs"]}; best={state["best_tag"]}; last={state["last_tag"]}; no historical fallback',flush=True)
    return done
