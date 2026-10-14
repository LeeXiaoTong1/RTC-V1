"""Explicit parent transfer, whole-utterance adaptation, joint PCL and atomic resume."""
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
from w2v_v3184.records import dev_partition
from .step import train_step,schedule
from w2v_v3184.inference import infer
from .monitor import summarize,measure,fit_state,print_epoch,render
from w2v_v3184.calibration import choose as choose_calibration
from .parent import initialize,baseline_check
from w2v_v3184.recovery import trim_step_log
from w2v_v3184.panel import scene_panel


def select_key(metrics):
    return (metrics['weighted_f1'],metrics['noisy_f1'],metrics['clean_f1'])


def promote(state,tag,current,metrics,logits):
    improved=state['best_metrics'] is None or select_key(metrics)>select_key(state['best_metrics'])
    if improved:state.update(best_tag=tag,best_model=current,best_metrics=metrics,best_logits=logits)
    return improved


def save(run,cfg,state,model,optimizer,current=None):
    # The end-of-epoch snapshot is also used for best selection. Reuse it;
    # no optimizer update occurs between selection and this atomic commit.
    state.update(model=partial_state(model) if current is None else current,
                 optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng())
    if state['best_tag']==state['last_tag']:state['best_model']=state['model']
    atomic_save(Path(run)/'last.pt',state,cfg['free_reserve_bytes']+cfg['disk_margin_bytes'])
    atomic_json(Path(run)/'training_history.json',state['history'])


def run_experiment(cfg,run,train,dev,model_factory=load_model,baseline_only=False):
    if model_factory is load_model:
        from .interface_smoke import main as native_check
        native_check(cfg['device'])
    run=Path(run);seed_all(cfg['seed'])
    print('[Startup] loading public Omni encoder and constructing full-layer fusion/AASIST',flush=True)
    model=model_factory(cfg)
    optimizer=optimizer_for(model,cfg)
    lora_count=sum(p.numel() for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b')))
    head=sum(p.numel() for p in model.head.parameters())
    fusion_count=sum(p.numel() for p in model.fusion.parameters())
    trainable=lora_count+head+fusion_count
    inventory=dict(total=trainable,head=head,lora=lora_count,fusion=fusion_count,lora_projections=model.lora_inventory,
        joint_only=True,external_window_pooling=False,pcl_parameters=0,
        encoder_base_frozen=True,tfcl_parameters=0,head_buffers_saved=True,
        omni_arch=cfg.get('omni_arch','1b'),encoder_dim=cfg['encoder_dim'],encoder_layers=cfg.get('encoder_layers'))
    atomic_json(run/'parameter_inventory.json',inventory)
    if shutil.disk_usage(run).free<cfg['free_reserve_bytes']+1024**3:raise OSError('Need 1 GiB for partial checkpoints plus free-space reserve')
    print(f'[Model] V3.18.6 {cfg["variant"]} Omni W2V {cfg.get("omni_arch","1b").upper()}: joint trainable={trainable:,}; AASIST={head:,}; LoRA={lora_count:,}; fusion={fusion_count:,}; PCL={cfg["consistency"]}; raw Offline/clean RTC CE=off',flush=True)
    plan=Plan(train,cfg['stream_sources'],cfg['seed'],cfg['group_quotas'],cfg['view_factors'],cfg['refresh_base_variants'],cfg=cfg)
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
    scenes,scene_tickets=scene_panel(cfg)
    atomic_json(run/'scene_panel.json',dict(rows=scenes,tickets=scene_tickets,
        note='New noise source diagnostic only; excluded from historical Weighted, calibration and checkpoint selection'))
    state=dict(schema=schema_for(cfg),identity=fingerprint(cfg),cursor=0,epoch=0,last_tag=None,best_tag=None,
        best_metrics=None,best_model=None,best_logits=None,last_logits=None,history=[],steps=[],patience_anchor=None,no_progress=0)
    atomic_json(run/'augmentation_budget_audit.json',cfg.get('augmentation_budget_audit') or plan.audit_budget(cfg['epochs']))
    if (run/'last.pt').is_file():
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if state.get('schema')!=schema_for(cfg) or state.get('identity')!=fingerprint(cfg):raise ValueError('Resume configuration changed')
        apply_partial(model,state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        if not state.get('baseline',{}).get('verified') or not state['baseline'].get('reference_only'):
            raise ValueError('Resume lacks verified cached parent reference')
        print(f'[Resume] committed epochs={state["epoch"]}; optimizer and BN restored',flush=True)
    else:
        state['initialization_audit']=initialize(model,cfg)
        atomic_json(run/'initialization_audit.json',state['initialization_audit'])
        print('[Parent] reading completed parent best reference; initialization audit records full continuation or explicit migration',flush=True)
        state['baseline']=baseline_check(model,cfg,run,dev,partition,infer,measure)
        print('[Initial scenes] fixed six-condition diagnostic before adaptation; no full Dev replay',flush=True)
        initial_logits,initial_rows=infer(model,scenes,cfg,scene_tickets,'dev')
        state['scene_baseline']=summarize(initial_rows,initial_logits)
        atomic_json(run/'scene_baseline.json',state['scene_baseline'])
        np.savez_compressed(run/'scene_baseline_scores.npz',logits=initial_logits)
        save(run,cfg,state,model,optimizer)
    trim_step_log(run/'steps.jsonl',state['cursor'])
    atomic_json(run/'training_history.json',state['history'])
    if baseline_only:
        print('INITIALIZATION_VERIFIED_RUN='+str(run)+'; use --resume to train this exact configuration',flush=True)
        return dict(status='initialization_verified',version='3.18.6')
    sampler=TicketSampler();batches=loader(Waves(train,cfg),cfg,batch_sampler=sampler)
    try:
        for epoch in range(state['epoch'],cfg['epochs']):
            if state.get('stopped_early'):break
            joint=epoch>=cfg['warm_epochs'];model.set_phase(joint);model.train()
            if cfg['device'].startswith('cuda'):torch.cuda.reset_peak_memory_stats(cfg['device'])
            coverage=plan.coverage(epoch);atomic_json(run/f'coverage_epoch_{epoch+1}.json',coverage)
            sampler.batches=list(plan.batches(epoch))
            phase='joint'
            print(f'\n[Train] V3.18.6 epoch {epoch+1}/{cfg["epochs"]} phase={phase} updates={plan.steps}; quotas={cfg["group_quotas"]}; Online/Noisy loss=50/50%; group loss=25% each',flush=True)
            totals=defaultdict(float);cells=defaultdict(lambda:defaultdict(float));processing=Counter();base_views=Counter();recent=[];start=time.monotonic()
            pcl_totals=defaultdict(Counter);pcl_coverage=defaultdict(Counter)
            epoch_start=start
            for step,units in enumerate(batches):
                wait=time.monotonic()-start;risk=schedule(optimizer,cfg,epoch,step,plan.steps);compute=time.monotonic()
                stats=train_step(model,optimizer,units,cfg,risk,audit=(step==0 or (step+1)%cfg.get('gradient_audit_interval',100)==0))
                # C3 already does a blocking telemetry .cpu() AFTER optimizer.step.
                # Its default-stream work has finished; no device-wide wait needed.
                seconds=time.monotonic()-compute
                totals['loss']+=stats['loss'];totals['raw_ce_sum']+=stats['raw_ce_sum'];totals['views']+=stats['views']
                totals['compute_seconds']+=seconds;totals['wait_seconds']+=wait;totals['updates']+=1
                for key,values in stats['cells'].items():
                    for name,value in values.items():cells[key][name]+=value
                processing.update(stats['processing_counts']);totals['unique_sources_per_update']+=stats['unique_original_sources']
                for name in ('classification_loss','encoded_views','auxiliary_reference_views'):
                    totals[name]+=stats.get(name,0.)
                for kind,values in stats['pcl_losses'].items():pcl_totals[kind].update(values)
                for cell,values in stats['pcl_coverage'].items():pcl_coverage[cell].update(values)
                base_views.update(stats['base_source_view_counts'])
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
            print('[Eval] full fixed Dev, fixed Train probe, independent mechanisms and six scenes',flush=True)
            evaluation_seconds={}
            def timed_infer(name,*args):
                print('[Eval phase] '+name,flush=True)
                began=time.monotonic();result=infer(*args)
                evaluation_seconds[name]=time.monotonic()-began
                return result
            logits,_=timed_infer('dev',model,dev,cfg)
            train_logits,probe_rows=timed_infer('train_probe',model,train,cfg,probes)
            panel_logits,panel_meta=timed_infer('mechanisms',model,panels,cfg,panel_tickets,'dev')
            scene_logits,scene_meta=timed_infer('six_scenes',model,scenes,cfg,scene_tickets,'dev')
            metrics=measure(dev,logits);selected_metrics=measure(selection_rows,logits[selection])
            snapshot_start=time.monotonic()
            current=partial_state(model)
            snapshot_seconds=time.monotonic()-snapshot_start
            epoch_logits=torch.from_numpy(logits.copy())
            updated=promote(state,tag,current,selected_metrics,epoch_logits)
            training=dict(loss=totals['loss']/plan.steps,raw_ce=totals['raw_ce_sum']/totals['views'],
                compute_seconds=totals['compute_seconds']/plan.steps,wait_seconds=totals['wait_seconds']/plan.steps,
                cells={k:dict(v) for k,v in cells.items()},views=int(totals['views']),processing_counts=dict(processing),
                mean_unique_sources_per_update=totals['unique_sources_per_update']/plan.steps,
                base_source_view_counts=dict(base_views),
                auxiliary=dict(
                    mean_per_update={name:totals[name]/plan.steps for name in ('classification_loss','encoded_views','auxiliary_reference_views')},
                    losses={kind:dict(mean_source_loss=v['source_loss_sum']/max(1,v['sources']),
                        mean_weighted_loss_per_update=v['weighted_loss']/plan.steps,
                        scheduled_sources=int(v['sources'])) for kind,v in pcl_totals.items()},
                    coverage={cell:dict(v) for cell,v in pcl_coverage.items()}))
            training['cuda_peak_memory']=None
            if cfg['device'].startswith('cuda'):
                training['cuda_peak_memory']=dict(
                    allocated_gib=torch.cuda.max_memory_allocated(cfg['device'])/1024**3,
                    reserved_gib=torch.cuda.max_memory_reserved(cfg['device'])/1024**3,
                    scope='training and all fixed evaluation passes in this epoch, including audit snapshots')
            entry=dict(epoch=epoch+1,tag=tag,phase=phase,metrics=metrics,
                selection_metrics=selected_metrics,train=summarize(probe_rows,train_logits),dev=summarize(dev,logits),
                panel=summarize(panel_meta,panel_logits),training=training,promoted=updated)
            entry['scenes']=summarize(scene_meta,scene_logits)
            entry['fusion']=model.fusion.describe()
            entry['parent_selection_delta']=selected_metrics['weighted_f1']-state['baseline']['selection_metrics']['weighted_f1']
            entry['fit']=fit_state(state['history']+[entry])
            if joint:
                score=selected_metrics['weighted_f1']
                if state['patience_anchor'] is None or score>state['patience_anchor']+cfg['min_delta']:
                    state['patience_anchor']=score;state['no_progress']=0
                else:state['no_progress']+=1
            state['history'].append(entry);state['epoch']=epoch+1;state['last_tag']=tag
            state['last_logits']=epoch_logits
            state['stopped_early']=bool(joint and cfg['patience'] and epoch+1-cfg['warm_epochs']>=cfg['min_joint_epochs'] and state['no_progress']>=cfg['patience'])
            np.savez_compressed(run/f'dev_scores_{tag}.npz',logits=logits)
            np.savez_compressed(run/f'scene_scores_{tag}.npz',logits=scene_logits)
            atomic_json(run/f'sampling_and_loss_{tag}.json',dict(coverage=coverage,actual=training))
            checkpoint_start=time.monotonic()
            save(run,cfg,state,model,optimizer,current)
            checkpoint_seconds=time.monotonic()-checkpoint_start
            report_start=time.monotonic()
            print_epoch(entry,state);render(run,state['history'],state['steps'])
            timing=dict(epoch=epoch+1,train_compute_seconds=totals['compute_seconds'],
                train_wait_seconds=totals['wait_seconds'],evaluation_seconds=evaluation_seconds,
                state_snapshot_seconds=snapshot_seconds,checkpoint_write_seconds=checkpoint_seconds,
                report_seconds=time.monotonic()-report_start,epoch_wall_seconds=time.monotonic()-epoch_start,
                note='Wall-clock timings include CPU work and blocking result transfers; not a CUDA kernel profiler')
            atomic_json(run/f'performance_{tag}.json',timing)
            print(f'[Timing] train compute/wait={totals["compute_seconds"]:.1f}/{totals["wait_seconds"]:.1f}s; '
                  f'eval={sum(evaluation_seconds.values()):.1f}s; checkpoint={snapshot_seconds+checkpoint_seconds:.1f}s; '
                  f'epoch wall={timing["epoch_wall_seconds"]:.1f}s; details=performance_{tag}.json',flush=True)
            memory=training['cuda_peak_memory']
            if memory:
                print(f'[CUDA memory] epoch={epoch+1}; peak allocated={memory["allocated_gib"]:.3f} GiB; peak reserved={memory["reserved_gib"]:.3f} GiB; includes training and evaluation',flush=True)
            if state['stopped_early']:
                print(f'[Early stop] no significant selection improvement for {state["no_progress"]} joint epochs; best/last both retained',flush=True);break
    finally:close(batches)
    if state['best_tag'] is None:raise RuntimeError('No completed V3.18.6 epoch')
    checkpoint_hash=digest(run/'last.pt')
    fitted={}
    for kind in ('best','last'):
        z=state['best_logits' if kind=='best' else 'last_logits'].numpy()
        tag=state['best_tag' if kind=='best' else 'last_tag']
        if tag not in fitted:fitted[tag]=choose_calibration(dev,z,partition)
        cal=fitted[tag]
        atomic_json(run/f'calibration_{kind}.json',dict(cal,checkpoint_sha256=checkpoint_hash,selected=state['best_tag' if kind=='best' else 'last_tag']))
    done=dict(version='3.18.6',recipe=cfg['recipe'],pooling=cfg['pooling'],status='complete',best_tag=state['best_tag'],last_tag=state['last_tag'],
        completed_epochs=state['epoch'],planned_epochs=cfg['epochs'],stopped_early=state.get('stopped_early',False),
        checkpoint_sha256=checkpoint_hash,baseline_fallback=False,selection=cfg['selection'],initialization=cfg['initialization'],
        parent_baseline=state['baseline'],best_selection_delta=state['best_metrics']['weighted_f1']-state['baseline']['selection_metrics']['weighted_f1'])
    atomic_json(run/'completed.json',done)
    print(f'[Complete] V3.18.6 epochs={state["epoch"]}/{cfg["epochs"]}; best={state["best_tag"]}; last={state["last_tag"]}; no historical fallback',flush=True)
    return done
