"""Complete-best continuation, exact source epochs and optimizer-boundary pause."""
from collections import Counter, defaultdict
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
from .inference import infer
from .step import train_step,schedule
from .monitor import summarize,measure,fit_state,print_epoch,render
from w2v_v3184.records import dev_partition
from w2v_v3184.calibration import choose as choose_calibration
from w2v_v3184.recovery import trim_step_log
from w2v_v3184.panel import scene_panel


def select_key(metrics):
    return metrics['weighted_f1'],metrics['noisy_f1'],metrics['clean_f1']


def promote(state,tag,current,metrics,logits):
    improved=state['best_metrics'] is None or select_key(metrics)>select_key(state['best_metrics'])
    if improved:
        state.update(best_tag=tag,best_model=current,best_metrics=metrics,best_logits=logits)
    return improved


def save(run,cfg,state,model,optimizer,current=None):
    state.update(model=partial_state(model) if current is None else current,
                 optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng())
    # A partial next epoch must never overwrite the completed best snapshot.
    if state.get('next_step',0)==0 and state['best_tag'] is not None and state['best_tag']==state['last_tag']:
        state['best_model']=state['model']
    atomic_save(Path(run)/'last.pt',state,cfg['free_reserve_bytes']+cfg['disk_margin_bytes'])
    atomic_json(Path(run)/'training_history.json',state['history'])


def calibrations(run,state,dev,partition,*,paused=False):
    checksum=digest(Path(run)/'last.pt');fitted={}
    for kind in (('best',) if paused else ('best','last')):
        tag=state[kind+'_tag']
        if tag is None:
            continue
        if tag not in fitted:
            fitted[tag]=choose_calibration(dev,state[kind+'_logits'].numpy(),partition)
        atomic_json(Path(run)/('calibration_'+kind+'.json'),
                    dict(fitted[tag],checkpoint_sha256=checksum,selected=tag))
    return checksum


def acknowledge_pause(run,cfg,state,model,optimizer,dev,partition,*,saved=False):
    if not saved:
        save(run,cfg,state,model,optimizer)
    checksum=calibrations(run,state,dev,partition,paused=True)
    paused=dict(status='paused',version='3.18.8',recipe=cfg['recipe'],completed_epochs=state['epoch'],
        next_step=state['next_step'],best_tag=state['best_tag'],checkpoint_sha256=checksum)
    atomic_json(run/'paused.json',paused)
    atomic_json(run/'progress.json',dict(paused,phase='paused'))
    print('PAUSED_RUN='+str(run),flush=True)
    return paused


def initial_audit(model,cfg):
    """Fresh tiny-model fixture only; the production entry requires a parent."""
    if cfg['initialization']['mode']!='pretrained' or cfg['initialization'].get('old_detector_loaded'):
        raise ValueError('Fresh-model test fixture must start from pretrained SSL only')
    adapters=[p for n,p in model.named_parameters() if n.endswith('.lora_b')]
    if any(torch.count_nonzero(p.detach()).item() for p in adapters):
        raise ValueError('Fresh LoRA B must start at zero')
    batch_norms=[m for m in model.head.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    if batch_norms:
        raise ValueError('V3.18.8 v2 detection backend must contain no BatchNorm')
    return dict(mode='pretrained',detector_checkpoint_loaded=False,optimizer_loaded=False,
        zero_lora_b=len(adapters),batch_norm_modules=0,normalization='GroupNorm/LayerNorm',
        scope='Official SSL backbone only; all detection parameters are new; no random-Dev baseline inference')


def _accumulator(value=None):
    value=value or {}
    return dict(totals=defaultdict(float,value.get('totals',{})),
        cells=defaultdict(Counter,{k:Counter(v) for k,v in value.get('cells',{}).items()}),
        processing=Counter(value.get('processing',{})),base_views=Counter(value.get('base_views',{})),
        pcl_totals=defaultdict(Counter,{k:Counter(v) for k,v in value.get('pcl_totals',{}).items()}),
        pcl_coverage=defaultdict(Counter,{k:Counter(v) for k,v in value.get('pcl_coverage',{}).items()}),
        confidence=Counter(value.get('confidence',{})),variance=Counter(value.get('variance',{})),
        variance_coverage=defaultdict(Counter,{k:Counter(v) for k,v in value.get('variance_coverage',{}).items()}))


def _plain(acc):
    return {name:{k:dict(v) if isinstance(v,dict) else v for k,v in values.items()} for name,values in acc.items()}


def run_experiment(cfg,run,train,dev,model_factory=load_model,baseline_only=False):
    run=Path(run);seed_all(cfg['seed'])
    # Consume only a request already acknowledged by the previous pause.
    # A new request during slow model loading must survive until the first
    # optimizer boundary. Keep acknowledgement until resume validation succeeds,
    # so a failed model load does not invalidate an intact paused best.
    request_path=run/'stop_requested.json';paused_path=run/'paused.json'
    if paused_path.is_file():
        if request_path.is_file() and request_path.stat().st_mtime_ns <= paused_path.stat().st_mtime_ns:
            request_path.unlink()
    if model_factory is load_model:
        from .interface_smoke import main as native_check
        native_check(cfg['device'])
    print('[Startup] official Omni backbone; complete V3.18.7 best detector; single-Noisy continuation',flush=True)
    model=model_factory(cfg);optimizer=optimizer_for(model,cfg)
    inventory=dict(head=sum(p.numel() for p in model.head.parameters()),
        lora=sum(p.numel() for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b'))),
        fusion=sum(p.numel() for p in model.fusion.parameters()),encoder_base_frozen=True,
        external_window_pooling=True,pooling='asp',classification_dim=model.head.aasist.out_layer.in_features,
        normalization='GroupNorm/LayerNorm',optimizer_groups=[dict(name=g['name'],lr=g['initial_lr']) for g in optimizer.param_groups],
        head_buffers_saved=True,pcl_scale_saved=True)
    atomic_json(run/'parameter_inventory.json',inventory)
    if shutil.disk_usage(run).free<cfg['free_reserve_bytes']+1024**3:
        raise OSError('Need 1 GiB plus free-space reserve for partial checkpoints')
    plan=Plan(train,cfg['stream_sources'],cfg['seed'],cfg['group_quotas'],cfg['view_factors'],cfg['refresh_base_variants'],cfg=cfg)
    partition=dev_partition(dev,cfg['dev_pairs'],cfg['seed'],cfg['calibration_fraction'])
    source=Path(cfg['data_source']['run'])
    if partition!=read_json(source/'dev_partition.json'):
        raise ValueError('Fixed source-isolated Dev partition changed')
    atomic_json(run/'dev_partition.json',partition)
    selection=partition['select'];selection_rows=[dev[i] for i in selection]
    probes=probe_tickets(plan,cfg['train_probe_per_group'])
    atomic_json(run/'train_probe_tickets.json',probes)
    panel=read_json(source/'independent_panel.json') if (source/'independent_panel.json').is_file() else None
    if panel:
        atomic_json(run/'independent_panel.json',panel)
    scenes,scene_tickets=scene_panel(cfg)
    atomic_json(run/'scene_panel.json',dict(rows=scenes,tickets=scene_tickets,
        note='Diagnostic only, every configured interval and final epoch; excluded from checkpoint selection'))
    state=dict(schema=schema_for(cfg),identity=fingerprint(cfg),cursor=0,epoch=0,next_step=0,
        epoch_accumulator=None,last_tag=None,best_tag=None,best_metrics=None,best_model=None,
        best_logits=None,last_logits=None,history=[],steps=[])
    atomic_json(run/'augmentation_budget_audit.json',cfg.get('augmentation_budget_audit') or plan.audit_budget(cfg['epochs']))
    if (run/'last.pt').is_file():
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if state.get('schema')!=schema_for(cfg) or state.get('identity')!=fingerprint(cfg):
            raise ValueError('Resume configuration changed')
        apply_partial(model,state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        print(f'[Resume] completed epochs={state["epoch"]}; next step={state["next_step"]}; optimizer and PCL scale restored',flush=True)
    elif cfg.get('parent_init'):
        from .parent import initialize
        baseline,weights,logits,audit=initialize(model,cfg,dev,partition)
        state.update(best_tag=baseline['tag'],last_tag=baseline['tag'],best_model=weights,
            best_logits=logits,last_logits=logits,best_metrics=baseline['selection_metrics'],
            history=[baseline],initialization_audit=audit)
        atomic_json(run/'initialization_audit.json',audit)
        atomic_json(run/'parent_baseline.json',baseline)
        np.savez_compressed(run/'parent_baseline_scores.npz',logits=logits.numpy())
        save(run,cfg,state,model,optimizer,current=weights)
        print_epoch(baseline,state)
    elif cfg.get('warm_start'):
        if model_factory is load_model:
            raise ValueError('Production V3.18.8 requires complete parent best, not a warmup checkpoint')
        origin=cfg['warm_start'];checkpoint=Path(origin['checkpoint'])
        if digest(checkpoint)!=origin['sha256']:
            raise ValueError('Warmup checkpoint changed')
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if (state.get('schema')!=schema_for(cfg) or state.get('identity')!=origin['parent_identity'] or
                state.get('epoch')!=cfg['warm_epochs'] or state.get('next_step')!=0):
            raise ValueError('Not the verified completed warmup state')
        apply_partial(model,state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        if cfg['consistency']=='dual' and not model.pcl_scale.is_frozen:
            raise ValueError('PCL warm start lacks frozen Train scale')
        state['identity']=fingerprint(cfg)
        atomic_json(run/'warm_start_audit.json',dict(origin,epoch=state['epoch'],optimizer_and_rng_restored=True))
        save(run,cfg,state,model,optimizer)
    else:
        if model_factory is load_model:
            raise ValueError('Production V3.18.8 requires --parent-run or --resume')
        state['initialization_audit']=initial_audit(model,cfg)
        atomic_json(run/'initialization_audit.json',state['initialization_audit'])
        save(run,cfg,state,model,optimizer)
    trim_step_log(run/'steps.jsonl',state['cursor'])
    atomic_json(run/'training_history.json',state['history'])
    if baseline_only:
        print('INITIALIZATION_VERIFIED_RUN='+str(run),flush=True)
        return dict(status='initialization_verified',version='3.18.8')
    paused_path.unlink(missing_ok=True)
    sampler=TicketSampler();batches=loader(Waves(train,cfg),cfg,batch_sampler=sampler)
    try:
        for epoch in range(state['epoch'],cfg['epochs']):
            joint=epoch>=cfg['warm_epochs'];model.set_phase(joint);model.train()
            if joint and cfg['consistency']=='dual' and not model.pcl_scale.is_frozen:
                raise ValueError('Joint PCL requires frozen Train-warmup scale')
            phase='joint' if joint else 'head_warmup'
            if cfg['device'].startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(cfg['device'])
            coverage=plan.coverage(epoch);atomic_json(run/f'coverage_epoch_{epoch+1}.json',coverage)
            begin=int(state.get('next_step',0));sampler.batches=list(plan.batches(epoch))[begin:]
            print(f'\n[Train] V3.18.8 epoch {epoch+1}/{cfg["epochs"]} phase={phase} updates={plan.steps}',flush=True)
            acc=_accumulator(state.get('epoch_accumulator'));totals=acc['totals'];start=time.monotonic();epoch_start=start
            for step,units in enumerate(batches,start=begin):
                wait=time.monotonic()-start;risk=schedule(optimizer,cfg,epoch,step,plan.steps);compute=time.monotonic()
                try:
                    stats=train_step(model,optimizer,units,cfg,risk,audit=(step==0 or (step+1)%cfg['gradient_audit_interval']==0))
                except torch.OutOfMemoryError:
                    context=dict(epoch=epoch+1,step=step+1,committed_epochs=state['epoch'],
                        last_saved_checkpoint=str(run/'last.pt'),microbatch=cfg['microbatch'],
                        frame_budget=cfg['frame_budget'],last_encode=getattr(model,'last_encode_geometry',None),
                        note='Failed optimizer update was not committed; last.pt remains the last safe boundary')
                    atomic_json(run/'oom_context.json',context)
                    if cfg['device'].startswith('cuda'):
                        (run/'cuda_memory_at_failure.txt').write_text(torch.cuda.memory_summary(device=cfg['device']),encoding='utf-8')
                    raise
                seconds=time.monotonic()-compute
                for name in ('loss','raw_ce_sum','views','classification_loss','encoded_views','auxiliary_reference_views'):
                    totals[name]+=stats.get(name,0.)
                totals['compute_seconds']+=seconds;totals['wait_seconds']+=wait;totals['updates']+=1
                totals['unique_sources_per_update']+=stats['unique_original_sources']
                for key,values in stats['cells'].items():acc['cells'][key].update(values)
                acc['processing'].update(stats['processing_counts']);acc['base_views'].update(stats['base_source_view_counts'])
                for key,values in stats['pcl_losses'].items():acc['pcl_totals'][key].update(values)
                for key,values in stats['pcl_coverage'].items():acc['pcl_coverage'][key].update(values)
                for name in ('weighted_loss','active_views','views'):
                    acc['confidence'][name]+=stats['en_real_confidence'][name]
                acc['variance']['weighted_loss']+=stats.get('pcl_variance',{}).get('weighted_loss',0.)
                for kind,values in stats.get('pcl_variance',{}).get('coverage',{}).items():
                    acc['variance_coverage'][kind].update(values)
                state['cursor']+=1;state['next_step']=step+1
                append_json(run/'steps.jsonl',dict(update=state['cursor'],epoch=epoch+1,step=step+1,**stats,
                                                 compute_seconds=seconds,wait_seconds=wait))
                if (step+1)%100==0 or step+1==plan.steps:
                    state['steps'].append(dict(update=state['cursor'],loss=totals['loss']/totals['updates'],
                                              raw_ce=totals['raw_ce_sum']/totals['views']))
                atomic_json(run/'progress.json',dict(status='training',phase=phase,epoch=epoch+1,epochs=cfg['epochs'],
                    step=step+1,steps=plan.steps,loss=stats['loss'],raw_ce=stats['raw_ce_sum']/stats['views'],
                    seconds_per_step=(totals['compute_seconds']+totals['wait_seconds'])/totals['updates'],
                    eta=(plan.steps-step-1)*(totals['compute_seconds']+totals['wait_seconds'])/totals['updates']))
                if (run/'stop_requested.json').is_file():
                    state['epoch_accumulator']=_plain(acc)
                    return acknowledge_pause(run,cfg,state,model,optimizer,dev,partition)
                start=time.monotonic()
            if int(totals['updates'])!=plan.steps:
                raise RuntimeError('Incomplete logical epoch')
            if epoch+1==cfg['warm_epochs'] and cfg['consistency']=='dual':
                model.pcl_scale.freeze()
                atomic_json(run/'pcl_scale_calibration.json',model.pcl_scale.describe())
            tag=f'epoch_{epoch+1}_step_{plan.steps}';evaluation_seconds={}
            def timed_infer(name,*args):
                atomic_json(run/'progress.json',dict(status='evaluating',phase=name,epoch=epoch+1,epochs=cfg['epochs'],step=plan.steps,steps=plan.steps,loss=totals['loss']/plan.steps))
                began=time.monotonic();result=infer(*args);evaluation_seconds[name]=time.monotonic()-began
                return result
            logits,_=timed_infer('dev',model,dev,cfg)
            train_logits,probe_rows=timed_infer('train_probe',model,train,cfg,probes)
            metrics=measure(dev,logits);selected_metrics=measure(selection_rows,logits[selection])
            current=partial_state(model);epoch_logits=torch.from_numpy(logits.copy())
            updated=promote(state,tag,current,selected_metrics,epoch_logits)
            training=dict(loss=totals['loss']/plan.steps,raw_ce=totals['raw_ce_sum']/totals['views'],
                compute_seconds=totals['compute_seconds']/plan.steps,wait_seconds=totals['wait_seconds']/plan.steps,
                views=int(totals['views']),cells={k:dict(v) for k,v in acc['cells'].items()},
                processing_counts=dict(acc['processing']),base_source_view_counts=dict(acc['base_views']),
                mean_unique_sources_per_update=totals['unique_sources_per_update']/plan.steps,
                en_real_confidence=dict(acc['confidence']),
                pcl_variance=dict(acc['variance'],coverage={k:dict(v) for k,v in acc['variance_coverage'].items()}),
                pcl_scale=model.pcl_scale.describe(),
                auxiliary=dict(mean_per_update={k:totals[k]/plan.steps for k in ('classification_loss','encoded_views','auxiliary_reference_views')},
                    losses={k:dict(v) for k,v in acc['pcl_totals'].items()},coverage={k:dict(v) for k,v in acc['pcl_coverage'].items()}),
                cuda_peak_memory=None)
            entry=dict(epoch=epoch+1,tag=tag,phase=phase,metrics=metrics,selection_metrics=selected_metrics,
                train=summarize(probe_rows,train_logits),dev=summarize(dev,logits),training=training,
                promoted=updated,fusion=model.fusion.describe())
            if (epoch+1)%cfg['diagnostic_interval']==0 or epoch+1==cfg['epochs']:
                if panel:
                    pz,pr=timed_infer('mechanisms',model,panel['rows'],cfg,panel['tickets'],'dev')
                    entry['panel']=summarize(pr,pz)
                sz,sr=timed_infer('six_scenes',model,scenes,cfg,scene_tickets,'dev')
                entry['scenes']=summarize(sr,sz);np.savez_compressed(run/f'scene_scores_{tag}.npz',logits=sz)
            if cfg['device'].startswith('cuda'):
                training['cuda_peak_memory']=dict(allocated_gib=torch.cuda.max_memory_allocated(cfg['device'])/1024**3,
                    reserved_gib=torch.cuda.max_memory_reserved(cfg['device'])/1024**3,scope='training and evaluation since process epoch start')
            entry['fit']=fit_state([e for e in state['history'] if e.get('phase')!='parent_baseline']+[entry])
            state['history'].append(entry);state.update(epoch=epoch+1,next_step=0,epoch_accumulator=None,last_tag=tag,last_logits=epoch_logits)
            np.savez_compressed(run/f'dev_scores_{tag}.npz',logits=logits)
            atomic_json(run/f'sampling_and_loss_{tag}.json',dict(coverage=coverage,actual=training))
            checkpoint_start=time.monotonic();save(run,cfg,state,model,optimizer,current);checkpoint_seconds=time.monotonic()-checkpoint_start
            if epoch+1==cfg['warm_epochs'] and not cfg.get('warm_start'):
                atomic_save(run/'warmup.pt',state,cfg['free_reserve_bytes']+cfg['disk_margin_bytes'])
            print_epoch(entry,state);render(run,state['history'],state['steps'])
            atomic_json(run/f'performance_{tag}.json',dict(epoch=epoch+1,train_compute_seconds=totals['compute_seconds'],
                train_wait_seconds=totals['wait_seconds'],evaluation_seconds=evaluation_seconds,checkpoint_write_seconds=checkpoint_seconds,
                epoch_wall_seconds_this_process=time.monotonic()-epoch_start,
                note='Resumed epochs accumulate compute/wait from both processes; wall time is this process only'))
            if (run/'stop_requested.json').is_file():
                return acknowledge_pause(run,cfg,state,model,optimizer,dev,partition,saved=True)
    finally:
        close(batches)
    if state['best_tag'] is None:
        raise RuntimeError('No completed V3.18.8 epoch')
    checksum=calibrations(run,state,dev,partition)
    done=dict(version='3.18.8',recipe=cfg['recipe'],pooling=cfg['pooling'],status='complete',best_tag=state['best_tag'],last_tag=state['last_tag'],
        completed_epochs=state['epoch'],planned_epochs=cfg['epochs'],checkpoint_sha256=checksum,
        baseline_fallback=state['best_tag'].startswith('parent_v3187_'),
        best_origin='v3187_parent' if state['best_tag'].startswith('parent_v3187_') else 'v3188_trained',
        selection=cfg['selection'],initialization=cfg['initialization'])
    atomic_json(run/'completed.json',done);atomic_json(run/'progress.json',dict(done,phase='complete'))
    print(f'[Complete] V3.18.8 epochs={state["epoch"]}/{cfg["epochs"]}; best={state["best_tag"]}; origin={done["best_origin"]}',flush=True)
    return done
