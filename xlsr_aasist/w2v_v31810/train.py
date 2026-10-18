"""Fresh detector, transactional first-order tasks and one locked final audit."""
from collections import defaultdict
from pathlib import Path
import time
import shutil
import torch
from .common import (schema_for,atomic_json,read_json,digest,fingerprint,seed_all,partial_state,apply_partial,
    atomic_save,to_cpu,capture_rng,restore_rng,append_json)
from .model import load_model,optimizer_for
from .sampling import Plan
from .data import Waves,TicketSampler,loader,close
from .meta import train_step
from .schedule import schedule
from .precision import scaler_for
from .monitor import print_epoch,render
from .validation import evaluate_protocol
from w2v_v3184.recovery import trim_step_log


def save(run,cfg,state,model,optimizer,current=None):
    state.update(model=partial_state(model) if current is None else current,
        optimizer=to_cpu(optimizer.state_dict()),grad_scaler=scaler_for(optimizer,cfg).state_dict(),rng=capture_rng())
    if state.get('next_step',0)==0 and state.get('best_tag') is not None and state['best_tag']==state['last_tag']:
        state['best_model']=state['model']
    atomic_save(Path(run)/'last.pt',state,cfg.get('free_reserve_bytes',0)+cfg.get('disk_margin_bytes',0))
    atomic_json(Path(run)/'training_history.json',state['history'])


def acknowledge_pause(run,cfg,state,model,optimizer,*,saved=False):
    if not saved:save(run,cfg,state,model,optimizer)
    paused=dict(status='paused',version='3.18.10',recipe=cfg['recipe'],completed_epochs=state['epoch'],
        next_step=state['next_step'],best_tag=state['best_tag'],last_tag=state['last_tag'],checkpoint_sha256=digest(run/'last.pt'))
    atomic_json(run/'paused.json',paused);atomic_json(run/'progress.json',dict(paused,phase='Paused at a saved outer-update boundary'))
    print('PAUSED_RUN='+str(run),flush=True)
    return paused


def _progress(run,cfg,epoch,step,steps,phase,**extra):
    atomic_json(run/'progress.json',dict(status='training',phase=phase,epoch=epoch,epochs=cfg['epochs'],step=step,steps=steps,**extra))


def _flatten(result):
    p=result['panels']
    return dict(online_f1=p['official_online']['macro_f1'],regular_f1=p['regular_noisy']['macro_f1'],
        heldout_f1=p['heldout_noisy']['macro_f1'],score=result['score'])


def _evaluate(evaluator,model,protocol,cfg,run,split,tag):
    kwargs=dict(split=split)
    if split=='audit':kwargs['selected_checkpoint']=tag
    if evaluator is evaluate_protocol:
        name=f'selection_scores_{tag}.npz' if split=='select' else 'locked_audit_scores.npz'
        kwargs.update(predictions_path=Path(run)/name,checkpoint_tag=tag)
    return evaluator(model,protocol,cfg,**kwargs)


def run_experiment(cfg,run,train,dev,protocol,model_factory=load_model,baseline_only=False,evaluator=evaluate_protocol):
    run=Path(run);seed_all(cfg['seed'])
    if fingerprint(protocol)!=cfg['validation_protocol_hash']:raise ValueError('Locked validation protocol changed')
    request=run/'stop_requested.json';paused=run/'paused.json'
    if paused.is_file() and request.is_file() and request.stat().st_mtime_ns<=paused.stat().st_mtime_ns:request.unlink()
    if model_factory is load_model:
        from .interface_smoke import main
        main(cfg['device'],cfg['backend_precision'])
    _progress(run,cfg,0,0,0,'Loading official Omni; initializing new detector')
    model=model_factory(cfg);optimizer=optimizer_for(model,cfg)
    if hasattr(model,'pcl_scale'):raise ValueError('V3.18.10 must not instantiate PCL state')
    if shutil.disk_usage(run).free<cfg.get('free_reserve_bytes',0)+1024**3:raise OSError('Need free-space reserve plus 1 GiB for checkpoints')
    plan=Plan(train,cfg=cfg,seed=cfg['seed'])
    state=dict(schema=schema_for(cfg),identity=fingerprint(cfg),epoch=0,next_step=0,cursor=0,alpha=0. if cfg['experiment']=='E0' else None,
        epoch_accumulator=None,best_tag=None,last_tag=None,best_model=None,best_score=None,history=[],steps=[])
    if (run/'last.pt').is_file():
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if state.get('schema')!=schema_for(cfg) or state.get('identity')!=fingerprint(cfg):raise ValueError('Resume configuration changed')
        apply_partial(model,state['model']);optimizer.load_state_dict(state['optimizer'])
        if state.get('grad_scaler'):scaler_for(optimizer,cfg).load_state_dict(state['grad_scaler'])
        restore_rng(state['rng'])
        print(f'[Resume] epoch={state["epoch"]} next_step={state["next_step"]} alpha={state["alpha"]}; no virtual parameters loaded',flush=True)
    else:
        adapters=[p for n,p in model.named_parameters() if n.endswith('.lora_b')]
        if any(bool(torch.count_nonzero(p)) for p in adapters):raise ValueError('Fresh LoRA B must be zero')
        inventory=dict(initialization=cfg['initialization'],experiment=cfg['experiment'],zero_lora_b=len(adapters),
            trainable_parameters=sum(p.numel() for n,p in model.named_parameters() if p.requires_grad or n.endswith(('.lora_a','.lora_b'))),
            classification_dim=model.head.aasist.out_layer.in_features,pcl_state=False,
            backend_precision=cfg.get('backend_precision','fp32'),trainable_snapshot_only=True,
            optimizer_groups=[dict(name=g['name'],lr=g['initial_lr']) for g in optimizer.param_groups])
        atomic_json(run/'parameter_inventory.json',inventory)
        save(run,cfg,state,model,optimizer)
    trim_step_log(run/'steps.jsonl',state['cursor'])
    if baseline_only:
        print('INITIALIZATION_VERIFIED_RUN='+str(run),flush=True)
        return dict(status='initialization_verified',version='3.18.10')
    paused.unlink(missing_ok=True)
    if request.is_file():return acknowledge_pause(run,cfg,state,model,optimizer)
    sampler=TicketSampler();batches=None
    try:
        for epoch in range(state['epoch'],cfg['epochs']):
            joint=epoch>=cfg['warm_epochs'];model.set_phase(joint);model.train()
            if joint and state['alpha'] is None:
                from .calibration import calibrate
                _progress(run,cfg,epoch+1,0,12,'Train-only inner-step calibration')
                def task_factory():
                    dataset=Waves(train,cfg)
                    for i,tickets in enumerate(plan.calibration_batches()):
                        _progress(run,cfg,epoch+1,i,12,'Train-only inner-step calibration')
                        yield [dataset[t] for t in tickets]
                try:calibration=calibrate(model,optimizer,task_factory,cfg)
                except Exception as exc:
                    atomic_json(run/'inner_calibration_failure.json',dict(error=str(exc),details=getattr(exc,'report',None),
                        retained_checkpoint=str(run/'last.pt'),policy='No silent zero-step fallback'))
                    raise
                state['alpha']=float(calibration['selected_alpha']);state['calibration']=calibration
                atomic_json(run/'inner_calibration.json',calibration)
                save(run,cfg,state,model,optimizer)
                if request.is_file():return acknowledge_pause(run,cfg,state,model,optimizer,saved=True)
            if batches is None:batches=loader(Waves(train,cfg),cfg,batch_sampler=sampler)
            coverage=plan.coverage(epoch);atomic_json(run/f'coverage_epoch_{epoch+1}.json',coverage)
            begin=state.get('next_step',0);sampler.batches=list(plan.batches(epoch))[begin:]
            phase='joint' if joint else 'head_warmup'
            print(f'[Train] V3.18.10 epoch {epoch+1}/{cfg["epochs"]} phase={phase} updates={plan.steps}',flush=True)
            totals=defaultdict(float,state.get('epoch_accumulator') or {});start=time.monotonic()
            if cfg['device'].startswith('cuda'):torch.cuda.reset_peak_memory_stats(cfg['device'])
            for step,units in enumerate(batches,start=begin):
                wait=time.monotonic()-start;joint,beta=schedule(optimizer,cfg,epoch,step,plan.steps)
                compute=time.monotonic();audit=(step%cfg.get('gradient_audit_interval',200)==0)
                try:
                    overflow_attempts=[]
                    for attempt in range(4):
                        stats=train_step(model,optimizer,units,cfg,alpha=state['alpha'] or 0.,beta=beta,joint=joint,audit=audit)
                        if not stats.get('amp',{}).get('skipped_update',False):break
                        overflow_attempts.append(dict(stats['amp'],encoded_views=stats.get('encoded_views',0)))
                    else:
                        atomic_json(run/'precision_failure.json',dict(epoch=epoch+1,step=step+1,attempts=overflow_attempts,
                            retained_checkpoint=str(run/'last.pt'),policy='Failed tasks never consume the successful update budget'))
                        raise FloatingPointError('Repeated mixed-precision overflow on the same task; inspect precision_failure.json')
                    stats['amp_retry_attempts']=overflow_attempts
                    stats['encoded_views']+=sum(item['encoded_views'] for item in overflow_attempts)
                except torch.OutOfMemoryError:
                    atomic_json(run/'oom_context.json',dict(epoch=epoch+1,step=step+1,microbatch=cfg['microbatch'],
                        frame_budget=cfg['frame_budget'],last_encode=getattr(model,'last_encode_geometry',None),
                        note='Virtual transaction must be restored; last.pt is the last committed saved boundary'))
                    raise
                seconds=time.monotonic()-compute
                for key in ('loss','classification_loss','views','encoded_views'):totals[key]+=stats.get(key,0.)
                totals['compute_seconds']+=seconds;totals['wait_seconds']+=wait;totals['updates']+=1
                totals['amp_retry_attempts']+=len(overflow_attempts)
                state['cursor']+=1;state['next_step']=step+1;state['epoch_accumulator']=dict(totals)
                append_json(run/'steps.jsonl',dict(update=state['cursor'],epoch=epoch+1,step=step+1,**stats,compute_seconds=seconds,wait_seconds=wait))
                if audit or (step+1)%100==0 or step+1==plan.steps:
                    state['steps'].append(dict(update=state['cursor'],loss=stats['loss'],delta_B=stats.get('meta',{}).get('delta_B')))
                mean=(totals['compute_seconds']+totals['wait_seconds'])/totals['updates']
                _progress(run,cfg,epoch+1,step+1,plan.steps,phase,loss=stats['loss'],eta=(plan.steps-step-1)*mean)
                if request.is_file():return acknowledge_pause(run,cfg,state,model,optimizer)
                start=time.monotonic()
            if totals['updates']!=plan.steps:raise RuntimeError('Incomplete logical epoch')
            tag=f'epoch_{epoch+1}_step_{plan.steps}'
            _progress(run,cfg,epoch+1,plan.steps,plan.steps,'Selection evaluation; no temporary adaptation')
            before=time.monotonic();result=_evaluate(evaluator,model,protocol,cfg,run,'select',tag);eval_seconds=time.monotonic()-before
            score=float(result['score']);current=partial_state(model)
            improved=state['best_score'] is None or score>state['best_score']
            if improved:state.update(best_tag=tag,best_score=score,best_model=current)
            training=dict(loss=totals['loss']/plan.steps,views=int(totals['views']),encoded_views=int(totals['encoded_views']),
                compute_seconds=totals['compute_seconds']/plan.steps,wait_seconds=totals['wait_seconds']/plan.steps,
                data_wait_fraction=totals['wait_seconds']/max(1e-9,totals['wait_seconds']+totals['compute_seconds']),
                amp_retry_attempts=int(totals['amp_retry_attempts']),amp_skipped_updates=0,alpha=state['alpha'],beta=beta)
            if cfg['device'].startswith('cuda'):training['cuda_peak_memory']=dict(allocated_gib=torch.cuda.max_memory_allocated(cfg['device'])/1024**3,reserved_gib=torch.cuda.max_memory_reserved(cfg['device'])/1024**3)
            entry=dict(epoch=epoch+1,tag=tag,phase=phase,metrics=_flatten(result),selection=result,
                dev=dict(groups=result.get('groups',{})),training=training,promoted=improved,fusion=model.fusion.describe())
            state['history'].append(entry);state.update(epoch=epoch+1,next_step=0,epoch_accumulator=None,last_tag=tag)
            atomic_json(run/f'selection_{tag}.json',result)
            save(run,cfg,state,model,optimizer,current);print_epoch(entry,state);render(run,state['history'],state['steps'])
            atomic_json(run/f'performance_{tag}.json',dict(train_compute_seconds=totals['compute_seconds'],train_wait_seconds=totals['wait_seconds'],selection_seconds=eval_seconds))
            if request.is_file():return acknowledge_pause(run,cfg,state,model,optimizer,saved=True)
    finally:
        if batches is not None:close(batches)
    if state['best_tag'] is None:raise RuntimeError('No completed checkpoint selected')
    checksum=digest(run/'last.pt');audit_path=run/'locked_audit.json'
    audit_identity=dict(checkpoint_sha256=checksum,selected=state['best_tag'],protocol_sha256=cfg['validation_protocol_hash'])
    previous=read_json(audit_path) if audit_path.is_file() else None
    if not previous or previous.get('identity')!=audit_identity:
        atomic_json(run/'progress.json',dict(status='evaluating_final',phase='Locked audit of selected best',epoch=state['epoch'],epochs=cfg['epochs']))
        apply_partial(model,state['best_model'])
        audit=_evaluate(evaluator,model,protocol,cfg,run,'audit',state['best_tag'])
        atomic_json(audit_path,dict(identity=audit_identity,metrics=audit,used_for_selection=False))
    done=dict(status='complete',version='3.18.10',recipe=cfg['recipe'],experiment=cfg['experiment'],best_tag=state['best_tag'],last_tag=state['last_tag'],
        completed_epochs=state['epoch'],planned_epochs=cfg['epochs'],checkpoint_sha256=checksum,best_origin='fresh_official_omni',threshold=.5,
        inner_alpha=state['alpha'],selection=cfg['selection'],initialization=cfg['initialization'])
    atomic_json(run/'completed.json',done);atomic_json(run/'progress.json',dict(done,phase='Training complete; preparing requested artifacts'))
    print(f'[Complete] V3.18.10 epochs={state["epoch"]}/{cfg["epochs"]}; best={state["best_tag"]}; selected by locked protocol',flush=True)
    return done
