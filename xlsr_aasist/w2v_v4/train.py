"""One shared detector; ordinary updates, finite auxiliary weights, safe resume."""
from collections import defaultdict
from pathlib import Path
import signal
import time
import torch
from .common import (seed_all,partial_state,apply_partial,atomic_save,atomic_json,read_json,
    fingerprint,digest,capture_rng,restore_rng,to_cpu,append_json)
from .model import load_model,optimizer_for
from w2v_v31810.data import loader,TicketSampler,close
from w2v_v31810.precision import scaler_for
from .model import schedule
from .validation import evaluate_protocol
from w2v_v3184.recovery import trim_step_log
from w2v_v31813.data import Waves
from w2v_v31813.pairing import Plan
from .step import train_step,calibrate
from w2v_v31813.diagnostics import compare
from . import SCHEMA,VERSION


def save(run,cfg,state,model,optimizer):
    state.update(model=partial_state(model),optimizer=to_cpu(optimizer.state_dict()),
                 grad_scaler=scaler_for(optimizer,cfg).state_dict(),rng=capture_rng())
    atomic_save(run/'last.pt',state,cfg.get('free_reserve_bytes',0))
    atomic_json(run/'training_history.json',state['history'])


def print_epoch(entry,epochs):
    print(f'\nEpoch {entry["epoch"]}/{epochs} completed'+(' *' if entry['promoted'] else ''),flush=True)
    p=entry['selection']['panels'];t=entry['training']
    print('Online Regular Heldout SelectS (EN/ZH equal) | CE Prior PCL',flush=True)
    print(' '.join(f'{100*p[k].get("language_macro_f1",p[k]["macro_f1"]):.2f}' for k in ('official_online','regular_noisy','heldout_noisy'))+
          f' {100*entry["selection"]["score"]:.2f} | {t["ce"]:.5f} {t["prior"]:.5f} {t["pcl"]:.5f}',flush=True)
    print('Group                       FakeR  RealR     F1    AUC',flush=True)
    for name,m in entry['selection']['groups'].items():
        print(f'{name:26} '+ ' '.join(f'{100*m[k]:6.2f}' for k in ('fake_recall','real_recall','macro_f1','auc')),flush=True)


def run_experiment(cfg,run,rows,protocol,model_factory=load_model,optimizer_factory=optimizer_for,evaluator=evaluate_protocol):
    run=Path(run);seed_all(cfg['seed']);model=model_factory(cfg);optimizer=optimizer_factory(model,cfg)
    request=run/'stop_requested.json';paused=run/'paused.json'
    if paused.exists() and request.exists() and request.stat().st_mtime_ns<=paused.stat().st_mtime_ns:request.unlink()
    plan=Plan(rows,cfg,read_json(run/'xeus_neighbors.json'),read_json(run/'original_phone_pairs.json')['pairs'])
    def evaluate(split,tag):
        kwargs=dict(split=split,checkpoint_tag=tag,predictions_path=run/f'{split}_{tag}.npz')
        if split=='audit':kwargs['selected_checkpoint']=tag
        rng=capture_rng()
        try:return evaluator(model,protocol,cfg,**kwargs)
        finally:restore_rng(rng)
    state=dict(schema=SCHEMA,identity=fingerprint(cfg),epoch=0,next_step=0,cursor=0,history=[],totals={},weights=None)
    if (run/'last.pt').exists():
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if state['schema']!=SCHEMA or state['identity']!=fingerprint(cfg):raise ValueError('Resume configuration changed')
        apply_partial(model,state['model']);optimizer.load_state_dict(state['optimizer'])
        scaler_for(optimizer,cfg).load_state_dict(state['grad_scaler']);restore_rng(state['rng'])
    else:
        apply_partial(model,torch.load(run/'init.pt',map_location='cpu',weights_only=True),parent=True)
        if cfg.get('paired_probe_per_cell',0):
            from .probe import paired_probe
            print('V4 paired Train gradient probe (no optimizer updates)',flush=True)
            atomic_json(run/'paired_gradient_probe.json',paired_probe(model,rows,cfg,plan))
        baseline=evaluate('select','parent');atomic_json(run/'parent_selection.json',baseline)
        state.update(best_model=partial_state(model),best_score=baseline['score'],best_tag='parent',last_tag='parent')
        save(run,cfg,state,model,optimizer)
    trim_step_log(run/'steps.jsonl',state['cursor'])
    handlers={}
    def stopping(signum,_frame):
        atomic_json(request,dict(signal=signum,requested_at=time.time()));print('\nStop requested; finishing this optimizer update before saving.',flush=True)
    for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,stopping)
    def pause():
        save(run,cfg,state,model,optimizer)
        done=dict(status='paused',version=VERSION,best_tag=state['best_tag'],epoch=state['epoch'],next_step=state['next_step'],checkpoint_sha256=digest(run/'last.pt'))
        atomic_json(paused,done);print('PAUSED_RUN='+str(run),flush=True);return done
    sampler=TicketSampler();batches=None
    try:
        for epoch in range(state['epoch'],cfg['epochs']):
            if request.exists():return pause()
            print(f'\nEpoch {epoch+1}: '+('CE + PCGrad; XEUS/PCL start in epoch 2' if epoch==0 else 'CE + PCGrad + ramped XEUS/PCL' if epoch<3 else 'CE + PCGrad + full XEUS/PCL'),flush=True)
            model.set_phase(epoch>=1);model.train();tickets=plan.epoch_batches(epoch)
            atomic_json(run/f'coverage_epoch_{epoch+1}.json',plan.audit13)
            if epoch>=1 and state['weights'] is None:
                dataset=Waves(rows,cfg)
                # Six batches evenly spaced through this fixed Train epoch,
                # including scarce cap-limited anchors; no Dev/calibration reuse.
                indices=[i for i,t in enumerate(tickets) if any('official_pcl' in v for v in t) and
                    (cfg['mode']=='baseline' or any('xeus_pair' in v for v in t))]
                indices=indices[::max(1,len(indices)//6)][:6]
                if not indices:raise ValueError('No usable Train calibration tasks')
                calibration=calibrate(model,([dataset[t] for t in tickets[i]] for i in indices),cfg)
                state['weights']=calibration['weights'];atomic_json(run/'auxiliary_calibration.json',calibration)
                save(run,cfg,state,model,optimizer)
            if batches is None:batches=loader(Waves(rows,cfg),cfg,batch_sampler=sampler)
            begin=state['next_step'];sampler.batches=tickets[begin:];totals=defaultdict(float,state['totals']);start=time.monotonic()
            for step,units in enumerate(batches,start=begin):
                schedule(optimizer,cfg,epoch,step,plan.steps);rng=capture_rng()
                for attempt in range(4):
                    restore_rng(rng)
                    stats=train_step(model,optimizer,units,cfg,state['weights'] or dict(prior=0.,pcl=0.),epoch,step,plan.steps)
                    if not stats['amp']['skipped_update']:break
                else:raise FloatingPointError('Repeated FP16 overflow; this task was not committed')
                for key in ('ce','prior','pcl','loss','views','pcl_sources','xeus_pairs'):totals[key]+=stats[key]
                totals['updates']+=1
                totals['pcgrad_applied']+=int(stats['coordination']['applied'])
                totals['clean_gradient_norm']+=stats['coordination']['clean_norm'] or 0.
                totals['noisy_gradient_norm']+=stats['coordination']['noisy_norm'] or 0.
                state.update(cursor=state['cursor']+1,next_step=step+1,totals=dict(totals))
                append_json(run/'steps.jsonl',dict(update=state['cursor'],epoch=epoch+1,step=step+1,**stats))
                if step%10==0 or step+1==plan.steps:
                    elapsed=time.monotonic()-start;eta=elapsed/max(1,step+1-begin)*(plan.steps-step-1)
                    print(f'\rEpoch {epoch+1}/{cfg["epochs"]} {step+1}/{plan.steps} {(step+1)/plan.steps:6.1%} | loss={stats["loss"]:.5f} CE={stats["ce"]:.5f} X={stats["prior"]:.5f} PCL={stats["pcl"]:.5f} pairs={stats["xeus_pairs"]} PCGrad={int(stats["coordination"]["applied"])} ETA={eta/60:.1f}m',end='',flush=True)
                if request.exists():return pause()
                if (step+1)%200==0:save(run,cfg,state,model,optimizer)
            if totals['updates']!=plan.steps:raise RuntimeError('Epoch did not complete its exposure budget')
            tag=f'epoch_{epoch+1}';result=evaluate('select',tag);improved=result['score']>state['best_score']
            if improved:state.update(best_tag=tag,best_score=result['score'],best_model=partial_state(model))
            entry=dict(epoch=epoch+1,tag=tag,selection=result,training={k:v/plan.steps for k,v in totals.items()},promoted=improved)
            state['history'].append(entry);state.update(epoch=epoch+1,next_step=0,totals={},last_tag=tag)
            atomic_json(run/f'selection_{tag}.json',result);save(run,cfg,state,model,optimizer);print_epoch(entry,cfg['epochs'])
            if (run/f'select_{tag}.npz').exists():atomic_json(run/f'diagnostic_{tag}.json',compare(run/'select_parent.npz',run/f'select_{tag}.npz'))
        if request.exists():return pause()
        apply_partial(model,state['best_model'])
        audit=evaluate('audit',state['best_tag']);atomic_json(run/'locked_audit.json',dict(selected=state['best_tag'],used_for_selection=False,metrics=audit))
        done=dict(status='complete',version=VERSION,best_tag=state['best_tag'],completed_epochs=state['epoch'],checkpoint_sha256=digest(run/'last.pt'),
                  baseline_retained=state['best_tag']=='parent',threshold=.5)
        atomic_json(run/'completed.json',done);print('\nV4 complete; best='+state['best_tag'],flush=True);return done
    finally:
        if batches is not None:close(batches)
        for sig,handler in handlers.items():signal.signal(sig,handler)
