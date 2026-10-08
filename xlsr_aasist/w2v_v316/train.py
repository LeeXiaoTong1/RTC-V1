"""Train a new Omni detector; historical V3.15 predictions are a fixed reference."""
import math
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch

from live_progress import Phase
from w2v_v39.common import atomic_json,read_json,digest,announce
from w2v_v39.metrics import change_audit
from w2v_v315.train import acceptance
from .config import verify_inputs
from .data import SourcePlan,TrainingLoader,bundles
from .model import load_model,optimizer_for
from .objectives import TFCL
from .step import train_step
from .performance import select_execution,probe_examples
from .inference import infer
from .metrics import measure,print_metrics
from .state import (SCHEMA,identity,partial_state,apply_partial,to_cpu,atomic_save,
    capture_rng,restore_rng,load_resume,budget,choose)


def schedule(optimizer,cursor,total,warmup,cfg):
    joint=cursor>=warmup
    progress=max(0.,(cursor-warmup)/max(1,total-warmup))
    cosine=.1+.9*.5*(1+math.cos(math.pi*min(1.,progress)))
    for group in optimizer.param_groups:
        ramp=min(1.,(cursor+1)/50) if group['name']=='head' else min(1.,max(0,cursor-warmup+1)/100)
        group['lr']=group['initial_lr']*cosine*ramp
    return joint


def run_experiment(cfg,run):
    run=Path(run); verify_inputs(cfg)
    torch.set_num_threads(1); torch.cuda.set_device(torch.device(cfg['device']))
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 GPU required')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    model=load_model(cfg)
    auxiliary=TFCL(128,cfg['tfcl_heads'],cfg['tfcl_bins']).to(cfg['device'])
    optimizer=optimizer_for(model,auxiliary,cfg)
    storage=budget(model,auxiliary,cfg); free=shutil.disk_usage(run).free
    atomic_json(run/'storage_budget.json',dict(storage,free_bytes=free))
    if free<storage['required_free_bytes']:
        raise OSError('Insufficient free space for atomic partial weights plus 10 GiB reserve')
    design=dict(backbone='omniASR_W2V_7B SSL',lora=model.lora_inventory,feature_layers=cfg['feature_layers'],
        new_head=True,old_detector_weights_imported=False,
        trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad},
        source_batch=cfg['source_batch'],views=3,ce_mass_final=[.3,.1,.6],group_ce_mass=.25,
        max_epochs=cfg['epochs'],feature_cache_bytes=0,audio_cache_bytes=0,
        no_audio_truncation=True,auxiliary_only_structure_bins=cfg['tfcl_bins'],detach_reference=False)
    atomic_json(run/'training_design.json',design)
    with bundles(cfg) as (train,dev):
        rows=dev['rows']
        if rows!=read_json(cfg['reference_rows']): raise ValueError('V3.15/V3.16 Dev row order differs')
        with np.load(cfg['reference_scores'],allow_pickle=False) as z: original=z['logits'].copy()
        reference=measure(rows,original,cfg['matched_fake_recall'])
        atomic_json(run/'dev_rows.json',rows); atomic_json(run/'reference_metrics.json',reference)
        np.savez_compressed(run/'dev_scores_reference.npz',logits=original)
        print_metrics('V3.15 reference (not Omni initialization)',reference)
        plan=SourcePlan(train['rows'],cfg['source_batch'],cfg['seed'])
        total=plan.steps*cfg['epochs']; warmup=min(cfg['head_warmup_updates'],max(1,plan.steps//4))
        pool=TrainingLoader(plan,cfg,run)
        try:
            # Profile rollback includes adapters, head, auxiliary, Adam, RNG.
            sample=None
            if not (run/'execution_plan.json').exists() and cfg['autotune']:
                sample=probe_examples(plan,cfg,run)
            execution=select_execution(model,auxiliary,optimizer,sample,cfg,run)
            effective=dict(cfg,**execution)
            if (run/'last.pt').exists():
                state=load_resume(run,cfg)
                apply_partial(model,state['model']); auxiliary.load_state_dict(state['auxiliary'])
                optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng'])
            else:
                state=dict(schema=SCHEMA,identity=identity(cfg),cursor=0,history=[],candidates={},
                    selections=dict(best_weighted=None,best_guarded=None),scores=dict(best_weighted=-1.,best_guarded=-1.),
                    stale=0,progress_best=-1.,stop=False,last_tag='untrained',execution_sha256=digest(run/'execution_plan.json'))
                state.update(model=partial_state(model),auxiliary=to_cpu(auxiliary.state_dict()),
                    optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng())
                atomic_save(run/'last.pt',state,cfg['free_reserve_bytes'])
            atomic_json(run/'training_history.json',state['history'])
            while state['cursor']<total and not state['stop']:
                epoch,within=divmod(state['cursor'],plan.steps)
                boundaries=sorted({math.ceil(plan.steps/2),plan.steps}); end=next(b for b in boundaries if b>within)
                atomic_json(run/f'sampling_epoch_{epoch+1}.json',plan.coverage(epoch))
                announce(f'V3.16 epoch {epoch+1}/{cfg["epochs"]} steps {within+1}-{end}')
                phase=Phase(f'V3.16 epoch {epoch+1}',end-within)
                start=within; started=ready=time.perf_counter(); waiting=computing=0.; stats={}; sums=[]
                torch.cuda.reset_peak_memory_stats()
                for examples in pool.segment(epoch,within,end):
                    now=time.perf_counter(); waiting+=now-ready
                    cursor=state['cursor']
                    joint=schedule(optimizer,cursor,total,warmup,cfg)
                    model.set_phase(joint,effective['checkpointing'])
                    ramp=min(1.,max(0,cursor-warmup+1)/max(1,plan.steps*cfg['objective_ramp_epochs']))
                    torch.cuda.synchronize(); began=time.perf_counter()
                    stats=train_step(model,auxiliary,optimizer,examples,effective,ramp,audit=(within==start and joint))
                    torch.cuda.synchronize(); computing+=time.perf_counter()-began
                    state['cursor']+=1; within+=1; sums.append(stats['total_loss'])
                    phase.update(within-start,loss=stats['total_loss'])
                    if within==start+1 or within%100==0 or within==end:
                        count=within-start
                        print(f'  STEP {within}/{plan.steps} phase={"lora_tfcl" if joint else "new_head_warmup"} '
                            f'TOTAL={stats["total_loss"]:.6f} CE={stats["classification_loss"]:.6f} '
                            f'time_raw={stats["time_loss"]:.5f} time_weighted={stats["weighted_time"]:.6f} '
                            f'CKA_raw={stats["structure_loss"]:.5f} CKA_weighted={stats["weighted_structure"]:.6f} '
                            f'max_CE={stats["maximum_example_ce"]:.3f} pairs={stats["eligible_pairs"]}/16 '
                            f'compute_s={computing/count:.2f} wait_s={waiting/count:.2f} '
                            f'peak_alloc_GiB={torch.cuda.max_memory_allocated()/1024**3:.2f} '
                            f'peak_reserved_GiB={torch.cuda.max_memory_reserved()/1024**3:.2f}',flush=True)
                        if stats['aux_gradient_audit']: print('  TFCL_GRADIENT_REACH='+str(stats['aux_gradient_audit']),flush=True)
                    ready=time.perf_counter()
                if within!=end: raise RuntimeError('Training segment incomplete')
                tag=f'epoch_{epoch+1}_step_{within}'
                logits=infer(model,rows,effective,'V3.16 '+tag+' fixed Dev')
                value=measure(rows,logits,cfg['matched_fake_recall'])
                eligible,reasons=acceptance(reference,value,cfg)
                current=partial_state(model)
                promoted=choose(state,tag,value,eligible,current)
                if value['weighted_f1']>=state['progress_best']+cfg['progress_min_delta']:
                    state['stale']=0; state['progress_best']=value['weighted_f1']
                else: state['stale']+=1
                # A newly initialized detector cannot use a 2% drop from the OLD
                # model as an early-abort rule; allow at least two complete epochs.
                state['stop']=state['cursor']>=cfg['minimum_epochs']*plan.steps and state['stale']>=cfg['patience']
                entry=dict(tag=tag,cursor=state['cursor'],metrics=value,eligible=eligible,reasons=reasons,
                    promoted=promoted,committed=True,changed=change_audit(rows,original,logits),last_training_step=stats,
                    seconds=time.perf_counter()-started,data_wait_seconds=waiting,compute_seconds=computing,
                    mean_total_loss=float(np.mean(sums)),checks_without_progress=state['stale'],stop=state['stop'])
                state['history'].append(entry); state['last_tag']=tag
                state.update(model=current,auxiliary=to_cpu(auxiliary.state_dict()),optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng())
                atomic_json(run/'validation_pending.json',entry)
                np.savez_compressed(run/('dev_scores_'+tag+'.npz'),logits=logits)
                announce('V3.16 committing LoRA/head/optimizer; immutable 7B base referenced')
                atomic_save(run/'last.pt',state,cfg['free_reserve_bytes'])
                atomic_json(run/'training_history.json',state['history']); (run/'validation_pending.json').unlink(missing_ok=True)
                print_metrics(tag,value)
                print('  guarded_eligible='+str(eligible)+'; reasons='+','.join(reasons)+'; selections='+str(state['selections']),flush=True)
                from .workflow import report
                report(run)
            verify_inputs(cfg)
            done=dict(version='3.16',status='complete',checkpoint_sha256=digest(run/'last.pt'),
                omni_sha256=cfg['omni_sha256'],selections=state['selections'],last_tag=state['last_tag'],
                committed_updates=state['cursor'],planned_updates=total,early_stopped=state['cursor']<total,
                reference_tag=cfg['reference_tag'],new_detector=True,automatic_baseline_fallback=False)
            atomic_json(run/'completed.json',done)
            for kind in ('best_guarded','best_weighted','last'):
                atomic_json(run/(kind+'.json'),dict(checkpoint='last.pt',selector=kind,
                    tag=state['last_tag'] if kind=='last' else state['selections'][kind]))
            print('V316_SELECTED='+str(state['selections']),flush=True)
            return done
        finally: pool.close()
