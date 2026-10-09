"""Measure real Train triplets, then roll back every parameter, moment and RNG."""
from dataclasses import replace
import gc
import json
from w2v_v315.augment import recipe
from pathlib import Path
import time

import torch
from live_progress import publish

from w2v_v39.common import atomic_json, read_json
from w2v_v316_tfcl.data import Triplets, TripletCollator, tensors, GROUPS
from .state import identity, partial_state, apply_partial, to_cpu, capture_rng, restore_rng
from .step import train_step, clear_features


def checkpointing(model, enabled):
    # The installed wrapper still skips the frozen prefix. This only changes
    # recomputation, not the layer inventory, parameter keys or objective.
    model.backbone.encoder.gradient_checkpointing = bool(enabled)


def probe_examples(plan, cfg, run):
    tickets = plan.batches(cfg['sampling_epoch_offset'])[0]
    rank = {g:sorted(pool,key=lambda s:max(plan.rows[plan.inventory[s]['indices'][kind]].get('output_samples',0)
                                        for kind in plan.inventory[s]['indices'] if kind in ('offline','online')))
            for g,pool in plan.pools.items()}
    result=[]
    quantiles=(.5,.9,.99,1.)
    for j,t in enumerate(tickets):
        row=plan.rows[t.original]; group=(row['language'],row['label']); pool=rank[group]
        source=pool[min(len(pool)-1,int((len(pool)-1)*quantiles[(j//4)%4]))]
        item=plan.inventory[source]
        result.append(replace(t,original=item['indices']['offline'],
            online=item['indices'].get('online'),occurrence='profile:'+t.occurrence,
            recipe_json=json.dumps(recipe(cfg['seed'],'profile:'+t.occurrence,t.phase,
                plan.rows[item['indices']['offline']]['group_id'],warm=t.warm),sort_keys=True)))
    dataset=Triplets(plan.rows,cfg,run)
    return tensors(TripletCollator(cfg['ssl_path'])([dataset[t] for t in result]))


def select_execution(model, auxiliary, optimizer, plan, cfg, run):
    path=Path(run)/'execution_plan.json'
    signature=identity(cfg)
    if path.exists():
        saved=read_json(path)
        if saved['config_identity']!=signature:
            raise ValueError('Profile belongs to another configuration')
        if cfg['device'].startswith('cuda') and saved['device_name']!=torch.cuda.get_device_name():
            raise ValueError('Resume hardware changed; recorded execution cannot be silently replaced')
        return saved
    selected=dict(microbatch=cfg['microbatch'],frame_budget=cfg['frame_budget'],checkpointing=True)
    saved=dict(config_identity=signature,selected=selected,attempts=[],rollback_verified=False,
        device_name=torch.cuda.get_device_name() if cfg['device'].startswith('cuda') else 'cpu',
        logical_sources=cfg['source_batch'],full_views_per_update=3*cfg['source_batch'])
    if not cfg['autotune'] or not cfg['device'].startswith('cuda'):
        saved.update(reason='explicit execution limits; no CUDA profiling',rollback_verified=True)
        atomic_json(path,saved)
        return saved
    publish('Preparing median/long/longest Train profiling views',force=True)
    examples=probe_examples(plan,cfg,run)
    original, aux_original, moments, rng = partial_state(model),to_cpu(auxiliary.state_dict()),to_cpu(optimizer.state_dict()),capture_rng()
    trials=[]
    for enabled in (True,False):
        for n,budget in ((min(8,cfg['microbatch']),min(4800,cfg['frame_budget'])),
                         (cfg['microbatch'],cfg['frame_budget'])):
            value=dict(microbatch=n,frame_budget=budget,checkpointing=enabled)
            if value not in trials:trials.append(value)

    def rollback():
        clear_features(model); optimizer.zero_grad(set_to_none=True)
        apply_partial(model,original); auxiliary.load_state_dict(aux_original,strict=True)
        # load_state_dict may alias CPU state tensors, notably Adam step counts.
        # Clone every restore so a trial cannot mutate the immutable snapshot.
        optimizer.load_state_dict(to_cpu(moments)); restore_rng(rng)
        gc.collect(); torch.cuda.empty_cache()

    try:
        for trial in trials:
            rollback(); checkpointing(model,trial['checkpointing'])
            torch.cuda.reset_peak_memory_stats()
            elapsed=[]
            result=dict(**trial)
            publish('Profiling full SSL TFCL: views='+str(trial['microbatch'])+
                    ' checkpointing='+str(trial['checkpointing']),force=True)
            print('V3161_PROFILE testing '+str(trial)+' on median/long/longest Train sources',flush=True)
            try:
                for _ in range(2):
                    torch.cuda.synchronize(); started=time.perf_counter()
                    train_step(model,auxiliary,optimizer,examples,dict(cfg,**trial),warm=1.)
                    torch.cuda.synchronize(); elapsed.append(time.perf_counter()-started)
                free,total=torch.cuda.mem_get_info()
                reserved=torch.cuda.memory_reserved()
                peak=torch.cuda.max_memory_reserved()
                # Includes other GPU occupants and allocator peaks, not only model tensors.
                usable_peak=free+reserved-cfg['gpu_reserve_bytes']
                result.update(seconds_per_update=sum(elapsed)/len(elapsed),peak_reserved_bytes=peak,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    eligible=peak<=usable_peak,status='ok')
            except torch.cuda.OutOfMemoryError:
                result.update(status='out_of_memory',eligible=False)
            saved['attempts'].append(result)
            print('V3161_PROFILE result '+str(result),flush=True)
    finally:
        rollback()
        checkpointing(model,True)
    if any(not torch.equal(p.detach().cpu(),original[n]) for n,p in model.named_parameters() if p.requires_grad):
        raise AssertionError('Performance probe changed detector parameters')
    if any(not torch.equal(v.detach().cpu(),aux_original[k]) for k,v in auxiliary.state_dict().items()):
        raise AssertionError('Performance probe changed TFCL parameters')
    good=[r for r in saved['attempts'] if r['eligible']]
    if not good:
        atomic_json(Path(run)/'execution_profile_failed.json',saved)
        raise RuntimeError('No profiled execution leaves 6 GiB GPU reserve; reduce --microbatch/--frame-budget in a NEW run')
    winner=min(good,key=lambda r:r['seconds_per_update'])
    saved.update(selected={k:winner[k] for k in ('microbatch','frame_budget','checkpointing')},
                 rollback_verified=True,reason='fastest measured full triplet update within GPU reserve')
    atomic_json(path,saved)
    return saved
