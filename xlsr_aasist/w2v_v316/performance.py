"""Short full-update benchmarks, with parameters/optimizer/RNG rolled back."""
import gc
import statistics
import time
from dataclasses import replace
import torch
from w2v_v39.common import atomic_json,read_json
from .state import identity,partial_state,apply_partial,to_cpu,capture_rng,restore_rng
from .step import train_step


def probe_examples(plan,cfg,run):
    from .data import Triplets
    tickets=plan.batches(0)[0]
    pools={g:sorted(pool,key=lambda s:plan.rows[plan.inventory[s]['indices']['offline']].get('output_samples',0))
        for g,pool in plan.pools.items()}
    dataset=Triplets(plan.rows,cfg,run); examples=[]
    for i,ticket in enumerate(tickets):
        row=plan.rows[ticket.original]; pool=pools[(row['language'],row['label'])]
        q=(.5,.9,.99,1.)[(i//4)%4]
        source=pool[int((len(pool)-1)*q)]; indices=plan.inventory[source]['indices']
        selected=replace(ticket,original=indices['offline'],ordinary=indices.get('online',indices['offline']),
            occurrence='profile:'+ticket.occurrence)
        examples.extend(dataset[selected])
    return examples


def select_execution(model,auxiliary,optimizer,examples,cfg,run):
    path=run/'execution_plan.json'
    if path.exists():
        plan=read_json(path)
        if plan['identity']!=identity(cfg): raise ValueError('Execution plan configuration differs')
        return plan['selected']
    selected=dict(microbatch=cfg['microbatch'],frame_budget=cfg['frame_budget'],checkpointing=True)
    trials=[]
    if cfg['autotune']:
        weights,aux,opt,rng=partial_state(model),to_cpu(auxiliary.state_dict()),to_cpu(optimizer.state_dict()),capture_rng()
        def rollback():
            optimizer.zero_grad(set_to_none=True)
            apply_partial(model,weights); auxiliary.load_state_dict(aux)
            # Adam's CPU step counters may alias load_state_dict inputs even on
            # CUDA; each trial needs a fresh copy of the immutable snapshot.
            optimizer.load_state_dict(to_cpu(opt)); restore_rng(rng)
            gc.collect(); torch.cuda.empty_cache()
        try:
            sizes=sorted({2,cfg['microbatch'],min(8,2*cfg['microbatch'])})
            for size,recompute in [(n,True) for n in sizes]+[(sizes[-1],False)]:
                rollback()
                model.set_phase(True,recompute)
                trial=dict(microbatch=size,frame_budget=cfg['frame_budget'],checkpointing=recompute)
                print('V316_PROFILE '+str(trial),flush=True)
                times=[]
                try:
                    torch.cuda.reset_peak_memory_stats()
                    for i in range(3):
                        torch.cuda.synchronize(); start=time.perf_counter()
                        train_step(model,auxiliary,optimizer,examples,dict(cfg,**trial),1.)
                        torch.cuda.synchronize()
                        if i: times.append(time.perf_counter()-start)
                    peak=torch.cuda.max_memory_reserved()
                    free,_=torch.cuda.mem_get_info()
                    capacity=free+torch.cuda.memory_reserved()
                    trials.append(dict(**trial,seconds=statistics.median(times),peak_reserved_bytes=peak,
                        eligible=peak<capacity-cfg['gpu_reserve_bytes']))
                except torch.cuda.OutOfMemoryError:
                    trials.append(dict(**trial,eligible=False,reason='CUDA OOM'))
            choices=[t for t in trials if t['eligible']]
            if not choices: raise RuntimeError('No paired 7B microbatch fits with reserve; source/checkpoints preserved')
            best=min(choices,key=lambda t:t['seconds'])
            selected={k:best[k] for k in selected}
        finally:
            rollback()
        # Verify exact master tensors, not just their shapes.
        restored=partial_state(model)
        if any(not torch.equal(v,restored[k]) for k,v in weights.items()):
            raise RuntimeError('Profiling rollback verification failed')
        if any(not torch.equal(v,auxiliary.state_dict()[k].cpu()) for k,v in aux.items()):
            raise RuntimeError('TFCL profiling rollback verification failed')
    atomic_json(path,dict(identity=identity(cfg),selected=selected,trials=trials,
        logical_sources=cfg['source_batch'],logical_views=3*cfg['source_batch'],
        note='Balanced median/90th/99th/longest Train triplets; three updates/trial (first excluded). No truncation.',
        rollback_verified=bool(cfg['autotune'])))
    return selected
