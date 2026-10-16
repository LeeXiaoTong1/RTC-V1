"""Bounded CPU parallelism; deterministic tickets do not depend on worker IDs."""
import math
import os
from pathlib import Path
import torch
from w2v_v3184.data import DiagnosticLoader, passthrough, worker_init


def _read(path):
    try:return Path(path).read_text().strip()
    except (OSError,ValueError):return ''


def host_resources():
    try:cores=len(os.sched_getaffinity(0))
    except (AttributeError,OSError):cores=os.cpu_count() or 1
    quota=None
    fields=_read('/sys/fs/cgroup/cpu.max').split()
    if len(fields)==2 and fields[0]!='max':
        quota=max(1,int(fields[0])//max(1,int(fields[1])))
    else:
        q=_read('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
        p=_read('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        if q and p and int(q)>0:quota=max(1,int(q)//max(1,int(p)))
    if quota is not None:cores=min(cores,quota)
    available=None
    for line in _read('/proc/meminfo').splitlines():
        if line.startswith('MemAvailable:'):available=int(line.split()[1])*1024
    limit=_read('/sys/fs/cgroup/memory.max');current=_read('/sys/fs/cgroup/memory.current')
    if limit and current and limit!='max':
        remaining=max(0,int(limit)-int(current))
        available=min(available,remaining) if available is not None else remaining
    return dict(usable_cpu_cores=max(1,cores),cgroup_cpu_quota=quota,available_memory_bytes=available,
        scope='Affinity and visible cgroup limits; best-effort host snapshot, not a memory guarantee')


def settings(args, old, host=None):
    host=host_resources() if host is None else host
    requested=args.workers
    workers=max(1,min(8,host['usable_cpu_cores']//2)) if requested is None else requested
    previous=max(1,int(old.get('workers',4)))
    raw_budget=previous*int(old.get('raw_audio_cache_mib',128))
    noise_budget=previous*int(old.get('noise_cache_mib',128))
    scene_budget=previous*int(old.get('scene_cache_mib',min(old.get('noise_cache_mib',128),32)))
    # A conservative queue estimate includes all CE views and both references.
    seconds=(int(old.get('whole_max_frames',3000))*320+400)/16000
    views=4*int(old.get('stream_sources',24))
    queue_mib=math.ceil(views*seconds*16000*4/1024**2)*args.prefetch_factor
    def values(count):
        n=max(1,count)
        return dict(raw_audio_cache_mib=raw_budget//n,noise_cache_mib=noise_budget//n,scene_cache_mib=scene_budget//n)
    def estimate(count):
        # Keep room for the persistent training pool plus a temporary eval pool.
        caches=sum(values(count).values())
        eval_count=min(4,count) if args.eval_workers is None else args.eval_workers
        return (count+eval_count)*(512+caches+queue_mib)*1024**2
    memory=host.get('available_memory_bytes')
    if requested is None and memory is not None:
        while workers>0 and estimate(workers)>.5*memory:workers-=1
    evaluation=min(4,workers) if args.eval_workers is None else args.eval_workers
    result=dict(workers=workers,eval_workers=evaluation,prefetch_factor=args.prefetch_factor,**values(workers))
    result['worker_resource_audit']=dict(host,requested_workers=requested,selected_workers=workers,
        eval_workers=evaluation,prefetch_factor=args.prefetch_factor,
        queued_train_batches=workers*args.prefetch_factor,estimated_pool_bytes=estimate(workers),
        cache_mib_per_process={k:v for k,v in result.items() if k.endswith('_cache_mib')},
        parent_aggregate_cache_mib=raw_budget+noise_budget+scene_budget,
        policy='Auto: half usable CPU cores, at most 8; RAM estimate may reduce further. Explicit counts override auto.',
        native_threads_per_worker=1)
    return result


def loader(dataset,cfg,**kwargs):
    options=dict(dataset=dataset,num_workers=cfg['workers'],collate_fn=passthrough,
        generator=torch.Generator().manual_seed(cfg['seed']),**kwargs)
    if cfg['workers']:
        options.update(multiprocessing_context='spawn',persistent_workers=True,
            prefetch_factor=cfg.get('prefetch_factor',1),worker_init_fn=worker_init)
    return DiagnosticLoader(**options)
