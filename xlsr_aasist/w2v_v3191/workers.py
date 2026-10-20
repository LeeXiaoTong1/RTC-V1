"""CPU and RAM bounds that include all receiver candidates and paired views."""
import torch
from w2v_v3184.data import DiagnosticLoader, passthrough, worker_init
from w2v_v3189.workers import host_resources


MIB = 1024**2


def settings(args, old, host=None):
    host = host_resources() if host is None else host
    cores = max(1, int(host['usable_cpu_cores']))
    requested = args.workers
    requested_eval = args.eval_workers
    prefetch = int(args.prefetch_factor)
    if prefetch < 1 or any(value is not None and (type(value) is not int or value < 0)
                           for value in (requested, requested_eval)):
        raise ValueError('Worker counts must be nonnegative integers; prefetch must be positive')
    if any(value is not None and value > cores for value in (requested, requested_eval)):
        raise ValueError(f'CPU-heavy receiver workers cannot exceed {cores} usable CPU cores')
    candidates = int(getattr(args, 'robust_candidate_count', 4))
    if candidates < 2:
        raise ValueError('Candidate memory accounting requires baseline and alternative views')
    source_count = int(old.get('stream_sources', 24))
    frames = int(old.get('whole_max_frames', 3000))
    # Max accepted geometry, not average duration. A Train task has S Online
    # and S Offline-source rows. Offline candidates hold two waves each and
    # at most one shared official PCL reference.
    samples = frames*320+400
    bytes_per_wave = samples*4
    train_views = source_count + source_count*(2*candidates+1)
    train_batch = train_views*bytes_per_wave
    evaluation_batch = int(getattr(args, 'eval_batch', old.get('eval_batch', 24)))*bytes_per_wave
    previous = max(1, int(old.get('workers', 4)))
    budgets = dict(raw_audio_cache_mib=previous*int(old.get('raw_audio_cache_mib', 128)),
        noise_cache_mib=previous*int(old.get('noise_cache_mib', 128)),
        scene_cache_mib=previous*int(old.get('scene_cache_mib', 32)))

    def cache_values(count):
        divisor = max(1, count)
        return {key: budget//divisor for key, budget in budgets.items()}

    def estimate(count, eval_count):
        caches = sum(cache_values(count).values())*MIB
        # Include generated and prefetched tasks, two PCM copies for IPC, the
        # temporary evaluation pool, runtime and one source's DSP workspaces.
        train_queue = (1+count*(prefetch+1))*2*train_batch
        eval_queue = (1+eval_count*(prefetch+1))*2*evaluation_batch
        process_bytes = (count+eval_count)*(512*MIB+caches+8*bytes_per_wave)
        if count == 0:
            process_bytes += caches
        return int(train_queue+eval_queue+process_bytes)

    workers = max(1, cores//2) if requested is None else requested
    evaluation = min(4, workers) if requested_eval is None else requested_eval
    memory = host.get('available_memory_bytes')
    if memory is not None:
        budget = int(memory*(.5 if requested is None and requested_eval is None else .75))
        while estimate(workers, evaluation) > budget:
            if requested is None and workers > 0:
                workers -= 1
                if requested_eval is None:
                    evaluation = min(4, workers)
            elif requested_eval is None and evaluation > 0:
                evaluation -= 1
            else:
                raise ValueError('Receiver candidate queue exceeds the host RAM estimate: '
                    f'{estimate(workers,evaluation)/MIB:.0f} MiB required, {budget/MIB:.0f} MiB budget. '
                    'Reduce --workers / --eval-workers / --prefetch-factor, or use a larger-memory host. '
                    'The estimate conservatively uses the accepted full-utterance length bound.')
    result = dict(workers=workers, eval_workers=evaluation, prefetch_factor=prefetch,
                  **cache_values(workers))
    result['worker_resource_audit'] = dict(host, requested_workers=requested,
        requested_eval_workers=requested_eval, selected_workers=workers,
        eval_workers=evaluation, prefetch_factor=prefetch,
        queued_train_batches=workers*prefetch,
        receiver_candidates_per_noisy_source=candidates,
        waveform_views_per_train_task_bound=train_views,
        maximum_samples_per_waveform=samples,
        estimated_train_task_pcm_bytes=train_batch,
        ipc_waveform_copy_factor=2,
        estimated_pool_bytes=estimate(workers,evaluation),
        cache_mib_per_process=cache_values(workers),
        parent_aggregate_cache_mib=sum(budgets.values()),
        policy='Auto: half usable cores, bounded by 50% available RAM; explicit counts <= CPU cores and <=75% RAM. Candidate-pair PCM, active work and concurrent eval included.',
        scope='Conservative queue bound, not measured process RSS or GPU memory',
        native_threads_per_worker=1)
    print(f'[Workers] train={workers}, eval={evaluation}, prefetch={prefetch}; '
          f'{candidates} paired candidates; conservative pool estimate '
          f'{estimate(workers,evaluation)/1024**3:.2f} GiB', flush=True)
    return result


def loader(dataset, cfg, **kwargs):
    options = dict(dataset=dataset, num_workers=cfg['workers'], collate_fn=passthrough,
        generator=torch.Generator().manual_seed(cfg['seed']), **kwargs)
    if cfg['workers']:
        options.update(multiprocessing_context='spawn', persistent_workers=True,
            prefetch_factor=cfg.get('prefetch_factor',1), worker_init_fn=worker_init)
    return DiagnosticLoader(**options)
