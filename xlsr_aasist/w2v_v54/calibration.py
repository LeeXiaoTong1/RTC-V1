"""Fixed Train-only frame scales and pair coverage; never use test feedback."""
from collections import Counter
import copy
import numpy as np
import torch
from .common import atomic_json, capture_rng, restore_rng, fingerprint, read_json, atomic_save
from .data import Waves, TicketSampler, loader, close
from w2v_v3183r2.common import GROUPS
from w2v_v3184.common import seed_for
from .repair import aligned_reference


def tickets_for(plan, rows, cfg):
    quota = cfg['repair_scale_sources_per_group']
    selected, seen, counts = [], set(), Counter()
    for batch in plan.batches(0):
        for ticket in batch:
            row = rows[ticket['index']]
            group = (row['language'], row['label'])
            if ticket['kind'] != 'offline' or row['source_id'] in seen or counts[group] >= quota:
                continue
            if row.get('split') != 'train':
                raise ValueError('Frame calibration may use allowed Train rows only')
            seen.add(row['source_id']); counts[group] += 1
            for lane in range(cfg['repair_scale_recipes']):
                t = copy.deepcopy(ticket)
                t.update(meta_side='A', pcl_enabled=False, pcl_noise=False, pcl_platform_budget=0,
                    recipe_seed=seed_for(cfg['sampling_seed'], 'v54-scale', lane), noise_recipe_view=lane)
                t.pop('pcl_online_index', None)
                # Do not preserve a condition override from an unrelated task.
                t.pop('domain_override', None)
                selected.append(t)
        if all(counts[group] == quota for group in GROUPS):
            break
    if any(counts[g] != quota for g in GROUPS):
        raise ValueError('Insufficient distinct sources for balanced frame calibration')
    return selected


def calibrate(model, cfg, rows, plan, run):
    """Cache only sufficient statistics after completion; checkpoint persists scale.

    Each source/recipe has equal mass independent of duration. A fixed maximum
    128 real frames per source is enough for scaling, not an estimator of EER.
    """
    rng, modes = capture_rng(), [(m, m.training) for m in model.modules()]
    path = run/'pair_preflight.json'
    progress_path = run/'pair_preflight_progress.pt'
    batches = None
    try:
        model.eval()
        if path.exists():
            cached = read_json(path)
            if cached.get('config') == fingerprint(cfg) and cached.get('schema') == 'v54_frame_scale_v1':
                scale = torch.tensor(cached['scale'], dtype=torch.float32)
                if scale.shape != (128,) or not torch.isfinite(scale).all() or (scale <= 0).any():
                    raise ValueError('Invalid cached frame scale')
                model.repair.frame_scale.copy_(scale)
                model.repair.scale_ready.fill_(True)
                print('[V5.4 prepare] verified completed frame-scale cache reused', flush=True)
                return
        if cfg['repair_arm'] == 'baseline':
            model.repair.scale_ready.fill_(True)
            atomic_json(path, dict(arm='baseline', scale='unused', pairs=0))
            return
        tickets = tickets_for(plan, rows, cfg)
        sampler = TicketSampler()
        # Two complete receiver pairs per loader batch: bounded padding/VRAM.
        dataset = Waves(rows, dict(cfg, repair_arm='paired'))
        mean = torch.zeros(128, dtype=torch.float64)
        m2 = torch.zeros_like(mean)
        count, pair_count, coverage = 0, 0, Counter()
        drift_square, drift_count = torch.zeros_like(mean), 0
        if progress_path.exists():
            saved = torch.load(progress_path, map_location='cpu', weights_only=True)
            if saved['config'] != fingerprint(cfg):
                raise ValueError('Preflight inputs changed')
            mean, m2, count, pair_count = (saved[k] for k in ('mean', 'm2', 'count', 'pairs'))
            coverage = Counter(saved['coverage'])
            drift_square, drift_count = saved['drift_square'], saved['drift_count']
        sampler.batches = [tickets[i:i+2] for i in range(pair_count, len(tickets), 2)]
        batches = loader(dataset, cfg, batch_sampler=sampler)
        with torch.no_grad():
            for units in batches:
                receiver_rows = [r for unit in units for r in (unit[0]['pcl_refs']['clean'], unit[0])]
                encoded = model.encode([r['wave'] for r in receiver_rows])
                predictions = model.classify_sequences(encoded).argmax(1).cpu().tolist()
                for i, unit in enumerate(units):
                    row = unit[0]
                    info = row['repair_pair']
                    clean = encoded[2*i].float()
                    # Scale estimates use actual non-artificial Clean frames,
                    # not only survivors of the Noisy reliability gate.
                    n = min(len(clean), row['pcl_refs']['clean']['pcl_valid_frames'])
                    if n:
                        index = torch.linspace(0, n-1, min(n, 128), device=clean.device).long()
                        x = clean[index].double().cpu()
                        mu, var = x.mean(0), x.var(0, unbiased=False)
                        count += 1
                        delta = mu-mean
                        mean += delta/count
                        m2 += var+delta*(mu-mean)
                    pair_count += 1
                    group = f'{row["language"]}/{row["label"]}'
                    correct_c, correct_n = predictions[2*i] == row['label'], predictions[2*i+1] == row['label']
                    outcome = ('both_correct' if correct_c and correct_n else 'noisy_only_wrong' if correct_c
                               else 'clean_only_wrong' if correct_n else 'both_wrong')
                    coverage[group+'/'+outcome] += 1
                    coverage['delay_valid'] += int(info['delay']['valid'])
                    coverage['eligible_pairs'] += int(np.count_nonzero(info['noisy_valid']) >= cfg['repair_min_frames'])
                    coverage['origin_'+row.get('v5_origin_split', 'train')] += 1
                    noisy = encoded[2*i+1].float()
                    target, mask = aligned_reference(clean, torch.as_tensor(info['clean_valid'],device=clean.device),
                        len(noisy), info['shift_frames'])
                    mask &= torch.as_tensor(info['noisy_valid'],device=noisy.device)
                    if int(mask.sum()) >= cfg['repair_min_frames']:
                        drift_square += (noisy[mask]-target[mask]).double().square().mean(0).cpu()
                        drift_count += 1
                        coverage[group+'/valid_frame_pair'] += 1
                if pair_count % 32 == 0 or pair_count == len(tickets):
                    print(f'[V5.4 prepare] receiver pairs {pair_count}/{len(tickets)}; valid={coverage["eligible_pairs"]}', flush=True)
                paused = (run.parent/'STOP_REQUESTED').exists()
                if pair_count % 32 == 0 or paused:
                    atomic_save(progress_path, dict(config=fingerprint(cfg), mean=mean, m2=m2,
                        count=count, pairs=pair_count, coverage=dict(coverage),
                        drift_square=drift_square, drift_count=drift_count), 0)
                if paused:
                    return False
        missing = [f'{language}/{label}' for language,label in GROUPS
                   if not coverage[f'{language}/{label}/valid_frame_pair']]
        if count < 4 or missing:
            atomic_json(path, dict(schema='v54_frame_scale_failed_v1', config=fingerprint(cfg),
                status='failed', coverage=dict(coverage), pairs=pair_count, missing_groups=missing))
            raise ValueError('No usable paired Train frames; inspect pair_preflight before training')
        scale = (m2/count).clamp_min(0).sqrt().float()
        floor = max(1e-4, .1*float(scale.median()))
        scale.clamp_min_(floor)
        if not torch.isfinite(scale).all():
            raise ValueError('Nonfinite Clean frame scale')
        model.repair.frame_scale.copy_(scale)
        model.repair.scale_ready.fill_(True)
        atomic_json(path, dict(schema='v54_frame_scale_v1', config=fingerprint(cfg),
            pairs=pair_count, source_recipe_mass='equal', frame_cap=128, valid_clean_records=count,
            coverage=dict(coverage), scale=scale.tolist(), floor=floor,
            normalized_channel_mse=(drift_square/max(1,drift_count)/scale.double().square()).tolist(),
            drift_valid_pairs=drift_count,
            scope='Allowed parent Train+Dev only; no threshold/recipe selection; not unseen validation'))
    finally:
        if batches is not None:
            close(batches)
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)
