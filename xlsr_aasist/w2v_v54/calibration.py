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
from .data import valid_frames
from .diagnostics import encode_bounded, aggregate


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
            if cached.get('config') == fingerprint(cfg) and cached.get('schema') == 'v54_frame_scale_v2' and (run/'fixed_pairs.pt').is_file():
                scale = torch.tensor(cached['scale'], dtype=torch.float32)
                if scale.shape != (128,) or not torch.isfinite(scale).all() or (scale <= 0).any():
                    raise ValueError('Invalid cached frame scale')
                model.repair.frame_scale.copy_(scale)
                model.repair.scale_ready.fill_(True)
                print('[V5.4 prepare] verified completed frame-scale cache reused', flush=True)
                return
        tickets = tickets_for(plan, rows, cfg)
        sampler = TicketSampler()
        # Two complete receiver pairs per loader batch: bounded padding/VRAM.
        dataset = Waves(rows, dict(cfg, repair_arm='paired'))
        mean = torch.zeros(128, dtype=torch.float64)
        m2 = torch.zeros_like(mean)
        count, pair_count, coverage = 0, 0, Counter()
        drift_square, drift_count = torch.zeros_like(mean), 0
        records, fixed, repeat_checks = [], [], {}
        if progress_path.exists():
            saved = torch.load(progress_path, map_location='cpu', weights_only=True)
            if saved['config'] != fingerprint(cfg):
                raise ValueError('Preflight inputs changed')
            mean, m2, count, pair_count = (saved[k] for k in ('mean', 'm2', 'count', 'pairs'))
            coverage = Counter(saved['coverage'])
            drift_square, drift_count = saved['drift_square'], saved['drift_count']
            records,fixed,repeat_checks = saved.get('records',[]),saved.get('fixed',[]),saved.get('repeat_checks',{})
        sampler.batches = [tickets[i:i+2] for i in range(pair_count, len(tickets), 2)]
        batches = loader(dataset, cfg, batch_sampler=sampler)
        with torch.no_grad():
            for units in batches:
                receiver_rows = [r for unit in units for r in (unit[0]['pcl_refs']['clean'], unit[0])]
                encoded = encode_bounded(model,[r['wave'] for r in receiver_rows],cfg)
                all_logits = model.classify_sequences(encoded).float().cpu()
                predictions = all_logits.argmax(1).tolist()
                encode_context=[torch.as_tensor(r['wave']).clone() for r in receiver_rows]
                for i, unit in enumerate(units):
                    row = unit[0]
                    info = row['repair_pair']
                    clean = encoded[2*i].float()
                    # Scale estimates use actual non-artificial Clean frames,
                    # not only survivors of the Noisy reliability gate.
                    # Exclude artificial/filter-boundary frames, without
                    # conditioning the scale on Noisy's reliability outcome.
                    scale_valid = valid_frames(row['pcl_refs']['clean'],activity=False)
                    eligible = torch.as_tensor(np.flatnonzero(scale_valid),device=clean.device)
                    if len(eligible):
                        index = eligible[torch.linspace(0,len(eligible)-1,min(len(eligible),128),device=clean.device).long()]
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
                    coverage['origin_'+row.get('v5_origin_split', 'train')] += 1
                    noisy = encoded[2*i+1].float()
                    target, mask = aligned_reference(clean, torch.as_tensor(info['clean_valid'],device=clean.device),
                        len(noisy), info['shift_frames'])
                    mask &= torch.as_tensor(info['noisy_valid'],device=noisy.device)
                    coverage['eligible_pairs'] += int(mask.sum() >= cfg['repair_min_frames'])
                    margin = all_logits[2*i:2*i+2,row['label']]-all_logits[2*i:2*i+2,1-row['label']]
                    identity = dict(source=row['source_id'],group=group,origin=row.get('v5_origin_split','train'),
                        condition=info.get('condition','unknown'),snr_db=info.get('snr_db'),
                        seconds=info.get('seconds',row.get('original_samples',len(row['wave']))/16000.),
                        recipe_hash=info.get('recipe_hash'),common_base_hash=info.get('common_base_hash'))
                    records.append(dict(identity,frames=len(noisy),valid_frames=int(mask.sum()),
                        clean_margin=float(margin[0]),noisy_margin=float(margin[1]),
                        margin_change=float(margin[1]-margin[0]),outcome=outcome,
                        delay=info['delay'],rejection_reason='ok' if int(mask.sum())>=cfg['repair_min_frames'] else info['delay']['reason'] if not info['delay']['valid'] else 'insufficient_reliable_frames'))
                    if int(mask.sum()) >= cfg['repair_min_frames']:
                        drift_square += (noisy[mask]-target[mask]).double().square().mean(0).cpu()
                        drift_count += 1
                        coverage[group+'/valid_frame_pair'] += 1
                    chosen = {entry['identity']['source'] for entry in fixed if entry['identity']['group']==group}
                    if row['source_id'] not in chosen and len(chosen)<cfg.get('repair_probe_sources_per_group',16):
                        fixed.append(dict(identity=identity,label=row['label'],
                            clean_wave=torch.as_tensor(receiver_rows[2*i]['wave']).clone(),
                            noisy_wave=torch.as_tensor(row['wave']).clone(),
                            encode_context=encode_context,encode_indices=[2*i,2*i+1],
                            clean_feature=clean.cpu(),noisy_feature=noisy.cpu(),
                            scale_valid=torch.as_tensor(scale_valid),initial_logits=all_logits[2*i:2*i+2].clone(),
                            info={k:torch.as_tensor(v) if isinstance(v,np.ndarray) else v for k,v in info.items()}))
                    if group not in repeat_checks:
                        same = encode_bounded(model,[r['wave'] for r in receiver_rows],cfg)[2*i].float()
                        # A deterministic alternative companion tests batch
                        # padding; this waveform is never training evidence.
                        limit=min(cfg.get('whole_max_frames',3000),cfg['frame_budget']//2)
                        extra=max(0,min(len(row['wave'])//2,limit*320+80-len(row['wave'])))
                        companion=np.pad(row['wave'],(0,extra),mode='reflect')
                        changed = encode_bounded(model,[companion,receiver_rows[2*i]['wave']],cfg)[1].float()
                        repeat_checks[group] = dict(repeat_max_abs=float((same-clean).abs().max()),
                            companion_max_abs=float((changed-clean).abs().max()),
                            repeat_relative_l2=float((same-clean).norm()/clean.norm().clamp_min(1e-8)),
                            companion_relative_l2=float((changed-clean).norm()/clean.norm().clamp_min(1e-8)),
                            companion_added_samples=extra)
                if pair_count % 32 == 0 or pair_count == len(tickets):
                    print(f'[V5.4 prepare] receiver pairs {pair_count}/{len(tickets)}; valid={coverage["eligible_pairs"]}', flush=True)
                paused = (run.parent/'STOP_REQUESTED').exists()
                if pair_count % 32 == 0 or paused:
                    atomic_save(progress_path, dict(config=fingerprint(cfg), mean=mean, m2=m2,
                        count=count, pairs=pair_count, coverage=dict(coverage),
                        drift_square=drift_square, drift_count=drift_count,
                        records=records,fixed=fixed,repeat_checks=repeat_checks), 0)
                if paused:
                    return False
        expected = cfg['repair_scale_sources_per_group']*cfg['repair_scale_recipes']
        missing = [f'{language}/{label}' for language,label in GROUPS
                   if coverage[f'{language}/{label}/valid_frame_pair']/expected < cfg.get('repair_min_group_coverage',.05)]
        if count < 4 or missing:
            atomic_json(path, dict(schema='v54_frame_scale_failed_v1', config=fingerprint(cfg),
                status='failed', coverage=dict(coverage), pairs=pair_count, missing_groups=missing))
            raise ValueError('Insufficient paired Train coverage; inspect pair_preflight missing_groups and the recorded minimum coverage')
        scale = (m2/count).clamp_min(0).sqrt().float()
        floor = max(1e-4, .1*float(scale.median()))
        scale.clamp_min_(floor)
        if not torch.isfinite(scale).all():
            raise ValueError('Nonfinite Clean frame scale')
        model.repair.frame_scale.copy_(scale)
        model.repair.scale_ready.fill_(True)
        # Probe stores only the fixed bounded subset, not all 1024 feature pairs.
        atomic_save(run/'fixed_pairs.pt',dict(config=fingerprint(cfg),pairs=fixed),0)
        from .diagnostics import probe
        probe(model,cfg,run,'pair_mechanism_initial',cached_initial=True)
        atomic_json(path, dict(schema='v54_frame_scale_v2', config=fingerprint(cfg),
            pairs=pair_count, source_recipe_mass='equal', frame_cap=128, valid_clean_records=count,
            coverage=dict(coverage), scale=scale.tolist(), floor=floor,
            normalized_channel_mse=(drift_square/max(1,drift_count)/scale.double().square()).tolist(),
            drift_valid_pairs=drift_count,
            groups=aggregate(records),records=records,repeat_checks=repeat_checks,
            fixed_probe_pairs=len(fixed),minimum_group_pair_coverage=cfg.get('repair_min_group_coverage',.05),
            normalized_channel_mse_scope='equal mass per eligible source/recipe, after relative-delay interpolation',
            margin_feature_relation='See fixed-pair mechanism records: standardized feature losses versus same-pair margins',
            scope='Allowed parent Train+Dev only; no threshold/recipe selection; not unseen validation'))
    finally:
        if batches is not None:
            close(batches)
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)
