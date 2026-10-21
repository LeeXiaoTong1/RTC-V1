"""Observe actual CE logits only, preserving view/source and loss accounting."""
from collections import Counter
from w2v_v3183r2.common import GROUPS
from w2v_v53_observe.observe import analyze


def capture_plan(micro, quotas, weights, side, batch):
    main, clean = [], []
    for index, unit in enumerate(micro):
        row = unit[0]
        if len(unit) != 1:
            raise ValueError('V5.4 expects one main receiver view per source')
        quota = quotas[GROUPS.index((row['language'], row['label']))]
        def entry(view, actual):
            return dict(row=actual, source=row, view=view, side=side,
                pair_id=(side, batch, index), mass=weights[view]/(8.*quota))
        main.append(entry(row['role'], row))
        if row['role'] == 'noisy':
            clean.append(entry('clean_rtc', row['pcl_refs']['clean']))
    return main+clean


def collect(a, b, cfg, result):
    captured = a.pop('_ce_captured')+b.pop('_ce_captured')
    plan = a.pop('_ce_plan')+b.pop('_ce_plan')
    result['hard_examples'] = analyze(captured, plan, cfg, result)
    records = a['repair_losses']['records']
    cells = {}
    for record in records:
        for key in (record['group'], 'origin/'+record['origin']):
            cell = cells.setdefault(key, Counter())
            cell.update(pairs=1, valid=int(record['valid']), kd_active=int(record['kd_active']),
                valid_frames=record['valid_frames'], frames=record['frames'],
                feature_loss_sum=record['feature_loss'], raw_feature_loss_sum=record['raw_feature_loss'],
                kd_loss_sum=record['kd_loss'])
            if not record['delay']['valid']:
                cell['rejected_'+record['delay']['reason']] += 1
    result['repair_losses'] = dict(feature=.5*a['repair_losses']['feature'],
        kd=.5*a['repair_losses']['kd'], cells={k: dict(v) for k, v in cells.items()})


def summarize(stage, maximum):
    from w2v_v53_observe.summary import summarize as hard_summary, events
    from .common import atomic_json
    hard_summary(stage, maximum)
    cells, losses, updates = {}, Counter(), 0
    for event in events(stage/'repair_diagnostics.jsonl', maximum):
        updates += 1
        losses.update(feature=event['feature'], kd=event['kd'])
        for key, counts in event['cells'].items():
            cells.setdefault(key, Counter()).update(counts)
    atomic_json(stage/'repair_summary.json', dict(schema='v54_repair_summary_v1',
        updates=updates, last_committed_update=maximum,
        mean_effective_loss={k: v/max(1,updates) for k,v in losses.items()},
        cells={k: dict(v) for k,v in cells.items()},
        scope='Training pairs only; coverage/updates do not establish generalization'))
    from statistics import median
    # Exclude startup and the first 50 joint updates; compare with a matched
    # control's profile, never infer a speedup from an isolated ETA.
    samples, first_joint = [], None
    for event in events(stage/'steps.jsonl', maximum):
        if event.get('repair_stage') != 'joint':
            continue
        first_joint = event['update'] if first_joint is None else first_joint
        if event['update'] >= first_joint+50:
            samples.append(event)
        if len(samples) == 50:
            break
    if samples:
        durations = sorted(e['timing']['update_seconds'] for e in samples)
        peaks = [e['timing']['peak_allocated_bytes'] for e in samples if e['timing']['peak_allocated_bytes'] is not None]
        atomic_json(stage/'training_profile.json', dict(updates=len(samples),
            median_update_seconds=median(durations), p90_update_seconds=durations[min(len(durations)-1,int(.9*len(durations)))],
            median_encoded_views=median(e['encoded_views'] for e in samples),
            peak_allocated_bytes=max(peaks) if peaks else None,
            scope='First 50 updates after 50 joint settling updates; timer excludes data fetch; cycle_seconds in steps includes it'))
