"""Check every scheduled metric group before the first GPU update."""
from w2v_v3191.resources import _samples
from w2v_v5.resources import audit_training as parent_audit
from .sampling import Plan


def audit_training(rows, dev, cfg):
    if dev:
        raise ValueError('V5.1 final has no untouched Dev for V5.5 selection')
    # Read each header once, and reuse sizes in the inherited resource audit.
    sized = [dict(r, output_samples=_samples(r)[0]) for r in rows]
    result = parent_audit(sized, [], cfg)
    repeat = max(3920, round(cfg['v5_input']['short_seconds']*16000))
    frame = lambda n: (max(n, repeat)-400)//320+1
    frames = [frame((r['output_samples']*1000+969)//970) for r in sized]
    official_frames = [frame(r['output_samples']) for r in sized]
    plan = Plan(rows, cfg['stream_sources'], cfg['sampling_seed'], cfg['group_quotas'],
                cfg['view_factors'], cfg['refresh_base_variants'], cfg=cfg)
    maximum_views, maximum_frames, group_count = 0, 0, 0
    failures = []
    for epoch in range(cfg['epochs']):
        for tickets in plan.batches(epoch):
            groups = {}
            for ticket in tickets:
                name = ticket.get('hp_group')
                if name is None:
                    continue
                n = frames[ticket['index']]
                sizes = [n] if ticket['kind'] == 'online' else [n,n]
                if ticket.get('hp_member_role') == 'anchor':
                    sizes.append(n)
                if 'pcl_online_index' in ticket:
                    sizes.append(official_frames[ticket['pcl_online_index']])
                groups.setdefault(name, []).extend(sizes)
            for name,sizes in groups.items():
                group_count += 1
                count, padded = len(sizes), len(sizes)*max(sizes)
                maximum_views, maximum_frames = max(maximum_views,count), max(maximum_frames,padded)
                if (count > cfg['microbatch'] or padded > cfg['frame_budget']) and len(failures)<8:
                    failures.append(dict(group=name, views=count, padded_frames=padded))
    result['hard_positive'] = dict(scheduled_groups=group_count, maximum_atomic_views=maximum_views,
        maximum_padded_frames=maximum_frames, all_epochs_checked=cfg['epochs'],
        extra_encoder_views_per_update_at_most=4,
        note='Conservative speed/phoneme-reference bound; same CE/PCL/metric forward, no scoring pass')
    if failures:
        raise ValueError(f'V5.5 metric group preflight exceeds physical budgets: {failures}; '
                         f'requires --microbatch {maximum_views} --frame-budget {maximum_frames}; '
                         'check available GPU memory; no data were cropped or dropped')
    return result
