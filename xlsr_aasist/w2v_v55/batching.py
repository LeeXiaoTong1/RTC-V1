"""Keep six-view metric groups intact, with no repeated encoder scoring."""
from collections import defaultdict
from w2v_v3191.side import _layout as parent_layout
from w2v_v5.data import microbatches as parent_microbatches


def layout(unit, weights):
    rows, slots = parent_layout(unit, weights)
    rows = [dict(row) for row in rows]
    first = unit[0]
    role = first.get('hp_member_role')
    for row in rows:
        row.pop('hp_view_role', None)
    if role == 'anchor':
        if first['label'] != 1 or 'hp_positive2' not in first:
            raise ValueError('A Real anchor needs its second processed positive')
        rows[0]['hp_view_role'] = 'positive'
        rows[slots['clean']]['hp_view_role'] = 'anchor'
        slots['positive2'] = len(rows)
        rows.append(dict(first['hp_positive2'], hp_view_role='positive'))
    elif role in ('negative_noisy', 'negative_online'):
        rows[0]['hp_view_role'] = 'negative'
    elif role == 'negative_clean':
        rows[slots['clean']]['hp_view_role'] = 'negative'
    elif role is not None:
        raise ValueError('Unknown hard-positive member role')
    return rows, slots


def microbatches(units, size, frame_budget, *, weights):
    if type(size) is not int or type(frame_budget) is not int or min(size, frame_budget) < 1:
        raise ValueError('Positive physical batch budgets required')
    marked, pending = defaultdict(list), []
    for unit in units:
        if unit[0].get('hp_group') is not None:
            marked[unit[0]['hp_group']].append(unit)
        else:
            pending.append(unit)

    def cost(batch):
        expanded = [row for unit in batch for row in layout(unit, weights)[0]]
        longest = max((max(1, (len(r['wave'])-400)//320+1) for r in expanded), default=0)
        return len(expanded), len(expanded) * longest

    def fits(batch):
        count, frames = cost(batch)
        return count <= size and frames <= frame_budget

    groups = []
    for name in sorted(marked):
        members = marked[name]
        if len(members) != 4 or sum(u[0].get('hp_member_role') == 'anchor' for u in members) != 1:
            raise ValueError('Expected one Real source and three distinct Fake sources per group')
        if not fits(members):
            raise ValueError(f'V5.5 complete hard-positive/PCL group {name} needs {cost(members)} '
                             f'(views, padded frames); increase --microbatch/--frame-budget; no cropping or skipping')
        groups.append(members)
    # Do not silently lose the parent's >=4-source PCL variance opportunity.
    # Fill metric batches with remaining paired sources before Online rows.
    for members in groups:
        while sum(u[0]['role'] == 'noisy' for u in members) < 4:
            available = [u for u in pending if u[0]['role'] == 'noisy' and fits(members + [u])]
            if not available:
                break
            chosen = min(available, key=lambda u: (cost(members + [u])[1], str(u[0]['id'])))
            members.append(chosen)
            pending = [u for u in pending if u is not chosen]
        yield members
    # The original packing remains in use outside metric groups.
    yield from parent_microbatches(pending, size, frame_budget, weights=weights)
