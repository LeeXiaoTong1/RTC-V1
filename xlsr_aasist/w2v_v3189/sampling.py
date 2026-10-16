"""Exact per-stream exposure budgets, continuous source cycles, fresh variants."""
from bisect import bisect_right
from collections import Counter
import math
import numpy as np
from w2v_v318.sampling import Plan as InventoryPlan, probe_tickets
from w2v_v3185.sampling import DEFAULT_VIEW_FACTORS, PAIR_FAMILIES
from w2v_v3186.sampling import canonical_pairs
from w2v_v3185.common import GROUPS, seed_for
from .views import FAMILIES

DEFAULT_SOURCE_DRAWS = (9668, 8000, 8000, 8000)


def validate_units(units, quotas):
    counts = Counter(); identities = {}
    for unit in units:
        if len(unit) != 1 or unit[0].get('role') not in ('online', 'noisy'):
            raise ValueError('One Online or one simulated Noisy view required; raw Offline is forbidden')
        row = unit[0]
        if row.get('split') != 'train':
            raise ValueError('Training split required')
        kind = 'online' if row['role'] == 'online' else 'offline'
        key = (kind, row['language'], row['label']); counts[key] += 1
        seen = identities.setdefault(key, set())
        if row['source_id'] in seen:
            raise ValueError('Repeated source within a logical stream/group batch')
        seen.add(row['source_id'])
    expected = Counter({(kind, *group): q for kind in ('online', 'offline') for group, q in zip(GROUPS, quotas)})
    if counts != expected:
        raise ValueError('Logical batch violates configured quotas')


def quota_schedule(budgets, batch):
    """Hamilton cumulative allocation: exact totals and no empty group steps."""
    if len(budgets) != 4 or any(type(x) is not int or x < 1 for x in budgets):
        raise ValueError('Four positive integer source exposure budgets are required')
    if type(batch) is not int or batch < 4:
        raise ValueError('A stream batch needs at least four sources')
    total = sum(budgets); steps = math.ceil(total / batch)
    # Rebalance the last two steps if the final remainder cannot contain all
    # four groups. This does not overfill, duplicate, or discard any draw.
    sizes = [batch] * (steps - 1) + [total - batch * (steps - 1)]
    if len(sizes) > 1 and sizes[-1] < 4:
        take = 4 - sizes[-1]; sizes[-2] -= take; sizes[-1] += take
    cumulative = [0] * 4; result = []; consumed = 0
    for size in sizes:
        consumed += size
        values = [consumed * b // total for b in budgets]
        remainder = consumed - sum(values)
        order = sorted(range(4), key=lambda j: (-(consumed * budgets[j] % total), j))
        for j in order[:remainder]: values[j] += 1
        quotas = tuple(values[j] - cumulative[j] for j in range(4))
        if min(quotas) < 1:
            raise ValueError('Exposure budgets are too skewed for four groups in every batch')
        result.append(quotas); cumulative = values
    return result


class Plan(InventoryPlan):
    def __init__(self, rows, stream_sources=24, seed=31801, quotas=(6, 6, 6, 6),
                 view_factors=DEFAULT_VIEW_FACTORS, refresh_base_variants=True, *, cfg=None):
        super().__init__(rows, stream_sources, seed)
        self.cfg = cfg or {}
        if len(view_factors) != 4 or any(type(v) is not int or v not in (1, 3, 5) for v in view_factors):
            raise ValueError('Four view budgets must each be 1, 3 or 5')
        if type(refresh_base_variants) is not bool:
            raise ValueError('refresh_base_variants must be boolean')
        values = self.cfg.get('source_draws_per_stream', DEFAULT_SOURCE_DRAWS)
        self.budgets = dict(zip(GROUPS, values))
        self.schedule = quota_schedule(tuple(values), stream_sources)
        self.steps = len(self.schedule)
        self.quotas = dict(zip(GROUPS, quotas))  # nominal display only
        self.prefixes = {g: [0] for g in GROUPS}
        for step in self.schedule:
            for j, g in enumerate(GROUPS): self.prefixes[g].append(self.prefixes[g][-1] + step[j])
        for kind, groups in self.pools.items():
            for j, (group, pool) in enumerate(groups.items()):
                if len(pool) < max(q[j] for q in self.schedule):
                    raise ValueError(f'{kind}/{group}: too few distinct sources for a scheduled batch')
        self.view_factors = dict(zip(GROUPS, view_factors))
        self.refresh_base_variants = refresh_base_variants
        self._cycles = {}; self._view_epoch = None; self._view_streams = {}; self.definitions = {}
        for kind, groups in self.pools.items():
            for group, pool in groups.items():
                ordered = sorted(pool, key=lambda i: (seed_for(seed, 'v3189-definition', kind, rows[i]['id']), rows[i]['id']))
                for rank, index in enumerate(ordered):
                    factor = self.view_factors[group]
                    families = tuple(range(1, 5)) if factor == 5 else PAIR_FAMILIES[rank % 8] if factor == 3 else ()
                    self.definitions[index] = dict(factor=factor, phase=rank % 5,
                                                   family_phase=rank % 4, families=families)
        self.consistency = self.cfg.get('consistency', 'dual') != 'none'
        self.platform_count = int(self.cfg.get('platform_sources_per_group', 3))
        if self.platform_count < 0: raise ValueError('Platform target must be nonnegative')
        self.pairs = canonical_pairs(rows) if self.consistency else {}

    def order(self, kind, group, cycle):
        key = (kind, group, cycle)
        if key in self._cycles: return self._cycles[key]
        size = len(self.pools[kind][group]); budget = self.budgets[group]
        prefix = self.prefixes[group]; j = GROUPS.index(group)
        for current in range(cycle + 1):
            key = (kind, group, current)
            if key in self._cycles: continue
            order = np.random.default_rng(seed_for(self.seed, 'v3189-cycle', kind, group, current)).permutation(size).tolist()
            offset = (current * size) % budget
            step = bisect_right(prefix, offset) - 1
            tail = offset - prefix[step]
            if tail:
                previous = set(self._cycles[kind, group, current - 1][-tail:])
                needed = self.schedule[step][j] - tail
                for position in range(needed):
                    if order[position] in previous:
                        swap = next(i for i in range(needed, size) if order[i] not in previous)
                        order[position], order[swap] = order[swap], order[position]
            self._cycles[key] = order
        return self._cycles[kind, group, cycle]

    def view_spec(self, index, kind, cycle):
        if cycle < 0: raise ValueError('Source cycle must be nonnegative')
        d = self.definitions[index]; phase = d['phase']
        original = (cycle + phase) % 5 == 0
        originals_before = (cycle + phase + 4) // 5 - (phase + 4) // 5
        augmented_before = cycle - originals_before
        dynamic = False; generation = 0; variant = 0; family = 0
        if not original:
            if d['factor'] == 1:
                dynamic = True; family = 1 + (augmented_before + d['family_phase']) % 4
                generation = augmented_before // 4
            else:
                count = d['factor'] - 1
                rotation = d['family_phase'] if count == 4 else 0
                variant = 1 + (augmented_before + rotation) % count
                family = d['families'][variant - 1]
                if self.refresh_base_variants: generation = augmented_before // count
        mode = 'dynamic' if dynamic else 'slot'
        identity = f'{self.rows[index]["id"]}#v3189:{kind}:{mode}:{variant}:g{generation}:{FAMILIES[family]}'
        return dict(base_variant=variant, base_family=FAMILIES[family], base_generation=generation,
                    base_dynamic=dynamic, base_view_id=identity)

    def view_stream(self, kind, group, epoch):
        if type(epoch) is not int or epoch < 0: raise ValueError('Epoch must be nonnegative')
        if self._view_epoch != epoch: self._view_epoch = epoch; self._view_streams = {}
        key = (kind, group)
        if key in self._view_streams: return self._view_streams[key]
        pool = self.pools[kind][group]; count = self.budgets[group]
        offset = (epoch + int(self.cfg.get('source_epoch_offset', 0))) * count; values = []
        while len(values) < count:
            cycle, pos = divmod(offset, len(pool)); order = self.order(kind, group, cycle)
            chunk = order[pos:pos + count - len(values)]
            for position in chunk:
                index = pool[position]; values.append((index, self.view_spec(index, kind, cycle), cycle))
            offset += len(chunk)
        self._view_streams[key] = values
        return values

    def stream(self, kind, group, epoch):
        return [index for index, _, _ in self.view_stream(kind, group, epoch)]

    def batches(self, epoch):
        streams = {(kind, g): self.view_stream(kind, g, epoch) for kind in self.pools for g in GROUPS}
        active = self.consistency and epoch >= self.cfg.get('warm_epochs', 1)
        for step, quotas in enumerate(self.schedule):
            tickets = []
            for j, group in enumerate(GROUPS):
                start = self.prefixes[group][step]
                for k in range(quotas[j]):
                    for kind in ('online', 'offline'):
                        index, spec, cycle = streams[kind, group][start + k]
                        tickets.append(dict(index=index, kind=kind,
                                            epoch=epoch+int(self.cfg.get('source_epoch_offset',0)), step=step, **spec,
                                            batch_group_quotas=quotas, source_cycle=cycle,
                                            occurrence=f'v3189:{epoch}:{step}:{kind}:{group[0]}:{group[1]}:{k}'))
            selected = set(); platform_budgets = {}
            if active and self.cfg.get('pcl_platform_weight', .15) > 0:
                for group in GROUPS:
                    candidates = [t for t in tickets if t['kind'] == 'offline' and
                                  (self.rows[t['index']]['language'], self.rows[t['index']]['label']) == group and t['index'] in self.pairs]
                    candidates.sort(key=lambda t: seed_for(self.seed, 'v3189-platform', epoch, step, self.rows[t['index']]['id']))
                    chosen = candidates[:self.platform_count]
                    selected.update(t['index'] for t in chosen)
                    platform_budgets[group] = len(chosen)
            for ticket in tickets:
                if ticket['kind'] != 'offline': continue
                row = self.rows[ticket['index']]; group = row['language'], row['label']
                ticket['pcl_noise'] = active and self.cfg.get('pcl_noise_weight', .15) > 0
                ticket['pcl_noisy_index'] = 0
                ticket['noise_recipe_view'] = ticket['source_cycle'] % 2
                ticket['pcl_platform_budget'] = platform_budgets.get(group, 0)
                if ticket['index'] in selected: ticket['pcl_online_index'] = self.pairs[ticket['index']]
                ticket['pcl_enabled'] = bool(ticket['pcl_noise'] or ticket['index'] in selected)
            order = np.random.default_rng(seed_for(self.seed, 'v3189-ticket-order', epoch, step)).permutation(len(tickets))
            yield [tickets[i] for i in order]

    def coverage(self, epoch):
        cells = {}
        for kind, groups in self.pools.items():
            for group, pool in groups.items():
                values = self.view_stream(kind, group, epoch); counts = Counter(i for i, _, _ in values)
                originals = sum(s['base_family'] == 'original' for _, s, _ in values)
                cells[f'{kind}/{group[0]}/{group[1]}'] = dict(
                    available=len(pool), source_count=len(pool), draws=len(values), unique=len(counts),
                    maximum_repeats=max(counts.values()), mean_draws=len(values) / len(pool), loss_mass=.125,
                    views_per_draw=1,
                    classification_views=len(values),
                    fixed_view_factor=self.view_factors[group], available_fixed_views=len(pool) * self.view_factors[group],
                    unique_virtual_views_this_epoch=len({s['base_view_id'] for _, s, _ in values}),
                    original_draws=originals, augmented_draws=len(values) - originals,
                    original_fraction=originals / len(values), target_original_fraction=.2,
                    base_families=dict(Counter(s['base_family'] for _, s, _ in values)),
                    generations=dict(Counter(str(s['base_generation']) for _, s, _ in values)))
        total = sum(self.budgets.values())
        return dict(epoch=epoch + 1, source_epoch_offset=int(self.cfg.get('source_epoch_offset',0)),
                    steps=self.steps, cells=cells, online_views=total, noisy_views=total,
                    stream_sources=self.batch, last_stream_sources=sum(self.schedule[-1]),
                    source_draws_per_stream=list(self.budgets.values()),
                    group_quota_ranges={f'{g[0]}/{g[1]}': [min(q[j] for q in self.schedule), max(q[j] for q in self.schedule)] for j, g in enumerate(GROUPS)},
                    stream_loss_mass={'online': .5, 'noisy': .5}, class_loss_mass={'fake': .5, 'real': .5},
                    class_view_counts={name: 2 * sum(b for g, b in self.budgets.items() if g[1] == label) for label, name in enumerate(('fake', 'real'))},
                    duplicate_waveform_ids_excluded=self.duplicates, refresh_base_variants=self.refresh_base_variants,
                    source_policy='Exact exposure budget; complete continuous cycles; no per-stream/group step duplicates; no stored augmented waveforms.',
                    pcl=dict(enabled=self.consistency, active_this_epoch=self.consistency and epoch >= self.cfg.get('warm_epochs', 1),
                             official_pairs=len(self.pairs), platform_target_per_group=self.platform_count))

    def audit_budget(self, epochs=20):
        if type(epochs) is not int or epochs < 1: raise ValueError('epochs must be positive')
        cells = {}
        for kind, groups in self.pools.items():
            for group, pool in groups.items():
                counts = Counter(); views = Counter(); slots = set(); families = Counter(); originals = 0
                for epoch in range(epochs):
                    for index, spec, _ in self.view_stream(kind, group, epoch):
                        counts[index] += 1; views[spec['base_view_id']] += 1
                        if not spec['base_dynamic']: slots.add((index, spec['base_variant']))
                        families[spec['base_family']] += 1; originals += spec['base_family'] == 'original'
                draws = self.budgets[group] * epochs; available = len(pool) * self.view_factors[group]
                cells[f'{kind}/{group[0]}/{group[1]}'] = dict(
                    source_count=len(pool), unique_sources_seen=len(counts), source_draws=draws,
                    source_draws_per_epoch=self.budgets[group], classification_views=draws,
                    fixed_view_factor=self.view_factors[group], available_fixed_views=available,
                    fixed_views_seen=len(slots), fixed_views_unseen=available-len(slots),
                    all_fixed_views_covered=len(slots) == available, realized_views_seen=len(views),
                    repeated_view_draws=draws-len(views), maximum_view_visits=max(views.values()),
                    minimum_source_visits=min(counts.get(i, 0) for i in pool), maximum_source_visits=max(counts.values()),
                    mean_source_visits=draws / len(pool), original_draws=originals,
                    augmented_draws=draws-originals, original_fraction=originals/draws, families=dict(families))
        return dict(epochs=epochs, steps_per_epoch=self.steps, cells=cells,
                    all_fixed_views_covered=all(c['all_fixed_views_covered'] for c in cells.values()),
                    all_expanded_views_covered=all(c['all_fixed_views_covered'] for c in cells.values() if c['fixed_view_factor'] > 1),
                    all_source_recordings_seen=all(c['unique_sources_seen'] == c['source_count'] for c in cells.values()),
                    storage='Generated on demand from original recordings; no waveform expansion on disk.',
                    note='A source draw, a generated view, and an independent source recording are different counts.')


def describe(rows, seed, stream_sources, quotas, view_factors=DEFAULT_VIEW_FACTORS,
             refresh_base_variants=True, *, cfg=None):
    plan = Plan(rows, stream_sources, seed, quotas, view_factors, refresh_base_variants, cfg=cfg)
    result = plan.coverage(0)
    result['full_budget'] = (cfg or {}).get('augmentation_budget_audit') or plan.audit_budget(int((cfg or {}).get('epochs', 20)))
    result.update(view_families=list(FAMILIES), waveform_cache=False)
    return result
