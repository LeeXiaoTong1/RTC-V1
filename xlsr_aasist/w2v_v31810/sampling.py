"""Exact source budgets, globally distinct recordings, rotating condition tasks."""
from collections import Counter, deque, defaultdict
import math
import numpy as np
from w2v_v3189.sampling import Plan as CyclePlan, DEFAULT_SOURCE_DRAWS
from w2v_v3185.sampling import DEFAULT_VIEW_FACTORS
from w2v_v3185.common import GROUPS, seed_for
from w2v_v3184.noise import CATEGORIES
from w2v_v318.sampling import probe_tickets
from .identity import canonical_sources

FAMILIES = ('webrtc', 'ffmpeg', 'light')
TASKS = ('noise', 'rtc', 'joint')


def quota_schedule(budgets, batch=24):
    if len(budgets) != 4 or any(type(n) is not int or n < 1 for n in budgets):
        raise ValueError('Four positive integer exposure budgets are required')
    if type(batch) is not int or batch < 8 or batch % 2:
        raise ValueError('Each stream requires an even capacity >= 8 for both eight-cell sides')
    total = sum(budgets); steps = math.ceil(total / batch)
    sizes = [batch] * (steps - 1) + [total - batch * (steps - 1)]
    if any(n % 2 for n in sizes):
        raise ValueError('Both sides need equal stream counts; total stream exposure must be even')
    if len(sizes) > 1 and sizes[-1] < 8:
        shift = 8 - sizes[-1]; sizes[-2] -= shift; sizes[-1] += shift
    cumulative = [0] * 4; consumed = 0; result = []
    for size in sizes:
        consumed += size
        values = [consumed * b // total for b in budgets]
        rank = sorted(range(4), key=lambda j: (-(consumed * budgets[j] % total), j))
        for j in rank[:consumed - sum(values)]: values[j] += 1
        quotas = tuple(values[j] - cumulative[j] for j in range(4))
        if min(quotas) < 2:
            raise ValueError('Exposure budgets cannot provide every stream/language/class on both sides')
        result.append(quotas); cumulative = values
    return result


def _side_quotas(quotas, step):
    left = [q // 2 for q in quotas]
    odd = [j for j, q in enumerate(quotas) if q % 2]
    # Rotate integer remainders; do not tie the extra X sample to one class.
    if odd:
        offset = step % len(odd); odd = odd[offset:] + odd[:offset]
        for j in odd[:len(odd) // 2]: left[j] += 1
    return tuple(left), tuple(q - a for q, a in zip(quotas, left))


def validate_units(units, quotas=None):
    origins = set(); counts = Counter()
    for unit in units:
        if len(unit) != 1: raise ValueError('Meta tasks contain exactly one classification view per source')
        row = unit[0]
        if row.get('split') != 'train' or row.get('role') not in ('online', 'noisy'):
            raise ValueError('Only Train Online/Noisy may enter classification')
        if any(key in row for key in ('pcl_refs', 'pcl_pairs')):
            raise ValueError('Meta classification must not encode auxiliary references')
        source = row.get('canonical_source')
        if not source or source in origins: raise ValueError('Original recording repeated in a logical task')
        origins.add(source)
        if row.get('meta_side') not in ('A', 'B'): raise ValueError('Explicit A/B side required')
        counts[row['meta_side'], row['role'], row['language'], row['label']] += 1
    if len(counts) != 16: raise ValueError('Both task sides must contain all eight classification cells')
    for side in ('A', 'B'):
        sizes = [sum(n for (s, r, _, _), n in counts.items() if s == side and r == role)
                 for role in ('online', 'noisy')]
        if sizes[0] != sizes[1]: raise ValueError('Each side requires equal Online/Noisy view counts')
    if quotas is not None:
        for role in ('online', 'noisy'):
            for j, group in enumerate(GROUPS):
                if sum(counts[s, role, *group] for s in ('A', 'B')) != quotas[j]:
                    raise ValueError('Task source exposure quotas changed')


class Plan(CyclePlan):
    def __init__(self, rows, stream_sources=24, seed=31801, quotas=(6, 6, 6, 6),
                 view_factors=DEFAULT_VIEW_FACTORS, refresh_base_variants=True, *, cfg=None):
        cfg = dict(cfg or {}); cfg['consistency'] = 'none'
        self.canonical, self.identity_audit = canonical_sources(rows,
            cfg.get('train_pairs') or cfg.get('train_pair_path'), require_online_pairs=True)
        if any(r.get('split') != 'train' for r in rows): raise ValueError('Training sampler accepts Train only')
        super().__init__(rows, stream_sources, seed, quotas, view_factors, refresh_base_variants, cfg=cfg)
        self.schedule = quota_schedule(tuple(cfg.get('source_draws_per_stream', DEFAULT_SOURCE_DRAWS)), stream_sources)
        self.steps = len(self.schedule)
        self.prefixes = {g: [0] for g in GROUPS}
        for counts in self.schedule:
            for j, group in enumerate(GROUPS): self.prefixes[group].append(self.prefixes[group][-1] + counts[j])
        self.canonical_duplicates = []
        for kind, groups in self.pools.items():
            for group, pool in groups.items():
                seen = set(); keep = []
                for index in pool:
                    key = self.canonical[index]
                    if key in seen: self.canonical_duplicates.append(rows[index]['id'])
                    else: seen.add(key); keep.append(index)
                groups[group] = keep
        for j, group in enumerate(GROUPS):
            need = max(q[j] for q in self.schedule)
            sets = [{self.canonical[i] for i in self.pools[kind][group]} for kind in ('online', 'offline')]
            if min(map(len, sets)) < need or len(sets[0] | sets[1]) < 2 * need:
                raise ValueError(f'{group}: insufficient distinct original recordings for {2 * need} views/task')
        self.experiment = cfg.get('experiment', 'E2')
        if self.experiment not in ('E0', 'E1', 'E2'): raise ValueError('Expected experiment E0, E1 or E2')
        self._planned_epoch = None; self._planned = None; self._repair_counts = {}

    def _repair(self, online, offline, group):
        """Permute existing exposure tickets only; no new draws or discarded variants.

        Fast two-slot exchanges handle ordinary sparse collisions. An augmenting
        path relocates existing tickets when a direct exchange is not sufficient.
        Each ticket retains its original cycle/variant/generation metadata.
        """
        prefix = self.prefixes[group]; bins = [list(offline[a:b]) for a, b in zip(prefix, prefix[1:])]
        forbidden = [{self.canonical[t[0]] for t in online[a:b]} for a, b in zip(prefix, prefix[1:])]
        repairs = 0
        def origin(token): return self.canonical[token[0]]
        def can_enter(token, b, slot):
            key = origin(token)
            return key not in forbidden[b] and all(i == slot or x is None or origin(x) != key for i, x in enumerate(bins[b]))
        for b, entries in enumerate(bins):
            for slot in range(len(entries)):
                token = bins[b][slot]
                if can_enter(token, b, slot): continue
                repaired = False
                for distance in range(1, len(bins)):
                    other = (b + distance) % len(bins)
                    for target, candidate in enumerate(bins[other]):
                        if can_enter(candidate, b, slot) and can_enter(token, other, target):
                            bins[b][slot], bins[other][target] = candidate, token
                            repaired = True; break
                    if repaired: break
                if not repaired:
                    bins[b][slot] = None
                    # Nodes are displaced ticket slots, beginning with the hole.
                    queue = deque([(b, slot)]); previous = {(b, slot): None}; tokens = {(b, slot): token}
                    terminal = None
                    while queue and terminal is None:
                        node = queue.popleft(); moving = tokens[node]
                        if can_enter(moving, b, slot): terminal = node; break
                        # Mutate at most one slot per destination bin along an
                        # augmenting path, so each admission test remains valid
                        # after all displaced tickets are committed together.
                        ancestry = set(); ancestor = node
                        while ancestor is not None:
                            ancestry.add(ancestor[0]); ancestor = previous[ancestor]
                        for target_bin, target_entries in enumerate(bins):
                            if target_bin in ancestry: continue
                            if origin(moving) in forbidden[target_bin]: continue
                            for target_slot, candidate in enumerate(target_entries):
                                target_node = target_bin, target_slot
                                if candidate is None or target_node in previous: continue
                                if can_enter(moving, target_bin, target_slot):
                                    previous[target_node] = node; tokens[target_node] = candidate; queue.append(target_node)
                    if terminal is None:
                        bins[b][slot] = token
                        raise ValueError(f'{group}: exact source-isolated exposure assignment is infeasible; no budget was changed')
                    bins[b][slot] = tokens[terminal]
                    while previous[terminal] is not None:
                        predecessor = previous[terminal]
                        bins[terminal[0]][terminal[1]] = tokens[predecessor]; terminal = predecessor
                repairs += 1
        # Exchanges move source visits, but must not expose a refreshed variant
        # before the same source's earlier cycle. Reattach that source's exact
        # queued variants in chronological cycle order; the multiset is unchanged.
        chronological = defaultdict(list)
        for token in offline: chronological[token[0]].append(token)
        chronological = {i: deque(sorted(tokens, key=lambda t: t[2])) for i, tokens in chronological.items()}
        result = [chronological[token[0]].popleft() for entries in bins for token in entries]
        if Counter((i, s['base_view_id'], c) for i, s, c in result) != Counter((i, s['base_view_id'], c) for i, s, c in offline):
            raise RuntimeError('Source isolation changed the exact exposure multiset')
        return result, repairs

    def _streams(self, epoch):
        if self._planned_epoch != epoch:
            streams = {(kind, g): self.view_stream(kind, g, epoch) for kind in ('online', 'offline') for g in GROUPS}
            repairs = {}
            for group in GROUPS:
                streams['offline', group], repairs[str(group)] = self._repair(streams['online', group], streams['offline', group], group)
            self._planned_epoch = epoch; self._planned = streams; self._repair_counts = repairs
        return self._planned

    def _conditions(self, number, block, rank):
        task = TASKS[number % 3]; round_ = number // 3
        scene = CATEGORIES[round_ % len(CATEGORIES)]
        family = FAMILIES[(round_ // len(CATEGORIES)) % len(FAMILIES)]
        scenes = [s for s in CATEGORIES if s != scene]; families = [f for f in FAMILIES if f != family]
        nuisance_seed = seed_for(self.seed, 'v31810-nuisance', number, rank)
        rng = np.random.default_rng(nuisance_seed)
        common_family = str(rng.choice(FAMILIES, p=(.5, .3, .2)))
        common_scene = CATEGORIES[(round_ + rank) % len(CATEGORIES)]
        selected_scene = (scene if block == 'Y' else scenes[(round_ + rank) % len(scenes)]) if task != 'rtc' else common_scene
        selected_family = (family if block == 'Y' else families[(round_ + rank) % len(families)]) if task != 'noise' else common_family
        return dict(scene=selected_scene, family=selected_family), nuisance_seed, dict(scene=scene, family=family)

    def batches(self, epoch):
        streams = self._streams(epoch)
        for step, quotas in enumerate(self.schedule):
            number = epoch * self.steps + step; qx, qy = _side_quotas(quotas, number)
            # Flip after every full (3 tasks x 6 scenes x 3 families) cycle too.
            # Parity alone would permanently tie each held-out scene to a role.
            forward = (number + number // (len(TASKS) * len(CATEGORIES) * len(FAMILIES))) % 2 == 0
            tickets = []; condition_slots = {}
            for j, group in enumerate(GROUPS):
                begin = self.prefixes[group][step]
                for kind in ('online', 'offline'):
                    for k in range(quotas[j]):
                        index, spec, cycle = streams[kind, group][begin + k]
                        block = 'X' if k < qx[j] else 'Y'; rank = k if block == 'X' else k - qx[j]
                        slot = block, rank
                        if slot not in condition_slots:
                            condition_slots[slot] = self._conditions(number, block, rank)
                        domain, nuisance, heldout = condition_slots[slot]
                        side = ('A' if block == 'X' else 'B') if forward else ('B' if block == 'X' else 'A')
                        tickets.append(dict(index=index, kind=kind, epoch=epoch, step=step, **spec,
                            source_cycle=cycle, canonical_source=self.canonical[index], batch_group_quotas=quotas,
                            occurrence=f'v31810:{epoch}:{step}:{kind}:{group[0]}:{group[1]}:{k}',
                            meta_side=side, condition_block=block, task_kind=TASKS[number % 3],
                            task_direction='X_to_Y' if forward else 'Y_to_X', condition_rank=rank,
                            domain_override=domain if kind == 'offline' else None,
                            nuisance_seed=nuisance, heldout_condition=heldout, noise_recipe_view=cycle % 2))
            if self.experiment == 'E1':
                # The waveform recipe does NOT read meta_side. Within every cell,
                # alternate X/Y provenance before splitting into the same quotas.
                for j, group in enumerate(GROUPS):
                    for kind in ('online', 'offline'):
                        cell = [t for t in tickets if t['kind'] == kind and (self.rows[t['index']]['language'], self.rows[t['index']]['label']) == group]
                        old_a = sum(t['meta_side'] == 'A' for t in cell)
                        xx = [t for t in cell if t['condition_block'] == 'X']; yy = [t for t in cell if t['condition_block'] == 'Y']
                        rng = np.random.default_rng(seed_for(self.seed, 'v31810-E1-mix', epoch, step, kind, j))
                        rng.shuffle(xx); rng.shuffle(yy)
                        if number % 2: xx, yy = yy, xx
                        interleaved = [t for pair in zip(xx, yy) for t in pair] + xx[min(len(xx), len(yy)):] + yy[min(len(xx), len(yy)):]
                        for k, ticket in enumerate(interleaved): ticket['meta_side'] = 'A' if k < old_a else 'B'
            if len({t['canonical_source'] for t in tickets}) != len(tickets):
                raise RuntimeError('Original recording repeated after task construction')
            order = np.random.default_rng(seed_for(self.seed, 'v31810-ticket-order', epoch, step)).permutation(len(tickets))
            yield [tickets[i] for i in order]

    def calibration_batches(self):
        cfg = dict(self.cfg, experiment='E2', source_epoch_offset=0)
        separate = Plan(self.rows, self.batch, seed_for(self.seed, 'v31810-alpha-calibration'),
            tuple(self.quotas.values()), tuple(self.view_factors.values()), self.refresh_base_variants, cfg=cfg)
        counts = Counter(); selected = []
        # Calibrate the joint-training input distribution, after head warmup.
        first_epoch = int(self.cfg.get('warm_epochs', 1))
        for epoch in range(first_epoch, first_epoch + 12):
            for batch in separate.batches(epoch):
                task = batch[0]['task_kind']
                if counts[task] < 4:
                    for ticket in batch:
                        ticket['calibration'] = True
                        ticket['occurrence'] = 'alpha-calibration:' + ticket['occurrence']
                        ticket['recipe_seed'] = separate.seed
                    selected.append(batch); counts[task] += 1
                if len(selected) == 12: return selected
        raise ValueError('Cannot construct twelve calibration tasks with all three conditions')

    def coverage(self, epoch):
        result = super().coverage(epoch); self._streams(epoch)
        result.update(canonical_identity=self.identity_audit,
            canonical_duplicate_rows_excluded=self.canonical_duplicates,
            source_assignment_repairs=self._repair_counts,
            source_policy='Exact continuous-cycle ticket multiset; reordered for 48 globally distinct original recordings.',
            experiment=self.experiment, task_kinds=list(TASKS), pcl=dict(enabled=False),
            side_classification_weights='Eight cell means, each 1/8, normalized over complete side counts',
            final_task_views=2 * sum(self.schedule[-1]))
        result['task_counts'] = dict(Counter(TASKS[(epoch * self.steps + step) % 3] for step in range(self.steps)))
        return result

    def audit_budget(self, epochs=15):
        result = super().audit_budget(epochs)
        result.update(canonical_identity=self.identity_audit, no_auxiliary_views=True,
            source_assignment='Permutation only: original source/cycle/variant tickets are neither dropped nor added')
        return result


def describe(rows, seed, stream_sources, quotas, view_factors=DEFAULT_VIEW_FACTORS,
             refresh_base_variants=True, *, cfg=None):
    plan = Plan(rows, stream_sources, seed, quotas, view_factors, refresh_base_variants, cfg=cfg)
    result = plan.coverage(0)
    result['full_budget'] = (cfg or {}).get('augmentation_budget_audit') or plan.audit_budget(int((cfg or {}).get('epochs', 15)))
    result['waveform_cache'] = False
    return result
