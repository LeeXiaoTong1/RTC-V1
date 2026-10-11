"""Source-centric sampling: tempered exposure, exactly equal group loss mass."""
from collections import Counter
import math
import numpy as np
from .common import GROUPS, seed_for


def inventory(rows):
    result = {}
    ids = set()
    audio_metadata = {}
    for index, row in enumerate(rows):
        if row['condition'] not in ('offline', 'online'):
            raise ValueError('Only official Offline/Online rows may enter the source plan')
        if row['id'] in ids:
            raise ValueError('Duplicate recording ID in canonical manifest')
        ids.add(row['id'])
        metadata = row['language'], row['label']
        sha = row.get('audio_sha256')
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Missing verified recording hash')
        if sha in audio_metadata and audio_metadata[sha] != metadata:
            raise ValueError('Identical waveform has conflicting language/label metadata')
        audio_metadata[sha] = metadata
        group = row['group_id']
        if (row.get('split') != 'train' or row.get('full_length') is not True
                or row.get('source_sha256') != group or len(group) != 64
                or any(c not in '0123456789abcdef' for c in group)
                or (row['language'], row['label']) not in GROUPS):
            raise ValueError('Missing verified full-wave Train source identity')
        item = result.setdefault(group, dict(language=row['language'], label=row['label'],
                                             offline=[], online=[], duplicate_ids=[]))
        if (item['language'], item['label']) != (row['language'], row['label']):
            raise ValueError('Same source has conflicting labels/language')
        if row['condition'] == 'offline' and row.get('audio_sha256') != group:
            raise ValueError('Offline hash differs from canonical source hash')
        item[row['condition']].append(index)
    for group, item in result.items():
        if not item['offline']:
            raise ValueError('Official Online has no verified original source: ' + group)
        originals = {rows[i]['id'] for i in item['offline']}
        for i in item['offline'] + item['online']:
            if rows[i]['source_id'] not in originals:
                raise ValueError('Pair source ID does not identify its original waveform')
        for kind in ('offline', 'online'):
            unique = {}
            for i in sorted(item[kind], key=lambda i: rows[i]['id']):
                waveform = rows[i]['audio_sha256']
                if waveform in unique:
                    item['duplicate_ids'].append(rows[i]['id'])
                else:
                    unique[waveform] = i
            item[kind] = list(unique.values())
    return result


class Plan:
    def __init__(self, rows, source_batch=16, seed=31801, power=.5):
        if type(source_batch) is not int or source_batch < 4 or not 0 <= power <= 1:
            raise ValueError('Require >=4 sources and a sampling power in [0,1]')
        self.rows, self.batch, self.seed, self.power = rows, source_batch, seed, power
        self.inventory = inventory(rows)
        self.pools = {g: sorted(k for k, v in self.inventory.items()
                              if (v['language'], v['label']) == g) for g in GROUPS}
        if any(not pool for pool in self.pools.values()):
            raise ValueError('All four source groups are required')
        sizes = np.array([len(self.pools[g]) for g in GROUPS], dtype=float) ** power
        target = source_batch * sizes / sizes.sum()
        quotas = np.maximum(1, np.floor(target).astype(int))
        while quotas.sum() < source_batch:
            quotas[np.argmax(target - quotas)] += 1
        while quotas.sum() > source_batch:
            quotas[np.argmax(np.where(quotas > 1, quotas - target, -np.inf))] -= 1
        self.quotas = dict(zip(GROUPS, map(int, quotas)))
        if any(len(self.pools[g]) < self.quotas[g] for g in GROUPS):
            raise ValueError('Not enough distinct sources for a within-batch unique plan')
        self.steps = math.ceil(len(self.inventory) / source_batch)

    def stream(self, group, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('Invalid epoch')
        pool, quota = self.pools[group], self.quotas[group]
        count = self.steps * quota
        start, stop = epoch * count, (epoch + 1) * count
        values, previous = [], None
        for cycle in range((stop - 1) // len(pool) + 1):
            order = np.random.default_rng(seed_for(self.seed, 'source-cycle', group, cycle)).permutation(len(pool)).tolist()
            # A reshuffle may straddle a logical batch. Do not repeat its tail
            # in the new cycle's prefix; every cycle still visits every source.
            remainder = cycle * len(pool) % quota
            if remainder:
                forbidden = set(previous[-remainder:])
                prefix = [i for i in order if i not in forbidden][:quota - remainder]
                chosen = set(prefix)
                order = prefix + [i for i in order if i not in chosen]
            left, right = max(0, start-cycle*len(pool)), min(len(pool), stop-cycle*len(pool))
            if right > left:
                values.extend(pool[i] for i in order[left:right])
            previous = order
        return values

    def ticket(self, source, epoch, step, position, cycle=0, probe=False):
        item = self.inventory[source]
        group = item['language'], item['label']
        partners = item['online']
        offset = seed_for(self.seed, 'partner', source) % max(1, len(partners))
        return dict(kind='source', index=item['offline'][0],
                    online_index=partners[(offset+cycle) % len(partners)] if partners else None,
                    epoch=epoch, step=step, source_mass=.25/self.quotas[group],
                    occurrence=('fixed-probe:'+source if probe else f'{epoch}:{step}:{position}:{source}'),
                    probe=probe)

    def batches(self, epoch):
        streams = {g: self.stream(g, epoch) for g in GROUPS}
        for step in range(self.steps):
            tickets = []
            for g in GROUPS:
                q = self.quotas[g]
                for j in range(q):
                    position = step*q+j
                    cycle = (epoch*self.steps*q+position)//len(self.pools[g])
                    tickets.append(self.ticket(streams[g][position], epoch, step, len(tickets), cycle))
            order = np.random.default_rng(seed_for(self.seed, 'physical-order', epoch, step)).permutation(self.batch)
            yield [tickets[i] for i in order]

    def coverage(self, epoch):
        cells, roles = {}, Counter()
        paired_draws = 0
        for g, pool in self.pools.items():
            counts = Counter(self.stream(g, epoch))
            paired = sum(n for s, n in counts.items() if self.inventory[s]['online'])
            total = sum(counts.values())
            paired_draws += paired
            cells[f'{g[0]}/{g[1]}'] = dict(available=len(pool), quota=self.quotas[g],
                draws=total, unique=len(counts), mean_repeats=total/len(pool),
                maximum_repeats=max(counts.values()), paired_draws=paired,
                missing_online_sources=sum(not self.inventory[s]['online'] for s in pool), loss_mass=.25)
            roles['offline'] += .25*.10
            roles['online'] += .25*.45*paired/total
            roles['noisy'] += .25*(.90-.45*paired/total)
        return dict(epoch=epoch+1, steps=self.steps, source_batch=self.batch,
            source_draws=self.steps*self.batch, available_sources=len(self.inventory),
            unique_sources=sum(c['unique'] for c in cells.values()), cells=cells,
            waveform_views=3*self.steps*self.batch, online_views=paired_draws,
            condition_loss_mass=dict(roles), class_loss_mass={'fake':.5, 'real':.5},
            sampling_power=self.power, within_batch_unique=True,
            duplicate_waveform_ids_excluded=sorted(i for v in self.inventory.values() for i in v['duplicate_ids']),
            note='Fewer exposures do not reduce the equal group objective mass or create new speakers.')


def validate_units(units, source_batch):
    if len(units) != source_batch:
        raise ValueError('Logical source count differs')
    sources, masses, counts = set(), Counter(), Counter()
    for unit in units:
        if len(unit) != 3 or [r['role'] for r in unit] not in (
                ['offline', 'online', 'noisy_a'], ['offline', 'noisy_a', 'noisy_b']):
            raise ValueError('Require Offline/Online/Noisy or Offline/Noisy/Noisy')
        first = unit[0]
        keys = ('source_id', 'source_sha256', 'group_id', 'language', 'label', 'occurrence', 'source_mass')
        if len({tuple(r[k] for k in keys) for r in unit}) != 1:
            raise ValueError('Source/view identity mismatch')
        if first['group_id'] in sources:
            raise ValueError('Repeated source in a logical batch')
        sources.add(first['group_id'])
        group = first['language'], first['label']
        counts[group] += 1
        mass = first['source_mass']
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError('Invalid source supervision mass')
        for row, ratio in zip(unit, (.1, .45, .45)):
            if row['split'] != 'train' or not math.isclose(row['ce_weight'], mass*ratio, abs_tol=1e-10):
                raise ValueError('Invalid view CE weight/split')
            masses[group] += row['ce_weight']
    if set(masses) != set(GROUPS) or any(not math.isclose(masses[g], .25, abs_tol=1e-9) for g in GROUPS):
        raise ValueError('Four groups must each own 25% CE mass')
    for unit in units:
        r = unit[0]
        if not math.isclose(r['source_mass'], .25/counts[r['language'], r['label']], abs_tol=1e-10):
            raise ValueError('Source weights must be uniform within a group')


def probe_tickets(plan, per_group=512):
    pools = {g: [s for s in p if plan.inventory[s]['online']] for g, p in plan.pools.items()}
    count = min(per_group, *(len(p) for p in pools.values()))
    if count < 1:
        raise ValueError('Fixed Online probe requires verified partners in all groups')
    return [plan.ticket(s, 0, 0, 0, probe=True) for g, p in pools.items()
            for s in sorted(p, key=lambda s: seed_for(plan.seed, 'paired-probe', s))[:count]]
