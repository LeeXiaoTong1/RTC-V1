"""Deterministic coverage of existing Train views, with committed-only history.

Four source pairs per step, two per class. A batch shares bank/family/SNR band;
English quotas are enforced within every condition and class. Thus labels see
identical processing distributions, without a second augmentation or forward.
"""
from collections import Counter, defaultdict
import copy
import math
import random

from rtc_noisy_v2.plan import digest_json, stable_seed
from rtc_noisy_v2.sampling import RotatingViewBatchSampler
from .language import language_id


def cell_key(bank, family, band, label, language):
    return f'{bank}|{family}|{band}|{label}|{language}'


class CoverageViewBatchSampler(RotatingViewBatchSampler):
    def __init__(self, source_sampler, dataset, seed=1234, en_real=.35, en_fake=.40):
        super().__init__(source_sampler, dataset.sources, seed, len(dataset.banks), 'cycle')
        if getattr(source_sampler, 'pairs_per_batch', None) != 4:
            raise ValueError('Coverage requires exactly four source pairs per step')
        self.targets = [float(en_fake), float(en_real)]
        if any(not math.isfinite(q) or not 0 < q < 1 for q in self.targets):
            raise ValueError('Coverage English targets must lie strictly between 0 and 1')
        self.pools, self.conditions, identities = {}, {}, []
        self.labels = [s['label'] for s in dataset.sources]
        self.languages = [language_id(s['offline']) for s in dataset.sources]
        self.families = {}
        for bank, rows_by_source in enumerate(dataset.banks):
            families = set()
            for index, source in enumerate(dataset.sources):
                for band, row in sorted(rows_by_source[source['offline']].items()):
                    family = row.get('processing', {}).get('family', 'ffmpeg')
                    if family not in ('ffmpeg', 'webrtc'):
                        raise ValueError('Only existing Train ffmpeg/webrtc views may enter coverage')
                    key = cell_key(bank, family, band, source['label'], self.languages[index])
                    self.pools.setdefault(key, []).append(index)
                    self.families[index, bank, band] = family
                    families.add(family)
                    identities.append([bank, index, band, family, source['label']])
            self.conditions[bank] = [(family, band) for family in sorted(families) for band in range(4)]
            for family, band in self.conditions[bank]:
                for label in (0, 1):
                    for language in (0, 1):
                        key = cell_key(bank, family, band, label, language)
                        if len(self.pools.get(key, [])) < 2:
                            raise ValueError(f'Train coverage cell needs >=2 distinct sources: {key}')
        self.pool_digest = digest_json(identities)
        self._history = {'bank_batches': [0]*self.banks, 'condition_batches': {}, 'pool_draws': {}}
        self._pending_history = None

    def _draw(self, key, history, shuffled, excluded):
        pool = self.pools[key]
        while True:
            visit = history['pool_draws'].get(key, 0)
            cycle, slot = divmod(visit, len(pool))
            identity = (key, cycle)
            if identity not in shuffled:
                order = pool.copy()
                random.Random(stable_seed(self.seed, 'coverage_sources', key, cycle)).shuffle(order)
                shuffled[identity] = order
            index = shuffled[identity][slot]
            history['pool_draws'][key] = visit+1
            if index not in excluded:
                return index

    def set_epoch(self, epoch):
        if self._plan is not None or epoch != self.completed_epoch+1:
            raise ValueError('Commit/discard the previous plan and use consecutive coverage epochs')
        if self.banks > 1 and self.mixture is None:
            raise ValueError('Coverage across banks requires an explicit scheduled mixture')
        history, shuffled, plan = copy.deepcopy(self._history), {}, []
        visits = self.visits.copy()
        bank_visits = [row.copy() for row in self.bank_visits] if self.mixture else None
        for step in range(len(self)):
            bank = 0
            if self.mixture:
                n = (epoch-1)*len(self)+step
                before, after = self._extra_batches(n), self._extra_batches(n+1)
                if after > before:
                    bank = 1+(after-1) % (self.banks-1)
            cycle, slot = divmod(history['bank_batches'][bank], len(self.conditions[bank]))
            conditions = self.conditions[bank].copy()
            random.Random(stable_seed(self.seed, 'coverage_conditions', bank, cycle)).shuffle(conditions)
            family, band = conditions[slot]
            history['bank_batches'][bank] += 1
            condition = f'{bank}|{family}|{band}'
            count = history['condition_batches'].get(condition, 0)
            tickets, used = [], set()
            for label in (0, 1):
                # Within each processing condition, cumulative English quota error <1.
                phase = random.Random(stable_seed(self.seed, 'coverage_phase', condition, label)).random()
                q = self.targets[label]
                english = math.floor(2*q*(count+1)+phase)-math.floor(2*q*count+phase)
                for language, take in ((0, english), (1, 2-english)):
                    key = cell_key(bank, family, band, label, language)
                    for _ in range(take):
                        index = self._draw(key, history, shuffled, used)
                        used.add(index)
                        tickets.append((index, bank, band))
                        visits[index] += 1
                        if bank_visits is not None:
                            bank_visits[index][bank] += 1
            history['condition_batches'][condition] = count+1
            random.Random(stable_seed(self.seed, 'coverage_batch', epoch, step)).shuffle(tickets)
            plan.append(tuple(tickets))
        self._plan, self._epoch, self._end_visits = tuple(plan), epoch, visits
        self._end_bank_visits, self._pending_history = bank_visits, history

    def plan_summary(self):
        result = super().plan_summary()
        cells, unique = Counter(), defaultdict(set)
        for batch in self._plan:
            for index, bank, band in batch:
                key = cell_key(bank, self.families[index, bank, band], band,
                               self.labels[index], self.languages[index])
                cells[key] += 1
                unique[key].add(index)
        result.update(schema='w2v_coverage_plan_v1', en_targets=self.targets,
                      cell_key='bank|family|band|label(fake=0,real=1)|language(en=0,zh=1)',
                      cells=[{'cell': key, 'views': cells[key], 'unique_sources': len(unique[key]),
                              'pool_sources': len(self.pools[key])} for key in sorted(self.pools)],
                      metadata_digest=self.pool_digest)
        return result

    def commit_epoch(self, completed_steps):
        pending = self._pending_history
        super().commit_epoch(completed_steps)
        self._history, self._pending_history = pending, None

    def discard_epoch(self):
        super().discard_epoch()
        self._pending_history = None

    def state_dict(self):
        result = super().state_dict()
        result.update(format='rtc_coverage_rotation_v1', en_targets=self.targets.copy(),
                      pool_digest=self.pool_digest, history=copy.deepcopy(self._history))
        return result

    def load_state_dict(self, state):
        if (state.get('format') != 'rtc_coverage_rotation_v1' or state.get('en_targets') != self.targets
                or state.get('pool_digest') != self.pool_digest):
            raise ValueError('Coverage pools or target distribution changed on resume')
        history = state.get('history')
        if not isinstance(history, dict) or set(history) != {'bank_batches', 'condition_batches', 'pool_draws'}:
            raise ValueError('Invalid coverage history')
        bank = history['bank_batches']
        if (not isinstance(bank, list) or len(bank) != self.banks
                or any(type(v) is not int or v < 0 for v in bank)):
            raise ValueError('Invalid coverage bank history')
        conditions = {f'{b}|{f}|{s}' for b, values in self.conditions.items() for f, s in values}
        for name, allowed in (('condition_batches', conditions), ('pool_draws', set(self.pools))):
            if (not isinstance(history[name], dict) or not set(history[name]) <= allowed
                    or any(type(v) is not int or v < 0 for v in history[name].values())):
                raise ValueError('Invalid coverage cell history')
        if sum(bank) != state.get('completed_epoch', -1)*len(self) or sum(history['condition_batches'].values()) != sum(bank):
            raise ValueError('Coverage completed steps do not match history')
        legacy = {**state, 'format': 'rtc_view_rotation_v2'}
        super().load_state_dict(legacy)
        self._history = copy.deepcopy(history)


class CoverageMeter:
    """Count only batches whose optimizer step completed, never prefetched rows."""
    def __init__(self):
        self.steps = 0
        self.crops, self.cells = Counter(), Counter()
        self.sources, self.windows = defaultdict(set), set()
        self.ordinary_sources = set()

    def update(self, branches):
        self.steps += 1
        for row in branches.get('ordinary', []):
            crop = row['crop']
            group = f'{row["label"]}|{row["language"]}'
            self.crops[group+'|'+crop['kind']] += 1
            self.ordinary_sources.add(row['source_id'])
            self.windows.add((row['source_id'], crop['start']))
        for row in branches.get('noisy_pair', []):
            key = cell_key(row['bank'], row['family'], row['band'], row['label'], row['language'])
            self.cells[key] += 1
            self.sources[key].add(row['source_id'])

    def result(self):
        return {'schema': 'w2v_coverage_actual_v1', 'completed_steps': self.steps,
                'ordinary_views': sum(self.crops.values()), 'ordinary_unique_files': len(self.ordinary_sources),
                'ordinary_unique_windows': len(self.windows), 'crop_counts': dict(sorted(self.crops.items())),
                'noisy_processed_views': sum(self.cells.values()),
                'cells': [{'cell': k, 'views': n, 'unique_sources': len(self.sources[k])}
                          for k, n in sorted(self.cells.items())]}

    def verify(self, planned, ordinary_count):
        if len(self.ordinary_sources) != ordinary_count or sum(self.crops.values()) != ordinary_count:
            raise RuntimeError('Coverage ordinary traversal omitted or repeated files')
        expected = {row['cell']: row['views'] for row in planned['cells'] if row['views']}
        if dict(self.cells) != expected:
            raise RuntimeError('Consumed noisy coverage differs from the planned epoch')
