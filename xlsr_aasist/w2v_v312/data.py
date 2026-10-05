"""Reuse audio, balance source budgets, isolate independent language probes."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from w2v_aasist.data import worker_init
from w2v_v36.features import load_cache, VerifiedAudioDataset, NumpyCollator
from w2v_v36.fit import sources
from w2v_v39.data import cache_path, verify_owner, validate_rows


@contextmanager
def bundles(cfg):
    opened = {}
    try:
        for split in ('train', 'dev'):
            path = cache_path(cfg['v37_run'], split)
            verify_owner(path, split, cfg)
            opened[split] = load_cache(path)
        validate_rows(opened['train']['rows'], opened['dev']['rows'], opened['train']['rows'])
        yield opened['train'], opened['dev']
    finally:
        for bundle in opened.values():
            for value in bundle.values():
                if getattr(value, '_mmap', None) is not None:
                    value._mmap.close()


def probe_split(rows, cfg):
    inventory = sources(rows)
    real_groups = {language: sorted({item['group_id'] for item in inventory.values()
                                    if item['label'] == 1 and item['language'] == language})
                   for language in ('en', 'zh')}
    # At most 20% of each language's real sources is held out from this adaptation.
    count = min(cfg['probe_sources_per_language'], *(len(g) // 5 for g in real_groups.values()))
    if count < 4:
        raise ValueError('Need >=20 independent real Train sources per language for disjoint fit/test probes')
    fit, test = set(), set()
    for language, groups in real_groups.items():
        ordered = sorted(groups, key=lambda g: hashlib.sha256(f'{cfg["seed"]}/{language}/{g}'.encode()).hexdigest())
        fit.update(ordered[:count//2])
        test.update(ordered[count//2:count])
    probe_indices = [i for i, row in enumerate(rows) if row['group_id'] in fit | test]
    training = [row for row in rows if row['group_id'] not in fit | test]
    probe_rows = [rows[i] for i in probe_indices]
    report = dict(fit_groups=sorted(fit), test_groups=sorted(test), heldout_per_language=count,
                  overlap=0, original_encoder_previously_saw_train=True,
                  scope='held out from V3.12 adaptation only; not an unseen-corpus test')
    if fit & test or {r['group_id'] for r in training} & (fit | test):
        raise AssertionError('Probe source leakage')
    return training, probe_rows, probe_indices, report


class SourcePlan:
    """A fixed source budget, balanced EN/ZH x real/fake; cyclic shuffled pools.

    Epoch means N source draws, not a guarantee of visiting every majority source.
    Pool offsets persist across epochs so majority coverage is not reset each time.
    """
    def __init__(self, rows, batch, seed):
        if batch < 4 or batch % 4:
            raise ValueError('Source batch must be a positive multiple of four')
        self.rows, self.batch, self.seed = rows, batch, seed
        self.inventory = sources(rows)
        self.pools = defaultdict(list)
        for source, item in sorted(self.inventory.items()):
            self.pools[(item['language'], item['label'])].append(source)
        if set(self.pools) != {('en',0), ('en',1), ('zh',0), ('zh',1)}:
            raise ValueError('All four source groups are required')
        self.steps = math.ceil(len(self.inventory) / batch)

    def batches(self, epoch, start=0):
        per_group = self.steps * (self.batch // 4)
        streams = {}
        for group_index, (key, pool) in enumerate(sorted(self.pools.items())):
            offset = epoch * per_group
            values = []
            while len(values) < per_group:
                cycle, position = divmod(offset, len(pool))
                rng = np.random.default_rng(self.seed + group_index * 1000003 + cycle)
                ordered = rng.permutation(len(pool))
                chunk = [pool[i] for i in ordered[position:position + per_group - len(values)]]
                values.extend(chunk)
                offset += len(chunk)
            streams[key] = values
        result = []
        for step in range(start, self.steps):
            selected = []
            for values in streams.values():
                a = step * (self.batch // 4)
                selected.extend(values[a:a + self.batch // 4])
            indices = [i for source in selected for i in self.inventory[source]['indices'].values()]
            result.append(indices)
        return result


def loss_weights(rows):
    # Every source occurrence has ordinary mass .5 and noisy mass .5. The sampler
    # repeats all views of an occurrence together; identical IDs may repeat.
    counts = Counter((r['source_id'], r['condition']) for r in rows)
    unique = {r['source_id'] for r in rows}
    source_occurrences = sum(counts[(s, 'offline')] for s in unique)
    ce = []
    for row in rows:
        ordinary = 1 + int((row['source_id'], 'online') in counts)
        ce.append((.25 if row['condition'].startswith('noisy') else .5/ordinary) / source_occurrences)
    cells = Counter((r['condition'], r['language']) for r in rows if r['label'] == 1)
    conditions = [c for c in ('offline', 'online', 'noisy_a', 'noisy_b')
                  if cells[(c,'en')] and cells[(c,'zh')]]
    adv = [1./(len(conditions)*2*cells[(r['condition'],r['language'])])
           if r['label'] == 1 and r['condition'] in conditions else 0. for r in rows]
    if not np.isclose(sum(ce), 1.):
        raise ValueError('Classification source budget does not sum to one')
    return ce, adv


def loader(plan, epoch, start, cfg):
    dataset = VerifiedAudioDataset(plan.rows, training=False, epoch=0, seed=cfg['seed'],
                                   max_seconds=0., rawboost=0, raw_config={})
    kwargs = dict(dataset=dataset, batch_sampler=plan.batches(epoch, start),
                  collate_fn=NumpyCollator(cfg['ssl_path']), num_workers=cfg['workers'],
                  pin_memory=False, generator=torch.Generator().manual_seed(cfg['seed'] + epoch))
    if cfg['workers']:
        kwargs.update(multiprocessing_context='spawn', worker_init_fn=worker_init,
                      prefetch_factor=1, persistent_workers=False)
    return DataLoader(**kwargs)
