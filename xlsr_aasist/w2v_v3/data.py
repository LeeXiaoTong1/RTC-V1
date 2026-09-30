"""Complete official Train traversal with complete, non-repeated two-view noisy coverage."""
from collections import Counter, defaultdict
import heapq
import math
import random
import torch
from torch.utils.data import DataLoader
from w2v_aasist.data import (AudioDataset as BaseDataset, FeatureCollator, build_data as checked_data,
                            read_protocol, stable_seed, worker_init)


def class_weights(rows):
    counts = torch.bincount(torch.tensor([r['label'] for r in rows]), minlength=2)
    if not bool((counts > 0).all()):
        raise ValueError('Both real and fake are required in each supervision component')
    return counts.sum().float() / (2 * counts.float())


def stratified_order(records, indices, seed):
    """Spread existing strata through the epoch without changing their total counts."""
    groups = defaultdict(list)
    for index in indices:
        r = records[index]
        groups[(r['label'], r['language'], r.get('version', -1))].append(index)
    for group, values in sorted(groups.items()):
        random.Random(stable_seed(seed, 0, group)).shuffle(values)
    # Midpoints distribute rare strata across the entire epoch, not just its beginning.
    # Materialized metadata avoids late-bound generator variables; no audio is cached here.
    streams = []
    for group, values in sorted(groups.items()):
        streams.append([((j+.5)/len(values), group, index) for j,index in enumerate(values)])
    return [index for _, _, index in heapq.merge(*streams)]


class FullEpochPlan:
    def __init__(self, ordinary, banks, ordinary_batch=16, noisy_batch=16, seed=1234):
        if len(banks) != 1 or not banks[0] or not all(r.get('full_length') for r in banks[0]):
            raise ValueError('V3 requires exactly one complete full-duration two-view Train cache')
        if ordinary_batch < 2 or noisy_batch < 2:
            raise ValueError('V3 component batch targets must be >= 2')
        self.ordinary, self.banks, self.seed = ordinary, banks, seed
        self.records = list(ordinary) + list(banks[0])
        self.full = True
        self.steps = max(math.ceil(len(ordinary)/ordinary_batch), math.ceil(len(banks[0])/noisy_batch))
        if self.steps > min(len(ordinary), len(banks[0])):
            raise ValueError('Batch targets leave an empty supervised component; increase target batch size')
        expected = Counter(r['id'] for r in ordinary if r['domain'] == 'offline')
        actual = Counter((r['id'], r.get('version')) for r in banks[0])
        if any(v != 1 for v in expected.values()) or actual != Counter({(k,v):1 for k in expected for v in (0,1)}):
            raise ValueError('Every Offline Train source must have exactly versions 0 and 1')
        self.noisy_weights = class_weights(banks[0])

    def batches(self, epoch):
        n, s = len(self.ordinary), len(self.banks[0])
        ordinary = stratified_order(self.records, range(n), self.seed + 1009*epoch)
        noisy = stratified_order(self.records, range(n, n+s), self.seed + 13007*epoch)
        result = []
        for step in range(self.steps):
            batch = ordinary[step*n//self.steps:(step+1)*n//self.steps]
            for index in noisy[step*s//self.steps:(step+1)*s//self.steps]:
                batch.append((index, index, 'full', (epoch-1)*s + index-n))
            result.append(batch)
        return result

    def coverage(self):
        def describe(rows):
            return dict(Counter('/'.join(map(str, (r['language'],r['label'],r.get('version','raw')))) for r in rows))
        return {'ordinary_rows_per_epoch':len(self.ordinary),
                'noisy_rows_per_epoch':len(self.banks[0]), 'steps':self.steps,
                'ordinary_groups':describe(self.ordinary), 'noisy_groups':describe(self.banks[0]),
                'note':'Each listed row exactly once per completed epoch; no class oversampling or extra language weights.'}


class AudioDataset(BaseDataset):
    def __init__(self, *args, rawboost_probability=.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= rawboost_probability <= 1:
            raise ValueError('rawboost_probability must be in [0,1]')
        self.rawboost_probability = rawboost_probability

    def __getitem__(self, ticket):
        index = ticket[0] if isinstance(ticket, (tuple,list)) else ticket
        gate = random.Random(stable_seed(self.seed,self.epoch,('rawboost-gate',index))).random()
        algorithm = self.rawboost
        use = self.training and not self.records[index]['noisy'] and gate < self.rawboost_probability
        try:
            self.rawboost = algorithm if use else 0
            row = super().__getitem__(ticket)
        finally:
            self.rawboost = algorithm
        row['augmentation'] = 'rawboost' if use and algorithm else 'unchanged'
        return row


def loader(records, cfg, *, training=False, epoch=0, batches=None):
    dataset = AudioDataset(records, training=training, epoch=epoch, seed=cfg['seed'],
                           max_seconds=0., rawboost=cfg['rawboost'] if training else 0,
                           raw_config=cfg['raw_config'], rawboost_probability=cfg.get('rawboost_probability',.5))
    kwargs = dict(dataset=dataset, collate_fn=FeatureCollator(cfg['ssl_path']),
                  num_workers=cfg['workers'], pin_memory=False,
                  generator=torch.Generator().manual_seed(cfg['seed']+epoch))
    if cfg['workers']:
        kwargs.update(multiprocessing_context='spawn',worker_init_fn=worker_init,prefetch_factor=2)
    if batches is None:
        kwargs.update(batch_size=cfg.get('eval_batch',8),shuffle=False)
    else:
        kwargs['batch_sampler'] = batches
    return DataLoader(**kwargs)


def build_data(cfg):
    if not cfg.get('full_noisy') or cfg.get('max_seconds',0.) != 0:
        raise ValueError('V3 requires full original and full noisy utterances without a time cap')
    old, validation, weights, counts, fingerprints = checked_data(cfg)
    plan = FullEpochPlan(old.ordinary,old.banks,cfg['ordinary_batch'],cfg['noisy_batch'],cfg['seed'])
    return plan, validation, weights, counts, fingerprints
