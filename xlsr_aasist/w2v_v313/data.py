"""Verified two-view sources, balanced online pairs, and bounded audio transport.

Canonical IDs come from the already verified official pair CSV in the V3.7
feature manifest. No filename pairing, crop, new waveform, or frame cache is
created here. One logical batch owns its CE mass before it is split for memory.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from w2v_aasist.data import worker_init
from w2v_v36.features import VerifiedAudioDataset, NumpyCollator
from w2v_v36.fit import sources
from w2v_v312.data import bundles, probe_split as _probe_split


GROUPS = (('en', 0), ('en', 1), ('zh', 0), ('zh', 1))
# Microgroups share a family, so each language has real/fake at the same condition.
# Half the pairs use the actual official Offline--Online recording relationship.
PAIR_CYCLE = (
    ('offline', 'online', 'noisy_a'),
    ('offline', 'noisy_a', 'noisy_a'),
    ('offline', 'online', 'noisy_b'),
    ('online', 'noisy_b', 'noisy_b'),
    ('offline', 'online', 'noisy_b'),
    ('offline', 'noisy_b', 'noisy_b'),
    ('offline', 'online', 'noisy_a'),
    ('online', 'noisy_a', 'noisy_a'),
)


def _inventory(rows):
    result = sources(rows)
    used_online = set()
    for source, item in result.items():
        original = rows[item['indices']['offline']]
        group = item['group_id']
        if (len(group) != 64 or any(c not in '0123456789abcdef' for c in group)
                or original.get('audio_sha256') != group or original['id'] != source):
            raise ValueError('Canonical source must be bound to its verified original waveform SHA256')
        for condition, index in item['indices'].items():
            row = rows[index]
            if (row.get('split') != 'train' or row.get('source_sha256') != group
                    or row.get('view') != 'full' or row.get('full_length') is not True):
                raise ValueError('Paired Train views must preserve full-wave source identity')
            if condition == 'online':
                if row['id'] in used_online:
                    raise ValueError('An official Online recording was paired to multiple sources')
                used_online.add(row['id'])
    return result


def probe_split(rows, cfg):
    training, probe, indices, report = _probe_split(rows, cfg)
    report['scope'] = 'held out from V3.13 adaptation only; original model previously saw Train'
    return training, probe, indices, report


def heldout_pairs(rows, cfg):
    """Hold out original Train hashes for treatment-induced error diagnostics.

    All conditions and duplicate IDs of an original waveform stay together.
    This is an adaptation holdout, not an unseen-corpus/generalization claim.
    """
    inventory = _inventory(rows)
    strata = {key: sorted({item['group_id'] for item in inventory.values()
                          if (item['language'], item['label']) == key}) for key in GROUPS}
    requested = cfg.get('pair_probe_sources_per_group', 64)
    if type(requested) is not int or requested < 4:
        raise ValueError('Pair diagnostics require at least four held-out sources per group')
    count = min(requested, *(len(values) // 5 for values in strata.values()))
    if count < 4:
        raise ValueError('Need >=20 original Train sources per group for paired holdout')
    selected = set()
    for (language, label), values in strata.items():
        ordered = sorted(values, key=lambda group: hashlib.sha256(
            f'{cfg["seed"]}/paired-diagnostic/{language}/{label}/{group}'.encode()).hexdigest())
        selected.update(ordered[:count])
    training = [row for row in rows if row['group_id'] not in selected]
    diagnostic = [row for row in rows if row['group_id'] in selected]
    if {r['group_id'] for r in training} & {r['group_id'] for r in diagnostic}:
        raise AssertionError('Pair diagnostic source leakage')
    report = dict(heldout_groups=sorted(selected), heldout_per_group=count,
                  conditions=dict(Counter(r['condition'] for r in diagnostic)),
                  rows=len(diagnostic), overlap=0, original_encoder_previously_saw_train=True,
                  scope='held out from V3.13 adaptation only; no Progress/Eval samples')
    return training, diagnostic, report


@dataclass(frozen=True)
class PairIndex:
    index: int
    pair_occurrence: str
    pair_position: int
    pair_kind: str
    requested_pair_kind: str
    microgroup: int
    ce_weight: float
    pair_weight: float


def _kind(a, b):
    if (a, b) == ('offline', 'online'):
        return 'official_offline_online'
    return a + '_noisy'


class SourcePlan:
    """A fixed balanced source budget with exactly two views per occurrence.

    Shuffled source pools continue across epoch boundaries. Minority sources are
    intentionally repeated; the exact unique coverage/repeat counts are audited.
    By default each microgroup has two sources per EN/ZH x fake/real cell, hence
    sixteen views and two opposite-class mining candidates in each language.
    A four-source/eight-view low-memory option is explicit. Occurrence IDs remain
    distinct when a minority source repeats inside the same logical batch.
    """
    def __init__(self, rows, batch, seed, micro_sources=8):
        if type(batch) is not int or batch < 4 or batch % 4:
            raise ValueError('Source batch must be a positive multiple of four')
        if type(seed) is not int or seed < 0:
            raise ValueError('Seed must be a nonnegative integer')
        if micro_sources not in (4, 8) or batch % micro_sources:
            raise ValueError('Pair microgroup must have four or eight sources and divide source batch')
        self.rows, self.batch, self.seed = rows, batch, seed
        self.micro_sources = micro_sources
        self.inventory = _inventory(rows)
        self.pools = defaultdict(list)
        for source, item in sorted(self.inventory.items()):
            self.pools[(item['language'], item['label'])].append(source)
        if set(self.pools) != set(GROUPS):
            raise ValueError('All four source groups are required')
        self.steps = math.ceil(len(self.inventory) / batch)

    def _streams(self, epoch):
        per_group = self.steps * (self.batch // 4)
        streams = {}
        for group_index, key in enumerate(GROUPS):
            pool = self.pools[key]
            offset = epoch * per_group
            values = []
            while len(values) < per_group:
                cycle, position = divmod(offset, len(pool))
                rng = np.random.default_rng(self.seed + group_index * 1000003 + cycle)
                ordered = rng.permutation(len(pool))
                remaining = per_group - len(values)
                chunk = [pool[i] for i in ordered[position:position + remaining]]
                values.extend(chunk)
                offset += len(chunk)
            streams[key] = values
        return streams

    def batches(self, epoch, start=0):
        if type(epoch) is not int or epoch < 0 or type(start) is not int or not 0 <= start <= self.steps:
            raise ValueError('Invalid deterministic epoch/resume cursor')
        streams = self._streams(epoch)
        count = self.batch // self.micro_sources
        rounds = self.micro_sources // 4
        result = []
        for step in range(start, self.steps):
            indices = []
            for microgroup in range(count):
                phase = (self.seed + epoch * (self.steps * count + 1)
                         + step * count + microgroup) % len(PAIR_CYCLE)
                first, second, fallback_noisy = PAIR_CYCLE[phase]
                requested = _kind(first, second)
                for pair_round in range(rounds):
                    position = step * (self.batch // 4) + microgroup * rounds + pair_round
                    for group_index, key in enumerate(GROUPS):
                        source = streams[key][position]
                        item = self.inventory[source]
                        a, b = first, second
                        kind = requested
                        if 'online' in (a, b) and 'online' not in item['indices']:
                            a, b = 'offline', fallback_noisy
                            kind = 'missing_online_offline_noisy'
                        occurrence = f'{epoch}:{step}:{microgroup * self.micro_sources + pair_round * 4 + group_index}'
                        for side, condition in enumerate((a, b)):
                            indices.append(PairIndex(item['indices'][condition], occurrence, side, kind,
                                                     requested, microgroup, 1. / (2 * self.batch),
                                                     1. / self.batch))
            result.append(indices)
        return result

    def coverage(self, epoch=0):
        all_batches = self.batches(epoch)
        pairs = [item for batch in all_batches for item in batch if item.pair_position == 0]
        counts = Counter(self.rows[item.index]['source_id'] for item in pairs)
        group_counts = {}
        for key in GROUPS:
            values = [counts[source] for source in self.pools[key]]
            group_counts[f'{key[0]}/{key[1]}'] = dict(available=len(values), draws=sum(values),
                unique=sum(value > 0 for value in values), minimum_repeats=min(values),
                maximum_repeats=max(values))
        return dict(epoch=epoch, steps=self.steps, sources_per_batch=self.batch,
                    sources_per_microgroup=self.micro_sources, views_per_microgroup=self.micro_sources * 2,
                    opposite_class_candidates_per_language=self.micro_sources // 4,
                    source_draws=len(pairs), audio_views=2 * len(pairs),
                    available_sources=len(self.inventory), unique_sources=len(counts),
                    missing_online_sources=sum('online' not in item['indices'] for item in self.inventory.values()),
                    actual_pair_kinds=dict(Counter(item.pair_kind for item in pairs)),
                    requested_pair_kinds=dict(Counter(item.requested_pair_kind for item in pairs)),
                    view_conditions=dict(Counter(self.rows[item.index]['condition']
                                                 for batch in all_batches for item in batch)),
                    source_groups=group_counts, original_audio_cache_bytes_added=0,
                    policy='two full views; official pairs primary; rotating existing noisy views')


class PairedAudioDataset(VerifiedAudioDataset):
    def __getitem__(self, index):
        if not isinstance(index, PairIndex):
            raise TypeError('Paired audio requires a source occurrence ticket')
        output = super().__getitem__(index.index)
        return dict(output, pair_occurrence=index.pair_occurrence, pair_position=index.pair_position,
                    pair_kind=index.pair_kind, requested_pair_kind=index.requested_pair_kind,
                    microgroup=index.microgroup, ce_weight=index.ce_weight, pair_weight=index.pair_weight)


def paired_microgroups(examples, micro_sources=None):
    """Validate then split a logical batch, never separating either pair endpoint."""
    groups = defaultdict(list)
    all_occurrences = set()
    for row in examples:
        groups[row['microgroup']].append(row)
    if not groups or sorted(groups) != list(range(len(groups))):
        raise ValueError('Paired microgroups must form a contiguous logical batch')
    output = []
    for group_index in sorted(groups):
        current = groups[group_index]
        pairs = defaultdict(list)
        for row in current:
            pairs[row['pair_occurrence']].append(row)
        if (len(pairs) not in (4, 8) or len(current) != 2 * len(pairs)
                or (micro_sources is not None and len(pairs) != micro_sources)
                or all_occurrences & set(pairs)):
            raise ValueError('Each microgroup needs four/eight complete source occurrences')
        all_occurrences.update(pairs)
        identities = []
        ordered = []
        for pair in pairs.values():
            if len(pair) != 2 or {r['pair_position'] for r in pair} != {0, 1}:
                raise ValueError('Missing or duplicated pair endpoint')
            pair = sorted(pair, key=lambda r: r['pair_position'])
            identity = tuple(pair[0][name] for name in ('source_id', 'group_id', 'language', 'label'))
            if (tuple(pair[1][name] for name in ('source_id', 'group_id', 'language', 'label')) != identity
                    or pair[0]['condition'] == pair[1]['condition']):
                raise ValueError('Pair endpoints do not share original source identity')
            if any(r.get('source_sha256') != r['group_id'] for r in pair):
                raise ValueError('Pair waveform source fingerprint changed')
            if any(pair[0][name] != pair[1][name]
                   for name in ('pair_kind', 'requested_pair_kind', 'pair_weight', 'ce_weight')):
                raise ValueError('Pair endpoint loss metadata differs')
            identities.append((pair[0]['language'], pair[0]['label']))
            ordered.extend(pair)
        if Counter(identities) != {key: len(pairs) // 4 for key in GROUPS}:
            raise ValueError('Microgroup must balance EN/ZH x real/fake')
        output.append(ordered)
    return output


def loss_weights(rows):
    paired_microgroups(rows)
    expected = 1. / len(rows)
    if any(not math.isclose(r['ce_weight'], expected, rel_tol=0, abs_tol=1e-12) for r in rows):
        raise ValueError('CE weights must belong to the full logical source batch')
    if any(not math.isclose(r['pair_weight'], expected * 2, rel_tol=0, abs_tol=1e-12) for r in rows):
        raise ValueError('Pair weights must belong to the full logical source batch')
    ce = [r['ce_weight'] for r in rows]
    if not math.isclose(sum(ce), 1., abs_tol=1e-12):
        raise ValueError('Classification mass does not sum to one')
    cells = Counter((r['condition'], r['language']) for r in rows if r['label'] == 1)
    conditions = [condition for condition in ('offline', 'online', 'noisy_a', 'noisy_b')
                  if cells[(condition, 'en')] and cells[(condition, 'zh')]]
    adv = [1. / (len(conditions) * 2 * cells[(r['condition'], r['language'])])
           if r['label'] == 1 and r['condition'] in conditions else 0. for r in rows]
    return ce, adv


def loader(plan, epoch, start, cfg):
    if cfg.get('pair_micro_sources', plan.micro_sources) != plan.micro_sources:
        raise ValueError('Configured pair graph budget differs from the sampler')
    dataset = PairedAudioDataset(plan.rows, training=False, epoch=0, seed=cfg['seed'],
                                 max_seconds=0., rawboost=0, raw_config={})
    kwargs = dict(dataset=dataset, batch_sampler=plan.batches(epoch, start),
                  collate_fn=NumpyCollator(cfg['ssl_path']), num_workers=cfg['workers'],
                  pin_memory=False, generator=torch.Generator().manual_seed(cfg['seed'] + epoch))
    if cfg['workers']:
        kwargs.update(multiprocessing_context='spawn', worker_init_fn=worker_init,
                      prefetch_factor=1, persistent_workers=False)
    return DataLoader(**kwargs)
