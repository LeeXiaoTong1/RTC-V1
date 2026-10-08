"""Verified Online anchors, matched RTC views, and reusable CPU worker pools.

Tickets carry their complete deterministic augmentation recipe. Workers never
read a mutable epoch value, so prefetching, validation and exact resume use the
same examples. Only decoded original waveforms and noise live in bounded RAM;
fresh occurrence-specific RTC pairs are deliberately not written to disk.
"""
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from w2v_aasist.data import worker_init
from w2v_v313.data import SourcePlan as PriorPlan, GROUPS
from w2v_v315.augment import recipe, NoiseBank, Engines, generate
from w2v_v315.data import bundles, verified_wave, extend_short, TripletCollator, tensors


@dataclass(frozen=True)
class Ticket:
    original: int
    online: int
    occurrence: str
    phase: int
    warm: float
    noisy_mass: float
    recipe_json: str

    @property
    def ordinary(self):
        """Compatibility for profiling code; this always means official Online."""
        return self.online


def _masses(cfg):
    cfg = cfg or {}
    online = float(cfg.get('online_ce_mass', .5))
    reference = float(cfg.get('reference_ce_mass', .1))
    initial = float(cfg.get('initial_noisy_ce_mass', .3))
    main = float(cfg.get('main_noisy_ce_mass', .4))
    ramp = float(cfg.get('objective_ramp_epochs', .25))
    if (not all(math.isfinite(x) for x in (online, reference, initial, main, ramp))
            or not math.isclose(online + reference + main, 1., abs_tol=1e-12)
            or not 0 < reference < 1 or not 0 <= initial <= main < 1-reference
            or ramp <= 0):
        raise ValueError('Invalid Online/reference/noisy source CE budget or ramp')
    return online, reference, initial, main, ramp


class SourcePlan(PriorPlan):
    """EN/ZH x fake/real each owns 25% of one logical source budget.

The canonical manifest remains authoritative. Sources without an official,
verified Online partner are explicitly excluded; there is no Offline fallback.
Excluded source IDs and group counts are exposed by coverage() for reporting.
"""
    def __init__(self, rows, batch, seed, cfg=None):
        super().__init__(rows, batch, seed, micro_sources=4)
        self.online_mass, self.reference_mass, self.initial_noisy_mass, self.main_noisy_mass, self.ramp_epochs = _masses(cfg)
        self.all_source_count = len(self.inventory)
        self.excluded = {source: item for source, item in self.inventory.items()
                         if 'online' not in item['indices']}
        self.inventory = {source: item for source, item in self.inventory.items()
                          if 'online' in item['indices']}
        self.pools = defaultdict(list)
        for source, item in sorted(self.inventory.items()):
            self.pools[(item['language'], item['label'])].append(source)
        if set(self.pools) != set(GROUPS):
            raise ValueError('Verified official Online pairs are required in all four language/class groups')
        self.steps = math.ceil(len(self.inventory) / batch)

    def batches(self, epoch, start=0, stop=None):
        stop = self.steps if stop is None else stop
        if (type(epoch) is not int or epoch < 0 or type(start) is not int
                or type(stop) is not int or not 0 <= start <= stop <= self.steps):
            raise ValueError('Invalid deterministic epoch/resume segment')
        streams, rounds = self._streams(epoch), self.batch // 4
        output = []
        for step in range(start, stop):
            warm = min(1., (epoch*self.steps+step+1) / max(1., self.steps*self.ramp_epochs))
            noisy_mass = self.initial_noisy_mass + (self.main_noisy_mass-self.initial_noisy_mass)*warm
            current = []
            for j in range(rounds):
                position = step*rounds+j
                phase = self.seed+epoch*self.steps*rounds+position
                for group_index, group in enumerate(GROUPS):
                    item = self.inventory[streams[group][position]]
                    occurrence = f'{epoch}:{step}:{j*4+group_index}'
                    r = recipe(self.seed, occurrence, phase, item['group_id'], warm=warm)
                    current.append(Ticket(item['indices']['offline'], item['indices']['online'],
                        occurrence, phase, warm, noisy_mass,
                        json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False)))
            output.append(current)
        return output

    def coverage(self, epoch=0):
        counts, families, kinds = Counter(), Counter(), Counter()
        for batch in self.batches(epoch):
            for ticket in batch:
                counts[self.rows[ticket.original]['source_id']] += 1
                r = json.loads(ticket.recipe_json)
                families[r['family']] += 1
                kinds[r['noise_type']+'/'+r['severity']] += 1
        exclusions = Counter(f'{item["language"]}/{item["label"]}' for item in self.excluded.values())
        return dict(epoch=epoch, steps=self.steps, sources_per_batch=self.batch,
            source_draws=sum(counts.values()), full_wave_views=3*sum(counts.values()),
            canonical_sources=self.all_source_count, available_sources=len(self.inventory), unique_sources=len(counts),
            missing_online_sources=len(self.excluded), excluded_missing_online_ids=sorted(self.excluded),
            excluded_source_groups=dict(exclusions),
            ordinary_conditions={'online': sum(counts.values())},
            processing_families=dict(families), noise_types=dict(kinds),
            source_groups={f'{g[0]}/{g[1]}': dict(available=len(pool), draws=sum(counts[s] for s in pool),
                unique=sum(counts[s]>0 for s in pool), maximum_repeats=max(counts[s] for s in pool))
                for g, pool in self.pools.items()},
            initial_ce_mass=dict(online=1-self.reference_mass-self.initial_noisy_mass,
                                 reference=self.reference_mass, noisy=self.initial_noisy_mass),
            main_ce_mass=dict(online=self.online_mass, reference=self.reference_mass, noisy=self.main_noisy_mass),
            group_ce_mass=.25, new_frame_cache_bytes=0, new_audio_disk_cache_bytes=0,
            policy='verified official Online / matched simulated RTC reference / noisy RTC; full waveforms')


class RawWaveCache:
    """Byte-bounded decoded audio, with source stat checks on every access."""
    def __init__(self, cap_bytes):
        if type(cap_bytes) is not int or cap_bytes < 0:
            raise ValueError('Decoded-wave cache cap must be a nonnegative integer')
        self.cap, self.used = cap_bytes, 0
        self.cache = OrderedDict()

    def get(self, row):
        path = Path(row['audio'])
        stat = path.stat()
        if stat.st_size != row['audio_size'] or stat.st_mtime_ns != row['audio_mtime_ns']:
            raise ValueError('Official waveform changed: '+str(path))
        key = (str(path), row['audio_sha256'], row['audio_size'], row['audio_mtime_ns'], row.get('output_samples'))
        if key in self.cache:
            wave = self.cache[key]
            self.cache.move_to_end(key)
        else:
            wave = verified_wave(row)
            if wave.nbytes <= self.cap:
                while self.cache and self.used+wave.nbytes > self.cap:
                    _, old = self.cache.popitem(last=False)
                    self.used -= old.nbytes
                self.cache[key] = wave
                self.used += wave.nbytes
        # Callers may augment or extend this wave. Never expose mutable cache storage.
        return wave.copy()


class Triplets(Dataset):
    def __init__(self, rows, cfg, run, split='train', panel_recipes=None):
        if split not in ('train', 'dev'):
            raise ValueError('Only official Train/Dev views are supported')
        if cfg.get('rolling_cache_bytes', 0) != 0:
            raise ValueError('V3.15.1 has no generated audio or feature disk cache')
        self.rows, self.cfg, self.run, self.split = rows, cfg, str(run), split
        self.panel_recipes = panel_recipes
        self.reference_mass = _masses(cfg)[1]
        self.bank = self.engines = self.raw_cache = None

    def __len__(self):
        return len(self.rows)

    def _ready(self):
        if self.bank is None:
            self.bank = NoiseBank(self.cfg['noise_records'][self.split], self.cfg.get('noise_cache_mib', 96)*1024**2)
            self.engines = Engines(self.cfg['augmentation_runtime'].get('ffmpeg_path'))
            self.raw_cache = RawWaveCache(int(self.cfg.get('raw_audio_cache_mib', 128)*1024**2))

    def __getitem__(self, ticket):
        self._ready()
        if self.split == 'train':
            if not isinstance(ticket, Ticket):
                raise TypeError('Train requires a deterministic three-view Ticket')
            original, online = self.rows[ticket.original], self.rows[ticket.online]
            keys = ('source_id', 'group_id', 'source_sha256', 'language', 'label')
            if (original['split'] != 'train' or online['split'] != 'train'
                    or original['condition'] != 'offline' or online['condition'] != 'online'
                    or any(original[key] != online[key] for key in keys)
                    or any(row.get('view') != 'full' or row.get('full_length') is not True for row in (original, online))):
                raise ValueError('Only verified same-source full official Train Offline/Online pairs may optimize')
            r = json.loads(ticket.recipe_json)
            if r.get('split') != 'train':
                raise ValueError('Training tickets cannot carry Dev augmentation recipes')
            occurrence, noisy_mass, source_index = ticket.occurrence, ticket.noisy_mass, ticket.original
        else:
            original = self.rows[ticket]
            if original['split'] != 'dev':
                raise ValueError('Robustness panel requires held-out official Dev')
            r = self.panel_recipes[ticket]
            if r.get('split') != 'dev':
                raise ValueError('Dev panel cannot use Train augmentation recipes')
            occurrence, noisy_mass, source_index = 'panel:'+str(ticket), 1., ticket
        reference, noisy, meta = generate(self.raw_cache.get(original), r, self.bank, self.engines)
        common = dict(pair_occurrence=occurrence, pair_eligible=meta['pair_eligible'],
                      noise_pair_eligible=meta['pair_eligible'], online_pair_eligible=self.split=='train',
                      recipe=meta, source_index=source_index)
        outputs = []
        if self.split == 'train':
            wave, _ = extend_short(self.raw_cache.get(online), [])
            outputs.append(dict(online, **common, wave=wave, role='online', aux_spans=[],
                ce_weight=(1-self.reference_mass-noisy_mass)/self.cfg['source_batch']))
        for role, wave, mass in (('reference', reference, self.reference_mass), ('noisy', noisy, noisy_mass)):
            if self.split == 'dev' and role == 'reference':
                continue
            # Exclude the known erased interval on BOTH aligned simulated arms.
            # Its timestamps do not describe official Online, which is independent.
            wave, spans = extend_short(wave, meta['erased_spans'])
            outputs.append(dict(original, **common, wave=wave, role=role, condition='rtc_'+role,
                aux_spans=spans, ce_weight=mass/self.cfg['source_batch'] if self.split=='train' else 0.))
        return outputs


class SegmentSampler:
    def __init__(self):
        self.batches = []

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _loader(dataset, cfg, persistent=False, **kwargs):
    workers = cfg['workers']
    options = dict(dataset=dataset, collate_fn=TripletCollator(cfg['ssl_path']), num_workers=workers,
        pin_memory=False, generator=torch.Generator().manual_seed(cfg['seed']), **kwargs)
    if workers:
        prefetch = cfg.get('prefetch_factor', 2)
        if type(prefetch) is not int or prefetch < 1:
            raise ValueError('Prefetch factor must be a positive integer')
        options.update(multiprocessing_context='spawn', worker_init_fn=worker_init,
                       prefetch_factor=prefetch, persistent_workers=persistent)
    return DataLoader(**options)


class TrainingLoader:
    """One worker pool and one feature extractor per worker for the entire run."""
    def __init__(self, plan, cfg, run):
        self.plan, self.sampler = plan, SegmentSampler()
        self.loader = _loader(Triplets(plan.rows, cfg, run), cfg, persistent=True, batch_sampler=self.sampler)
        self.closed = False

    def segment(self, epoch, start, stop):
        if self.closed:
            raise RuntimeError('Training worker pool is already closed')
        self.sampler.batches = self.plan.batches(epoch, start, stop)
        return iter(self.loader)

    def close(self):
        iterator = getattr(self.loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()
            self.loader._iterator = None
        self.closed = True


def loader(plan, epoch, start, cfg, run, stop=None):
    """Short-lived loader for isolated profiling; training uses TrainingLoader."""
    return _loader(Triplets(plan.rows, cfg, run), cfg, batch_sampler=plan.batches(epoch, start, stop))


def panel_loader(rows, recipes, cfg, run):
    return _loader(Triplets(rows, cfg, run, 'dev', recipes), cfg,
                   batch_size=cfg['feature_batch'], shuffle=False)


def validate_batch(rows, source_batch, cfg=None):
    _, reference_mass, initial_noisy_mass, main_noisy_mass, _ = _masses(cfg)
    groups, roles, occurrences = Counter(), Counter(), {}
    for row in rows:
        if row.get('split') != 'train':
            raise ValueError('Dev/Progress/Eval cannot enter optimization')
        weight = row['ce_weight']
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Classification weights must be finite and nonnegative')
        groups[(row['language'], row['label'])] += weight
        roles[row['role']] += weight
        occurrences.setdefault(row['pair_occurrence'], []).append(row)
    if (len(occurrences) != source_batch or set(groups) != set(GROUPS)
            or any(abs(value-.25)>1e-7 for value in groups.values())):
        raise ValueError('Logical source/group CE budget differs')
    if set(roles) != {'online', 'reference', 'noisy'} or abs(sum(roles.values())-1)>1e-7:
        raise ValueError('Online/reference/noisy CE mass differs')
    if (abs(roles['reference']-reference_mass)>1e-7
            or not initial_noisy_mass-1e-7 <= roles['noisy'] <= main_noisy_mass+1e-7):
        raise ValueError('Configured Online/reference/noisy CE schedule differs')
    for pair in occurrences.values():
        if len(pair) != 3 or {row['role'] for row in pair} != {'online', 'reference', 'noisy'}:
            raise ValueError('Each source needs exactly three full views')
        keys = ('source_id', 'group_id', 'source_sha256', 'language', 'label')
        if len({tuple(row[key] for key in keys) for row in pair}) != 1:
            raise ValueError('Triplet endpoints have different source identities')
        if any(row['role']=='online' and row['condition']!='online' for row in pair):
            raise ValueError('Official Online cannot be replaced with an Offline view')
        if abs(sum(row['ce_weight'] for row in pair)-1/source_batch)>1e-7:
            raise ValueError('One source occurrence must retain its complete CE budget')
    return occurrences
