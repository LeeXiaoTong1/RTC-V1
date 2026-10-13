"""Sixfold English views, source-balanced triplets, and persistent CPU workers."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from w2v_aasist.data import worker_init
from w2v_v313.data import GROUPS, _inventory
from w2v_v315.data import bundles, extend_short, TripletCollator, tensors
from w2v_v3151.data import RawWaveCache, SegmentSampler
from .augment import NoiseBank, Engines, generate, clean, make_recipe, seed_for
from .views import ViewSchedule, DEFAULT_VIEW_FACTORS, FAMILIES, augment, recipe as base_recipe


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _masses(cfg):
    cfg = cfg or {}
    values = tuple(float(cfg.get(key, default)) for key, default in
                   (('online_ce_mass', .3), ('reference_ce_mass', .1), ('main_noisy_ce_mass', .6)))
    if any(not math.isfinite(v) or v <= 0 for v in values) or not math.isclose(sum(values), 1., abs_tol=1e-12):
        raise ValueError('Positive Online/reference/noisy CE masses must sum to one')
    return values


@dataclass(frozen=True)
class Ticket:
    original: int
    online: object
    occurrence: str
    phase: int
    warm: float
    noisy_mass: float
    recipe_json: str
    source_weight: float
    offline_view_json: str
    online_view_json: str
    source_cycle: int
    fallback_recipe_json: str

    @property
    def ordinary(self):
        return self.online


class SourcePlan:
    """Complete source cycles; 25% CE/auxiliary mass per language/class group."""
    def __init__(self, rows, batch, seed, cfg=None):
        self.cfg = dict(cfg or {}, seed=seed)
        quotas = tuple(self.cfg.get('group_quotas', (4, 3, 5, 4)))
        if (type(batch) is not int or batch < 4 or len(quotas) != 4
                or any(type(q) is not int or q < 1 for q in quotas) or sum(quotas) != batch):
            raise ValueError('Four positive source quotas must sum to the logical source batch')
        if type(seed) is not int or seed < 0:
            raise ValueError('Nonnegative source seed required')
        self.rows, self.batch, self.seed = rows, batch, seed
        self.quotas = dict(zip(GROUPS, quotas))
        self.online_mass, self.reference_mass, self.main_noisy_mass = _masses(self.cfg)
        self.inventory = _inventory(rows)
        self.pools = defaultdict(list)
        for source, item in sorted(self.inventory.items()):
            self.pools[item['language'], item['label']].append(source)
        if set(self.pools) != set(GROUPS) or any(len(self.pools[g]) < self.quotas[g] for g in GROUPS):
            raise ValueError('Each language/class pool needs at least its quota of distinct sources')
        epochs = self.cfg.get('epochs', 5)
        if type(epochs) is not int or epochs < 1:
            raise ValueError('Positive integer epochs required for English six-version coverage')
        self.original_steps = math.ceil(len(self.inventory)/batch)
        self.coverage_steps = max(math.ceil(6*len(self.pools[g])/(self.quotas[g]*epochs))
                                  for g in GROUPS if g[0] == 'en')
        # Increasing the update count is explicit. Quotas, per-group loss and
        # physical batches retain their meaning; no hidden expansion of epochs.
        self.steps = max(self.original_steps, self.coverage_steps)
        self.epochs = epochs
        self.all_source_count = len(self.inventory)
        self.missing_online = {s: item for s, item in self.inventory.items() if 'online' not in item['indices']}
        self.views = ViewSchedule(rows, self.inventory, self.pools, seed,
                                  self.cfg.get('view_factors', DEFAULT_VIEW_FACTORS))
        self._cycles, self._streams_cache, self._epoch = {}, {}, None

    def order(self, group, cycle):
        size, quota = len(self.pools[group]), self.quotas[group]
        for c in range(cycle+1):
            key = (group, c)
            if key in self._cycles:
                continue
            order = np.random.default_rng(seed_for(self.seed, 'v319-source-cycle', group, c)).permutation(size).tolist()
            # A logical batch may straddle permutations. Repair only that
            # boundary, preserving every source exactly once per full cycle.
            tail = (c*size) % quota
            if tail:
                previous = set(self._cycles[group, c-1][-tail:])
                needed = quota-tail
                for j in range(needed):
                    if order[j] in previous:
                        swap = next(i for i in range(needed, size) if order[i] not in previous)
                        order[j], order[swap] = order[swap], order[j]
            self._cycles[key] = order
        return self._cycles[group, cycle]

    def _streams(self, epoch):
        if self._epoch == epoch:
            return self._streams_cache
        streams = {}
        for group in GROUPS:
            pool = self.pools[group]
            count = self.steps*self.quotas[group]
            offset, values = epoch*count, []
            while len(values) < count:
                cycle, position = divmod(offset, len(pool))
                order = self.order(group, cycle)
                chunk = order[position:position+count-len(values)]
                values.extend((pool[i], cycle) for i in chunk)
                offset += len(chunk)
            streams[group] = values
        self._epoch, self._streams_cache = epoch, streams
        return streams

    def ticket(self, source, epoch, step, slot, cycle, occurrence=None, profile=False):
        item = self.inventory[source]
        group = (item['language'], item['label'])
        occurrence = occurrence or f'v319:{epoch}:{step}:{group[0]}:{group[1]}:{slot}'
        phase = epoch*self.steps*self.batch+step*self.batch+slot
        offline = self.views.specification(source, 'offline', cycle)
        online = self.views.specification(source, 'online', cycle)
        if profile:
            for kind, value in (('offline', offline), ('online', online)):
                value['recipe'] = base_recipe(self.seed, source, kind, 1, family='speed_slow')
                value['recipe']['rate'] = 970
                value['view_id'] = 'profile:'+value['view_id']
        r = make_recipe(self.cfg, occurrence, item['group_id'], epoch=epoch)
        fallback = (make_recipe(self.cfg, occurrence+':missing-online', item['group_id'], epoch=epoch)
                    if 'online' not in item['indices'] else None)
        return Ticket(item['indices']['offline'], item['indices'].get('online'), occurrence, phase, 1.,
            self.main_noisy_mass, _json(r), .25/self.quotas[group], _json(offline), _json(online), cycle, _json(fallback))

    def batches(self, epoch, start=0, stop=None):
        stop = self.steps if stop is None else stop
        if (type(epoch) is not int or epoch < 0 or type(start) is not int or type(stop) is not int
                or not 0 <= start <= stop <= self.steps):
            raise ValueError('Invalid deterministic epoch/resume segment')
        streams, output = self._streams(epoch), []
        for step in range(start, stop):
            current = []
            for group in GROUPS:
                q = self.quotas[group]
                for j in range(q):
                    source, cycle = streams[group][step*q+j]
                    current.append(self.ticket(source, epoch, step, j, cycle))
            order = np.random.default_rng(seed_for(self.seed, 'v319-batch-order', epoch, step)).permutation(self.batch)
            output.append([current[i] for i in order])
        return output

    def coverage(self, epoch=0):
        streams = self._streams(epoch)
        groups, families, kinds = {}, Counter(), Counter()
        total, missing, candidates = 0, 0, 0
        for group in GROUPS:
            pool = self.pools[group]
            values = streams[group]
            counts = Counter(s for s, _ in values)
            cell = dict(available=len(pool), quota=self.quotas[group], draws=len(values),
                        unique=len(counts), maximum_repeats=max(counts.values()), loss_mass=.25,
                        per_source_loss_mass=.25/self.quotas[group], streams={})
            for kind in ('offline', 'online'):
                available = [s for s in pool if kind == 'offline' or s not in self.missing_online]
                selected = [(s, c) for s, c in values if kind == 'offline' or s not in self.missing_online]
                specs = [self.views.description(s, kind, c) for s, c in selected]
                cumulative = dict(fixed=0, dynamic=0, realized=0)
                all_draws = (epoch+1)*len(values)
                cycles, remainder = divmod(all_draws, len(pool))
                extra = {pool[i] for i in self.order(group, cycles)[:remainder]}
                for source in available:
                    report = self.views.cumulative(source, kind, cycles+int(source in extra))
                    for key in cumulative:
                        cumulative[key] += report[key]
                originals = sum(s['original'] for s in specs)
                cell['streams'][kind] = dict(fixed_view_factor=self.views.factors[group],
                    available_sources=len(available), available_fixed_views=len(available)*self.views.factors[group],
                    realized_this_epoch=len({s['view_id'] for s in specs}),
                    unique_fixed_views_this_epoch=len({s['view_id'] for s in specs if not s['dynamic']}),
                    dynamic_draws=sum(s['dynamic'] for s in specs), draws=len(specs), original_draws=originals,
                    augmented_draws=len(specs)-originals, original_fraction=originals/len(specs) if specs else None,
                    target_original_fraction=1/6, families=dict(Counter(s['family'] for s in specs)),
                    cumulative=cumulative)
            total += len(values)
            missing += sum(s in self.missing_online for s, _ in values)
            candidates += sum(s not in self.missing_online and
                              self.views.description(s, 'offline', c)['family'] != 'temporal_context'
                              for s, c in values)
            groups[f'{group[0]}/{group[1]}'] = cell
        # Count the exact occurrence recipes without constructing or storing a
        # second epoch of full tickets/base transforms/fallback recipes.
        for group, values in streams.items():
            q = self.quotas[group]
            for position, (source, _) in enumerate(values):
                step, slot = divmod(position, q)
                occurrence = f'v319:{epoch}:{step}:{group[0]}:{group[1]}:{slot}'
                r = make_recipe(self.cfg, occurrence, self.inventory[source]['group_id'], epoch=epoch)
                families[r['family']] += 1
                kinds[r['noise_type']] += 1
        return dict(epoch=epoch, steps=self.steps, original_steps=self.original_steps,
            coverage_steps=self.coverage_steps, planned_epochs=self.epochs,
            coverage_policy='Every EN-fake and EN-real source receives all six fixed versions over the planned run; verified Online partners follow the same six source visits.',
            planned_source_draws=self.steps*self.epochs*self.batch,
            planned_full_wave_views=3*self.steps*self.epochs*self.batch,
            additional_steps_per_epoch=self.steps-self.original_steps,
            sources_per_batch=self.batch, source_draws=total,
            full_wave_views=3*total, canonical_sources=self.all_source_count,
            available_sources=len(self.inventory), unique_sources=sum(c['unique'] for c in groups.values()),
            missing_online_sources=len(self.missing_online), missing_online_draws=missing,
            excluded_missing_online_ids=[], source_groups=groups,
            main_ce_mass=dict(online=self.online_mass, reference=self.reference_mass, noisy=self.main_noisy_mass),
            group_ce_mass=.25, processing_families=dict(families), noise_types=dict(kinds),
            bridge_guaranteed_noncrop_candidates=candidates,
            bridge_note='Missing official Online and actual temporal crops skip bridge only; envelope fallback remains eligible.',
            new_audio_disk_cache_bytes=0, new_frame_cache_bytes=0,
            policy='Same-source triplets; Offline only supplies simulated clean/noisy RTC; no raw Offline classification.',
            version_semantics='6 total English versions = 1 original + 5 transforms; augmented views are not independent speakers.')


class Triplets(Dataset):
    def __init__(self, rows, cfg, run, split='train', panel_recipes=None):
        if split not in ('train', 'dev') or cfg.get('rolling_cache_bytes', 0) != 0:
            raise ValueError('Only Train/Dev and no generated audio disk cache are supported')
        self.rows, self.cfg, self.run, self.split = rows, cfg, str(run), split
        self.panel_recipes = panel_recipes
        self.masses = _masses(cfg)
        self.bank = self.engines = self.raw_cache = None

    def __len__(self):
        return len(self.rows)

    def _ready(self):
        if self.bank is None:
            self.bank = NoiseBank(self.cfg['noise_records'][self.split],
                int(self.cfg.get('noise_cache_mib', 96)*1024**2),
                self.cfg['scene_noise_records'][self.split])
            self.engines = Engines(self.cfg['augmentation_runtime'].get('ffmpeg_path'))
            self.raw_cache = RawWaveCache(int(self.cfg.get('raw_audio_cache_mib', 128)*1024**2))

    def __getitem__(self, ticket):
        self._ready()
        if self.split == 'dev':
            original = self.rows[ticket]
            r = self.panel_recipes[ticket]
            if original['split'] != 'dev' or original['condition'] != 'offline' or r['split'] != 'dev':
                raise ValueError('Scene panel requires verified Dev Offline and a Dev recipe')
            _, noisy, meta = generate(self.raw_cache.get(original), r, self.bank, self.engines,
                                      include_reference=False)
            wave, spans = extend_short(noisy, meta['erased_spans'])
            return [dict(original, wave=wave, role='noisy', condition='rtc_noisy', recipe=meta,
                source_index=ticket, pair_occurrence='panel:'+str(ticket), pair_eligible=meta['pair_eligible'],
                noise_pair_eligible=meta['pair_eligible'], online_pair_eligible=False,
                bridge_unavailable_reason='dev_panel', aux_spans=spans, ce_weight=0., source_weight=0.)]
        if not isinstance(ticket, Ticket):
            raise TypeError('Train requires a deterministic source Ticket')
        original = self.rows[ticket.original]
        keys = ('source_id', 'group_id', 'source_sha256', 'language', 'label')
        if (original['split'] != 'train' or original['condition'] != 'offline'
                or original.get('view') != 'full' or original.get('full_length') is not True):
            raise ValueError('Original source must be a verified full Train Offline recording')
        online = self.rows[ticket.online] if ticket.online is not None else None
        if online is not None and (online['split'] != 'train' or online['condition'] != 'online'
                or any(online[k] != original[k] for k in keys)
                or online.get('view') != 'full' or online.get('full_length') is not True):
            raise ValueError('Official Online must be a verified full same-source Train recording')
        r, offline_spec, online_spec = map(json.loads,
            (ticket.recipe_json, ticket.offline_view_json, ticket.online_view_json))
        if r['split'] != 'train':
            raise ValueError('Training cannot use Dev recipes')
        x, offline_meta = augment(self.raw_cache.get(original), offline_spec['recipe'])
        reference, noisy, meta = generate(x, r, self.bank, self.engines)
        z, online_meta = augment(self.raw_cache.get(online if online is not None else original), online_spec['recipe'])
        if online is None:
            fallback = json.loads(ticket.fallback_recipe_json)
            if fallback['split'] != 'train':
                raise ValueError('Fallback must use a Train recipe')
            z = clean(z, fallback, self.engines)
            online_row = dict(original, condition='rtc_online_fallback')
            bridge_reason = 'missing_official_online'
        else:
            online_row = online
            bridge_reason = ('independent_crop_overlap' if not (online_meta['normalized_time_bridge_compatible']
                and offline_meta['normalized_time_bridge_compatible']) else None)
        common = dict(pair_occurrence=ticket.occurrence, source_index=ticket.original,
            source_weight=ticket.source_weight, pair_eligible=meta['pair_eligible'],
            noise_pair_eligible=meta['pair_eligible'], online_pair_eligible=bridge_reason is None,
            bridge_unavailable_reason=bridge_reason, recipe=meta,
            offline_base_augmentation=offline_meta, online_base_augmentation=online_meta)
        online_wave, _ = extend_short(z, [])
        outputs = [dict(online_row, **common, wave=online_wave, role='online', aux_spans=[],
            base_view_id=online_spec['view_id'], official_online=online is not None,
            ce_weight=ticket.source_weight*self.masses[0])]
        for role, wave, mass in (('reference', reference, self.masses[1]), ('noisy', noisy, self.masses[2])):
            wave, spans = extend_short(wave, meta['erased_spans'])
            outputs.append(dict(original, **common, wave=wave, role=role, condition='rtc_'+role,
                base_view_id=offline_spec['view_id'], official_online=False,
                aux_spans=spans, ce_weight=ticket.source_weight*mass))
        return outputs


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
    return _loader(Triplets(plan.rows, cfg, run), cfg, batch_sampler=plan.batches(epoch, start, stop))


def panel_loader(rows, recipes, cfg, run):
    return _loader(Triplets(rows, cfg, run, 'dev', recipes), cfg,
                   batch_size=cfg['feature_batch'], shuffle=False)


def probe_examples(plan, cfg, run):
    """Real shortest-to-longest source triplets with worst allowed speed length."""
    tickets = []
    for group in GROUPS:
        pool = sorted(plan.pools[group], key=lambda s: max(plan.rows[i].get('output_samples', 0)
            for kind, i in plan.inventory[s]['indices'].items() if kind in ('offline', 'online')))
        q = plan.quotas[group]
        quantiles = np.array([1.]) if q == 1 else np.linspace(.5, 1., q)
        used = set()
        for j, quantile in enumerate(quantiles):
            position = min(len(pool)-1, int((len(pool)-1)*quantile))
            if position in used:
                position = next(i for i in range(len(pool)-1, -1, -1) if i not in used)
            used.add(position)
            tickets.append(plan.ticket(pool[position], 0, 0, j, 0,
                occurrence=f'profile:{group[0]}:{group[1]}:{j}', profile=True))
    dataset = Triplets(plan.rows, cfg, run)
    return tensors(TripletCollator(cfg['ssl_path'])([dataset[t] for t in tickets]))


def validate_batch(rows, source_batch, cfg=None):
    cfg = cfg or {}
    masses = dict(zip(('online', 'reference', 'noisy'), _masses(cfg)))
    quotas = dict(zip(GROUPS, cfg.get('group_quotas', (4, 3, 5, 4))))
    if sum(quotas.values()) != source_batch:
        raise ValueError('Configured quotas do not match source batch')
    groups, roles, occurrences = Counter(), Counter(), {}
    for row in rows:
        if row.get('split') != 'train':
            raise ValueError('Dev/Progress/Eval cannot enter optimization')
        weight = row['ce_weight']
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Classification weights must be finite and positive')
        groups[row['language'], row['label']] += weight
        roles[row['role']] += weight
        occurrences.setdefault(row['pair_occurrence'], []).append(row)
    if len(occurrences) != source_batch or set(groups) != set(GROUPS) or any(abs(v-.25)>1e-7 for v in groups.values()):
        raise ValueError('Logical source/group CE budget differs')
    if set(roles) != set(masses) or any(abs(roles[r]-masses[r])>1e-7 for r in masses):
        raise ValueError('Online/reference/noisy CE schedule differs')
    counts, sources = Counter(), set()
    for pair in occurrences.values():
        if len(pair) != 3 or {r['role'] for r in pair} != set(masses):
            raise ValueError('Each source needs Online/reference/noisy')
        keys = ('source_id', 'group_id', 'source_sha256', 'language', 'label')
        if len({tuple(r[k] for k in keys) for r in pair}) != 1:
            raise ValueError('Triplet endpoints have different source identities')
        row = pair[0]
        group = (row['language'], row['label'])
        counts[group] += 1
        source = (group, row['source_id'])
        if source in sources:
            raise ValueError('Repeated source inside one logical batch')
        sources.add(source)
        expected = .25/quotas[group]
        for r in pair:
            if (not math.isfinite(r['source_weight']) or r['source_weight'] <= 0
                    or abs(r['source_weight']-expected)>1e-7
                    or abs(r['ce_weight']-expected*masses[r['role']])>1e-7):
                raise ValueError('Source CE/auxiliary budget differs')
            condition = r['condition']
            allowed = ('online', 'rtc_online_fallback') if r['role'] == 'online' else ('rtc_'+r['role'],)
            if condition not in allowed:
                raise ValueError('Raw Offline must never enter detector classification')
            if condition == 'rtc_online_fallback' and r.get('online_pair_eligible'):
                raise ValueError('A simulated fallback cannot claim an official Online bridge')
    if counts != Counter(quotas):
        raise ValueError('Logical batch source quotas differ')
    return occurrences
