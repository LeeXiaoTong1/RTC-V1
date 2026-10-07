"""Three complete views per balanced source; no saved features or tensor IPC."""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from w2v_aasist.data import FeatureCollator, read_wave, worker_init
from w2v_v313.data import SourcePlan as PriorPlan, GROUPS
from w2v_v36.features import MODE, FORMAT, _digest
from w2v_v39.data import cache_path, verify_owner, validate_rows
from w2v_v39.common import read_json, digest
from .augment import recipe, NoiseBank, Engines, generate, cache_key
from .cache import RollingCache


def metadata(path, split, cfg):
    """Verify the producer/row manifest without opening obsolete x.npy caches."""
    path = Path(path)
    verify_owner(path, split, cfg)
    owner, complete = read_json(path/'owner.json'), read_json(path/'complete.json')
    if owner.get('format') != FORMAT or complete.get('format') != FORMAT or owner.get('mode') != MODE:
        raise ValueError('Unknown canonical metadata producer')
    bound = dict(identity=owner['identity'], records_digest=owner['records_digest'], mode=MODE,
                 preprocessing_sha256=owner['preprocessing_sha256'])
    if owner['identity_digest'] != _digest(bound) or complete['owner_sha256'] != digest(path/'owner.json'):
        raise ValueError('Canonical metadata owner changed')
    if complete['files']['rows.json'] != digest(path/'rows.json'):
        raise ValueError('Canonical row manifest changed')
    rows = read_json(path/'rows.json')
    if _digest(rows) != owner['records_digest'] or len(rows) != owner['shape'][0]:
        raise ValueError('Canonical source count/order changed')
    return dict(rows=rows)


@contextmanager
def bundles(cfg):
    train, dev = [metadata(cache_path(cfg['v37_run'], s), s, cfg) for s in ('train','dev')]
    validate_rows(train['rows'],dev['rows'],train['rows'])
    yield train,dev


@dataclass(frozen=True)
class Ticket:
    original: int
    ordinary: int
    occurrence: str
    phase: int
    warm: float
    noisy_mass: float


class SourcePlan(PriorPlan):
    def __init__(self, rows, batch, seed):
        super().__init__(rows,batch,seed,micro_sources=4)

    def batches(self, epoch, start=0):
        if type(epoch) is not int or epoch < 0 or type(start) is not int or not 0 <= start <= self.steps:
            raise ValueError('Invalid epoch/resume cursor')
        streams, rounds = self._streams(epoch), self.batch//4
        output = []
        for step in range(start,self.steps):
            current = []
            warm = min(1.,(epoch*self.steps+step+1)/max(1,self.steps*.25))
            for j in range(rounds):
                position = step*rounds+j
                phase = self.seed+epoch*self.steps*rounds+position
                for group_index, group in enumerate(GROUPS):
                    item = self.inventory[streams[group][position]]
                    ordinary = 'offline' if phase % 3 == 0 or 'online' not in item['indices'] else 'online'
                    current.append(Ticket(item['indices']['offline'],item['indices'][ordinary],
                        f'{epoch}:{step}:{j*4+group_index}',phase,warm,.5+.1*warm))
            output.append(current)
        return output

    def coverage(self, epoch=0):
        counts, views, families, kinds = Counter(),Counter(),Counter(),Counter()
        for batch in self.batches(epoch):
            for t in batch:
                counts[self.rows[t.original]['source_id']] += 1
                views[self.rows[t.ordinary]['condition']] += 1
                r = recipe(self.seed,t.occurrence,t.phase,self.rows[t.original]['group_id'],warm=t.warm)
                families[r['family']] += 1; kinds[r['noise_type']+'/'+r['severity']] += 1
        return dict(epoch=epoch, steps=self.steps, source_draws=sum(counts.values()),
            full_wave_views=3*sum(counts.values()), available_sources=len(self.inventory), unique_sources=len(counts),
            ordinary_conditions=dict(views), processing_families=dict(families), noise_types=dict(kinds),
            source_groups={f'{g[0]}/{g[1]}':dict(available=len(pool),draws=sum(counts[s] for s in pool),
                unique=sum(counts[s]>0 for s in pool), maximum_repeats=max(counts[s] for s in pool))
                for g,pool in self.pools.items()}, main_ce_mass=dict(ordinary=.3,reference=.1,noisy=.6),
            group_ce_mass=.25, new_frame_cache_bytes=0)


def verified_wave(row):
    path = Path(row['audio'])
    def check():
        stat = path.stat()
        if stat.st_size != row['audio_size'] or stat.st_mtime_ns != row['audio_mtime_ns']:
            raise ValueError('Official waveform changed: '+str(path))
    check(); wave = read_wave(path); check()
    if 'output_samples' in row and len(wave) != row['output_samples']:
        raise ValueError('Official full-wave length changed')
    return wave


def extend_short(wave, spans):
    n = len(wave)
    if n >= 6400:
        return wave,spans
    repeats = math.ceil(6400/n)
    return np.tile(wave,repeats)[:6400], [[a+i*n,min(b+i*n,6400)] for i in range(repeats)
        for a,b in spans if a+i*n<6400]


class Triplets(Dataset):
    def __init__(self, rows, cfg, run, split='train', panel_recipes=None):
        self.rows, self.cfg, self.run, self.split = rows,cfg,str(run),split
        self.panel_recipes = panel_recipes
        self.bank = self.engines = self.cache = None

    def __len__(self):
        return len(self.rows)

    def _ready(self):
        if self.bank is None:
            self.bank = NoiseBank(self.cfg['noise_records'][self.split],self.cfg['noise_cache_mib']*1024**2)
            self.engines = Engines(self.cfg['augmentation_runtime'].get('ffmpeg_path'))
            from .state import identity
            self.cache = RollingCache(Path(self.run)/'rolling_pairs',identity(self.cfg),self.cfg['rolling_cache_bytes'])

    def __getitem__(self, ticket):
        self._ready()
        if self.split == 'train':
            if not isinstance(ticket,Ticket):
                raise TypeError('Train requires a deterministic source occurrence ticket')
            original, ordinary = self.rows[ticket.original],self.rows[ticket.ordinary]
            if original['split'] != 'train' or ordinary['split'] != 'train':
                raise ValueError('Only official Train can supply optimizer examples')
            r = recipe(self.cfg['seed'],ticket.occurrence,ticket.phase,original['group_id'],warm=ticket.warm)
            occurrence, noisy_mass = ticket.occurrence,ticket.noisy_mass
        else:
            original = self.rows[ticket]
            if original['split'] != 'dev':
                raise ValueError('Robustness panel requires held-out official Dev')
            r = self.panel_recipes[ticket]
            occurrence,noisy_mass = 'panel:'+str(ticket),1.
        key = cache_key(original['audio_sha256'],r,self.cfg['noise_catalogs'][self.split],self.cfg['augmentation_runtime'])
        value = self.cache.get(key)
        if value is None:
            value = generate(verified_wave(original),r,self.bank,self.engines)
            self.cache.put(key,*value)
        reference,noisy,meta = value
        common = dict(original, pair_occurrence=occurrence, pair_eligible=meta['pair_eligible'],
                      recipe=meta, ce_weight=0., source_index=ticket.original if self.split=='train' else ticket)
        outputs = []
        if self.split == 'train':
            w,_ = extend_short(verified_wave(ordinary),[])
            outputs.append(dict(common, **{k:ordinary[k] for k in ('id','audio','condition')},
                wave=w, role='ordinary', aux_spans=[], ce_weight=(.9-noisy_mass)/self.cfg['source_batch']))
        for role,w,spans,mass in (('reference',reference,meta['erased_spans'],.1),
                                  ('noisy',noisy,meta['erased_spans'],noisy_mass)):
            if self.split == 'dev' and role == 'reference':
                continue
            w,spans = extend_short(w,spans)
            outputs.append(dict(common,wave=w, role=role, condition='rtc_'+role,
                aux_spans=spans, ce_weight=mass/self.cfg['source_batch'] if self.split=='train' else 0.))
        return outputs


class TripletCollator:
    def __init__(self, ssl_path):
        self.features = FeatureCollator(ssl_path)

    def __call__(self, bundles):
        rows = [r for group in bundles for r in group]
        lengths = [len(r['wave']) for r in rows]
        result = self.features(rows)
        for row, samples in zip(result,lengths):
            n = row['features'].shape[1]
            valid = np.ones(n,dtype=bool)
            # Expanded by one acoustic frame at each edge to cover filterbank windows.
            for a,b in row['aux_spans']:
                lo,hi = max(0,math.floor(a*n/samples)-1),min(n,math.ceil(b*n/samples)+1)
                valid[lo:hi] = False
            row.update(features=row['features'].numpy(), mask=row['mask'].numpy(),aux_valid=valid)
        return result


def loader(plan, epoch, start, cfg, run, stop=None):
    batches = plan.batches(epoch,start)
    if stop is not None:
        batches = batches[:stop-start]
    return _loader(Triplets(plan.rows,cfg,run),cfg,batch_sampler=batches)


def panel_loader(rows, recipes, cfg, run):
    return _loader(Triplets(rows,cfg,run,'dev',recipes),cfg,batch_size=cfg['feature_batch'],shuffle=False)


def _loader(dataset, cfg, **kwargs):
    workers = cfg['workers']
    options = dict(dataset=dataset, collate_fn=TripletCollator(cfg['ssl_path']),
        num_workers=workers, pin_memory=False, generator=torch.Generator().manual_seed(cfg['seed']), **kwargs)
    if workers:
        options.update(multiprocessing_context='spawn',worker_init_fn=worker_init,prefetch_factor=1,persistent_workers=False)
    return DataLoader(**options)


def tensors(batch):
    return [dict(r,features=torch.from_numpy(r['features']) if isinstance(r['features'],np.ndarray) else r['features'],
        mask=torch.from_numpy(r['mask']) if isinstance(r['mask'],np.ndarray) else r['mask']) for r in batch]


def validate_batch(rows, source_batch):
    groups, roles, occurrences = Counter(),Counter(),{}
    for row in rows:
        if row.get('split') != 'train':
            raise ValueError('Dev/Progress/Eval cannot enter optimization')
        groups[(row['language'],row['label'])] += row['ce_weight']
        roles[row['role']] += row['ce_weight']
        occurrences.setdefault(row['pair_occurrence'],[]).append(row)
    if len(occurrences) != source_batch or set(groups)!=set(GROUPS) or any(abs(x-.25)>1e-7 for x in groups.values()):
        raise ValueError('Logical source/group CE budget differs')
    if abs(sum(roles.values())-1)>1e-7 or abs(roles['reference']-.1)>1e-7 or not .5-1e-7<=roles['noisy']<=.6+1e-7:
        raise ValueError('Ordinary/reference/noisy CE mass differs')
    for pair in occurrences.values():
        if len(pair)!=3 or {r['role'] for r in pair}!={'ordinary','reference','noisy'}:
            raise ValueError('Each source needs exactly three full views')
        keys = ('source_id','group_id','source_sha256','language','label')
        if len({tuple(r[k] for k in keys) for r in pair}) != 1:
            raise ValueError('Triplet endpoints have different source identities')
    return occurrences
