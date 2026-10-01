"""Canonical official Train sources, with full/short budgets kept per condition."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from w2v_aasist.data import (read_protocol, read_wave, safe_audio_path, stable_seed,
                             worker_init, cache_index)
from w2v_aasist.runtime import sha256
from w2v_v3.data import class_weights, stratified_order, loader as validation_loader
from w2v_v31.data import crop_spec, short_settings
from w2v_v32.batching import PreparedCollator

CONDITIONS = ('offline', 'online', 'noisy_a', 'noisy_b')


def pair_manifest(cfg):
    explicit = cfg.get('train_pair_manifest') or cfg.get('train_pairs') or cfg.get('rtc_pairs')
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    roots = {Path(cfg['train_protocol']).resolve().parent,
             Path(cfg['train_data_path']).resolve().parent}
    roots |= {p.parent for p in tuple(roots)}
    candidates = set()
    for root in roots:
        for relative in ('train_offline_online_pairs.csv', 'meta/train_offline_online_pairs.csv',
                         'metadata/train_offline_online_pairs.csv', 'data/meta/train_offline_online_pairs.csv'):
            path = root / relative
            if path.is_file(): candidates.add(path.resolve())
    if len(candidates) != 1:
        raise ValueError('Set --train-pair-manifest to the official Train CSV; expected one candidate, found '
                         + repr(sorted(str(p) for p in candidates)))
    return candidates.pop()


def resolve_pair_manifest(cfg):
    path = pair_manifest(cfg)
    cfg['train_pair_manifest'] = str(path)
    return path


def _pair_id(value):
    value = (value or '').strip().replace('\\', '/')
    if value.startswith('train/'): value = value[len('train/'):]
    if value:
        safe_audio_path(Path.cwd(), value)  # Reject absolute paths and traversal, without opening audio.
    return value


def canonical_sources(records, csv_path):
    """No filename arithmetic; empty official Online cells mean a missing view."""
    by_id = {r['id']: r for r in records}
    if len(by_id) != len(records): raise ValueError('Duplicate official protocol identity')
    offline = {k: v for k, v in by_id.items() if v['domain'] == 'offline'}
    online = {k: v for k, v in by_id.items() if v['domain'] == 'online'}
    if len(offline) + len(online) != len(records): raise ValueError('Unknown official audio domain')
    seen, used_online, result = set(), set(), []
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'offline_id', 'online_id'} <= set(reader.fieldnames or ()):
            raise ValueError('Official pairing CSV requires offline_id and online_id')
        for row in reader:
            a, b = _pair_id(row['offline_id']), _pair_id(row['online_id'])
            if a not in offline or a in seen: raise ValueError('Unknown/duplicate Offline pairing ID: ' + a)
            if b and (b not in online or b in used_online):
                raise ValueError('Unknown/duplicate Online pairing ID: ' + b)
            base = offline[a]
            if b and (online[b]['label'] != base['label'] or online[b]['language'] != base['language']):
                raise ValueError('Official paired language/label mismatch: ' + a)
            conditions = {'offline': dict(base, condition='offline', source_id=a)}
            if b:
                conditions['online'] = dict(online[b], condition='online', source_id=a)
                used_online.add(b)
            result.append(dict(base, source_id=a, conditions=conditions, missing_online=not bool(b)))
            seen.add(a)
    if seen != set(offline) or used_online != set(online):
        raise ValueError('Pair CSV must cover every official Offline and every available Online exactly once')
    if not result: raise ValueError('Empty canonical Train sources')
    return sorted(result, key=lambda r: r['source_id'])


class PairedEpochPlan:
    def __init__(self, sources, cache_rows, source_batch=16, seed=1234):
        if type(source_batch) is not int or source_batch < 2:
            raise ValueError('source_batch must be an integer >=2')
        self.records = [dict(s, conditions=dict(s['conditions'])) for s in sources]
        lookup = {s['source_id']: s for s in self.records}
        for row in cache_rows:
            source, condition = row['source_id'], row['condition']
            if source not in lookup or condition not in ('noisy_a', 'noisy_b'):
                raise ValueError('Unknown paired noisy cache identity')
            target = lookup[source]
            if condition in target['conditions'] or row['label'] != target['label'] or row['language'] != target['language']:
                raise ValueError('Duplicate/incorrect paired noisy condition')
            target['conditions'][condition] = dict(row)
        if any(not {'offline', 'noisy_a', 'noisy_b'} <= set(s['conditions']) for s in self.records):
            raise ValueError('Every source requires both complete full-noisy families')
        self.source_batch, self.seed = source_batch, seed
        self.steps = math.ceil(len(self.records) / source_batch)
        self.source_count = len(self.records)
        self.full = True
        self.weights = class_weights(self.records)
        self.noisy_weights = self.weights  # Backward-compatible diagnostic alias; step weights once per source.
        self.ordinary = [r for s in self.records for c, r in s['conditions'].items() if c in ('offline', 'online')]
        self.banks = [[r for s in self.records for c, r in s['conditions'].items() if c.startswith('noisy_')]]

    def batches(self, epoch):
        order = stratified_order(self.records, range(self.source_count), self.seed + 1009 * epoch)
        return [order[start:start + self.source_batch] for start in range(0, len(order), self.source_batch)]

    def tickets(self, epoch, start=0, stop=None):
        yield from self.batches(epoch)[start:stop]

    def coverage(self):
        return {'sources_per_epoch': self.source_count, 'source_batch': self.source_batch,
                'steps': self.steps, 'missing_online': sum(s['missing_online'] for s in self.records),
                'source_groups': dict(Counter(f"{s['language']}/{s['label']}" for s in self.records)),
                'condition_rows_per_epoch': dict(Counter(c for s in self.records for c in s['conditions'])),
                'note': 'Every canonical source once; natural source proportions; no oversampling or language weights.'}


def _activity(wave):
    # One feature token contains two 25ms mel frames, beginning 10ms apart.
    # This is an AUXILIARY audibility mask, never the encoder's padding mask.
    frames = max(0, (len(wave) - 400) // 160 + 1) // 2
    starts = np.arange(frames, dtype=np.int64) * 320
    stops = np.minimum(starts + 560, len(wave))
    energy = np.concatenate(([0.], np.cumsum(np.asarray(wave, np.float64) ** 2)))
    power = (energy[stops] - energy[starts]) / np.maximum(1, stops - starts)
    threshold = max(1e-12, float(power.max(initial=0.)) * 10 ** (-35. / 10.))
    return power > threshold


class PairedDataset(Dataset):
    def __init__(self, records, cfg, epoch):
        self.records, self.cfg, self.epoch = records, cfg, epoch
        self.short = short_settings(cfg)
        self.seed = cfg['seed']
        probability = cfg.get('short_rawboost_probability', .25)
        if not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('short_rawboost_probability must be in [0,1]')
        self.raw_probability = probability
        self.silence_probability = cfg.get('processing_silence_probability', .05)
        if not math.isfinite(self.silence_probability) or not 0 <= self.silence_probability <= 1:
            raise ValueError('processing_silence_probability must be in [0,1]')

    def __len__(self): return len(self.records)

    def __getitem__(self, ticket):
        source = self.records[ticket]
        output = []
        for condition in CONDITIONS:
            if condition not in source['conditions']: continue
            row = source['conditions'][condition]
            wave = read_wave(row['audio'])
            if row.get('full_length') and len(wave) != row['output_samples']:
                raise ValueError('Full paired cache audio length changed')
            if len(wave) < 6400:
                wave = np.tile(wave, math.ceil(6400 / len(wave)))[:6400]
            start, count, mode = crop_spec(source['source_id'], len(wave), seed=self.seed, epoch=self.epoch,
                **{k: v for k, v in self.short.items() if k != 'short_loss_weight'})
            weight = self.short['short_loss_weight']
            group = hashlib.sha256(json.dumps([self.epoch, source['source_id'], condition]).encode()).hexdigest()
            common = dict(row, source_id=source['source_id'], source_group=group, condition=condition,
                          source_audio_seconds=len(wave) / 16000, source_samples=len(wave),
                          short_crop_mode=mode, augmentation='unchanged', composition='paired_' + condition,
                          audio_seconds=len(wave) / 16000, noisy=condition.startswith('noisy_'))
            full = dict(common, wave=wave, aux_audibility=_activity(wave), view='full', view_weight=1.,
                        crop_start_sample=0, crop_samples=len(wave))
            output.append(full)
            if count == len(wave) or weight == 0: continue
            full['view_weight'] = 1 - weight
            short = np.array(wave[start:start + count], dtype=np.float32, copy=True)
            augmentation = 'unchanged'
            draw = stable_seed(self.seed, self.epoch, (source['source_id'], condition, 'short-rawboost'))
            # Clean full reference is never RawBoosted; cached processed views are not processed again.
            if condition in ('offline', 'online') and self.cfg.get('rawboost', 0) and random.Random(draw).random() < self.raw_probability:
                from utils.data_utils import process_rawboost_feature
                py, np_state = random.getstate(), np.random.get_state()
                try:
                    random.seed(draw); np.random.seed(draw)
                    short = process_rawboost_feature(short, 16000, SimpleNamespace(**self.cfg.get('raw_config', {})), self.cfg['rawboost'])
                finally:
                    random.setstate(py); np.random.set_state(np_state)
                augmentation = 'short_rawboost'
            elif condition.startswith('noisy_'):
                silence_rng = np.random.default_rng(stable_seed(self.seed, self.epoch,
                    (source['source_id'], condition, 'short-silence')))
                if float(silence_rng.random()) < self.silence_probability:
                    from w2v_v32.silence import apply_local_silence
                    parameters = dict(operator='local_silence', duration_seconds=float(silence_rng.uniform(.04, .16)),
                                      position=float(silence_rng.random()), maximum_fraction=.05, fade_seconds=.005)
                    short, details = apply_local_silence(short, parameters)
                    if details['silence_applied']: augmentation = 'short_local_silence'
            if len(short) != count or not np.isfinite(short).all():
                raise ValueError('Short augmentation changed length or produced nonfinite audio')
            output.append(dict(common, wave=np.ascontiguousarray(short), aux_audibility=_activity(short),
                               view='short', view_weight=weight, crop_start_sample=start, crop_samples=count,
                               audio_seconds=count / 16000, augmentation=augmentation))
        return output


class PairedCollator(PreparedCollator):
    def __call__(self, rows):
        result = super().__call__(rows)
        for ex in result:
            audible = torch.as_tensor(ex.pop('aux_audibility'), dtype=torch.bool)
            if audible.numel() != ex['mask'].shape[1]:
                raise ValueError('Auxiliary audibility length differs from official feature frames')
            ex['audibility_mask'] = audible
        return result


def loader(records, cfg, *, training=False, epoch=0, batches=None):
    if not training: return validation_loader(records, cfg, training=False, epoch=epoch, batches=batches)
    records = records.records if isinstance(records, PairedEpochPlan) else records
    kwargs = dict(dataset=PairedDataset(records, cfg, epoch),
                  collate_fn=PairedCollator(cfg['ssl_path'], cfg.get('microbatch', 4), cfg.get('frame_budget', 1600)),
                  num_workers=cfg['workers'], pin_memory=str(cfg.get('device', 'cpu')).startswith('cuda'),
                  generator=torch.Generator().manual_seed(cfg['seed'] + epoch))
    if cfg['workers']:
        prefetch = cfg.get('prefetch_factor', 2)
        if type(prefetch) is not int or prefetch < 1: raise ValueError('Invalid prefetch_factor')
        kwargs.update(multiprocessing_context='spawn', worker_init_fn=worker_init, prefetch_factor=prefetch)
    if batches is None: kwargs.update(batch_size=cfg.get('source_batch', 16), shuffle=False)
    else: kwargs['batch_sampler'] = batches
    return DataLoader(**kwargs)


def build_data(cfg):
    from .cache import read_index
    train = read_protocol(cfg['train_protocol'], cfg['train_data_path'])
    pairs = resolve_pair_manifest(cfg)
    sources = canonical_sources(train, pairs)
    folder = cfg.get('train_noisy_cache_v33') or cfg.get('paired_cache')
    if not folder: raise ValueError('V3.3 requires its independent complete paired cache')
    rows, metadata = read_index(folder, cfg, train, verify_audio_hash=cfg.get('verify_cache_hashes', True))
    plan = PairedEpochPlan(sources, rows, cfg.get('source_batch', 16), cfg['seed'])
    if Path(cfg['train_data_path']).resolve() == Path(cfg['dev_data_path']).resolve():
        raise ValueError('Train and Dev audio roots must differ')
    dev = read_protocol(cfg['dev_protocol'], cfg['dev_data_path'])
    validation, dev_meta = {'clean': dev}, {}
    for name, role, key in (('seen', 'dev_seen', 'dev_noisy_cache'), ('heldout', 'dev_heldout', 'dev_heldout_cache')):
        validation[name], dev_meta[name] = cache_index(cfg[key], role, cfg['dev_protocol'], dev)
    from rtc_noisy_v2.cache import check_suite
    check_suite([metadata], dev_meta['seen'], dev_meta['heldout'])
    files = [cfg['train_protocol'], cfg['dev_protocol'], pairs, Path(cfg['ssl_path']) / 'preprocessor_config.json']
    for directory in (folder, cfg['dev_noisy_cache'], cfg['dev_heldout_cache']):
        files.extend(Path(directory) / name for name in ('config.json', 'manifest.jsonl'))
        if (Path(directory) / 'complete.json').is_file(): files.append(Path(directory) / 'complete.json')
    files.extend(Path(folder) / name for name in ('assignments.json', 'sources.json'))
    fingerprints = {str(Path(p).resolve()): sha256(p) for p in files}
    counts = torch.bincount(torch.tensor([s['label'] for s in sources]), minlength=2)
    return plan, validation, plan.weights, counts, fingerprints
