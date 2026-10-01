"""Full-plus-short training views; existing V3 cache and validation remain unchanged."""
import hashlib
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from w2v_aasist.data import FeatureCollator, stable_seed, worker_init
from w2v_v3.data import (AudioDataset as FullAudioDataset, build_data as full_build_data,
                         loader as full_loader, read_protocol)


def short_settings(cfg):
    values = {name: float(cfg.get(name, default)) for name, default in (
        ('short_min_seconds', 3.), ('short_max_seconds', 6.),
        ('short_prefix_probability', .5), ('short_loss_weight', .3))}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Short-view settings must be finite')
    if not 0 < values['short_min_seconds'] <= values['short_max_seconds']:
        raise ValueError('Short-view durations require 0 < minimum <= maximum')
    if not 0 <= values['short_prefix_probability'] <= 1:
        raise ValueError('Short prefix probability must be in [0,1]')
    if not 0 <= values['short_loss_weight'] < 1:
        raise ValueError('Short-view loss budget must be in [0,1), retaining full supervision')
    return values


def crop_spec(source_id, samples, *, seed, epoch, short_min_seconds=3.,
              short_max_seconds=6., short_prefix_probability=.5):
    """Source-only randomness: labels, language and noisy version never set a crop."""
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError('A nonempty waveform is required')
    rng = np.random.default_rng(stable_seed(seed, epoch, ('v31-short-view', source_id)))
    wanted = max(1, round(float(rng.uniform(short_min_seconds, short_max_seconds))*16000))
    prefix = bool(rng.random() < short_prefix_probability)
    if wanted >= samples:
        return 0, samples, 'full_consolidated'
    start = 0 if prefix else int(rng.integers(0, samples-wanted+1))
    return start, wanted, 'prefix' if prefix else 'random'


class AudioDataset(FullAudioDataset):
    """Read and augment each original row once, then crop its waveform in memory.

    The inherited <0.4-second safety extension remains unchanged. Valid short
    recordings are not repeated to the sampled 3-6-second duration or to 4 s.
    """
    def __init__(self, *args, short_min_seconds=3., short_max_seconds=6.,
                 short_prefix_probability=.5, short_loss_weight=.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.short = short_settings(dict(short_min_seconds=short_min_seconds,
                short_max_seconds=short_max_seconds, short_prefix_probability=short_prefix_probability,
                short_loss_weight=short_loss_weight))

    def __getitem__(self, ticket):
        row = super().__getitem__(ticket)
        if not self.training:
            return row
        # Dataset index/ticket distinguishes ordinary vs both cached versions of
        # a recording; source_id alone would incorrectly merge three exposures.
        identity = json.dumps([self.epoch, ticket], sort_keys=True, separators=(',', ':'))
        group = hashlib.sha256(identity.encode('utf-8')).hexdigest()
        wave = row['wave']
        weight = self.short['short_loss_weight']
        start, count, mode = crop_spec(row['id'], len(wave), seed=self.seed, epoch=self.epoch,
                                      **{k:v for k,v in self.short.items() if k != 'short_loss_weight'})
        common = {**row, 'source_group': group, 'source_id': row['id'],
                  'source_audio_seconds': len(wave)/16000,
                  'source_samples': len(wave), 'short_crop_mode': mode}
        full = {**common, 'view': 'full', 'view_weight': 1.,
                'crop_start_sample': 0, 'crop_samples': len(wave)}
        if count == len(wave) or weight == 0:
            # A second identical encoder invocation would not add supervision.
            return [full]
        full['view_weight'] = 1-weight
        short = {**common, 'wave': np.ascontiguousarray(wave[start:start+count]),
                 'audio_seconds': count/16000, 'view': 'short', 'view_weight': weight,
                 'crop_start_sample': start, 'crop_samples': count}
        return [full, short]


class ViewCollator:
    """Run independent official fbank+CMVN only after waveform-level cropping."""
    def __init__(self, ssl_path):
        self.features = FeatureCollator(ssl_path)

    def __call__(self, rows):
        expanded = [view for views in rows for view in views]
        return self.features(expanded)


def loader(records, cfg, *, training=False, epoch=0, batches=None):
    if not training:
        # Exact original validation behavior, including fixed noisy Dev lengths.
        return full_loader(records, cfg, training=False, epoch=epoch, batches=batches)
    dataset = AudioDataset(records, training=True, epoch=epoch, seed=cfg['seed'],
                           max_seconds=0., rawboost=cfg['rawboost'], raw_config=cfg['raw_config'],
                           rawboost_probability=cfg.get('rawboost_probability', .5), **short_settings(cfg))
    kwargs = dict(dataset=dataset, collate_fn=ViewCollator(cfg['ssl_path']),
                  num_workers=cfg['workers'], pin_memory=False,
                  generator=torch.Generator().manual_seed(cfg['seed']+epoch))
    if cfg['workers']:
        kwargs.update(multiprocessing_context='spawn', worker_init_fn=worker_init, prefetch_factor=2)
    if batches is None:
        kwargs.update(batch_size=cfg.get('eval_batch', 8), shuffle=False)
    else:
        kwargs['batch_sampler'] = batches
    return DataLoader(**kwargs)


def build_data(cfg):
    short_settings(cfg)
    return full_build_data(cfg)
