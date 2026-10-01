"""V3.2 on-read processing coverage; original full caches and Dev remain fixed."""
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from w2v_aasist.data import worker_init
from w2v_v3.data import (AudioDataset as FullAudioDataset, build_data as full_build_data,
                         loader as full_loader, read_protocol)
from w2v_v31.data import crop_spec, short_settings, ViewCollator
from .augment import processing_settings, process_wave
from .batching import PreparedCollator


class AudioDataset(FullAudioDataset):
    """Read once, process complete noisy waveform once, then produce both views."""
    def __init__(self, *args, short_min_seconds=3., short_max_seconds=6.,
                 short_prefix_probability=.5, short_loss_weight=.3,
                 processing_enabled=True, processing_identity_probability=.5,
                 processing_single_probability=.4, processing_silence_probability=0., **kwargs):
        super().__init__(*args, **kwargs)
        self.short = short_settings(dict(short_min_seconds=short_min_seconds,
                short_max_seconds=short_max_seconds, short_prefix_probability=short_prefix_probability,
                short_loss_weight=short_loss_weight))
        self.processing = processing_settings(dict(processing_enabled=processing_enabled,
                processing_identity_probability=processing_identity_probability,
                processing_single_probability=processing_single_probability,
                processing_silence_probability=processing_silence_probability))

    def __getitem__(self, ticket):
        row = super().__getitem__(ticket)
        if not self.training:
            return row
        identity = json.dumps([self.epoch, ticket], sort_keys=True, separators=(',', ':'))
        group = hashlib.sha256(identity.encode('utf-8')).hexdigest()
        wave = row['wave']
        weight = self.short['short_loss_weight']
        start, count, mode = crop_spec(row['id'], len(wave), seed=self.seed, epoch=self.epoch,
                                      **{k:v for k,v in self.short.items() if k != 'short_loss_weight'})
        protected=[(start,count)] if count<len(wave) and weight>0 else []
        if row['noisy']:
            if not row.get('full_length') or row.get('version') not in (0, 1):
                raise ValueError('V3.2 processing requires existing full-length noisy versions 0 and 1')
            wave, processing = process_wave(wave, row['id'], row['version'], seed=self.seed,
                                           epoch=self.epoch, protected_views=protected, **self.processing)
        else:
            processing = dict(processing_condition='ordinary_unchanged',
                              processing_parameters=[], processing_applied=False)
        common = {**row, **processing, 'wave':wave, 'source_group':group, 'source_id':row['id'],
                  'source_audio_seconds':len(wave)/16000, 'source_samples':len(wave),
                  'short_crop_mode':mode}
        full = {**common, 'view':'full', 'view_weight':1.,
                'crop_start_sample':0, 'crop_samples':len(wave)}
        if count == len(wave) or weight == 0:
            return [full]
        full['view_weight'] = 1-weight
        short = {**common, 'wave':np.ascontiguousarray(wave[start:start+count]),
                 'audio_seconds':count/16000, 'view':'short', 'view_weight':weight,
                 'crop_start_sample':start, 'crop_samples':count}
        return [full, short]


def loader(records, cfg, *, training=False, epoch=0, batches=None):
    if not training:
        return full_loader(records, cfg, training=False, epoch=epoch, batches=batches)
    dataset = AudioDataset(records, training=True, epoch=epoch, seed=cfg['seed'],
                           max_seconds=0., rawboost=cfg['rawboost'], raw_config=cfg['raw_config'],
                           rawboost_probability=cfg.get('rawboost_probability',.5),
                           **short_settings(cfg), **processing_settings(cfg))
    pin_memory = str(cfg.get('device', 'cpu')).startswith('cuda')
    kwargs = dict(dataset=dataset, collate_fn=PreparedCollator(cfg['ssl_path'],
                  cfg.get('microbatch',4),cfg.get('frame_budget',1600)),
                  num_workers=cfg['workers'], pin_memory=pin_memory,
                  generator=torch.Generator().manual_seed(cfg['seed']+epoch))
    if cfg['workers']:
        prefetch = cfg.get('prefetch_factor',2)
        if type(prefetch) is not int or prefetch < 1:
            raise ValueError('prefetch_factor must be a positive integer')
        kwargs.update(multiprocessing_context='spawn', worker_init_fn=worker_init, prefetch_factor=prefetch)
    if batches is None:
        kwargs.update(batch_size=cfg.get('eval_batch',8), shuffle=False)
    else:
        kwargs['batch_sampler'] = batches
    return DataLoader(**kwargs)


def build_data(cfg):
    short_settings(cfg)
    processing_settings(cfg)
    return full_build_data(cfg)
