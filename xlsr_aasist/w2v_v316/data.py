"""Unchanged V3.15 source/view budgets, raw waves, persistent worker pool."""
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from w2v_aasist.data import read_wave, worker_init
from w2v_v315.data import SourcePlan, bundles, Triplets as PriorTriplets, validate_batch, verified_wave, extend_short
from w2v_v315.augment import NoiseBank, Engines


class NoDiskCache:
    def get(self,key): return None
    def put(self,*args): pass


class Triplets(PriorTriplets):
    def _ready(self):
        if self.bank is None:
            self.bank = NoiseBank(self.cfg['noise_records'][self.split],self.cfg['noise_cache_mib']*1024**2)
            self.engines = Engines(self.cfg['augmentation_runtime'].get('ffmpeg_path'))
            self.cache = NoDiskCache()


def flatten(groups):
    # NumPy over IPC: no tensor file-descriptor sharing or filterbank duplication.
    return [row for group in groups for row in group]


class SegmentSampler:
    def __init__(self): self.batches = []
    def __iter__(self): return iter(self.batches)
    def __len__(self): return len(self.batches)


class TrainingLoader:
    def __init__(self,plan,cfg,run):
        self.plan = plan; self.sampler = SegmentSampler()
        self.loader = make_loader(Triplets(plan.rows,cfg,run),cfg,batch_sampler=self.sampler,collate_fn=flatten)
    def segment(self,epoch,start,stop):
        self.sampler.batches = self.plan.batches(epoch,start)[:stop-start]
        # Same DataLoader object and persistent workers across ALL validation segments.
        return iter(self.loader)
    def close(self):
        iterator = getattr(self.loader,'_iterator',None)
        if iterator is not None: iterator._shutdown_workers()


def make_loader(dataset,cfg,**kwargs):
    workers = cfg['workers']
    options = dict(dataset=dataset,num_workers=workers,pin_memory=False,
        generator=torch.Generator().manual_seed(cfg['seed']),**kwargs)
    if workers:
        options.update(multiprocessing_context='spawn',worker_init_fn=worker_init,
                       prefetch_factor=1,persistent_workers=True)
    return DataLoader(**options)


class WaveRows(Dataset):
    def __init__(self,rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row = self.rows[i]
        wave = verified_wave(row) if 'audio_size' in row else read_wave(Path(row['audio']))
        wave,_ = extend_short(wave,[])
        return dict(row,wave=wave,row_index=i)


def rows_list(rows): return rows


def grouped(units,size,frame_budget):
    """units are single ordinary views or indivisible reference/noisy pairs."""
    batch = []
    for unit in sorted(units,key=lambda u:max(len(r['wave']) for r in u)):
        longest = max(len(r['wave']) for r in unit)
        count = sum(map(len,batch))+len(unit)
        frames = max(1,(longest-400)//320+1)
        if batch and (count>size or count*frames>frame_budget or longest/len(batch[0][0]['wave'])>1.5):
            yield [r for u in batch for r in u]; batch = []
        batch.append(unit)
    if batch: yield [r for u in batch for r in u]


def auxiliary_mask(row,frames,device):
    # Omni W2V CNN has stride 320 and receptive field 400 samples at 16 kHz.
    expected = (len(row['wave'])-400)//320+1
    if frames!=expected:
        raise ValueError('Omni frontend frame geometry changed; refusing approximate missing-frame masks')
    index = np.arange(frames)*320
    valid = np.ones(frames,dtype=bool)
    for start,stop in row.get('aux_spans',[]):
        valid &= ~((index<stop+320)&(index+400>start-320))
    return torch.from_numpy(valid).to(device)
