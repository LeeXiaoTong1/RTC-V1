"""Persistent CPU decode/augment workers; NumPy IPC; no feature/waveform disk cache."""
from collections import OrderedDict
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset,DataLoader
from .augment import NoiseBank,Engines,recipe,generate


def read_wave(row):
    path=Path(row['audio']);stat=path.stat()
    if 'audio_size' in row and (stat.st_size!=row['audio_size'] or stat.st_mtime_ns!=row['audio_mtime_ns']):
        raise ValueError('Audio file changed: '+str(path))
    wave,sr=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(1)
    if sr!=16000:
        divisor=math.gcd(sr,16000);wave=resample_poly(wave,16000//divisor,sr//divisor).astype(np.float32)
    if not len(wave) or not np.isfinite(wave).all():raise ValueError('Empty or nonfinite audio: '+str(path))
    return wave


def ready(wave):
    # Preserve every sample; only exceptionally short utterances repeat to 12 frames.
    return np.tile(wave,math.ceil(3920/len(wave)))[:3920] if len(wave)<3920 else wave


class Waves(Dataset):
    def __init__(self,rows,cfg,split='train'):
        self.rows,self.cfg,self.split=rows,cfg,split
        self.raw=OrderedDict();self.raw_bytes=0;self.bank=self.engines=None
    def __len__(self):return len(self.rows)
    def original(self,row):
        key=row['audio']
        if key in self.raw:
            # Stat is cheap and still checked on cache hits.
            stat=Path(key).stat()
            if 'audio_size' in row and (stat.st_size!=row['audio_size'] or stat.st_mtime_ns!=row['audio_mtime_ns']):raise ValueError('Cached source changed')
            self.raw.move_to_end(key);return self.raw[key]
        x=read_wave(row);cap=self.cfg.get('raw_audio_cache_mib',128)*1024**2
        if x.nbytes<=cap:
            while self.raw and self.raw_bytes+x.nbytes>cap:
                _,old=self.raw.popitem(last=False);self.raw_bytes-=old.nbytes
            self.raw[key]=x;self.raw_bytes+=x.nbytes
        return x
    def __getitem__(self,t):
        if isinstance(t,int):
            r=self.rows[t];return [dict(r,wave=ready(self.original(r)),row_index=t)]
        r=self.rows[t['index']];x=self.original(r)
        base=dict(r,occurrence=t['occurrence'])
        if t['kind']=='online':return [dict(base,role='online',wave=ready(x))]
        if self.bank is None:
            self.bank=NoiseBank(self.cfg['noise_records'][self.split],self.cfg['noise_cache_mib']*1024**2)
            self.engines=Engines(self.cfg['ffmpeg'])
        result=[]
        views=(0,) if t.get('family') else (0,1)
        for v in views:
            rec=recipe(self.cfg['seed'],t['occurrence'],r['group_id'],v,t['epoch'],self.cfg['warm_epochs'],
                       self.split,t.get('family'),t.get('probe',False))
            try:
                wave,meta=generate(x,rec,self.bank,self.engines)
            except Exception as exc:
                raise RuntimeError(f'V3.18 augmentation failed: audio={r["audio"]!r} '
                    f'occurrence={t["occurrence"]!r} family={rec["family"]} '
                    f'codec={rec["codec"]}; {exc}') from exc
            result.append(dict(base,role=('noisy_a','noisy_b')[v],condition=t.get('family') or ('noisy_a','noisy_b')[v],wave=ready(wave),augmentation=meta))
        return result


def passthrough(rows):return rows


def worker_init(_):
    torch.set_num_threads(1)


class TicketSampler:
    def __init__(self):self.batches=[]
    def __iter__(self):return iter(self.batches)
    def __len__(self):return len(self.batches)


class DiagnosticLoader(DataLoader):
    def __iter__(self):
        iterator=None
        try:
            iterator=super().__iter__()
            yield from iterator
        except Exception as exc:
            # Capture status before caller cleanup terminates surviving workers.
            workers=[dict(pid=p.pid,exitcode=p.exitcode,alive=p.is_alive())
                     for p in getattr(iterator,'_workers',())]
            print('V318_DATA_FAILURE='+json.dumps(dict(num_workers=self.num_workers,
                  workers=workers,error=str(exc)),ensure_ascii=True),flush=True)
            raise


def loader(dataset,cfg,**kwargs):
    options=dict(dataset=dataset,num_workers=cfg['workers'],collate_fn=passthrough,
                 generator=torch.Generator().manual_seed(cfg['seed']),**kwargs)
    if cfg['workers']:
        options.update(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2,worker_init_fn=worker_init)
    return DiagnosticLoader(**options)


def close(loader):
    iterator=getattr(loader,'_iterator',None)
    if iterator is not None:iterator._shutdown_workers();loader._iterator=None


def microbatches(units,size,frame_budget):
    batch=[];count=0;longest=0
    for unit in units:
        n=max((len(r['wave'])-400)//320+1 for r in unit);total=count+len(unit);length=max(longest,n)
        if batch and (total>size or total*length>frame_budget):
            yield batch;batch=[];count=0;longest=0
        batch.append(unit);count+=len(unit);longest=max(longest,n)
    if batch:yield batch
