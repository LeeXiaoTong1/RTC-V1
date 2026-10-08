"""Full Offline / verified Online / noise-before-RTC; no generated disk cache."""
from collections import Counter
from dataclasses import dataclass
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from w2v_v313.data import SourcePlan as BasePlan,GROUPS
from w2v_v315.data import bundles,extend_short,tensors,TripletCollator as BaseCollator
from w2v_v315.augment import recipe,NoiseBank,Engines as BaseEngines,noise_wave,common_inputs
from w2v_v3151.data import RawWaveCache,SegmentSampler
from w2v_aasist.data import worker_init


class Engines(BaseEngines):
    def __call__(self,x,r):
        if r['family'] not in ('g711_mulaw','g711_alaw'):return super().__call__(x,r)
        codec=r['family'].split('_')[1]
        raw=self.local._run(['-f','f32le','-ar','16000','-ac','1','-i','pipe:0','-ar','8000',
            '-c:a','pcm_'+codec,'-f',codec,'pipe:1'],x.astype('<f4').tobytes())
        y=self.local._run(['-f',codec,'-ar','8000','-ac','1','-i','pipe:0','-ar','16000','-f','f32le','pipe:1'],raw)
        out=np.frombuffer(y,dtype='<f4').copy()
        if abs(len(out)-len(x))>2:raise RuntimeError('Unexpected G.711 length change')
        return np.pad(out,(0,max(0,len(x)-len(out))))[:len(x)]


def generate_noisy(wave,r,bank,engines):
    # Same V3.15 noise/gain/echo/RTC distribution, only the unused RTC reference
    # forward is removed. Known mapping refers to original input sample time.
    rng=np.random.default_rng(r['seed'])
    from utils.env_noise import active_mask
    for attempt in range(8):
        noise,used=noise_wave(bank,len(wave),r['noise_type'],rng)
        a,b,meta=common_inputs(wave,noise,r['snr_db'],r['input_gain_db'],r['echo'])
        if meta['pair_eligible'] or not active_mask(wave,16000).any():break
    noisy=engines(b,r)
    if len(noisy)!=len(b) or not np.isfinite(noisy).all():raise RuntimeError('RTC duration/nonfinite failure')
    erased=[]
    if r['dropout']:
        d=r['dropout'];count=min(round(d['seconds']*16000),max(1,len(noisy)//50))
        start=min(len(noisy)-count,round(d['position']*len(noisy)))
        noisy[start:start+count]*=d['gain'];erased.append([start,start+count])
    meta.update(recipe=r,noise_hashes=used,noise_attempts=attempt+1,erased_spans=erased,
        original_samples=len(wave),output_samples=len(noisy),mapping='identity_input_samples',
        uncompensated_delay_samples=0,appended_tail_samples=len(noisy)-len(wave),fresh_state_per_utterance=True)
    return noisy,meta


@dataclass(frozen=True)
class Ticket:
    original:int
    online:object
    occurrence:str
    phase:int
    warm:float
    recipe_json:str


class SourcePlan(BasePlan):
    def __init__(self,rows,batch,seed,cfg=None):
        super().__init__(rows,batch,seed,micro_sources=4);self.cfg=cfg or {}
    def batches(self,epoch,start=0,stop=None):
        stop=self.steps if stop is None else stop
        if not 0<=start<=stop<=self.steps:raise ValueError('Invalid training segment')
        streams=self._streams(epoch);rounds=self.batch//4;result=[]
        for step in range(start,stop):
            current=[]
            for j in range(rounds):
                position=step*rounds+j;phase=self.seed+epoch*self.steps*rounds+position
                warm=min(1.,(epoch*self.steps+step+1)/max(1,self.steps*.25))
                for g,key in enumerate(GROUPS):
                    item=self.inventory[streams[key][position]];occurrence=f'{epoch}:{step}:{j*4+g}'
                    r=recipe(self.seed,occurrence,phase,item['group_id'],warm=warm)
                    current.append(Ticket(item['indices']['offline'],item['indices'].get('online'),occurrence,
                        phase,warm,json.dumps(r,sort_keys=True)))
            result.append(current)
        return result
    def coverage(self,epoch=0):
        # Coverage is pure source accounting: do not regenerate tens of thousands
        # of unused noise/RTC recipes just to print a sampling summary.
        counts=Counter(source for stream in self._streams(epoch).values() for source in stream)
        missing=sum(counts[source] for source,item in self.inventory.items() if 'online' not in item['indices'])
        return dict(steps=self.steps,available_sources=len(self.inventory),unique_sources=len(counts),
            missing_online_sources=sum('online' not in x['indices'] for x in self.inventory.values()),
            missing_online_draws=missing,source_draws=sum(counts.values()),
            full_wave_views=3*sum(counts.values())-missing,excluded_sources=0,
            source_groups={f'{k[0]}/{k[1]}':dict(available=len(v),unique=sum(counts[s]>0 for s in v),
                draws=sum(counts[s] for s in v),maximum_repeats=max(counts[s] for s in v)) for k,v in self.pools.items()},
            full_ce_mass=dict(offline=.1,online=.5,noisy=.4),missing_online_ce_mass=dict(offline=.2,noisy=.8))


class Triplets(Dataset):
    def __init__(self,rows,cfg,run,split='train',panel_recipes=None):
        self.rows,self.cfg,self.split,self.panel_recipes=rows,cfg,split,panel_recipes
        if split not in ('train','dev') or cfg.get('rolling_cache_bytes',0):raise ValueError('Train/Dev only; no generated cache')
        self.bank=self.engines=self.raw=None
    def __len__(self):return len(self.rows)
    def __getitem__(self,t):
        if self.bank is None:
            self.bank=NoiseBank(self.cfg['noise_records'][self.split],self.cfg.get('noise_cache_mib',64)*1024**2)
            self.engines=Engines(self.cfg['augmentation_runtime'].get('ffmpeg_path'))
            self.raw=RawWaveCache(self.cfg.get('raw_audio_cache_mib',128)*1024**2)
        original=self.rows[t.original if self.split=='train' else t]
        if original['split']!=self.split or original['condition']!='offline':raise ValueError('Expected verified original source')
        r=json.loads(t.recipe_json) if self.split=='train' else self.panel_recipes[t]
        if r['split']!=self.split:raise ValueError('Recipe split mismatch')
        x=self.raw.get(original);noisy,meta=generate_noisy(x,r,self.bank,self.engines)
        occurrence=t.occurrence if self.split=='train' else 'panel:'+str(t)
        common=dict(pair_occurrence=occurrence,recipe=meta,pair_eligible=bool(meta['pair_eligible']),
                    source_index=t.original if self.split=='train' else t)
        outputs=[]
        online=None
        if self.split=='train' and t.online is not None:
            online=self.rows[t.online]
            if online['condition']!='online' or online['split']!='train' or any(
                original[k]!=online[k] for k in ('source_id','source_sha256','group_id','language','label')):
                raise ValueError('Invalid official pair; never guess from filenames')
        masses={'offline':.1,'online':.5,'noisy':.4} if online else {'offline':.2,'noisy':.8}
        if self.split=='train':
            wave,_=extend_short(x,[])
            outputs.append(dict(original,**common,wave=wave,role='offline',aux_spans=[],
                native_samples=len(x),ce_weight=masses['offline']/self.cfg['source_batch']))
            if online is not None:
                y=self.raw.get(online);wave,_=extend_short(y,[])
                outputs.append(dict(online,**common,wave=wave,role='online',aux_spans=[],
                    native_samples=len(y),ce_weight=masses['online']/self.cfg['source_batch']))
        noisy,spans=extend_short(noisy,meta['erased_spans'])
        outputs.append(dict(original,**common,wave=noisy,role='noisy',condition='rtc_noisy',aux_spans=spans,
            native_samples=meta['output_samples'],ce_weight=masses['noisy']/self.cfg['source_batch'] if self.split=='train' else 0.))
        return outputs


class TripletCollator(BaseCollator):
    def __call__(self,groups):
        lengths=[len(r['wave']) for group in groups for r in group]
        result=super().__call__(groups)
        if self.features.extractor.sampling_rate!=16000 or self.features.extractor.stride!=2:
            raise ValueError('Unknown fbank time geometry; refusing an incorrect sample map')
        for row,samples in zip(result,lengths):
            # Fbank stack geometry is checked against the extractor config; the
            # parent project uses 10 ms hops with stride 2 -> 320 samples/frame.
            row['sample_hop']=320.
            row['aux_valid']&=(np.arange(len(row['aux_valid']))+.5)*320 < row['native_samples']
            if row['role']=='noisy':
                row['aux_valid']&=(np.arange(len(row['aux_valid']))+.5)*320 < row['recipe']['original_samples']
        return result


def _loader(dataset,cfg,persistent=False,**kw):
    workers=cfg['workers'];options=dict(dataset=dataset,collate_fn=TripletCollator(cfg['ssl_path']),
        num_workers=workers,pin_memory=False,generator=torch.Generator().manual_seed(cfg['seed']),**kw)
    if workers:options.update(multiprocessing_context='spawn',worker_init_fn=worker_init,
        prefetch_factor=cfg.get('prefetch_factor',2),persistent_workers=persistent)
    return DataLoader(**options)


class TrainingLoader:
    def __init__(self,plan,cfg,run):
        self.plan,self.sampler=plan,SegmentSampler()
        self.loader=_loader(Triplets(plan.rows,cfg,run),cfg,True,batch_sampler=self.sampler)
    def segment(self,epoch,start,stop):
        self.sampler.batches=self.plan.batches(epoch,start,stop);return iter(self.loader)
    def close(self):
        iterator=getattr(self.loader,'_iterator',None)
        if iterator is not None:iterator._shutdown_workers();self.loader._iterator=None


def panel_loader(rows,recipes,cfg,run):
    return _loader(Triplets(rows,cfg,run,'dev',recipes),cfg,batch_size=cfg['feature_batch'],shuffle=False)


def validate_batch(rows,source_batch,cfg=None):
    groups=Counter();occurrences={}
    for r in rows:
        if r['split']!='train' or not math.isfinite(r['ce_weight']) or r['ce_weight']<=0:raise ValueError('Invalid Train mass')
        groups[(r['language'],r['label'])]+=r['ce_weight'];occurrences.setdefault(r['pair_occurrence'],[]).append(r)
    if len(occurrences)!=source_batch or set(groups)!=set(GROUPS) or any(abs(x-.25)>1e-6 for x in groups.values()):
        raise ValueError('Four-group/source CE budget changed')
    for group in occurrences.values():
        roles={r['role'] for r in group}
        if len(roles)!=len(group) or roles not in ({'offline','online','noisy'},{'offline','noisy'}):raise ValueError('Invalid source views')
        if len({tuple(r[k] for k in ('source_id','group_id','source_sha256','label','language')) for r in group})!=1:raise ValueError('Source mismatch')
        expected={'offline':.1,'online':.5,'noisy':.4} if 'online' in roles else {'offline':.2,'noisy':.8}
        if any(abs(r['ce_weight']*source_batch-expected[r['role']])>1e-6 for r in group):raise ValueError('View CE budget changed')
    return occurrences
