"""Smoothed source exposure, exactly balanced CE/TFCL group budgets."""
from collections import Counter
import json
import math
import numpy as np
from w2v_v313.data import GROUPS
from w2v_v316_tfcl.data import (SourcePlan as BasePlan, Triplets as BaseTriplets, Ticket,
    _loader, bundles, tensors, TripletCollator)
from w2v_v3151.data import SegmentSampler
from w2v_v315.augment import recipe


class SourcePlan(BasePlan):
    def __init__(self, rows, batch, seed, cfg=None):
        super().__init__(rows,batch,seed,cfg)
        power = self.cfg.get('source_sampling_power', .5)
        if not 0 <= power <= 1: raise ValueError('Sampling power must be in [0,1]')
        sizes = np.asarray([len(self.pools[k]) for k in GROUPS], dtype=np.float64) ** power
        target = sizes/sizes.sum()*batch
        quotas = np.maximum(1,np.floor(target).astype(int))
        while quotas.sum() < batch: quotas[int(np.argmax(target-quotas))] += 1
        while quotas.sum() > batch:
            excess = np.where(quotas>1,quotas-target,-np.inf)
            quotas[int(np.argmax(excess))] -= 1
        self.quotas = dict(zip(GROUPS,map(int,quotas)))

    def _streams(self, epoch):
        streams = {}
        for index,key in enumerate(GROUPS):
            pool=self.pools[key]; count=self.steps*self.quotas[key]; offset=epoch*count; values=[]
            while len(values)<count:
                cycle,pos=divmod(offset,len(pool))
                order=np.random.default_rng(self.seed+index*1000003+cycle).permutation(len(pool))
                chunk=[pool[i] for i in order[pos:pos+count-len(values)]]
                values.extend(chunk); offset+=len(chunk)
            streams[key]=values
        return streams

    def batches(self, epoch, start=0, stop=None):
        stop=self.steps if stop is None else stop
        if epoch<0 or not 0<=start<=stop<=self.steps: raise ValueError('Invalid source cursor')
        streams=self._streams(epoch); result=[]; rounds=max(self.quotas.values())
        for step in range(start,stop):
            current=[]
            for j in range(rounds):
                phase=self.seed+(epoch*self.steps+step)*rounds+j
                warm=min(1.,(epoch*self.steps+step+1)/max(1,self.steps*.25))
                for key in GROUPS:
                    if j>=self.quotas[key]: continue
                    source=streams[key][step*self.quotas[key]+j];item=self.inventory[source]
                    occurrence=f'{epoch}:{step}:{len(current)}'
                    r=recipe(self.seed,occurrence,phase,item['group_id'],warm=warm)
                    current.append(Ticket(item['indices']['offline'],item['indices'].get('online'),
                        occurrence,phase,warm,json.dumps(r,sort_keys=True)))
            result.append(current)
        return result

    def coverage(self,epoch=0):
        counts=Counter(s for stream in self._streams(epoch).values() for s in stream)
        return dict(steps=self.steps,available_sources=len(self.inventory),unique_sources=len(counts),
            source_draws=sum(counts.values()),group_loss_mass=.25,
            source_groups={f'{k[0]}/{k[1]}':dict(available=len(v),draws=self.steps*self.quotas[k],
                quota=self.quotas[k],mean_draws=self.steps*self.quotas[k]/len(v),
                unique=sum(counts[s]>0 for s in v),maximum_repeats=max(counts[s] for s in v))
                for k,v in self.pools.items()})


def execution_config(cfg,plan):
    return dict(cfg,source_quotas={f'{k[0]}/{k[1]}':v for k,v in plan.quotas.items()})


class Triplets(BaseTriplets):
    def __getitem__(self,ticket):
        rows=super().__getitem__(ticket)
        for row in rows:
            quota=self.cfg['source_quotas'][f'{row["language"]}/{row["label"]}']
            row['source_mass']=.25/quota
            row['ce_weight']*=self.cfg['source_batch']*row['source_mass']
        return rows


class TrainingLoader:
    def __init__(self,plan,cfg,run):
        self.plan,self.sampler=plan,SegmentSampler()
        self.loader=_loader(Triplets(plan.rows,execution_config(cfg,plan),run),cfg,True,batch_sampler=self.sampler)
    def segment(self,epoch,start,stop):
        self.sampler.batches=self.plan.batches(epoch,start,stop)
        return iter(self.loader)
    def close(self):
        iterator=getattr(self.loader,'_iterator',None)
        if iterator is not None: iterator._shutdown_workers();self.loader._iterator=None


def validate_batch(rows,source_batch,cfg=None):
    groups=Counter(); occurrences={}
    for r in rows:
        if r['split']!='train' or not math.isfinite(r['ce_weight']) or r['ce_weight']<=0:
            raise ValueError('Invalid supervised Train mass')
        groups[(r['language'],r['label'])]+=r['ce_weight']
        occurrences.setdefault(r['pair_occurrence'],[]).append(r)
    if len(occurrences)!=source_batch or set(groups)!=set(GROUPS) or any(abs(x-.25)>1e-6 for x in groups.values()):
        raise ValueError('Expected four equally weighted group budgets')
    for group in occurrences.values():
        roles={r['role'] for r in group}
        if len(roles)!=len(group) or roles not in ({'offline','online','noisy'},{'offline','noisy'}):
            raise ValueError('Invalid source views')
        if len({tuple(r[k] for k in ('source_id','group_id','source_sha256','label','language','source_mass')) for r in group})!=1:
            raise ValueError('Paired source identities/weights disagree')
        expected={'offline':.1,'online':.5,'noisy':.4} if 'online' in roles else {'offline':.2,'noisy':.8}
        mass=group[0]['source_mass']
        if not math.isfinite(mass) or mass<=0 or any(abs(r['ce_weight']-mass*expected[r['role']])>1e-6 for r in group):
            raise ValueError('View CE budget differs')
    return occurrences
