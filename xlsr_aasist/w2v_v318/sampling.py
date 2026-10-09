"""Independent balanced Online and Offline-source streams, cyclic source coverage."""
from collections import Counter, defaultdict
import math
import numpy as np
from .common import GROUPS, seed_for


class Plan:
    def __init__(self, rows, stream_sources=16, seed=31801):
        if stream_sources<4 or stream_sources%4: raise ValueError('stream_sources must be a positive multiple of four')
        self.rows,self.batch,self.seed=rows,stream_sources,seed
        self.pools={kind:{g:[] for g in GROUPS} for kind in ('online','offline')}
        identities={};self.duplicates=[]
        for i,r in enumerate(rows):
            key=(r['condition'],r.get('audio_sha256',r['id']))
            if r['condition'] not in self.pools: continue
            if key in identities:
                other=rows[identities[key]]
                if (other['language'],other['label'])!=(r['language'],r['label']):
                    raise ValueError('Identical audio has conflicting metadata')
                self.duplicates.append(r['id']);continue
            identities[key]=i;self.pools[r['condition']][r['language'],r['label']].append(i)
        if any(not p for groups in self.pools.values() for p in groups.values()):
            raise ValueError('Both independent streams need all four groups')
        self.steps=math.ceil(max(sum(map(len,groups.values())) for groups in self.pools.values())/self.batch)

    def stream(self,kind,group,epoch):
        pool=self.pools[kind][group];count=self.steps*(self.batch//4);offset=epoch*count;values=[]
        while len(values)<count:
            cycle,pos=divmod(offset,len(pool))
            order=np.random.default_rng(seed_for(self.seed,kind,group,cycle)).permutation(len(pool))
            chunk=[pool[i] for i in order[pos:pos+count-len(values)]]
            values.extend(chunk);offset+=len(chunk)
        return values

    def batches(self,epoch):
        streams={(kind,g):self.stream(kind,g,epoch) for kind in self.pools for g in GROUPS}
        q=self.batch//4
        for step in range(self.steps):
            tickets=[]
            for g in GROUPS:
                for j in range(q):
                    for kind in ('online','offline'):
                        i=streams[kind,g][step*q+j]
                        tickets.append(dict(index=i,kind=kind,epoch=epoch,step=step,
                            occurrence=f'{epoch}:{step}:{kind}:{g[0]}:{g[1]}:{j}'))
            order=np.random.default_rng(seed_for(self.seed,'physical-order',epoch,step)).permutation(len(tickets))
            yield [tickets[i] for i in order]

    def coverage(self,epoch):
        cells={}
        for kind,groups in self.pools.items():
            for group,pool in groups.items():
                counts=Counter(self.stream(kind,group,epoch))
                cells[f'{kind}/{group[0]}/{group[1]}']=dict(available=len(pool),draws=sum(counts.values()),
                    unique=len(counts),maximum_repeats=max(counts.values()),loss_mass=.125,
                    views_per_draw=1 if kind=='online' else 2)
        online=self.steps*self.batch;noisy=online*2
        return dict(epoch=epoch+1,steps=self.steps,cells=cells,online_views=online,noisy_views=noisy,
                    stream_loss_mass={'online':.5,'noisy':.5},class_loss_mass={'fake':.5,'real':.5},
                    class_view_counts={'fake':(online+noisy)//2,'real':(online+noisy)//2},
                    duplicate_waveform_ids_excluded=self.duplicates,
                    source_policy='cycle each group before reshuffle; do not reset minority cycles at epoch boundaries')


def validate_units(units,stream_sources):
    counts=Counter()
    for unit in units:
        row=unit[0];kind='online' if row['role']=='online' else 'offline'
        if row['split']!='train' or len(unit)!=(1 if kind=='online' else 2): raise ValueError('Invalid training unit')
        if len({(r['source_id'],r['language'],r['label'],r['occurrence']) for r in unit})!=1:
            raise ValueError('Condition pair identity mismatch')
        counts[kind,row['language'],row['label']]+=1
    if counts!=Counter({(k,*g):stream_sources//4 for k in ('online','offline') for g in GROUPS}):
        raise ValueError('Actual batch violates equal stream/group budgets')


def probe_tickets(plan,per_group=512):
    tickets=[]
    for kind,groups in plan.pools.items():
        count=min(per_group,*(len(p) for p in groups.values()))
        for group,pool in groups.items():
            selected=sorted(pool,key=lambda i:seed_for(plan.seed,'train-probe',plan.rows[i]['id']))[:count]
            for i in selected:
                tickets.append(dict(index=i,kind=kind,epoch=0,step=0,occurrence='fixed-probe:'+str(i),probe=True))
    return tickets
