"""Independent 16 Online + 16 paired-Noisy sources, balanced loss, unequal draws."""
from collections import Counter
import numpy as np
from w2v_v318.sampling import Plan as LegacyPlan, probe_tickets
from .common import GROUPS, seed_for


class Plan(LegacyPlan):
    def __init__(self, rows, stream_sources=16, seed=31801, quotas=(4,2,7,3)):
        super().__init__(rows, stream_sources, seed)
        if len(quotas) != 4 or min(quotas) < 1 or sum(quotas) != stream_sources:
            raise ValueError('Four positive quotas must sum to stream_sources')
        self.quotas = dict(zip(GROUPS, quotas))
        for kind, pools in self.pools.items():
            for group, pool in pools.items():
                if len(pool) < self.quotas[group]:
                    raise ValueError(f'{kind}/{group}: too few distinct sources for a batch')
        self._cycles, self._streams, self._epoch = {}, {}, None

    def order(self, kind, group, cycle):
        # Cache permutations; repairing a cycle boundary prevents a source from
        # appearing twice in one logical batch without dropping any source.
        key=(kind,group,cycle)
        if key in self._cycles: return self._cycles[key]
        size=len(self.pools[kind][group]); quota=self.quotas[group]
        for c in range(cycle+1):
            k=(kind,group,c)
            if k in self._cycles: continue
            order=np.random.default_rng(seed_for(self.seed,'v3183-cycle',kind,group,c)).permutation(size).tolist()
            tail=(c*size)%quota
            if tail:
                previous=set(self._cycles[kind,group,c-1][-tail:])
                needed=quota-tail
                for j in range(needed):
                    if order[j] in previous:
                        swap=next(i for i in range(needed,size) if order[i] not in previous)
                        order[j],order[swap]=order[swap],order[j]
            self._cycles[k]=order
        return self._cycles[key]

    def stream(self, kind, group, epoch):
        if self._epoch != epoch: self._epoch, self._streams = epoch, {}
        key=(kind,group)
        if key in self._streams: return self._streams[key]
        pool=self.pools[kind][group]; count=self.steps*self.quotas[group]
        offset=epoch*count; values=[]
        while len(values)<count:
            cycle,pos=divmod(offset,len(pool));order=self.order(kind,group,cycle)
            chunk=[pool[i] for i in order[pos:pos+count-len(values)]]
            values.extend(chunk);offset+=len(chunk)
        self._streams[key]=values
        return values

    def batches(self, epoch):
        streams={(k,g):self.stream(k,g,epoch) for k in self.pools for g in GROUPS}
        for step in range(self.steps):
            tickets=[]
            for group in GROUPS:
                q=self.quotas[group]
                for j in range(q):
                    for kind in ('online','offline'):
                        tickets.append(dict(index=streams[kind,group][step*q+j],kind=kind,epoch=epoch,step=step,
                            occurrence=f'v3183:{epoch}:{step}:{kind}:{group[0]}:{group[1]}:{j}'))
            order=np.random.default_rng(seed_for(self.seed,'v3183-order',epoch,step)).permutation(len(tickets))
            yield [tickets[i] for i in order]

    def coverage(self, epoch):
        report=super().coverage(epoch)
        report['group_quotas']={f'{g[0]}/{g[1]}':q for g,q in self.quotas.items()}
        report['class_view_counts']={label:self.steps*3*sum(q for g,q in self.quotas.items() if g[1]==i)
                                     for i,label in enumerate(('fake','real'))}
        report['source_policy']='cyclic, no within-stream/group batch duplicates; independent Online and Noisy pools'
        for kind in self.pools:
            for g,q in self.quotas.items():
                cell=report['cells'][f'{kind}/{g[0]}/{g[1]}']
                cell.update(quota=q,per_source_loss_mass=.125/q,mean_draws=cell['draws']/cell['available'])
        return report


def validate_units(units, quotas):
    counts=Counter();identities={}
    for unit in units:
        row=unit[0];kind='online' if row['role']=='online' else 'offline'
        roles={'online'} if kind=='online' else {'noisy_a','noisy_b'}
        if len(unit)!=(1 if kind=='online' else 2) or {r['role'] for r in unit}!=roles:
            raise ValueError('Expected Online or an indivisible Noisy pair; raw Offline is forbidden')
        if any(r['split']!='train' for r in unit):raise ValueError('Training split required')
        if len({(r['source_id'],r['language'],r['label'],r['occurrence']) for r in unit})!=1:
            raise ValueError('Noisy pair identity mismatch')
        key=(kind,row['language'],row['label']);counts[key]+=1
        seen=identities.setdefault(key,set())
        if row['id'] in seen:raise ValueError('Repeated source within a logical stream/group batch')
        seen.add(row['id'])
    expected=Counter({(k,*g):q for k in ('online','offline') for g,q in zip(GROUPS,quotas)})
    if counts!=expected:raise ValueError('Logical batch violates configured quotas')
