"""Cycle ORIGINAL sources, then rotate views: expansion never inflates source counts."""
from collections import Counter
import numpy as np
from w2v_v3183r2.sampling import Plan as SourcePlan,validate_units,probe_tickets
from .common import GROUPS,seed_for
from .views import FAMILIES


class Plan(SourcePlan):
    def __init__(self,rows,stream_sources=16,seed=31801,quotas=(4,3,5,4),view_factor=5):
        if view_factor not in (1,5):raise ValueError('Use 5 views, or 1 for a no-base-augmentation ablation')
        super().__init__(rows,stream_sources,seed,quotas)
        self.view_factor=view_factor;self._view_epoch=None;self._view_streams={}
        self.phases={i:seed_for(seed,'aug5-phase',kind,rows[i]['id'])%view_factor
                     for kind,groups in self.pools.items() for pool in groups.values() for i in pool}

    def view_stream(self,kind,group,epoch):
        if self._view_epoch!=epoch:self._view_epoch=epoch;self._view_streams={}
        key=(kind,group)
        if key in self._view_streams:return self._view_streams[key]
        pool=self.pools[kind][group];size=len(pool);count=self.steps*self.quotas[group]
        offset=epoch*count;values=[]
        while len(values)<count:
            cycle,pos=divmod(offset,size);order=self.order(kind,group,cycle)
            chunk=order[pos:pos+count-len(values)]
            for index in chunk:
                row=pool[index]
                # Stagger first views across sources, independent of the label.
                phase=self.phases[row]
                values.append((row,(cycle+phase)%self.view_factor,cycle))
            offset+=len(chunk)
        self._view_streams[key]=values
        return values

    def stream(self,kind,group,epoch):
        return [i for i,_,_ in self.view_stream(kind,group,epoch)]

    def batches(self,epoch):
        streams={(k,g):self.view_stream(k,g,epoch) for k in self.pools for g in GROUPS}
        for step in range(self.steps):
            tickets=[]
            for group in GROUPS:
                q=self.quotas[group]
                for j in range(q):
                    for kind in ('online','offline'):
                        index,variant,cycle=streams[kind,group][step*q+j]
                        tickets.append(dict(index=index,kind=kind,epoch=epoch,step=step,
                            base_variant=variant,source_cycle=cycle,
                            occurrence=f'v3184-aug5:{epoch}:{step}:{kind}:{group[0]}:{group[1]}:{j}'))
            order=np.random.default_rng(seed_for(self.seed,'aug5-order',epoch,step)).permutation(len(tickets))
            yield [tickets[i] for i in order]

    def coverage(self,epoch):
        result=super().coverage(epoch)
        for kind in self.pools:
            for group in GROUPS:
                values=self.view_stream(kind,group,epoch);n=len(self.pools[kind][group])
                cell=result['cells'][f'{kind}/{group[0]}/{group[1]}']
                cell.update(available_virtual_views=n*self.view_factor,
                    unique_virtual_views_this_epoch=len({(i,v) for i,v,_ in values}),
                    unique_virtual_views_through_epoch=min((epoch+1)*len(values),n*self.view_factor),
                    base_families=dict(Counter(FAMILIES[v] for _,v,_ in values)))
        result['view_factor']=self.view_factor
        result['source_policy']='Original-source cycles; one distinct source per stream/group batch; each source sees all view slots before repeat.'
        return result


def describe(rows,seed,stream_sources,quotas,view_factor=5):
    plan=Plan(rows,stream_sources,seed,quotas,view_factor)
    result=plan.coverage(0)
    result.update(virtual_definition='original ID + Online/Offline + fixed view slot; seeds reconstruct every waveform',
        view_families=list(FAMILIES[:view_factor]),waveform_cache=False,
        warning='Virtual views are not new speakers or independent recording sources.')
    return result
