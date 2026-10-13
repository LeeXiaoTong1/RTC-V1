"""Version budgets do not change source coverage or original/augmented odds."""
from collections import Counter
import numpy as np
from w2v_v3183r2.sampling import Plan as SourcePlan,validate_units,probe_tickets
from .common import GROUPS,seed_for
from .views import FAMILIES

DEFAULT_VIEW_FACTORS=(5,5,1,3)
# Two distinct families/source; each slot is balanced over eight source IDs.
PAIR_FAMILIES=((1,2),(2,3),(3,4),(4,1),(1,3),(3,1),(2,4),(4,2))


class Plan(SourcePlan):
    def __init__(self,rows,stream_sources=16,seed=31801,quotas=(4,3,5,4),
                 view_factors=DEFAULT_VIEW_FACTORS,refresh_base_variants=False):
        if len(view_factors)!=4 or any(type(v) is not int or v not in (1,3,5) for v in view_factors):
            raise ValueError('Four fixed version budgets must each be 1, 3, or 5')
        if type(refresh_base_variants) is not bool:raise ValueError('refresh_base_variants must be boolean')
        super().__init__(rows,stream_sources,seed,quotas)
        self.view_factors=dict(zip(GROUPS,view_factors))
        self.refresh_base_variants=refresh_base_variants
        self._view_epoch=None;self._view_streams={};self.definitions={}
        for kind,groups in self.pools.items():
            for group,pool in groups.items():
                ordered=sorted(pool,key=lambda i:(seed_for(seed,'v3185-definition',kind,rows[i]['id']),rows[i]['id']))
                for rank,index in enumerate(ordered):
                    factor=self.view_factors[group]
                    families=tuple(range(1,5)) if factor==5 else PAIR_FAMILIES[rank%8] if factor==3 else ()
                    self.definitions[index]=dict(factor=factor,phase=rank%5,
                        family_phase=rank%4,families=families)

    def view_spec(self,index,kind,cycle):
        d=self.definitions[index];phase=d['phase']
        if cycle<0:raise ValueError('source cycle must be nonnegative')
        original=(cycle+phase)%5==0
        # Augmented visits before this source visit, excluding the staggered
        # original slots. This starts at zero and can be reconstructed on resume.
        originals_before=(cycle+phase+4)//5-(phase+4)//5
        augmented_before=cycle-originals_before
        dynamic=False;generation=0;variant=0;family=0
        if not original:
            if d['factor']==1:
                dynamic=True;family=1+(augmented_before+d['family_phase'])%4
                generation=augmented_before//4
            else:
                count=d['factor']-1
                rotation=d['family_phase'] if count==4 else 0
                variant=1+(augmented_before+rotation)%count
                family=d['families'][variant-1]
                if self.refresh_base_variants:generation=augmented_before//count
        mode='dynamic' if dynamic else 'fixed'
        identity=f'{self.rows[index]["id"]}#v3185:{kind}:{mode}:{variant}:g{generation}:{FAMILIES[family]}'
        return dict(base_variant=variant,base_family=FAMILIES[family],base_generation=generation,
            base_dynamic=dynamic,base_view_id=identity)

    def view_stream(self,kind,group,epoch):
        if epoch<0:raise ValueError('epoch must be nonnegative')
        if self._view_epoch!=epoch:self._view_epoch=epoch;self._view_streams={}
        key=(kind,group)
        if key in self._view_streams:return self._view_streams[key]
        pool=self.pools[kind][group];size=len(pool);count=self.steps*self.quotas[group]
        offset=epoch*count;values=[]
        while len(values)<count:
            cycle,pos=divmod(offset,size);order=self.order(kind,group,cycle)
            chunk=order[pos:pos+count-len(values)]
            for index in chunk:
                row=pool[index];values.append((row,self.view_spec(row,kind,cycle),cycle))
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
                        index,spec,cycle=streams[kind,group][step*q+j]
                        tickets.append(dict(index=index,kind=kind,epoch=epoch,step=step,
                            **spec,source_cycle=cycle,
                            # Retain the V3.18.4 noise/RTC seed key for a
                            # controlled augmentation-only comparison.
                            occurrence=f'v3184-aug5:{epoch}:{step}:{kind}:{group[0]}:{group[1]}:{j}'))
            order=np.random.default_rng(seed_for(self.seed,'aug5-order',epoch,step)).permutation(len(tickets))
            yield [tickets[i] for i in order]

    def _cumulative_coverage(self,kind,group,draws):
        """Exact cumulative counts without replaying earlier epochs."""
        pool=self.pools[kind][group];cycles,remainder=divmod(draws,len(pool))
        extra=set(self.order(kind,group,cycles)[:remainder]) if remainder else set()
        fixed_slots=realized=dynamic=0
        for position,index in enumerate(pool):
            visits=cycles+int(position in extra);d=self.definitions[index];phase=d['phase']
            originals=(visits+phase+4)//5-(phase+4)//5
            augmented=visits-originals;has_original=int(originals>0)
            fixed_slots+=has_original+min(d['factor']-1,augmented)
            if d['factor']==1:
                dynamic+=augmented;realized+=has_original+augmented
            else:
                realized+=has_original+(augmented if self.refresh_base_variants else min(d['factor']-1,augmented))
        return dict(unique_fixed_slots_through_epoch=fixed_slots,
            unique_realized_views_through_epoch=realized,unique_dynamic_views_through_epoch=dynamic)

    def coverage(self,epoch):
        result=super().coverage(epoch)
        for kind in self.pools:
            for group in GROUPS:
                values=self.view_stream(kind,group,epoch);n=len(self.pools[kind][group])
                specs=[s for _,s,_ in values];originals=sum(s['base_family']=='original' for s in specs)
                cell=result['cells'][f'{kind}/{group[0]}/{group[1]}']
                cell.update(fixed_view_factor=self.view_factors[group],available_virtual_views=n*self.view_factors[group],
                    available_fixed_views=n*self.view_factors[group],
                    unique_virtual_views_this_epoch=len({s['base_view_id'] for s in specs}),
                    unique_fixed_views_this_epoch=len({s['base_view_id'] for s in specs if not s['base_dynamic']}),
                    unique_dynamic_views_this_epoch=len({s['base_view_id'] for s in specs if s['base_dynamic']}),
                    original_draws=originals,augmented_draws=len(specs)-originals,
                    original_fraction=originals/len(specs),target_original_fraction=.2,
                    dynamic_augmentation_draws=sum(s['base_dynamic'] for s in specs),
                    base_families=dict(Counter(s['base_family'] for s in specs)),
                    fixed_variant_draws=dict(Counter(str(s['base_variant']) for s in specs if not s['base_dynamic'])),
                    generations=dict(Counter(str(s['base_generation']) for s in specs)))
                cell.update(self._cumulative_coverage(kind,group,(epoch+1)*len(values)))
        result['view_factors']={f'{g[0]}/{g[1]}':v for g,v in self.view_factors.items()}
        result['refresh_base_variants']=self.refresh_base_variants
        result['source_policy']='Complete source cycles; no within-stream/group batch repeats; 20% original and 80% augmented per five source visits, independent of version budget.'
        return result


def describe(rows,seed,stream_sources,quotas,view_factors=DEFAULT_VIEW_FACTORS,refresh_base_variants=False):
    plan=Plan(rows,stream_sources,seed,quotas,view_factors,refresh_base_variants)
    result=plan.coverage(0)
    result.update(virtual_definition='Source ID + stream + fixed slot + generation; dynamic common augmentation is counted separately from fixed version budgets.',
        view_families=list(FAMILIES),waveform_cache=False,
        warning='Augmented views are not new speakers or independent recordings. Per-epoch exposure may differ from the long-run 20/80 target.')
    return result
