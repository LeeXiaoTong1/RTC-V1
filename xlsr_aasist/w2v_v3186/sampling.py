"""Unchanged V3.18.5 classification exposure plus explicit auxiliary tickets."""
from collections import Counter
import numpy as np
from w2v_v3185.sampling import Plan as BasePlan, DEFAULT_VIEW_FACTORS
from w2v_v3183r2.sampling import validate_units, probe_tickets
from w2v_v3185.common import GROUPS, seed_for


def canonical_pairs(rows):
    """Use saved official source identities; filenames never establish pairs."""
    offline={};online={}
    for index,row in enumerate(rows):
        if row.get('condition') not in ('online','offline'):continue
        if row.get('split')!='train':raise ValueError('Auxiliary inventory must be Train only')
        source=row.get('source_id')
        if not source:raise ValueError('Canonical source_id is required for PCL')
        target=offline if row['condition']=='offline' else online
        if source in target:raise ValueError('Repeated canonical source/condition: '+str(source))
        target[source]=index
    result={}
    for source,index in online.items():
        if source not in offline:raise ValueError('Official Online lacks its canonical Offline source')
        off=rows[offline[source]];on=rows[index]
        keys=('language','label','group_id','source_sha256')
        if any(off.get(key)!=on.get(key) for key in keys):
            raise ValueError('Official source pair metadata disagrees: '+str(source))
        if not off.get('full_length',True) or not on.get('full_length',True):
            raise ValueError('Canonical source audio must contain the full recording')
        result[offline[source]]=index
    return result


class Plan(BasePlan):
    def __init__(self,rows,stream_sources=16,seed=31801,quotas=(4,3,5,4),
                 view_factors=DEFAULT_VIEW_FACTORS,refresh_base_variants=False,*,cfg=None):
        super().__init__(rows,stream_sources,seed,quotas,view_factors,refresh_base_variants)
        self.cfg=cfg or {}
        self.consistency=self.cfg.get('consistency','pcl')!='none'
        self.platform_count=int(self.cfg.get('platform_sources_per_group',2))
        if self.platform_count<0 or self.platform_count>min(quotas):
            raise ValueError('Platform PCL target must fit every Offline source quota')
        self.pairs=canonical_pairs(rows) if self.consistency else {}

    def batches(self,epoch):
        for tickets in super().batches(epoch):
            if not self.consistency:yield tickets;continue
            selected=set()
            if self.cfg.get('pcl_platform_weight',.15)>0:
                for group in GROUPS:
                    candidates=[t for t in tickets if t['kind']=='offline' and
                        (self.rows[t['index']]['language'],self.rows[t['index']]['label'])==group and
                        t['index'] in self.pairs]
                    candidates.sort(key=lambda t:seed_for(self.seed,'v3186-platform',epoch,
                        t['step'],self.rows[t['index']]['id']))
                    selected.update(t['index'] for t in candidates[:self.platform_count])
            for ticket in tickets:
                if ticket['kind']!='offline':continue
                # Each visit to the SAME source alternates A/B. Independent
                # stream ordering or resume cannot reset this source schedule.
                ticket['pcl_noisy_index']=ticket['source_cycle']%2
                ticket['pcl_noise']=self.cfg.get('pcl_noise_weight',.15)>0
                if ticket['index'] in selected:
                    ticket['pcl_online_index']=self.pairs[ticket['index']]
                ticket['pcl_enabled']=bool(ticket['pcl_noise'] or ticket['index'] in selected)
            yield tickets

    def coverage(self,epoch):
        result=super().coverage(epoch)
        result['pcl']=dict(enabled=self.consistency,
            official_pairs=len(self.pairs),platform_target_per_group=self.platform_count,
            classification_schedule_unchanged=True,
            missing_pair_policy='retain classification and noise PCL; never replace sources')
        return result

    def audit_budget(self,epochs=6):
        """Exact budget audit, separate fixed slots from dynamic augmentation.

        No enlarged on-disk dataset is claimed. Available fixed views and views
        actually visited are distinct; uncovered slots are reported explicitly.
        """
        if type(epochs) is not int or epochs<1:raise ValueError('epochs must be positive')
        cells={}
        for kind,groups in self.pools.items():
            for group,pool in groups.items():
                source_counts=Counter();fixed=set();realized=Counter();families=Counter();original=0
                for epoch in range(epochs):
                    for index,spec,_ in self.view_stream(kind,group,epoch):
                        source_counts[index]+=1;realized[spec['base_view_id']]+=1
                        if not spec['base_dynamic']:fixed.add((index,spec['base_variant']))
                        families[spec['base_family']]+=1
                        original+=int(spec['base_family']=='original')
                available=len(pool)*self.view_factors[group];draws=sum(source_counts.values())
                cells[f'{kind}/{group[0]}/{group[1]}']=dict(
                    source_count=len(pool),unique_sources_seen=len(source_counts),
                    fixed_view_factor=self.view_factors[group],available_fixed_views=available,
                    fixed_views_seen=len(fixed),fixed_views_unseen=available-len(fixed),
                    all_fixed_views_covered=len(fixed)==available,realized_views_seen=len(realized),
                    repeated_view_draws=draws-len(realized),maximum_view_visits=max(realized.values()),
                    source_draws=draws,minimum_source_visits=min(source_counts.get(i,0) for i in pool),
                    maximum_source_visits=max(source_counts.values()),mean_source_visits=draws/len(pool),
                    original_draws=original,augmented_draws=draws-original,
                    original_fraction=original/draws,families=dict(families))
        return dict(epochs=epochs,steps_per_epoch=self.steps,cells=cells,
            all_fixed_views_covered=all(c['all_fixed_views_covered'] for c in cells.values()),
            all_expanded_views_covered=all(c['all_fixed_views_covered'] for c in cells.values() if c['fixed_view_factor']>1),
            all_source_recordings_seen=all(c['unique_sources_seen']==c['source_count'] for c in cells.values()),
            storage='virtual deterministic views generated on demand; no disk audio expansion',
            note='Different views do not add independent speakers or source recordings. '
                 'ZH-fake has no fixed expansion; its original slot need not appear for every source '
                 'within a short budget because shared dynamic augmentation retains 20/80 exposure.')


def describe(rows,seed,stream_sources,quotas,view_factors=DEFAULT_VIEW_FACTORS,
             refresh_base_variants=False,*,cfg=None):
    plan=Plan(rows,stream_sources,seed,quotas,view_factors,refresh_base_variants,cfg=cfg)
    result=plan.coverage(0)
    saved=(cfg or {}).get('augmentation_budget_audit')
    result['full_budget']=saved if saved is not None else plan.audit_budget(int((cfg or {}).get('epochs',6)))
    result['waveform_cache']=False
    return result
