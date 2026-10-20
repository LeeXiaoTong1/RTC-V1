"""Original exposure budgets, no neighbor swaps; balanced dual-PCL references."""
from collections import Counter
from w2v_v31810.sampling import Plan as BasePlan


class Plan(BasePlan):
    def __init__(self,rows,cfg,phones):
        super().__init__(rows,stream_sources=cfg['stream_sources'],seed=cfg['seed'],quotas=cfg['group_quotas'],
            view_factors=cfg['view_factors'],refresh_base_variants=cfg['refresh_base_variants'],cfg=cfg)
        self.rows4=rows;self.cfg4=cfg;self.phones=phones;self.audit={}

    def epoch_batches(self,epoch):
        batches=list(super().batches(epoch));counts=Counter()
        active=epoch>=self.cfg4.get('warm_epochs',1) and self.cfg4.get('consistency','dual')!='none'
        for step,batch in enumerate(batches):
            for t in batch:t['epoch']=epoch
            if not active:continue
            for kind,branch in (('online','official'),('offline','noise')):
                if branch=='noise' and self.cfg4.get('consistency')!='dual':continue
                for lang in ('en','zh'):
                    for label in (0,1):
                        candidates=[t for t in batch if t['kind']==kind and
                            (self.rows4[t['index']]['language'],self.rows4[t['index']]['label'])==(lang,label) and
                            self.rows4[t['index']]['id'] in self.phones[branch]]
                        if candidates:
                            t=candidates[(epoch+step)%len(candidates)]
                            t['pcl_'+branch]=self.phones[branch][self.rows4[t['index']]['id']]
                            counts[branch+'/'+lang+'/'+str(label)]+=1
        self.audit=dict(pcl_scheduled=dict(counts),classification_views=sum(map(len,batches)),
            maximum_pairs_per_update=8 if active and self.cfg4.get('consistency')=='dual' else 4 if active else 0,
            max_reference_views_per_update=12 if active and self.cfg4.get('consistency')=='dual' else 8 if active else 0,
            classification_source_budget_unchanged=True,source_epoch=epoch)
        return batches
