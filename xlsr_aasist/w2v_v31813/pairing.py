"""Reorder existing source tickets; never expand Real/Fake exposure budgets."""
from collections import defaultdict,Counter
import copy
from w2v_v31810.sampling import Plan as BasePlan


def source_fields(ticket):
    return {k:copy.deepcopy(v) for k,v in ticket.items() if k.startswith('base_') or k in
            ('index','canonical_source','source_cycle','noise_recipe_view')}


def pair_batches(batches,rows,neighbors,epoch,cap=2):
    batches=copy.deepcopy(batches);positions=defaultdict(set);members=[];used=Counter();locked=set();counts=Counter()
    for b,batch in enumerate(batches):
        members.append(Counter(t['canonical_source'] for t in batch))
        if any(n!=1 for n in members[-1].values()):raise ValueError('Duplicate source before XEUS pairing')
        for j,t in enumerate(batch):
            if rows[t['index']]['label']==0:positions[t['kind'],t['canonical_source']].add((b,j))
    for b,batch in enumerate(batches):
        for kind in ('online','offline'):
            for lang in ('en','zh'):
                real=[j for j,t in enumerate(batch) if t['kind']==kind and
                      (rows[t['index']]['language'],rows[t['index']]['label'])==(lang,1)]
                fake=[j for j,t in enumerate(batch) if t['kind']==kind and
                      (rows[t['index']]['language'],rows[t['index']]['label'])==(lang,0)]
                # One pair per stream/language/update; rotate anchors deterministically.
                real=real[(b+epoch)%len(real):]+real[:(b+epoch)%len(real)] if real else []
                found=False
                for rpos in real:
                    anchor=batch[rpos];rid=anchor['canonical_source']
                    if used[rid]>=cap:continue
                    candidates=neighbors.get(rid,[])
                    if candidates:
                        offset=(epoch+b)%len(candidates);candidates=candidates[offset:]+candidates[:offset]
                    for candidate in candidates:
                        fid=candidate['source']
                        if used[fid]>=cap:continue
                        for ob,oj in sorted(positions[kind,fid]):
                            if ob<b or (ob,oj) in locked:continue
                            targets=[oj] if ob==b else fake
                            for fj in targets:
                                if (b,fj) in locked:continue
                                old=batch[fj]['canonical_source']
                                if ob!=b and (members[b].get(fid,0) or members[ob].get(old,0)):continue
                                if ob!=b:
                                    a,bits=source_fields(batch[fj]),source_fields(batches[ob][oj])
                                    batch[fj].update(bits);batches[ob][oj].update(a)
                                    positions[kind,old].remove((b,fj));positions[kind,old].add((ob,oj))
                                    positions[kind,fid].remove((ob,oj));positions[kind,fid].add((b,fj))
                                    members[b][old]-=1;members[b][fid]+=1;members[ob][fid]-=1;members[ob][old]+=1
                                partner=batch[fj];pid=f'{epoch}:{b}:{kind}:{lang}'
                                for t in (anchor,partner):t['xeus_pair']=pid;t['xeus_similarity']=candidate['similarity']
                                # Match RTC family, scene and strength. Online base
                                # augmentations retain the same label-neutral policy.
                                if kind=='offline':
                                    for key in ('domain_override','nuisance_seed','condition_rank'):
                                        partner[key]=copy.deepcopy(anchor[key])
                                locked.add((b,fj));used[rid]+=1;used[fid]+=1;counts[kind+'/'+lang]+=1
                                found=True;break
                            if found:break
                        if found:break
                    if found:break
    for batch in batches:
        if len({t['canonical_source'] for t in batch})!=len(batch):raise ValueError('XEUS swap broke source isolation')
    return batches,dict(pairs=dict(counts),source_cap=cap,distinct_paired_sources=len(used),max_source_uses=max(used.values(),default=0))


class Plan(BasePlan):
    def __init__(self,rows,cfg,neighbors,phones):
        super().__init__(rows,stream_sources=cfg['stream_sources'],seed=cfg['seed'],quotas=cfg['group_quotas'],
                         view_factors=cfg['view_factors'],refresh_base_variants=cfg['refresh_base_variants'],cfg=cfg)
        self.rows13=rows;self.cfg13=cfg;self.neighbors=neighbors;self.phones=phones
        self.audit13={}

    def epoch_batches(self,epoch):
        # Continue augmentation visits from the parent's completed best epoch.
        visit=epoch+self.cfg13['source_epoch_offset']
        batches=list(super().batches(epoch))
        # Keep the new-run warmup/ramp counter independent of augmentation cycles.
        for batch in batches:
            for ticket in batch:ticket['epoch']=epoch
        if self.cfg13['mode'] in ('full','random') and epoch>=1:
            batches,audit=pair_batches(batches,self.rows13,self.neighbors,epoch)
        else:audit=dict(pairs={},source_cap=2,distinct_paired_sources=0,max_source_uses=0)
        selected=Counter()
        if epoch>=1:
            for step,batch in enumerate(batches):
                for lang in ('en','zh'):
                    for label in (0,1):
                        candidates=[t for t in batch if t['kind']=='online' and
                            (self.rows13[t['index']]['language'],self.rows13[t['index']]['label'])==(lang,label) and
                            self.rows13[t['index']]['id'] in self.phones]
                        if candidates:
                            t=candidates[(epoch+step)%len(candidates)]
                            t['official_pcl']=self.phones[self.rows13[t['index']]['id']]
                            selected[lang+'/'+str(label)]+=1
        self.audit13=dict(audit,pcl_sources=dict(selected),classification_views=sum(map(len,batches)),
                          max_reference_views_per_update=8,mode=self.cfg13['mode'],source_epoch=visit)
        return batches
