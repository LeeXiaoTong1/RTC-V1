"""Small same-source Train check; separate augmentation conflict from source mix."""
from collections import Counter
import torch
from torch.nn import functional as F
from .data import Waves
from .pcl import expand,pack

def packs(rows,cfg):
    for items in pack([expand(r,False) for r in rows],cfg['microbatch'],cfg['frame_budget']):
        yield [x['rows'][0] for x in items]
from w2v_v3189.data import prepared
from .common import capture_rng, restore_rng
from .gradients import coordinate


def paired_probe(model,rows,cfg,plan,dataset_factory=Waves):
    count=cfg.get('paired_probe_per_cell',2)
    online={}
    for i,r in enumerate(rows):
        if r['condition']=='online':online.setdefault(r['canonical_source'],i)
    selected=[];counts=Counter();seen=set()
    rng=capture_rng();modes=[(m,m.training) for m in model.modules()]
    try:
        model.eval();dataset=dataset_factory(rows,cfg)
        for batch in plan.epoch_batches(0):
            for ticket in batch:
                row=rows[ticket['index']];group=row['language'],row['label'];source=row['canonical_source']
                if ticket['kind']!='offline' or source not in online or source in seen or counts[group]>=count:continue
                noisy=dataset[ticket][0];reference=rows[online[source]]
                clean=prepared(reference,dataset.original(reference),role='online')
                selected.append((clean,noisy));counts[group]+=1;seen.add(source)
            if len(counts)==4 and min(counts.values())>=count:break
        if len(counts)!=4:
            return dict(status='insufficient_paired_train_groups',sources=len(selected),optimizer_updates=0)
        parameters=[p for p in model.parameters() if p.requires_grad]
        gradients=[];losses=[];scale=256. if cfg.get('backend_precision')=='fp16' else 1.
        for side in (0,1):
            accum=[torch.zeros_like(p) for p in parameters];total=0.
            for batch in packs([pair[side] for pair in selected],cfg):
                logits=model([r['wave'] for r in batch]).float()
                labels=torch.tensor([r['label'] for r in batch],device=logits.device)
                loss=sum(v/(4*counts[r['language'],r['label']]) for v,r in zip(
                    F.cross_entropy(logits,labels,reduction='none',label_smoothing=cfg.get('label_smoothing',.02)),batch))
                gs=torch.autograd.grad(loss*scale,parameters,allow_unused=True)
                for target,g in zip(accum,gs):
                    if g is not None:target.add_(g.detach())
                total+=float(loss.detach())
            gradients.append(accum);losses.append(total)
        _,stats=coordinate(*gradients,[torch.zeros_like(p) for p in parameters],enabled=False,scale=scale)
        return dict(status='measured',sources=len(selected),source_ids=sorted(seen),
            paired=True,clean_reference='same-source official original Online',noisy_view='current Offline base augmentation -> noise -> RTC',
            online_ce=losses[0],noisy_ce=losses[1],gradients=stats,optimizer_updates=0,
            limitation='Small fixed Train probe, not proof of the cause of unseen-Noisy errors')
    finally:
        for module,mode in modes:module.training=mode
        restore_rng(rng)
