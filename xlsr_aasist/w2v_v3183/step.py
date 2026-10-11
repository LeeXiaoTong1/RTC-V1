"""Single-pass C3 with one telemetry transfer per logical update."""
from collections import Counter, defaultdict
import math
import torch
from .common import require_finite
from .data import microbatches
from .sampling import validate_units
from .objective import micro_losses


def schedule(optimizer,cfg,epoch,step,steps):
    index=epoch*steps+step+1;total=cfg['epochs']*steps
    warm=min(cfg['lr_warmup_updates'],max(1,total//10))
    fraction=(index-warm)/max(1,total-warm)
    multiplier=index/warm if index<=warm else cfg['lr_floor']+(1-cfg['lr_floor'])*.5*(1+math.cos(math.pi*fraction))
    for group in optimizer.param_groups:group['lr']=group['initial_lr']*multiplier
    return cfg['risk_coefficient']


def train_step(model,optimizer,units,cfg,coefficient,audit=False):
    validate_units(units,cfg['group_quotas']);model.train();optimizer.zero_grad(set_to_none=True)
    processing=Counter(); telemetry=[]; metadata=[]
    for unit in units:
        for row in unit:
            r=row.get('augmentation',{}).get('recipe')
            if r:
                processing[f'{row["language"]}/{row["label"]}/{r["family"]}/{r["codec"]}/{r["noise_type"]}']+=1
                if r.get('stress'):processing['stress/'+r['stress']['mechanism']]+=1
    for micro in microbatches(units,cfg['microbatch'],cfg['frame_budget']):
        rows=[r for u in micro for r in u];logits=model([r['wave'] for r in rows]);offset=0;terms=[]
        for unit,(objective,raw) in zip(micro,micro_losses(logits,micro,cfg['group_quotas'],coefficient,cfg['label_smoothing'])):
            z=logits[offset:offset+len(unit)];offset+=len(unit)
            terms.append(objective);first=unit[0]
            # Detach only compact statistics; do not retain forward graphs.
            telemetry.append(torch.stack((objective.detach(),raw.detach().sum(),raw.detach().max(),
                (z.argmax(1)==first['label']).sum(),(z.argmax(1)==0).sum())).float())
            metadata.append((f'{"online" if len(unit)==1 else "noisy"}/{first["language"]}/{first["label"]}',len(unit),first['label']))
        loss=torch.stack(terms).sum();require_finite(loss,'C3 loss');loss.backward()
    module_norms={}
    if audit:
        squares={}
        for group in optimizer.param_groups:
            terms=[p.grad.detach().float().square().sum() for p in group['params'] if p.grad is not None]
            if terms:
                value=torch.stack(terms).sum();name=group['name']
                squares[name]=squares.get(name,0)+value
        if squares:
            values=torch.stack(list(squares.values())).sqrt().cpu().tolist()
            module_norms=dict(zip(squares,values))
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['max_grad_norm'],error_if_nonfinite=True)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    rows=torch.stack(telemetry).cpu().tolist();cells=defaultdict(lambda:defaultdict(float))
    predictions={'fake':[0,0],'real':[0,0]}
    for (cell,views,label),(loss,raw,maximum,correct,fake) in zip(metadata,rows):
        for key,value in dict(objective=loss,raw_ce_sum=raw,views=views,correct=correct).items():cells[cell][key]+=value
        pred=predictions['fake' if label==0 else 'real'];pred[0]+=int(fake);pred[1]+=views
    return dict(loss=sum(r[0] for r in rows),raw_ce_sum=sum(r[1] for r in rows),max_example_ce=max(r[2] for r in rows),
        views=sum(m[1] for m in metadata),cells={k:dict(v) for k,v in cells.items()},predicted_fake_by_true_class=predictions,
        grad_norm=float(norm),risk_coefficient=coefficient,lrs={g['name']:g['lr'] for g in optimizer.param_groups},
        processing_counts=dict(processing),unique_original_sources=len({u[0]['group_id'] for u in units}),
        module_grad_norms_before_clip=module_norms)
