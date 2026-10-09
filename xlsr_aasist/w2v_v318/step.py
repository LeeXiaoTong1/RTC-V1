"""Streaming gradients keep logical budgets intact across physical microbatches."""
from collections import defaultdict,Counter
import math
import torch
from .data import microbatches
from .sampling import validate_units
from .objective import unit_loss
from .common import require_finite


def schedule(optimizer,cfg,epoch,step,steps):
    warm=epoch<cfg['warm_epochs']
    if warm:
        progress=(epoch*steps+step+1)/max(1,cfg['warm_epochs']*steps)
        multiplier=min(1.,progress/.1)
    else:
        index=(epoch-cfg['warm_epochs'])*steps+step+1
        total=(cfg['epochs']-cfg['warm_epochs'])*steps
        warmup=max(1,int(.05*total))
        multiplier=index/warmup if index<=warmup else .1+.9*.5*(1+math.cos(math.pi*(index-warmup)/max(1,total-warmup)))
    for group in optimizer.param_groups:
        if warm:rate=cfg['warm_lr'] if group['name']=='aasist' else 0.
        else:rate=group['initial_lr']
        group['lr']=rate*multiplier
    risk=0. if warm else cfg['risk_coefficient']*min(1.,((epoch-cfg['warm_epochs'])*steps+step+1)/steps)
    return risk


def train_step(model,optimizer,units,cfg,coefficient):
    validate_units(units,cfg['stream_sources']);model.train();optimizer.zero_grad(set_to_none=True)
    values=defaultdict(float);cells=defaultdict(lambda:defaultdict(float));predictions=defaultdict(lambda:[0,0]);processing=Counter()
    for unit in units:
        for row in unit:
            r=row.get('augmentation',{}).get('recipe')
            if r:processing[f'{row["language"]}/{row["label"]}/{r["family"]}/{r["codec"]}/{r["noise_type"]}']+=1
    for micro in microbatches(units,cfg['microbatch'],cfg['frame_budget']):
        rows=[r for u in micro for r in u]
        logits=model([r['wave'] for r in rows]);offset=0;loss=logits.sum()*0
        for unit in micro:
            z=logits[offset:offset+len(unit)];offset+=len(unit)
            objective,raw,smoothed=unit_loss(z,unit,cfg['stream_sources'],coefficient,cfg['label_smoothing'])
            loss=loss+objective
            first=unit[0];cell=f'{"online" if len(unit)==1 else "noisy"}/{first["language"]}/{first["label"]}'
            cells[cell]['objective']+=float(objective.detach());cells[cell]['raw_ce_sum']+=float(raw.detach().sum())
            cells[cell]['views']+=len(unit);cells[cell]['correct']+=int((z.argmax(1)==first['label']).sum().detach())
            predictions['fake' if first['label']==0 else 'real'][0]+=int((z.argmax(1)==0).sum().detach())
            predictions['fake' if first['label']==0 else 'real'][1]+=len(unit)
            values['raw_ce_sum']+=float(raw.detach().sum());values['views']+=len(unit)
            values['max_example_ce']=max(values['max_example_ce'],float(raw.detach().max()))
        require_finite(loss,'loss');loss.backward();values['loss']+=float(loss.detach())
    params=[p for p in model.parameters() if p.requires_grad]
    norm=torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'],error_if_nonfinite=True)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    return dict(values,cells={k:dict(v) for k,v in cells.items()},predicted_fake_by_true_class=dict(predictions),
                grad_norm=float(norm),risk_coefficient=coefficient,lrs={g['name']:g['lr'] for g in optimizer.param_groups},
                processing_counts=dict(processing),unique_original_sources=len({u[0]['group_id'] for u in units}))
