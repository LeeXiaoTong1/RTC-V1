"""Keep C3 and its one-pass updates; extend the schedule and view telemetry."""
from collections import Counter
import math
from w2v_v3183r2.step import train_step as c3_step


def schedule(optimizer,cfg,epoch,step,steps):
    if not 0<=epoch<cfg['epochs'] or not 0<=step<steps:raise ValueError('Invalid schedule cursor')
    if epoch<cfg['warm_epochs']:
        index=epoch*steps+step+1;ramp=min(cfg['lr_warmup_updates'],cfg['warm_epochs']*steps)
        multiplier=min(1.,index/max(1,ramp))
        rates=dict(aasist=cfg['warm_lr'],fusion=cfg['warm_fusion_lr'],layer_mix=cfg['warm_layer_mix_lr'],
                   evidence=cfg['warm_evidence_lr'],lora=0.,pooling=cfg['pool_lr'],classifier=cfg['classifier_lr'])
        for group in optimizer.param_groups:group['lr']=rates[group['name']]*multiplier
    else:
        index=(epoch-cfg['warm_epochs'])*steps+step+1;total=(cfg['epochs']-cfg['warm_epochs'])*steps
        ramp=min(cfg['lr_warmup_updates'],max(1,total//10));fraction=(index-1)/max(1,total-1)
        multiplier=cfg['lr_floor']+(1-cfg['lr_floor'])*.5*(1+math.cos(math.pi*fraction))
        for group in optimizer.param_groups:
            factor=min(1.,index/ramp) if group['name']=='lora' else 1.
            group['lr']=group['initial_lr']*multiplier*factor
    return cfg['risk_coefficient']


def train_step(model,optimizer,units,cfg,coefficient,audit=False):
    result=c3_step(model,optimizer,units,cfg,coefficient,audit)
    counts=Counter()
    for unit in units:
        row=unit[0];meta=row.get('base_augmentation')
        if meta:
            kind='online' if len(unit)==1 else 'offline'
            counts[f'{kind}/{row["language"]}/{row["label"]}/{meta["recipe"]["family"]}']+=1
    result['base_source_view_counts']=dict(counts)
    return result
