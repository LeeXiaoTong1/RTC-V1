"""Fixed outer schedule; independent calibrated inner step never decays."""
import math


def schedule(optimizer,cfg,epoch,step,steps):
    if not 0<=epoch<cfg['epochs'] or not 0<=step<steps:raise ValueError('Invalid schedule cursor')
    joint=epoch>=cfg.get('warm_epochs',1)
    index=(epoch-cfg.get('warm_epochs',1))*steps+step if joint else 0
    total=(cfg['epochs']-cfg.get('warm_epochs',1))*steps
    fraction=index/max(1,total-1)
    factor=cfg.get('lr_floor',.1)+(1-cfg.get('lr_floor',.1))*.5*(1+math.cos(math.pi*fraction))
    for group in optimizer.param_groups:
        lora=group.get('role')=='lora'
        if not joint:group['lr']=0. if lora else group['initial_lr']
        else:group['lr']=group['initial_lr']*factor*(min(1.,(index+1)/cfg.get('lr_warmup_updates',200)) if lora else 1.)
    beta=cfg.get('meta_beta',.5)*min(1.,index/max(1,cfg.get('meta_beta_ramp_epochs',2)*steps-1)) if joint else 0.
    return joint,beta
