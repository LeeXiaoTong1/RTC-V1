"""Head warmup, then 24-layer LoRA warmup and a fresh cosine schedule."""
import math


def schedule(optimizer, cfg, epoch, step, steps):
    joint = epoch >= cfg.get('warm_epochs', 1)
    index = (epoch-cfg.get('warm_epochs', 1))*steps+step if joint else 0
    total = (cfg['epochs']-cfg.get('warm_epochs', 1))*steps
    factor = cfg.get('lr_floor', .1)+(1-cfg.get('lr_floor', .1))*.5*(1+math.cos(math.pi*index/max(1,total-1)))
    for group in optimizer.param_groups:
        lora = group.get('role') == 'lora'
        warm = min(1., (index+1)/cfg.get('lr_warmup_updates',200)) if lora else 1.
        group['lr'] = group['initial_lr']*(factor*warm if joint else 0. if lora else 1.)
