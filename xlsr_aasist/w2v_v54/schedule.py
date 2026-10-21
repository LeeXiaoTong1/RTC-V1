"""One committed cursor controls warmup, ramp and all continuation rates."""
import math


def warmup_steps(cfg, total):
    if cfg['repair_arm'] == 'baseline':
        return 0
    return min(cfg['repair_warmup_max'], math.ceil(cfg['repair_warmup_fraction']*total))


def schedule(optimizer, cfg, epoch, step, steps):
    if not 0 <= epoch < cfg['epochs'] or not 0 <= step < steps:
        raise ValueError('Invalid continuation cursor')
    index, total = epoch*steps+step, cfg['epochs']*steps
    warm = warmup_steps(cfg, total)
    joint = index >= warm
    for group in optimizer.param_groups:
        new = group['role'] == 'repair'
        progress = index/max(1, total-1) if new else max(0, index-warm)/max(1, total-warm-1)
        factor = cfg['lr_floor']+(1-cfg['lr_floor'])*.5*(1+math.cos(math.pi*progress))
        group['lr'] = group['initial_lr']*factor if new or joint else 0.
    optimizer._v3189_joint_update = index+1
    optimizer._v3189_update = index+1
    optimizer._v3189_joint = joint
    optimizer._v54_warm_steps = warm
    optimizer._v54_total_updates = total
    return True, 1.  # Old driver return value describes the CE recipe, not phase.


def auxiliary_coefficients(cfg):
    if cfg['repair_arm'] != 'paired':
        return 0., 0.
    index = cfg.get('current_update', 1)
    warm = cfg.get('_repair_warm_steps', cfg['repair_warmup_max'])
    f = cfg['repair_feature_weight']*min(1., index/max(1, min(cfg['repair_feature_ramp'], warm or 1)))
    ramp = min(cfg['repair_kd_ramp'],max(1,cfg.get('_repair_total_updates',warm+cfg['repair_kd_ramp'])-warm))
    d = cfg['repair_kd_weight']*min(1., max(0, index-warm)/max(1, ramp))
    return f, d
