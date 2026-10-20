"""Continue parent-best rates without restarting head or adapter warmup."""
import math


def schedule(optimizer, cfg, epoch, step, steps):
    if not 0 <= epoch < cfg["epochs"] or not 0 <= step < steps:
        raise ValueError("Invalid continuation cursor")
    index = epoch * steps + step
    fraction = index / max(1, cfg["epochs"] * steps - 1)
    multiplier = cfg["lr_floor"] + (1 - cfg["lr_floor"]) * .5 * (1 + math.cos(math.pi * fraction))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * multiplier
    optimizer._v3189_joint_update = index+1
    optimizer._v3189_update = index+1
    optimizer._v3189_joint = True
    return True, 1.
