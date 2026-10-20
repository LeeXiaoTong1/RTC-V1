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
    # A one-epoch experiment must actually reach the declared meta strength.
    beta = cfg["meta_beta"] * min(1., (index + 1) / cfg["meta_ramp_updates"])
    return True, beta
