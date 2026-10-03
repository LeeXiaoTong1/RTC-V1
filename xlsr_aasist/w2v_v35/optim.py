"""Explicit head/joint schedules and an EMA that never recopies frozen blocks."""
from contextlib import contextmanager
import math
import torch
from w2v_v34.optim import optimizer_for as joint_optimizer, apply_learning_rates


def optimizer_for(model, cfg, phase):
    if phase == 'joint':
        return joint_optimizer(model, {**cfg, 'trainable_layers': cfg.get('trainable_layers', 24),
            'encoder_lr': cfg.get('encoder_lr', 1e-6), 'joint_head_lr': cfg.get('joint_head_lr', 1e-5),
            'layer_decay': cfg.get('layer_decay', .9)})
    if phase != 'head':
        raise ValueError('Unknown training phase')
    model.configure_trainable_layers(0)
    lr = cfg.get('head_lr', 1e-4)
    decay = cfg.get('weight_decay', 1e-4)
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(decay) or decay < 0:
        raise ValueError('Invalid head optimizer rates')
    groups = []
    for use_decay in (True, False):
        params = [p for name, p in model.head.named_parameters()
                  if (p.ndim > 1 and not name.endswith('bias')) == use_decay]
        if params:
            groups.append(dict(name='head.' + ('decay' if use_decay else 'no_decay'), params=params,
                               lr=lr, base_lr=lr, weight_decay=decay if use_decay else 0.))
    return torch.optim.AdamW(groups, eps=1e-8)


def schedule_scale(step, total_steps, warmup_fraction=.05, min_ratio=.1):
    if (isinstance(step, bool) or not isinstance(step, int) or step < 0
            or not isinstance(total_steps, int) or total_steps < 1
            or not 0 <= warmup_fraction < 1 or not 0 <= min_ratio <= 1):
        raise ValueError('Invalid phase schedule')
    warmup = min(total_steps, max(1, math.ceil(total_steps * warmup_fraction))) if warmup_fraction else 0
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1., max(0., (step - warmup) / max(1, total_steps - warmup - 1)))
    return min_ratio + (1 - min_ratio) * .5 * (1 + math.cos(math.pi * progress))


class EMA:
    """Shadow only trainable FP32 parameters; add newly unfrozen blocks once.

    Shadows stay on the model device by default. A head-only epoch therefore
    copies no frozen encoder weights at every step. Validation temporarily swaps
    trainable tensors while raw weights are held on CPU, then restores them.
    """
    def __init__(self, model, decay=.999, device=None):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0,1)')
        self.decay, self.device, self.steps = float(decay), device, 0
        self.shadow = {}
        self.add_trainable(model)

    @torch.no_grad()
    def add_trainable(self, model):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name not in self.shadow:
                self.shadow[name] = parameter.detach().to(device=self.device or parameter.device,
                                                         dtype=torch.float32, copy=True)

    @torch.no_grad()
    def update(self, model):
        self.add_trainable(model)
        self.steps += 1
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        by_device = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                shadow = self.shadow[name]
                value = parameter.detach().to(device=shadow.device, dtype=shadow.dtype)
                old, new = by_device.setdefault(str(shadow.device), ([], []))
                old.append(shadow); new.append(value)
        for old, new in by_device.values():
            torch._foreach_lerp_(old, new, 1 - decay)

    def state_dict(self):
        return dict(decay=self.decay, steps=self.steps, shadow=self.shadow)

    def load_state_dict(self, state, model):
        required = {name for name, p in model.named_parameters() if p.requires_grad}
        if state.get('decay') != self.decay or set(state.get('shadow', {})) != required:
            raise ValueError('EMA configuration or trainable layout differs on resume')
        params = dict(model.named_parameters())
        self.steps = state['steps']
        if not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError('Invalid EMA update count')
        for name, value in state['shadow'].items():
            if value.shape != params[name].shape or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid EMA shadow tensor')
            self.shadow[name] = value.to(device=self.device or params[name].device, dtype=torch.float32, copy=True)

    @contextmanager
    def average_parameters(self, model):
        params = dict(model.named_parameters())
        raw = {}
        try:
            with torch.no_grad():
                for name, value in self.shadow.items():
                    raw[name] = params[name].detach().to('cpu', copy=True)
                    params[name].copy_(value)
            yield model
        finally:
            with torch.no_grad():
                for name, value in raw.items():
                    params[name].copy_(value)
