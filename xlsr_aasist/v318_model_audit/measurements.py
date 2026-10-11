"""Observe the real training step, and recalibrate only BatchNorm buffers."""
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import math
from unittest.mock import patch
import torch
from torch import nn


def tensor_digest(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().contiguous().cpu()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def bn_buffers(model):
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
            for key in ('running_mean', 'running_var', 'num_batches_tracked'):
                result[name + '.' + key] = getattr(module, key).detach().cpu().clone()
    return result


@contextmanager
def recalibration_mode(model):
    """Cumulative per-forward BN statistics, dropout off, no weight gradients.

    This is the usual batch-averaged estimate, not a pooled exact population
    variance. Variable window counts and upstream batch normalization matter;
    the subsequent full Dev comparison is the experiment's outcome.
    """
    modes = [(m, m.training) for m in model.modules()]
    norms = [(m, m.momentum) for m in model.modules()
             if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    if not norms:
        raise ValueError('No tracked BatchNorm modules')
    model.eval()
    try:
        for module, _ in norms:
            module.reset_running_stats()
            module.momentum = None
            module.train()
        with torch.no_grad():
            yield
    finally:
        for module, momentum in norms:
            module.momentum = momentum
        # Restore flags directly: Detector.train() deliberately freezes joint BN.
        for module, training in modes:
            module.training = training


def check_bn_only(before, after, allowed):
    if before.keys() != after.keys():
        raise RuntimeError('Partial-state inventory changed')
    changed = [n for n in before if not torch.equal(before[n], after[n])]
    if set(changed) - set(allowed):
        raise RuntimeError('BN diagnostic changed weights/non-BN buffers: ' + repr(set(changed)-set(allowed)))
    details = []
    for name in allowed:
        a, b = before[name].double(), after[name].double()
        if not torch.isfinite(b).all() or (name.endswith('running_var') and (b < 0).any()):
            raise RuntimeError('Invalid recalibrated BN buffer: ' + name)
        if name.endswith('num_batches_tracked') and int(b) < 1:
            raise RuntimeError('A BatchNorm module received no samples: ' + name)
        details.append(dict(name=name, changed=name in changed, before_norm=float(a.norm()),
                            after_norm=float(b.norm()), delta_norm=float((b-a).norm())))
    return dict(only_bn_buffers_changed=True, changed_count=len(changed), buffers=details)


def snapshot(model, gradients=False):
    return {n: (None if p.grad is None else p.grad.detach().cpu().clone()) if gradients
            else p.detach().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad}


def norm(values):
    return math.sqrt(sum(float(x.double().square().sum()) for x in values if x is not None))


def adapter_delta(a0, b0, a1, b1, scale):
    """Frobenius norm of scale*(B1 A1-B0 A0), without a dense 2048x2048 product."""
    b = torch.cat((b1.double(), b0.double()), dim=1)
    a = torch.cat((a1.double(), -a0.double()), dim=0)
    squared = ((b.T @ b) * (a @ a.T).T).sum()
    return float(squared.clamp_min(0).sqrt()) * abs(scale)


class StepObserver:
    """Read-only hooks around the production clip and AdamW step.

    No loss/sampler/optimizer reimplementation. The clip wrapper exists only
    inside this isolated synchronous diagnostic process, and always restores.
    """
    def __init__(self, model, optimizer):
        self.model, self.optimizer = model, optimizer
        self.names = {id(p): n for n, p in model.named_parameters()}
        self.groups = defaultdict(list)
        for group in optimizer.param_groups:
            self.groups[group['name']].extend(self.names[id(p)] for p in group['params'])
        self.pre = self.post = self.before = self.after = None
        self.calls = 0

    @contextmanager
    def active(self):
        original = torch.nn.utils.clip_grad_norm_
        def clip(*args, **kwargs):
            self.calls += 1
            self.pre = snapshot(self.model, True)
            value = original(*args, **kwargs)
            self.post = snapshot(self.model, True)
            return value
        def before_step(*_):
            self.before = snapshot(self.model)
        def after_step(*_):
            self.after = snapshot(self.model)
        pre = self.optimizer.register_step_pre_hook(before_step)
        post = self.optimizer.register_step_post_hook(after_step)
        try:
            with patch.object(torch.nn.utils, 'clip_grad_norm_', clip):
                yield self
            if self.calls != 1 or self.after is None:
                raise RuntimeError('Expected exactly one clip and optimizer update')
        finally:
            pre.remove(); post.remove()

    def result(self, include_effective=True):
        gradients = {}; changes = {}; per_parameter = []
        for name, before in self.before.items():
            pre, post = self.pre[name], self.post[name]
            delta = self.after[name].double() - before.double()
            base_norm = norm([before]); delta_norm = norm([delta])
            if not all(torch.isfinite(x).all() for x in (before, self.after[name], delta)
                       if x is not None):
                raise FloatingPointError('Nonfinite parameter update: ' + name)
            item = dict(name=name, numel=before.numel(), grad_present=pre is not None,
                        grad_pre=norm([pre]), grad_post=norm([post]), weight_norm=base_norm,
                        update_norm=delta_norm, update_rms=delta_norm/math.sqrt(before.numel()),
                        relative_update=delta_norm/base_norm if base_norm else None,
                        changed_elements=int(torch.count_nonzero(delta)))
            per_parameter.append(item)
        lookup = {p['name']: p for p in per_parameter}
        for group, names in self.groups.items():
            gradients[group] = dict(pre=norm([self.pre[n] for n in names]),
                                    post=norm([self.post[n] for n in names]),
                                    missing=[n for n in names if self.pre[n] is None])
            size = sum(lookup[n]['numel'] for n in names)
            base = math.sqrt(sum(lookup[n]['weight_norm']**2 for n in names))
            update = math.sqrt(sum(lookup[n]['update_norm']**2 for n in names))
            changes[group] = dict(norm=update, rms=update/math.sqrt(size),
                                  relative=update/base if base else None,
                                  changed_elements=sum(lookup[n]['changed_elements'] for n in names))
        adapters = []
        if include_effective:
            for name, module in self.model.named_modules():
                a, b = name+'.lora_a', name+'.lora_b'
                if a not in self.before or b not in self.before:
                    continue
                scale = module.scale
                a0, b0, a1, b1 = self.before[a], self.before[b], self.after[a], self.after[b]
                effective_norm = adapter_delta(a0, b0, a1, b1, scale)
                adapters.append(dict(name=name, effective_weight_update_norm=effective_norm,
                                     a_update_norm=lookup[a]['update_norm'], b_update_norm=lookup[b]['update_norm']))
        return dict(gradients=gradients, updates=changes, parameters=per_parameter, adapters=adapters)
