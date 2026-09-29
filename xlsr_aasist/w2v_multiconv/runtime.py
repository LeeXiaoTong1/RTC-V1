"""Shared checkpoint, metric and small-batch training utilities."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from . import SCHEMA
from .model import diversity_cka


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def storage_size(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(storage_size(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(storage_size(v) for v in value)
    return 0


def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needed = storage_size(state) + 512 * 1024**2
    if shutil.disk_usage(path.parent).free < needed:
        raise OSError(f'Insufficient checkpoint space: need {needed / 1024**3:.2f} GiB free')
    tmp = path.with_name(path.name + '.tmp')
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_checkpoint(path, schema=SCHEMA):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(state, dict) or state.get('schema') != schema:
        raise ValueError(f'Expected checkpoint schema {schema}')
    return state


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_rng(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_torch_rng(state, device):
    torch.set_rng_state(state[0].cpu())
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1].cpu(), device)


def amp_context(device, amp):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and amp == 'bf16' else nullcontext()


def loss_function(logits, blocks, labels, noisy, weights, noisy_weight=.3, cka_weight=.05):
    ce = F.cross_entropy(logits.float(), labels, reduction='none')
    ordinary = ~noisy
    if not bool(ordinary.any()):
        raise ValueError('An ordinary Train component is required')
    ordinary_ce = (ce[ordinary] * weights[labels[ordinary]]).mean()
    if bool(noisy.any()):
        # Noisy sources are class-balanced: do not also apply frequency weights.
        noisy_ce = ce[noisy].mean()
        classification = (1 - noisy_weight) * ordinary_ce + noisy_weight * noisy_ce
    else:
        noisy_ce = ce.sum() * 0
        classification = ordinary_ce
    cka = diversity_cka(blocks)
    loss = classification + cka_weight * cka
    return loss, {'ce': classification, 'ordinary_ce': ordinary_ce, 'noisy_ce': noisy_ce, 'cka': cka}


def replay_step(model, examples, optimizer, weights, device, amp='bf16',
                noisy_weight=.3, cka_weight=.05, grad_clip=1., check_replay=False):
    """Exact VJP replay: logical-batch CKA, one utterance graph at a time.

    Collect logits/features with dropout RNG, compute their logical-batch loss
    gradients, replay each utterance with identical RNG, then update once.
    """
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Replay requires no BatchNorm running-state updates')
    optimizer.zero_grad(set_to_none=True)
    states, zs, hs = [], [], []
    with torch.no_grad():
        for ex in examples:
            states.append(torch_rng(device))
            with amp_context(device, amp):
                z, h = model(ex['features'].to(device), ex['mask'].to(device))
            zs.append(z.float())
            hs.append(h.float())
    after_forward = torch_rng(device)
    z = torch.cat(zs).detach().requires_grad_(True)
    h = torch.cat(hs).detach().requires_grad_(True)
    labels = torch.tensor([e['label'] for e in examples], device=device)
    noisy = torch.tensor([e['noisy'] for e in examples], device=device, dtype=torch.bool)
    loss, stats = loss_function(z, h, labels, noisy, weights, noisy_weight, cka_weight)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Non-finite logical loss')
    gz, gh = torch.autograd.grad(loss, (z, h))
    try:
        for i, ex in enumerate(examples):
            restore_torch_rng(states[i], device)
            with amp_context(device, amp):
                rz, rh = model(ex['features'].to(device), ex['mask'].to(device))
            if check_replay and not (torch.allclose(rz.float(), zs[i], atol=2e-3, rtol=2e-3)
                                     and torch.allclose(rh.float(), hs[i], atol=2e-3, rtol=2e-3)):
                raise RuntimeError('Replay mismatch; no optimizer update was performed')
            torch.autograd.backward((rz, rh), (gz[i:i+1].to(rz.dtype), gh[i:i+1].to(rh.dtype)))
    finally:
        restore_torch_rng(after_forward, device)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return {**{k: float(v.detach()) for k, v in stats.items()}, 'loss': float(loss.detach()),
            'grad_norm': float(norm)}, z.detach()


class Metrics:
    def __init__(self):
        self.cm = torch.zeros(2, 2, dtype=torch.int64)
        self.ce = torch.zeros(2, dtype=torch.float64)

    def update(self, logits, labels):
        z, y = logits.detach().float().cpu(), torch.as_tensor(labels, dtype=torch.long).cpu()
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError('Non-finite score')
        pred = (z.softmax(1)[:, 0] < .5).long()
        self.cm += torch.bincount(y * 2 + pred, minlength=4).reshape(2, 2)
        ce = F.cross_entropy(z, y, reduction='none')
        for c in (0, 1):
            self.ce[c] += ce[y == c].double().sum()

    def result(self):
        cm = self.cm.double()
        count = cm.sum(1)
        f1 = 2 * cm.diag() / (count + cm.sum(0)).clamp_min(1)
        return {'count': int(count.sum()), 'class_counts': count.long().tolist(),
                'confusion': self.cm.tolist(), 'macro_f1': float(f1.mean()),
                'recall': [float(cm[c, c] / count[c]) if count[c] else None for c in (0, 1)],
                'balanced_ce': float((self.ce / count.clamp_min(1)).mean())}
