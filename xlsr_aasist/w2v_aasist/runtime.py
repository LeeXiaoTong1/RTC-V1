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
from .model import microbatches


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


def amp_context(device, amp):
    device = torch.device(device)
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and amp == 'bf16' else nullcontext()


def supervised_step(model, examples, optimizer, weights, device, amp='bf16',
                    noisy_weight=.3, grad_clip=1., microbatch=4, frame_budget=1600):
    """One forward/backward per sample; global CE denominators survive microbatching."""
    optimizer.zero_grad(set_to_none=True)
    labels = torch.tensor([e['label'] for e in examples], device=device)
    noisy = torch.tensor([e['noisy'] for e in examples], device=device, dtype=torch.bool)
    n, s = int((~noisy).sum()), int(noisy.sum())
    if not n:
        raise ValueError('An ordinary classification component is required')
    coefficients = weights[labels] * ((1 - noisy_weight) if s else 1.) / n
    if s:
        coefficients = torch.where(noisy, torch.full_like(coefficients, noisy_weight / s), coefficients)
    scores = torch.empty(len(examples), 2, device=device)
    stats = {'loss': 0., 'ordinary_ce': 0., 'noisy_ce': 0.}
    for indices, f, m in microbatches(examples, microbatch, frame_budget):
        with amp_context(device, amp):
            z, _ = model(f.to(device), m.to(device))
        ce = F.cross_entropy(z.float(), labels[indices], reduction='none')
        loss = (ce * coefficients[indices]).sum()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite loss; optimizer has not advanced')
        loss.backward()
        scores[indices] = z.detach().float()
        stats['loss'] += float(loss.detach())
        local_noisy = noisy[indices]
        stats['ordinary_ce'] += float((ce.detach()[~local_noisy] * weights[labels[indices][~local_noisy]]).sum()) / n
        if s:
            stats['noisy_ce'] += float(ce.detach()[local_noisy].sum()) / s
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return {**stats, 'grad_norm': float(norm)}, scores


@torch.inference_mode()
def predict(model, examples, device, amp='bf16', microbatch=4, frame_budget=1600):
    scores = torch.empty(len(examples), 2)
    for indices, f, m in microbatches(examples, microbatch, frame_budget):
        with amp_context(device, amp):
            z, _ = model(f.to(device), m.to(device))
        scores[indices] = z.detach().float().cpu()
    return scores


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
