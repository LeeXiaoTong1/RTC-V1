"""Loss bookkeeping, optimization, metric accumulation and safe state serialization."""
import hashlib
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import SCHEMA


def class_weights(ids, labels):
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate IDs in Train protocol; do not silently count the label map')
    y = torch.tensor([labels[x] for x in ids], dtype=torch.long)
    if not bool(((y == 0) | (y == 1)).all()):
        raise ValueError('Expected fake=0, real=1')
    count = torch.bincount(y, minlength=2)
    if (count == 0).any():
        raise ValueError('Both classes are required')
    return count.sum().float() / (2 * count.float()), count


def pair_loss(off, on, labels, temperature=.1):
    """V2 symmetric same-source positive / opposite-authenticity negative InfoNCE."""
    if off.shape != on.shape or off.ndim != 2 or len(labels) != len(off):
        raise ValueError('Pair shape mismatch')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid contrastive temperature')
    off, on = F.normalize(off.float(), dim=1), F.normalize(on.float(), dim=1)
    negative = labels[:, None] != labels[None, :]
    if not bool(negative.any(1).all()):
        raise ValueError('Each pair batch must contain fake and real sources')
    logits = off @ on.T / temperature
    mask = negative | torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    masked = logits.masked_fill(~mask, -torch.inf)
    pos = logits.diag()
    return ((masked.logsumexp(1) - pos).mean() + (masked.logsumexp(0) - pos).mean()) / 2


def objective(logits, features, labels, n, r, s, weights, real_weight, noisy_weight, beta=.3):
    if logits.shape != (n + 2*r + 2*s, 2) or labels.shape != (len(logits),):
        raise ValueError('Logical batch layout mismatch')
    if n <= 0 or (s and not r) or not 0 <= beta <= 1:
        raise ValueError('Invalid group sizes or beta')
    if not torch.isfinite(logits).all() or not torch.isfinite(features).all():
        raise FloatingPointError('Non-finite model output')
    ce = F.cross_entropy(logits.float(), labels, reduction='none')
    w = weights.to(logits.device)[labels[:n]]
    parts = {'ordinary': (ce[:n] * w).sum() / w.sum()}
    if r:
        if not torch.equal(labels[n:n+r], labels[n+r:n+2*r]):
            raise ValueError('Real-pair labels do not match')
        parts['real_pair'] = ce[n:n+2*r].mean()
    if s:
        a = n + 2*r
        if not torch.equal(labels[a:a+s], labels[a+s:]):
            raise ValueError('Noisy-pair labels do not match')
        parts['noisy_reference'], parts['noisy_processed'] = ce[a:a+s].mean(), ce[a+s:].mean()
        other = n + 2*r + s
        coeff = {'ordinary': (1-beta)*n/other, 'real_pair': (1-beta)*2*r/other,
                 'noisy_reference': (1-beta)*s/other, 'noisy_processed': beta}
    elif r:
        coeff = {'ordinary': n/(n+2*r), 'real_pair': 2*r/(n+2*r)}
    else:
        coeff = {'ordinary': 1.}
    classification = sum(parts[k] * v for k, v in coeff.items())
    real = pair_loss(features[n:n+r], features[n+r:n+2*r], labels[n:n+r]) if r else classification*0
    a = n + 2*r
    noisy = pair_loss(features[a:a+s], features[a+s:], labels[a:a+s]) if s else classification*0
    loss = classification + real_weight*real + noisy_weight*noisy
    return loss, {'ce': float(classification.detach()), 'rtc': float(real.detach()),
                  'noisy_pair': float(noisy.detach()), 'coefficients': coeff,
                  **{'ce_'+k: float(v.detach()) for k, v in parts.items()}}


class Metrics:
    def __init__(self):
        self.cm = torch.zeros(2, 2, dtype=torch.int64)
        self.ce = torch.zeros(2, dtype=torch.float64)
        self.prob = torch.zeros(2, dtype=torch.float64)

    def update(self, logits, labels):
        z, y = logits.detach().float().cpu(), labels.detach().long().cpu()
        if not len(y):
            return
        if not torch.isfinite(z).all():
            raise FloatingPointError('Non-finite validation logits')
        p = z.softmax(1)[:, 0]
        pred = (p < .5).long()  # fake=0 including a tie at the official fixed threshold
        self.cm += torch.bincount(2*y + pred, minlength=4).reshape(2, 2)
        ce = F.cross_entropy(z, y, reduction='none')
        for c in (0, 1):
            self.ce[c] += ce[y == c].double().sum()
            self.prob[c] += p[y == c].double().sum()

    def result(self):
        cm = self.cm.double()
        count = cm.sum(1)
        if bool((count == 0).any()):
            raise ValueError('Both classes must be present in each validation condition')
        f1 = 2*cm.diag()/(count + cm.sum(0)).clamp_min(1)
        return {'macro_f1': f1.mean().item(), 'balanced_ce': (self.ce/count).mean().item(),
                'accuracy': (cm.diag().sum()/cm.sum()).item(), 'recall': (cm.diag()/count).tolist(),
                'fake_prediction_fraction': (cm[:, 0].sum()/cm.sum()).item(),
                'mean_fake_score_by_class': (self.prob/count).tolist(), 'confusion': self.cm.tolist(),
                'count': int(count.sum())}


def build_optimizer(model, encoder_lr, head_lr, decay):
    """Explicit parameter coverage; do not apply decay to biases or normalization scales."""
    groups, seen = [], set()
    for prefix, module, lr in [('encoder', model.backbone, encoder_lr), ('head', model.head, head_lr)]:
        buckets = {True: [], False: []}
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen:
                raise ValueError('Parameter is duplicated between optimizer groups')
            seen.add(id(p))
            buckets[p.ndim > 1 and not name.endswith('bias')].append(p)
        for decayed, ps in buckets.items():
            if ps:
                groups.append({'params': ps, 'lr': lr, 'base_lr': lr,
                               'weight_decay': decay if decayed else 0., 'name': prefix+('/decay' if decayed else '/no_decay')})
    if seen != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Trainable parameters omitted from optimizer')
    # AdamW separates decay from data-loss gradients. FP32 parameters/moments; BF16 compute only.
    return torch.optim.AdamW(groups, eps=1e-8)


class Schedule:
    """Step LR warmup, then plateau reductions, WITHOUT weight rollback or Adam resets."""
    def __init__(self, optimizer, warmup_steps, patience=3, factor=.5, floor=.05):
        if warmup_steps < 0 or patience < 1 or not 0 < factor < 1 or not 0 < floor <= 1:
            raise ValueError('Invalid scheduler settings')
        self.opt, self.warmup_steps, self.patience = optimizer, int(warmup_steps), patience
        self.factor, self.floor, self.scale, self.best, self.bad, self.cooldown = factor, floor, 1., math.inf, 0, 0

    def before_step(self, step):
        warm = min(1., (step+1)/max(1, self.warmup_steps))
        warm = .1 + .9*warm if self.warmup_steps else 1.
        for g in self.opt.param_groups:
            g['lr'] = g['base_lr'] * self.scale * warm

    def validate(self, value, steps):
        if not math.isfinite(value):
            raise FloatingPointError('Non-finite validation objective')
        if steps < self.warmup_steps:
            return False
        if value < self.best * (1 - 1e-3):
            self.best, self.bad = value, 0
        elif self.cooldown:
            self.cooldown -= 1
        else:
            self.bad += 1
        if self.bad >= self.patience:
            new = max(self.floor, self.scale*self.factor)
            changed = new < self.scale
            self.scale, self.bad, self.cooldown = new, 0, 1
            return changed
        return False

    def state_dict(self):
        return {k: v for k, v in vars(self).items() if k != 'opt'}

    def load_state_dict(self, state):
        for k, v in state.items():
            if k not in vars(self) or k == 'opt':
                raise ValueError('Unexpected scheduler state')
            setattr(self, k, v)


def rng_state():
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (n[0], n[1].tolist(), n[2], n[3], n[4]),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get('schema') != SCHEMA:
        raise ValueError('This entry point requires a REBUILD checkpoint, not an old XLS-R/w2v state_dict')
    return ckpt


def gradient_audit(model):
    """Check real gradients, not just requires_grad flags; no need to keep huge parameter copies."""
    records, probes = {}, {}
    modules = {'feature_projection': model.backbone.feature_projection,
               **{f'layer_{i:02d}': m for i, m in enumerate(model.backbone.encoder.layers)}, 'head': model.head}
    for key, m in modules.items():
        params = [(name, p) for name, p in m.named_parameters() if p.requires_grad and p.ndim >= 2]
        if not params:
            raise RuntimeError(f'No trainable matrix in {key}')
        name, p = params[0]
        if p.grad is None or not torch.isfinite(p.grad).all() or not bool(p.grad.abs().max() > 0):
            raise RuntimeError(f'Broken/non-finite/zero gradient in {key}.{name}')
        # Choose the strongest 32 gradient coordinates. Record real FP32 weight updates afterwards.
        idx = p.grad.detach().flatten().abs().topk(min(32, p.numel())).indices
        before = p.detach().flatten()[idx].clone()
        probes[key] = (p, idx, before)
        records[key] = {'parameter': name, 'grad_norm': float(p.grad.detach().float().norm())}
    return records, probes


def finish_audit(records, probes):
    for key, (p, idx, before) in probes.items():
        delta = p.detach().flatten()[idx] - before
        records[key]['sampled_update_norm'] = float(delta.float().norm())
        if not bool(delta.abs().max() > 0):
            raise RuntimeError(f'No actual FP32 parameter update in {key}; check LR/precision/optimizer')
    return records
