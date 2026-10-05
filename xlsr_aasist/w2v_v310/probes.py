"""Retrain independent linear/nonlinear language probes on disjoint real sources."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import CONDITIONS


def run_probes(features, rows, split, cfg):
    if len(features) != len(rows) or any(r['label'] != 1 for r in rows):
        raise ValueError('Probes require aligned genuine Train representations')
    fit_groups, test_groups = set(split['fit_groups']), set(split['test_groups'])
    if fit_groups & test_groups:
        raise ValueError('Probe fit/test source overlap')
    device = torch.device(cfg['device'])
    device_ids = [device.index or 0] if device.type == 'cuda' else []
    result = dict(conditions={}, fit_groups=len(fit_groups), test_groups=len(test_groups),
                  representation_variance=float(np.asarray(features).var(0).mean()),
                  original_encoder_previously_saw_train=True)
    # Probe initialization/training must not alter subsequent detector RNG.
    with torch.random.fork_rng(devices=device_ids):
        for ci, condition in enumerate(CONDITIONS):
            fit = [i for i,r in enumerate(rows) if r['condition'] == condition and r['group_id'] in fit_groups]
            test = [i for i,r in enumerate(rows) if r['condition'] == condition and r['group_id'] in test_groups]
            if not fit or not test:
                continue
            x = torch.tensor(np.asarray(features)[fit], dtype=torch.float32, device=device)
            v = torch.tensor(np.asarray(features)[test], dtype=torch.float32, device=device)
            y = torch.tensor([int(rows[i]['language']=='zh') for i in fit], device=device)
            target = torch.tensor([int(rows[i]['language']=='zh') for i in test], device=device)
            if y.unique().numel() != 2 or target.unique().numel() != 2:
                raise ValueError('Both languages required in each conditional probe partition')
            mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-4)
            x, v = (x-mean)/scale, (v-mean)/scale
            weights = torch.stack([(y==i).sum() for i in (0,1)]).float().reciprocal()
            weights = weights / weights.sum() * 2
            output = {}
            for kind in ('linear', 'nonlinear'):
                torch.manual_seed(cfg['seed'] + ci)
                probe = (nn.Linear(x.shape[1],2) if kind == 'linear' else nn.Sequential(
                    nn.Linear(x.shape[1], cfg['probe_hidden']), nn.GELU(), nn.Linear(cfg['probe_hidden'],2))).to(device)
                optimizer = torch.optim.AdamW(probe.parameters(), lr=.01 if kind=='linear' else .003, weight_decay=.01)
                for _ in range(cfg['probe_steps']):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(probe(x), y, weight=weights)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite independent probe fit')
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    predicted = probe(v).argmax(1)
                    train_predicted = probe(x).argmax(1)
                    accuracy = float(torch.stack([(predicted[target==i]==i).float().mean() for i in (0,1)]).mean())
                    train_accuracy = float(torch.stack([(train_predicted[y==i]==i).float().mean() for i in (0,1)]).mean())
                output[kind] = dict(balanced_accuracy=accuracy, fit_balanced_accuracy=train_accuracy,
                                    test_rows=len(test), fit_rows=len(fit))
            result['conditions'][condition] = output
    if not result['conditions']:
        raise ValueError('No independent language probe conditions')
    # Best of two fresh readers: fooling only the adversary or one weak probe is insufficient.
    result['readability'] = float(np.mean([max(.5, *(p['balanced_accuracy'] for p in values.values()))
                                          for values in result['conditions'].values()]))
    return result


def compare(baseline, current, cfg):
    drop = baseline['readability'] - current['readability']
    variance_ratio = current['representation_variance'] / max(1e-12, baseline['representation_variance'])
    return dict(readability_drop=drop, variance_ratio=variance_ratio,
                evidence=bool(baseline['readability'] >= .55 and drop >= cfg['probe_min_drop'] and variance_ratio >= .1),
                interpretation='held-out probe evidence only; not causal attribution or proof all language information is erased')
