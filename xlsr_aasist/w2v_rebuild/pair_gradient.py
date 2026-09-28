"""Exact, non-updating loss-gradient measurements for the existing Stage3 loss.

Only scalar diagnostics leave this module. Gradients are raw Euclidean
gradients, not Adam updates. CPU copies avoid retaining several large gradient
vectors on the GPU. Error probes keep the ORIGINAL CE denominator/budget.
"""
from collections import defaultdict
import math

import torch
from torch.nn import functional as F

from .core import objective, pair_loss


def loss_terms(logits, features, labels, layout, weights, real_cost=1., language_weights=None,
               error_masks=None):
    n, r, s = layout
    ce, stats = objective(logits, features, labels, n, r, s, weights, 0., 0.,
                          real_ce_weight=real_cost, language_weights=language_weights,
                          tensor_stats=True)
    a = n + 2*r
    result = {'ce': ce,
              'rtc': pair_loss(features[n:n+r], features[n+r:a], labels[n:n+r]),
              'noisy': pair_loss(features[a:a+s], features[a+s:], labels[a:a+s])}
    element = F.cross_entropy(logits.float(), labels, reduction='none')
    if language_weights is not None:
        element = element * language_weights
    cost = torch.where(labels == 1, real_cost, 1.)
    # Actual per-view coefficients in the classification objective, not a new
    # mean over the error subset (which would amplify one mistake arbitrarily).
    noisy = element[a+s:] * cost[a+s:] / cost[a+s:].sum() * stats['coefficients']['noisy_processed']
    rtc = element[n+r:a] * cost[n+r:a] / cost[n:a].sum() * stats['coefficients']['real_pair']
    result['noisy_processed_ce'] = noisy.sum()
    for name, values in (('noisy_error_ce', noisy), ('rtc_error_ce', rtc)):
        mask = (error_masks or {}).get(name)
        if mask is not None:
            mask = torch.as_tensor(mask, device=labels.device, dtype=torch.bool)
            if mask.shape != values.shape:
                raise ValueError('Fixed error mask shape differs from pair count')
            if bool(mask.any()):
                result[name] = values[mask].sum()
    return result


def parameter_groups(name):
    groups = ['all']
    if name.startswith('backbone.'):
        groups.extend(['shared', 'encoder'])
        if name.startswith('backbone.encoder.layers.'):
            groups.append('encoder_layer_' + name.split('.')[3])
    elif name.startswith('head.classifier.'):
        groups.append('classifier')
    elif name.startswith('head.'):
        groups.extend(['shared', 'head_shared'])
    else:
        raise ValueError('Unrecognized detector parameter: '+name)
    return groups


def inner_products(left, right, named):
    """Float64 dot accumulation; missing gradients mean zero, not missing data."""
    sums = defaultdict(float)
    for (name, _), a, b in zip(named, left, right):
        groups = parameter_groups(name)
        value = 0.
        if a is not None and b is not None:
            # A tensor-sized FP64 temporary, never a whole-model concatenation.
            value = torch.dot(a.reshape(-1).double(), b.reshape(-1).double()).item()
        for group in groups:
            sums[group] += value
    return dict(sums)


def gradient_gram(terms, named):
    """Backprop each term through ONE shared forward graph; no .grad or optimizer.

    Retain CPU vectors only for the three principal terms. Additional error
    probes are compared to those three and discarded immediately.
    """
    named = [(n, p) for n, p in named if p.requires_grad]
    if not named:
        raise ValueError('No trainable parameters to measure')
    if any(p.grad is not None for _, p in named):
        raise ValueError('Audit requires empty parameter .grad fields')
    gram, saved, values = {}, {}, {}
    items = list(terms.items())
    for i, (name, loss) in enumerate(items):
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite audit loss: '+name)
        values[name] = float(loss.detach())
        grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True,
                                    retain_graph=i+1 < len(items))
        cpu = []
        for grad in grads:
            if grad is not None and not bool(torch.isfinite(grad).all()):
                raise FloatingPointError('Non-finite audit gradient: '+name)
            # clone on CPU too: a grad may share an upstream buffer.
            cpu.append(None if grad is None else grad.detach().float().to('cpu', copy=True))
        del grads
        gram[name+'|'+name] = inner_products(cpu, cpu, named)
        for previous, vector in saved.items():
            gram[previous+'|'+name] = inner_products(vector, cpu, named)
        if name in ('ce', 'rtc', 'noisy'):
            saved[name] = cpu
    if any(p.grad is not None for _, p in named):
        raise RuntimeError('Audit unexpectedly populated parameter .grad')
    return {'losses': values, 'gram': gram}


def dot(record, left, right, group):
    gram = record['gram']
    values = gram.get(left+'|'+right, gram.get(right+'|'+left))
    if values is None:
        raise KeyError('Unmeasured gradient cross product')
    return values[group]


def comparison_rows(record):
    result = []
    groups = record['gram']['ce|ce']
    for target in record['losses']:
        if target in ('rtc', 'noisy'):
            continue
        for auxiliary in ('rtc', 'noisy'):
            for group in groups:
                target_sq = max(0., dot(record, target, target, group))
                aux_sq = max(0., dot(record, auxiliary, auxiliary, group))
                product = dot(record, target, auxiliary, group)
                denom = math.sqrt(target_sq) * math.sqrt(aux_sq)
                cosine = max(-1., min(1., product/denom)) if denom else None
                for weight in ((.1,) if auxiliary == 'rtc' else (.05, .1)):
                    row = dict(target=target, auxiliary=auxiliary, group=group, weight=weight,
                               target_norm=math.sqrt(target_sq), raw_aux_norm=math.sqrt(aux_sq),
                               weighted_aux_norm=weight*math.sqrt(aux_sq), cosine=cosine,
                               weighted_dot=weight*product,
                               norm_ratio=weight*math.sqrt(aux_sq/target_sq) if target_sq else None,
                               opposition_fraction=-weight*product/target_sq if target_sq else None)
                    # Positive opposition means it cancels some of this target's
                    # self-descent in a hypothetical common-LR gradient step.
                    if auxiliary == 'noisy':
                        before = dot(record, target, 'ce', group) + .1*dot(record, target, 'rtc', group)
                        row.update(descent_without_noisy=before,
                                   descent_with_noisy=before+weight*product)
                    result.append(row)
    return result


def pair_diagnostics(reference, processed, labels):
    """Inspect saturation without mistaking rounded zero loss for zero gradient."""
    reference = reference.detach().float().cpu().requires_grad_(True)
    processed = processed.detach().float().cpu().requires_grad_(True)
    labels = labels.detach().long().cpu()
    loss = pair_loss(reference, processed, labels)
    grad = torch.autograd.grad(loss, (reference, processed))
    norm = math.sqrt(sum(float(g.double().square().sum()) for g in grad))
    with torch.no_grad():
        x, y = F.normalize(reference.double(), dim=1), F.normalize(processed.double(), dim=1)
        similarity = x @ y.T
        negative = labels[:, None] != labels[None, :]
        diagonal = similarity.diag()
        masked = (similarity/.1).masked_fill(~(negative | torch.eye(len(x), dtype=torch.bool)), -torch.inf)
        pos = diagonal/.1
        precise = ((masked.logsumexp(1)-pos).mean() + (masked.logsumexp(0)-pos).mean())/2
        hardest = similarity.masked_fill(~negative, -torch.inf)
    return dict(loss_fp32=float(loss.detach()), loss_fp64_reference=float(precise),
                feature_gradient_norm=norm, positive_cosine=diagonal.tolist(),
                hardest_negative_cosine_off_to_on=hardest.max(1).values.tolist(),
                hardest_negative_cosine_on_to_off=hardest.max(0).values.tolist(),
                opposite_class_negatives=negative.sum(1).tolist())
