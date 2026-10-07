"""Convex Train-only output adaptation, using PyTorch rather than SciPy BLAS."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .data import head_weights


def predict(features, head, device, chunk=8192):
    output = np.empty((len(features), 2), dtype=np.float32)
    weight, bias = head['weight'].to(device), head['bias'].to(device)
    with torch.inference_mode():
        for start in range(0, len(features), chunk):
            x = torch.from_numpy(np.array(features[start:start+chunk], copy=True)).to(device)
            output[start:start+len(x)] = F.linear(x, weight, bias).cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError('Nonfinite adapted-head scores')
    return output


def fit(bundle, original, cfg):
    rows = bundle['rows']
    if any(r.get('split') != 'train' for r in rows):
        raise ValueError('Head gradients require official Train only')
    weights = head_weights(rows)
    device = cfg['device']
    # Only the final layer is optimized. Double precision and deterministic
    # full-population gradients make this small convex solve numerically stable.
    x = torch.tensor(np.asarray(bundle['x']), dtype=torch.float64, device=device)
    labels = torch.tensor([r['label'] for r in rows], device=device)
    mass = torch.tensor(weights, dtype=torch.float64, device=device)
    w0, b0 = original['weight'].to(device, torch.float64), original['bias'].to(device, torch.float64)
    w, b = w0.clone().requires_grad_(), b0.clone().requires_grad_()
    optimizer = torch.optim.LBFGS([w, b], lr=1., max_iter=1, max_eval=12, history_size=20,
                                 line_search_fn='strong_wolfe', tolerance_grad=cfg['head_grad_tolerance'],
                                 tolerance_change=1e-12)
    calls, latest = 0, {}

    def closure():
        nonlocal calls, latest
        optimizer.zero_grad(set_to_none=True)
        ce_value = 0.
        for start in range(0, len(x), cfg['head_chunk']):
            stop = start + cfg['head_chunk']
            ce = (F.cross_entropy(F.linear(x[start:stop], w, b), labels[start:stop], reduction='none') * mass[start:stop]).sum()
            ce.backward()
            ce_value += float(ce.detach())
        regularizer = .5 * cfg['head_anchor'] * ((w-w0).square().sum() + (b-b0).square().sum())
        regularizer.backward()
        value = ce_value + float(regularizer.detach())
        if not math.isfinite(value) or not all(bool(torch.isfinite(p.grad).all()) for p in (w, b)):
            raise FloatingPointError('Nonfinite frozen-head objective/gradient')
        calls += 1
        latest = dict(objective=value, ce=ce_value, parameter_penalty=float(regularizer.detach()),
                      gradient_max=max(float(p.grad.abs().max()) for p in (w, b)))
        return w.new_tensor(value)

    trace = []
    closure()
    initial = dict(latest)
    for step in range(cfg['head_steps']):
        optimizer.step(closure)
        closure()  # Diagnostics and convergence refer to the actual accepted parameters.
        trace.append(dict(step=step+1, **latest))
        if step == 0 or (step+1) % 10 == 0:
            print(f'V3.14 head solve {step+1}/{cfg["head_steps"]} CE={latest["ce"]:.7f} '
                  f'anchor={latest["parameter_penalty"]:.7f} gradient={latest["gradient_max"]:.3g}', flush=True)
        if latest['gradient_max'] <= cfg['head_grad_tolerance']:
            break
    if latest['objective'] > initial['objective'] + 1e-9:
        raise FloatingPointError('Frozen-head optimizer increased its starting objective')
    state = dict(weight=w.detach().cpu().float().clone(), bias=b.detach().cpu().float().clone())
    return state, dict(initial=initial, final=latest, iterations=len(trace), objective_evaluations=calls,
        trace=trace, train_rows=len(rows), optimized_parameters=w.numel()+b.numel(),
        parameter_displacement_l2=float(((w-w0).square().sum()+(b-b0).square().sum()).sqrt()),
        train_only=True, noisy_ce_mass=float(sum(v for r, v in zip(rows, weights) if r['condition'].startswith('noisy_'))),
        optimizer='torch LBFGS strong_wolfe; FP64; anchored parameters, no teacher/logit loss')
