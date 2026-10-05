"""Real-only centered distillation, measured against the fitted constant mean."""
import math
import time

import numpy as np
import torch

from live_progress import Phase
from w2v_v37.fit import real_source_coefficients
from .common import cpu_state
from .model import CenteredStudent


def weighted_normalization(x, mass):
    mean = (x * mass[:, None]).sum(0)
    variance = ((x - mean).square() * mass[:, None]).sum(0)
    return mean, variance.sqrt().clamp_min(1e-3)


def _real(x, g, rows):
    if any(row.get('split') != 'train' for row in rows):
        raise ValueError('Language student accepts official Train rows only')
    mass = torch.as_tensor(real_source_coefficients(rows), device=x.device, dtype=torch.float32)
    ids = torch.where(mass > 0)[0]
    x, g = x[ids], g[ids]
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(g).all()):
        raise ValueError('Genuine Train language features must be finite')
    return x, g, mass[ids], [rows[int(i)] for i in ids.cpu()]


def _predict(module, x, chunk):
    with torch.inference_mode():
        return torch.cat([module(x[i:i+chunk]) for i in range(0, len(x), chunk)])


def train_student(x, g, rows, cfg, *, validation=None, epochs=None):
    """Teacher centering, input normalization, and optimization use genuine Train only."""
    horizon = int(cfg['student_epochs'])
    total = horizon if epochs is None else int(epochs)
    if horizon < 1 or not 0 <= total <= horizon:
        raise ValueError('Student epochs must lie between zero and the configured schedule horizon')
    x, g, mass, rows = _real(x, g, rows)
    mean, scale = weighted_normalization(x, mass)
    center = (g * mass[:, None]).sum(0)
    energy = ((g - center).square().sum(1) * mass).sum()
    if float(energy) < 1e-10:
        return None, dict(status='uninformative_teacher', best_epoch=0)
    sigma = energy.sqrt()
    target = (g - center) / sigma
    torch.manual_seed(cfg['seed'])
    model = CenteredStudent(x.shape[1], g.shape[1], cfg['student_hidden'], mean, scale).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['student_learning_rate'],
                                  weight_decay=cfg.get('weight_decay', 1e-4))
    chunk = cfg['batch_rows']
    if validation is not None:
        vx, vg, vm, vrows = _real(*validation)
        vt = (vg - center) / sigma
    else:
        vx, vg, vm, vrows, vt = x, g, mass, rows, target
    initial_loss = float((vt.square().sum(1) * vm).sum())
    best_loss = final_loss = initial_loss
    best, best_epoch, trace, learning_rates = cpu_state(model), 0, [], []
    phase = Phase('V3.9 centered language student ' + ('Train holdout' if validation is not None else 'full Train'), total)
    started = time.monotonic()
    generator = torch.Generator(device='cpu').manual_seed(cfg['seed'])
    for epoch in range(1, total + 1):
        # A shorter full-data refit follows the same schedule prefix as selection.
        # Compressing the cosine into selected_epochs would change that recipe.
        learning_rate = cfg['student_learning_rate'] * .5 * (1. + math.cos(math.pi * (epoch - 1) / horizon))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        learning_rates.append(learning_rate)
        order = torch.randperm(len(x), generator=generator).to(x.device)
        model.train()
        for start in range(0, len(x), chunk):
            ids = order[start:start+chunk]
            error = (model(x[ids]) - target[ids]).square().sum(1)
            loss = (error * mass[ids]).sum() * (len(x) / len(ids))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite centered student loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        prediction = _predict(model, vx, chunk)
        measured = float(((prediction - vt).square().sum(1) * vm).sum())
        if not math.isfinite(measured):
            raise FloatingPointError('Nonfinite centered student evaluation loss')
        final_loss = measured
        trace.append(measured)
        if measured < best_loss:
            best_loss, best_epoch, best = measured, epoch, cpu_state(model)
        phase.update(epoch, measured)
        print(f'[Student] epoch={epoch}/{total} centered_mse={measured:.6f} '
              f'best_epoch={best_epoch} best_mse={best_loss:.6f} lr={learning_rate:.3g} '
              f'elapsed_s={time.monotonic()-started:.1f}', flush=True)
    model.load_state_dict(best)
    model.eval().requires_grad_(False)
    fit_prediction, prediction = _predict(model, x, chunk), _predict(model, vx, chunk)
    sse = float(((prediction - vt).square().sum(1) * vm).sum())
    sst = float((vt.square().sum(1) * vm).sum())
    quality = dict(status='fitted', best_epoch=best_epoch, selected_epoch=best_epoch,
        epochs_completed=total, centered_mse_trace=trace,
        initial_centered_mse=initial_loss, best_centered_mse=best_loss,
        final_centered_mse=final_loss,
        selected_training_mse=float(((fit_prediction - target).square().sum(1) * mass).sum()),
        selection_metric='source-heldout centered MSE' if validation is not None else 'full-Train centered MSE',
        learning_rate_schedule='cosine, fixed configured epoch horizon',
        schedule_horizon=horizon, learning_rate_trace=learning_rates,
        centered_r2_vs_train_mean=1. - sse / max(sst, 1e-12),
        constant_mean_mse=sst, prediction_mse=sse,
        student_language_probe=language_probe(fit_prediction, rows, mass, prediction, vrows, vm),
        teacher_language_probe=language_probe(target, rows, mass, vt, vrows, vm),
        mean_only_language_balanced_accuracy=.5,
        estimation='genuine official Train only; no fake or Dev rows estimate language variation',
        validation='Train source holdout' if validation is not None else 'full Train fitting diagnostics, not validation')
    print(f'[Student] selected_epoch={best_epoch}/{total} selected_mse={sse:.6f} '
          f'last_epoch_mse={final_loss:.6f}', flush=True)
    return model.spec(), quality


def language_probe(fit, fit_rows, mass, validation, validation_rows, validation_mass):
    """Fixed ridge probe; no tuning on Dev, no SciPy or NumPy BLAS solver."""
    y = torch.tensor([1. if r['language'] == 'en' else -1. for r in fit_rows], device=fit.device)
    feature_mean = (fit * mass[:, None]).sum(0)
    a = fit - feature_mean
    gram = a.T @ (a * mass[:, None])
    gram += .01 * torch.eye(a.shape[1], device=a.device)
    coef = torch.linalg.solve(gram, a.T @ (mass * y))
    estimate = ((validation - feature_mean) @ coef >= 0).cpu().numpy()
    truth = np.asarray([r['language'] == 'en' for r in validation_rows])
    correct = torch.as_tensor(estimate == truth, device=fit.device, dtype=torch.float32)
    conditions = {}
    for name in ('offline', 'online', 'noisy_a', 'noisy_b'):
        ids = [i for i, r in enumerate(validation_rows) if r['condition'] == name]
        if not ids:
            continue
        recalls = [float(np.mean(estimate[np.asarray(ids)[truth[ids] == c]] == c))
                   for c in (False, True) if np.any(truth[ids] == c)]
        conditions[name] = float(np.mean(recalls))
    return dict(source_balanced_accuracy=float((correct * validation_mass).sum()),
                condition_balanced_accuracy=conditions)


def qualified(quality, cfg):
    reasons = []
    if quality.get('status') != 'fitted':
        return False, [quality.get('status', 'student_fit_failed')]
    if quality['centered_r2_vs_train_mean'] < cfg['min_student_r2']:
        reasons.append('student_does_not_explain_enough_beyond_constant_mean')
    if quality['student_language_probe']['source_balanced_accuracy'] < cfg['min_student_language_accuracy']:
        reasons.append('student_language_information_weak')
    if quality['teacher_language_probe']['source_balanced_accuracy'] < cfg['min_teacher_language_accuracy']:
        reasons.append('teacher_language_information_weak')
    return not reasons, reasons
