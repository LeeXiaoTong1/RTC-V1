"""All optimization and hyperparameter selection see Train only, never Dev."""
import math
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from live_progress import Phase
from w2v_v36.fit import source_coefficients, split_sources
from .common import cpu_state
from .model import CenteredStudent, ResidualClassifier
from .student import train_student, qualified, weighted_normalization


def objective_parts(margin, delta, fake_target, mass, regularization, cfg,
                    fake_mass, real_mass, *, fit_mass=None):
    """Protect correct decisions with a margin buffer, not their old confidence.

    T-squared compensated BCE prevents almost-certain Train examples from
    supplying vanishing gradients. Hard mining affects CE only: protection and
    the normalized correction penalty retain the original balanced mass.
    """
    temperature = float(cfg.get('loss_temperature', 2.))
    safety = float(cfg.get('safety_margin', 1.))
    if not math.isfinite(temperature) or temperature < 1 or not math.isfinite(safety) or safety < 0:
        raise ValueError('Finite temperature >= 1 and nonnegative safety margin required')
    fitting = mass if fit_mass is None else fit_mass
    ce = (F.binary_cross_entropy_with_logits((margin + delta) / temperature,
          fake_target, reduction='none') * fitting).sum() * temperature ** 2
    fake = (fake_target > .5) & (margin >= 0)
    real = (fake_target < .5) & (margin < 0)
    signed = (2 * fake_target - 1) * margin
    corrected_signed = (2 * fake_target - 1) * (margin + delta)
    violation = F.relu(signed.clamp(min=0, max=safety) - corrected_signed).square()
    fp = (violation * mass * fake).sum() / fake_mass.clamp_min(1e-12)
    rp = (violation * mass * real).sum() / real_mass.clamp_min(1e-12)
    budget = float(cfg.get('residual_cap', 2.)) + margin.abs()
    anchor = regularization * ((delta / budget).square() * mass).sum()
    return dict(ce=ce, correction_penalty=anchor,
                fake_protection=float(cfg.get('fake_protection', 2.)) * fp,
                real_protection=float(cfg.get('real_protection', 1.)) * rp)


def objective(margin, delta, fake_target, mass, regularization, cfg,
              fake_mass, real_mass, *, fit_mass=None):
    return sum(objective_parts(margin, delta, fake_target, mass, regularization,
               cfg, fake_mass, real_mass, fit_mass=fit_mass).values())


def hard_example_coefficients(rows, mass, margin, cfg):
    """Bounded baseline-only mining preserves every language/class/view budget.

    Weights are fixed before fitting, detached, and cannot chase an evolving
    prediction or silently increase the English-real class prior.
    """
    gain = float(cfg.get('hard_example_gain', 3.))
    temperature = float(cfg.get('hard_example_temperature', 2.))
    if not math.isfinite(gain) or not 0 <= gain <= 8 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Bounded finite hard-example gain and positive temperature required')
    target = torch.tensor([r['label'] == 0 for r in rows], device=margin.device, dtype=margin.dtype)
    raw = 1 + gain * torch.sigmoid(-(2 * target - 1) * margin.detach() / temperature)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row['language'], row['label'], row['condition'])].append(i)
    result = mass.detach().clone()
    for indices in groups.values():
        ids = torch.as_tensor(indices, device=mass.device)
        weighted = mass[ids] * raw[ids]
        result[ids] = weighted * mass[ids].sum() / weighted.sum().clamp_min(1e-12)
    return result.detach()


def _train_metrics(rows, margin):
    """Cheap fixed-threshold proxy; no per-epoch AUC sorting or audio inference."""
    if not bool(torch.isfinite(margin).all()):
        raise FloatingPointError('Nonfinite V3.9 Train selection margins')
    labels = np.asarray([r['label'] for r in rows], dtype=np.int64)
    predicted = np.where(margin.detach().cpu().numpy() >= 0, 0, 1)
    values = {}
    for condition in ('online', 'noisy_a', 'noisy_b'):
        ids = np.asarray([i for i, r in enumerate(rows) if r['condition'] == condition], dtype=np.int64)
        counts = np.bincount(labels[ids] * 2 + predicted[ids], minlength=4).reshape(2, 2)
        if np.any(counts.sum(1) == 0):
            raise ValueError('Train selection requires both classes in Online and both noisy pools')
        denominator = counts.sum(0) + counts.sum(1)
        values[condition] = float(np.mean(2 * np.diag(counts) / denominator))
    return dict(clean_f1=values['online'], noisy_f1=.5 * (values['noisy_a'] + values['noisy_b']),
                weighted_f1=.3 * values['online'] + .35 * (values['noisy_a'] + values['noisy_b']))


def _change_diagnostics(rows, margin, delta):
    old = margin >= 0
    new = margin + delta >= 0
    truth = torch.tensor([r['label'] == 0 for r in rows], device=margin.device)
    groups = {}
    for language in ('en', 'zh'):
        for label in (0, 1):
            ids = torch.as_tensor([i for i, r in enumerate(rows) if r['language'] == language and r['label'] == label], device=margin.device, dtype=torch.long)
            groups[language + '/' + ('fake' if label == 0 else 'real')] = dict(
                rescued=int(((old[ids] != truth[ids]) & (new[ids] == truth[ids])).sum()),
                new_errors=int(((old[ids] == truth[ids]) & (new[ids] != truth[ids])).sum()))
    wrong = old != truth
    return dict(changed_decisions=int((old != new).sum()), groups=groups,
        delta_mean=float(delta.mean()), delta_abs_mean=float(delta.abs().mean()),
        delta_min=float(delta.min()), delta_max=float(delta.max()),
        wrong_sample_delta_abs_mean=float(delta[wrong].abs().mean()) if bool(wrong.any()) else 0.)


def _better(weighted, loss, best_weighted, best_loss):
    return weighted > best_weighted + 1e-12 or (abs(weighted - best_weighted) <= 1e-12 and loss < best_loss - 1e-7)


def _predict_delta(module, x, margin, context, chunk):
    with torch.inference_mode():
        return torch.cat([module.adjustment(x[i:i+chunk], margin[i:i+chunk],
            context[i:i+chunk] if context is not None else None) for i in range(0, len(x), chunk)])


def _context(student, x, chunk):
    if student is None:
        return None
    with torch.inference_mode():
        return torch.cat([student(x[i:i+chunk]) for i in range(0, len(x), chunk)])


def _recall_guards(rows, baseline_margin, margin, max_drop=.005):
    for condition in ('offline', 'online', 'noisy_a', 'noisy_b'):
        for language in ('en', 'zh'):
            for label in (0, 1):
                ids = [i for i, r in enumerate(rows) if r['condition'] == condition and r['language'] == language and r['label'] == label]
                if not ids:
                    continue
                sign = 1 if label == 0 else -1
                old = baseline_margin[ids] >= 0 if sign == 1 else baseline_margin[ids] < 0
                new = margin[ids] >= 0 if sign == 1 else margin[ids] < 0
                if float(new.float().mean()) < float(old.float().mean()) - max_drop:
                    return False
    return True


def _train(arm, x, rows, weight, bias, cfg, regularization, *, tune=None, epochs=None,
           student_spec=None, max_updates=None, schedule_steps=None):
    mass = torch.as_tensor(source_coefficients(rows, 'group_balanced'), device=x.device, dtype=torch.float32)
    target = torch.tensor([r['label'] == 0 for r in rows], device=x.device, dtype=torch.float32)
    mean, scale = weighted_normalization(x, mass)
    student = CenteredStudent.restore(student_spec).to(x.device) if student_spec else None
    # The two nonlinear controls have nearly identical trainable parameter counts.
    context_dim = student.output.out_features if student is not None else 0
    hidden = cfg['residual_hidden'] if student else round(cfg['residual_hidden'] * (x.shape[1] + cfg.get('teacher_dim', 256)) / x.shape[1])
    torch.manual_seed(cfg['seed'] + 1)
    module = ResidualClassifier(weight, bias, arm=arm, mean=mean, scale=scale,
                                hidden=hidden, cap=cfg['residual_cap'], student=student).to(x.device)
    parameters = [p for p in module.parameters() if p.requires_grad]
    learning_rate = float(cfg.get('learning_rate', .0005))
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate,
                                  weight_decay=float(cfg.get('weight_decay', 1e-4)))
    chunk = cfg['batch_rows']
    with torch.no_grad():
        base_logits = F.linear(x, module.weight, module.bias)
        margin = base_logits[:, 0] - base_logits[:, 1]
    fit_mass = hard_example_coefficients(rows, mass, margin, cfg)
    context = _context(student, x, chunk)
    fake_mass = (mass * (target > .5) * (margin >= 0)).sum()
    real_mass = (mass * (target < .5) * (margin < 0)).sum()
    if tune is None:
        vx, vrows, vmass, vtarget, vmargin, vcontext = x, rows, mass, target, margin, context
    else:
        vx, vrows = tune
        vmass = torch.as_tensor(source_coefficients(vrows, 'group_balanced'), device=x.device, dtype=torch.float32)
        vtarget = torch.tensor([r['label'] == 0 for r in vrows], device=x.device, dtype=torch.float32)
        with torch.no_grad():
            vl = F.linear(vx, module.weight, module.bias)
            vmargin = vl[:, 0] - vl[:, 1]
        vcontext = _context(student, vx, chunk)
    temperature = float(cfg.get('loss_temperature', 2.))
    best_loss = float((F.binary_cross_entropy_with_logits(vmargin / temperature, vtarget, reduction='none') * vmass).sum()) * temperature ** 2
    if not math.isfinite(best_loss):
        raise FloatingPointError('Nonfinite V3.9 baseline validation loss')
    initial_loss = best_loss
    baseline_metrics = _train_metrics(vrows, vmargin)
    best_weighted = baseline_metrics['weighted_f1']
    best, best_epoch, trace = cpu_state(module), 0, []
    total = cfg.get('epochs', 20) if epochs is None else epochs
    batches = math.ceil(len(x) / chunk)
    horizon = max(1, int(schedule_steps or cfg.get('epochs', 20) * batches))
    updates = best_updates = 0
    phase = Phase(f'V3.9 {arm} ' + ('Train holdout' if tune is not None else 'full Train'), total)
    started = time.monotonic()
    generator = torch.Generator(device='cpu').manual_seed(cfg['seed'] + 2)
    for epoch in range(1, total + 1):
        module.train()
        order = torch.randperm(len(x), generator=generator).to(x.device)
        for start in range(0, len(x), chunk):
            if max_updates is not None and updates >= max_updates:
                break
            rate = .1 + .9 * .5 * (1 + math.cos(math.pi * min(updates / horizon, 1.)))
            for group in optimizer.param_groups:
                group['lr'] = learning_rate * rate
            ids = order[start:start+chunk]
            delta = module.adjustment(x[ids], margin[ids], context[ids] if context is not None else None)
            loss = objective(margin[ids], delta, target[ids], mass[ids], regularization,
                             cfg, fake_mass, real_mass, fit_mass=fit_mass[ids]) * (len(x) / len(ids))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite V3.9 residual objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
        module.eval()
        module.validate()
        delta = _predict_delta(module, vx, vmargin, vcontext, chunk)
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError('Nonfinite V3.9 validation correction')
        measured = float((F.binary_cross_entropy_with_logits((vmargin + delta) / temperature, vtarget, reduction='none') * vmass).sum()) * temperature ** 2
        if not math.isfinite(measured):
            raise FloatingPointError('Nonfinite V3.9 validation loss')
        metrics = _train_metrics(vrows, vmargin + delta)
        allowed = _recall_guards(vrows, vmargin, vmargin + delta, cfg.get('train_max_recall_drop', .005))
        diagnostics = _change_diagnostics(vrows, vmargin, delta)
        with torch.no_grad():
            vm_fake = (vmass * (vtarget > .5) * (vmargin >= 0)).sum()
            vm_real = (vmass * (vtarget < .5) * (vmargin < 0)).sum()
            parts = objective_parts(vmargin, delta, vtarget, vmass, regularization,
                                    cfg, vm_fake, vm_real)
        components = {k: float(v) for k, v in parts.items()}
        if any(not math.isfinite(v) for v in components.values()):
            raise FloatingPointError('Nonfinite V3.9 validation objective component')
        trace.append(dict(epoch=epoch, balanced_ce=measured, recall_guards_passed=allowed,
                          **metrics, **diagnostics, objective_components=components,
                          optimizer_steps=updates, learning_rate=optimizer.param_groups[0]['lr']))
        if allowed and _better(metrics['weighted_f1'], measured, best_weighted, best_loss):
            best, best_loss, best_epoch = cpu_state(module), measured, epoch
            best_weighted, best_updates = metrics['weighted_f1'], updates
        phase.update(epoch, measured)
        print(f'[Fit] {arm} lambda={regularization:g} epoch={epoch}/{total} '
              f'balanced_ce={measured:.6f} Train_proxy_weighted={100*metrics["weighted_f1"]:.3f} '
              f'changed={diagnostics["changed_decisions"]} recall_guard={allowed} '
              f'elapsed_s={time.monotonic()-started:.1f}', flush=True)
        if max_updates is not None and updates >= max_updates:
            break
    module.load_state_dict(best)
    module.eval()
    return module.spec(), dict(best_epoch=best_epoch, balanced_ce=best_loss, baseline_ce=initial_loss,
        regularization=regularization, trace=trace, trainable_parameters=sum(p.numel() for p in parameters),
        context_dim=context_dim, best_epoch_zero_means_identity=(best_epoch == 0),
        weighted_f1=best_weighted, baseline_weighted_f1=baseline_metrics['weighted_f1'],
        optimizer_steps_at_best=best_updates, optimizer_steps_executed=updates, schedule_steps=horizon,
        selection='fixed-threshold Train proxy Weighted first, tempered balanced CE on ties; class recall guards',
        loss_temperature=temperature, temperature_squared_compensation=True,
        hard_mining=dict(gain=cfg.get('hard_example_gain', 3.), temperature=cfg.get('hard_example_temperature', 2.),
                         grouping='language/class/condition mass preserved; fixed baseline scores only'))


def fit_all(bundle, weight, bias, cfg, stage):
    """The function deliberately has no Dev argument. Stage artifacts are resumable."""
    rows = bundle['rows']
    if any(r.get('split') != 'train' for r in rows):
        raise ValueError('V3.9 fitting accepts official Train only')
    ids, val_ids, split = split_sources(rows, cfg['holdout_fraction'], cfg['seed'])
    device = torch.device(cfg['device'])
    x = torch.from_numpy(np.array(bundle['x'], dtype=np.float32, copy=True)).to(device)
    g = torch.from_numpy(np.array(bundle['lid'], dtype=np.float32, copy=True)).to(device)
    cfg = dict(cfg, teacher_dim=g.shape[1])
    fit_rows, val_rows = [rows[int(i)] for i in ids], [rows[int(i)] for i in val_ids]
    xi, xv, gi, gv = x[ids], x[val_ids], g[ids], g[val_ids]

    def language_stage():
        spec, quality = train_student(xi, gi, fit_rows, cfg, validation=(xv, gv, val_rows))
        passed, reasons = qualified(quality, cfg)
        if quality.get('status') == 'fitted':
            print('[Student evidence: Train holdout] '
                  f'centered_R2_vs_mean={quality["centered_r2_vs_train_mean"]:.4f} '
                  f'student_language_accuracy={100*quality["student_language_probe"]["source_balanced_accuracy"]:.2f}% '
                  f'teacher_language_accuracy={100*quality["teacher_language_probe"]["source_balanced_accuracy"]:.2f}% '
                  'constant_language_accuracy=50.00%', flush=True)
        full_spec, full_quality = None, None
        if passed:
            full_spec, full_quality = train_student(x, g, rows, cfg, epochs=quality['best_epoch'])
            # A numerically degenerate refit cannot inherit the holdout pass.
            full_pass, full_reasons = qualified(full_quality, cfg)
            if not full_pass:
                passed, reasons = False, ['full_refit_' + r for r in full_reasons]
        print('[Student check] qualified=' + str(passed) + ' reasons=' + str(reasons), flush=True)
        return dict(qualified=passed, reasons=reasons, holdout=quality, full_train=full_quality,
                    holdout_spec=spec, full_spec=full_spec)

    language = stage('language', language_stage)
    arms = []
    for arm in ('calibration', 'residual_control', 'language_residual'):
        if arm == 'language_residual' and not language['qualified']:
            arms.append(dict(name=arm, status='skipped', reason='; '.join(language['reasons'])))
            continue

        def run_arm(arm=arm):
            trials = []
            for regularization in cfg['lambda_grid']:
                spec, detail = _train(arm, xi, fit_rows, weight, bias, cfg, regularization,
                    tune=(xv, val_rows), student_spec=language['holdout_spec'] if arm == 'language_residual' else None)
                trials.append((spec, detail))
            _, chosen = min(trials, key=lambda v: (-v[1]['weighted_f1'], v[1]['balanced_ce'], v[1]['best_epoch'], -v[1]['regularization']))
            # Preserve optimizer exposure and the same LR horizon. A nonidentity
            # refit still receives at least one complete pass over all Train.
            full_batches = math.ceil(len(x) / cfg['batch_rows'])
            updates = max(full_batches, chosen['optimizer_steps_at_best']) if chosen['best_epoch'] else 0
            full_epochs = math.ceil(updates / full_batches) if updates else 0
            spec, full = _train(arm, x, rows, weight, bias, cfg, chosen['regularization'],
                epochs=full_epochs, max_updates=updates, schedule_steps=chosen['schedule_steps'],
                student_spec=language['full_spec'] if arm == 'language_residual' else None)
            return dict(name=arm, status='fitted', spec=spec, train_selection=chosen,
                trials=[v[1] for v in trials], full_train=full, student_qualified=language['qualified'] if arm == 'language_residual' else None)

        arms.append(stage(arm, run_arm))
    return dict(candidates=arms, split=split,
        language={k: v for k, v in language.items() if not k.endswith('_spec')},
        fitting_data='official Train only; source grouped holdout; final refit covers all cached Train rows',
        source_count=len({r['source_id'] for r in rows}), train_views=len(rows),
        optimization='PyTorch AdamW, FP32, one CPU math thread, no SciPy L-BFGS or OpenBLAS solve')
