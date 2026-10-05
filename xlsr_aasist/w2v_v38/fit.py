"""All optimization and hyperparameter selection see Train only, never Dev."""
import time

import numpy as np
import torch
from torch.nn import functional as F

from live_progress import Phase
from w2v_v36.fit import source_coefficients, split_sources
from .common import cpu_state
from .model import CenteredStudent, ResidualClassifier
from .student import train_student, qualified, weighted_normalization


def objective(margin, delta, fake_target, mass, regularization, cfg, fake_mass, real_mass):
    ce = (F.binary_cross_entropy_with_logits(margin + delta, fake_target, reduction='none') * mass).sum()
    fake = (fake_target > .5) & (margin >= 0)
    real = (fake_target < .5) & (margin < 0)
    slack = cfg['protection_slack']
    fp = (F.relu(-delta - slack).square() * mass * fake).sum() / fake_mass.clamp_min(1e-12)
    rp = (F.relu(delta - slack).square() * mass * real).sum() / real_mass.clamp_min(1e-12)
    return ce + regularization * (delta.square() * mass).sum() + cfg['fake_protection'] * fp + cfg['real_protection'] * rp


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


def _train(arm, x, rows, weight, bias, cfg, regularization, *, tune=None, epochs=None, student_spec=None):
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
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=1e-4)
    chunk = cfg['batch_rows']
    with torch.no_grad():
        base_logits = F.linear(x, module.weight, module.bias)
        margin = base_logits[:, 0] - base_logits[:, 1]
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
    best_loss = float((F.binary_cross_entropy_with_logits(vmargin, vtarget, reduction='none') * vmass).sum())
    initial_loss = best_loss
    best, best_epoch, trace = cpu_state(module), 0, []
    total = cfg['epochs'] if epochs is None else epochs
    phase = Phase(f'V3.8 {arm} ' + ('Train holdout' if tune is not None else 'full Train'), total)
    started = time.monotonic()
    generator = torch.Generator(device='cpu').manual_seed(cfg['seed'] + 2)
    for epoch in range(1, total + 1):
        module.train()
        order = torch.randperm(len(x), generator=generator).to(x.device)
        for start in range(0, len(x), chunk):
            ids = order[start:start+chunk]
            delta = module.adjustment(x[ids], margin[ids], context[ids] if context is not None else None)
            loss = objective(margin[ids], delta, target[ids], mass[ids], regularization, cfg, fake_mass, real_mass) * (len(x) / len(ids))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite V3.8 residual objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
            optimizer.step()
        module.eval()
        delta = _predict_delta(module, vx, vmargin, vcontext, chunk)
        measured = float((F.binary_cross_entropy_with_logits(vmargin + delta, vtarget, reduction='none') * vmass).sum())
        allowed = tune is None or _recall_guards(vrows, vmargin, vmargin + delta)
        trace.append(dict(epoch=epoch, balanced_ce=measured, recall_guards_passed=allowed))
        if tune is None or (allowed and measured < best_loss - 1e-7):
            best, best_loss, best_epoch = cpu_state(module), measured, epoch
        phase.update(epoch, measured)
        print(f'[Fit] {arm} lambda={regularization:g} epoch={epoch}/{total} '
              f'balanced_ce={measured:.6f} recall_guard={allowed} elapsed_s={time.monotonic()-started:.1f}', flush=True)
    module.load_state_dict(best)
    module.eval()
    return module.spec(), dict(best_epoch=best_epoch, balanced_ce=best_loss, baseline_ce=initial_loss,
        regularization=regularization, trace=trace, trainable_parameters=sum(p.numel() for p in parameters),
        context_dim=context_dim, best_epoch_zero_means_identity=(best_epoch == 0))


def fit_all(bundle, weight, bias, cfg, stage):
    """The function deliberately has no Dev argument. Stage artifacts are resumable."""
    rows = bundle['rows']
    if any(r.get('split') != 'train' for r in rows):
        raise ValueError('V3.8 fitting accepts official Train only')
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
            _, chosen = min(trials, key=lambda v: (v[1]['balanced_ce'], v[1]['best_epoch'], -v[1]['regularization']))
            spec, full = _train(arm, x, rows, weight, bias, cfg, chosen['regularization'],
                epochs=chosen['best_epoch'], student_spec=language['full_spec'] if arm == 'language_residual' else None)
            return dict(name=arm, status='fitted', spec=spec, train_selection=chosen,
                trials=[v[1] for v in trials], full_train=full, student_qualified=language['qualified'] if arm == 'language_residual' else None)

        arms.append(stage(arm, run_arm))
    return dict(candidates=arms, split=split,
        language={k: v for k, v in language.items() if not k.endswith('_spec')},
        fitting_data='official Train only; source grouped holdout; final refit covers all cached Train rows',
        source_count=len({r['source_id'] for r in rows}), train_views=len(rows),
        optimization='PyTorch AdamW, FP32, one CPU math thread, no SciPy L-BFGS or OpenBLAS solve')
