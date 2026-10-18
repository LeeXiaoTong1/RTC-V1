"""Bounded Train-only inner-step calibration, preserving the warmup detector."""
from collections import Counter
import copy
import math
import statistics

import torch

from .meta import (Transaction, NonfiniteGradient, raw_gradients, restore_rng,
                   side_pass, validate_units, virtual_step, virtual_update_statistics,
                   logit_change)
from .precision import scaler_for

CANDIDATES = (.001, .003, .01, .03, .1)


class CalibrationError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def choose_alpha(records):
    summaries = []
    for alpha in CANDIDATES:
        selected = [row for row in records if row['alpha'] == alpha]
        finite = len(selected) == 12 and all(row.get('finite', False) for row in selected)
        changes = [row['relative_drop'] for row in selected if row.get('finite', False)]
        nonincrease = sum(value >= 0 for value in changes)
        median = statistics.median(changes) if len(changes) == 12 else None
        worst = max((-value for value in changes), default=None)
        passed = bool(finite and nonincrease >= 9 and .005 <= median <= .05 and worst <= .2)
        summaries.append(dict(alpha=alpha, finite=finite, nonincrease_tasks=nonincrease,
                              median_relative_drop=median, worst_relative_increase=worst,
                              accepted=passed))
    chosen = next((item['alpha'] for item in summaries if item['accepted']), None)
    return chosen, summaries


def calibrate(model, optimizer, task_factory, cfg, progress=None):
    """12 fixed tasks, 4/type, with each waveform generated exactly once.

    task_factory may be an iterable or zero-argument callable. Each element is
    either units or {'units': units, 'task_kind': 'noise'|'rtc'|'joint'}.
    Candidate sweeps are task-major to avoid retaining 576 full waveforms.
    All candidates start at the exact same original detector. B never scores
    the candidates; only A is encoded. The caller must enable LoRA first.
    """
    if hasattr(model, 'joint') and not model.joint:
        raise ValueError('Enable full LoRA training before inner-step calibration')
    transaction = Transaction(model)
    live_scaler = scaler_for(optimizer, cfg)
    scaler_state = copy.deepcopy(live_scaler.state_dict())
    scale = float(live_scaler.get_scale())
    # Unfreezing LoRA may require a different numerical loss scale. Keep this
    # calibration-only scaler private: no change to the live optimizer/scaler
    # state, and no change to alpha's loss-reduction gates.
    scaler = torch.amp.GradScaler(transaction.parameters[0][1].device.type,
                                 enabled=live_scaler.is_enabled(), init_scale=scale)
    scaler.load_state_dict(scaler_state)
    records, kinds = [], Counter()
    report = dict(status='running', candidates=list(CANDIDATES), records=records,
                  task_count=0, task_counts={}, encoded_views=0,
                  selection_data='fixed Train A only; no B, Dev, Progress or Eval',
                  initial_scale=scale, gradient_retries=[], final_calibration_scale=scale)
    try:
        model.train()
        tasks = task_factory() if callable(task_factory) else iter(task_factory)
        for index, task in enumerate(tasks):
            if index >= 12:
                raise ValueError('Calibration requires exactly 12 fixed Train tasks')
            units = task['units'] if isinstance(task, dict) else task
            sides = validate_units(units)
            kind = task.get('task_kind') if isinstance(task, dict) else units[0][0].get('task_kind')
            if kind not in ('noise', 'rtc', 'joint'):
                raise ValueError('Calibration task_kind must be noise, rtc, or joint')
            kinds[kind] += 1
            for attempt in range(4):
                transaction.restore()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                before = side_pass(model, sides['A'], cfg, scaler, backward=True, capture_logits=True)
                report['encoded_views'] += len(sides['A'])
                current_scale = float(scaler.get_scale())
                try:
                    gradients, norm = raw_gradients(transaction, current_scale)
                    break
                except NonfiniteGradient as exc:
                    if not scaler.is_enabled() or attempt == 3:
                        raise NonfiniteGradient('Calibration A gradients remain nonfinite after bounded loss-scale retries') from exc
                    next_scale = current_scale * float(scaler.get_backoff_factor())
                    scaler.update(new_scale=next_scale)
                    report['gradient_retries'].append(dict(task=index, attempt=attempt + 1,
                        previous_scale=current_scale, scale=next_scale, reason=str(exc),
                        same_waveforms=True, same_rng=True))
                    report['final_calibration_scale'] = next_scale
            for alpha in CANDIDATES:
                transaction.restore()
                model.train()
                inner_norm, inner_factor = virtual_step(transaction, gradients, alpha,
                    float(cfg.get('max_grad_norm', 1.)), norm=norm)
                update = virtual_update_statistics(optimizer, transaction)
                # Same dropout/RNG and unchanged A waveform as the before pass.
                restore_rng(transaction.rng)
                try:
                    after = side_pass(model, sides['A'], cfg, scaler, backward=False, capture_logits=True)
                    relative_drop = (before['loss'] - after['loss']) / max(before['loss'], 1e-12)
                    finite = math.isfinite(after['loss']) and math.isfinite(norm)
                    record = dict(task=index, task_kind=kind, alpha=alpha,
                                  before=before['loss'], after=after['loss'],
                                  relative_drop=relative_drop, gradient_norm=norm, finite=finite,
                                  loss_scale=current_scale, A_logit_change=logit_change(before, after))
                except NonfiniteGradient as exc:
                    record = dict(task=index, task_kind=kind, alpha=alpha,
                                  before=before['loss'], after=None, relative_drop=None,
                                  gradient_norm=norm, finite=False, error=str(exc), loss_scale=current_scale)
                record.update(update, planned_virtual_update_norm=alpha * inner_factor * inner_norm,
                              virtual_update_measured=True)
                records.append(record)
                report['encoded_views'] += len(sides['A'])
            del gradients
            report.update(task_count=index + 1, task_counts=dict(kinds))
            if progress is not None:
                progress(index + 1, 12, copy.deepcopy(report))
        if report['task_count'] != 12 or kinds != Counter(noise=4, rtc=4, joint=4):
            raise ValueError('Calibration requires exactly four tasks per condition type')
        selected, summaries = choose_alpha(records)
        report.update(selected_alpha=selected, summaries=summaries,
                      status='complete' if selected is not None else 'failed')
        if selected is None:
            raise CalibrationError('No inner-step candidate passed the fixed Train calibration gates', report)
        return report
    except CalibrationError:
        raise
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise CalibrationError('Inner-step calibration failed: ' + str(exc), report) from exc
    finally:
        transaction.restore()
        optimizer.zero_grad(set_to_none=True)
        # The live scaler was never used or mutated; the private scaler dies.
