"""Train-only language debiasing and anchored final-linear adaptation.

The external LID teacher supplies Train targets only. A nonlinear language
student is fitted on the fitting partition before any holdout comparison. At
deployment the student uses this detector's own frozen hidden vector. The
centered ridge map is fitted exclusively on genuine Train recordings, with
equal EN/ZH mass and the existing ordinary/noisy source budgets.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from w2v_v36.fit import (_numpy, _save_dev_scores, _solve, predict,
                         source_coefficients, sources, split_sources)
from w2v_v36.metrics import evaluate

from .student import fit_student, predict_student


HIDDEN_DIM = 512
LID_DIM = 256
CANDIDATES = ('head_only_control', 'language_debias')
RECALL_GROUPS = ('online', 'seen', 'heldout')


def _features(x, g=None, *, name='Features'):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != HIDDEN_DIM or x.dtype != np.float32 or not len(x):
        raise ValueError(name + ' requires nonempty FP32 [N,512] hidden vectors')
    if not np.isfinite(x).all():
        raise ValueError(name + ' contains nonfinite hidden vectors')
    if g is not None:
        g = np.asarray(g)
        if g.shape != (len(x), LID_DIM) or g.dtype != np.float32:
            raise ValueError(name + ' requires FP32 [N,256] continuous LID vectors')
        if not np.isfinite(g).all():
            raise ValueError(name + ' contains nonfinite continuous LID vectors')
    return x, g


def real_source_coefficients(rows):
    """Real-only: EN=ZH=0.5, each source ordinary=0.5, noise=0.25+0.25."""
    records = sources(rows)
    real = [item for item in records.values() if item['label'] == 1]
    counts = Counter(item['language'] for item in real)
    if set(counts) != {'en', 'zh'}:
        raise ValueError('Language map needs genuine EN and ZH Train sources')
    coefficients = np.zeros(len(rows), dtype=np.float64)
    for item in real:
        budget = .5 / counts[item['language']]
        indices = item['indices']
        ordinary_count = 1 + int('online' in indices)
        for condition, index in indices.items():
            coefficients[index] = budget * (.25 if condition.startswith('noisy') else .5 / ordinary_count)
    if not np.isclose(coefficients.sum(), 1., rtol=0, atol=1e-12):
        raise AssertionError('Real-only source mass must be one')
    return coefficients


def _language_moments(x, g, rows, chunk_rows=4096):
    x, g = _features(x, g, name='Language fitting')
    if len(rows) != len(x) or chunk_rows < 1:
        raise ValueError('Language fitting row count/chunk size is invalid')
    coefficients = real_source_coefficients(rows)
    indices = np.flatnonzero(coefficients)
    mean = np.zeros(LID_DIM, dtype=np.float64)
    hidden_mean = np.zeros(HIDDEN_DIM, dtype=np.float64)
    for start in range(0, len(indices), chunk_rows):
        idx = indices[start:start + chunk_rows]
        mass = coefficients[idx]
        mean += mass @ g[idx].astype(np.float64)
        hidden_mean += mass @ x[idx].astype(np.float64)
    variance = np.zeros(LID_DIM, dtype=np.float64)
    for start in range(0, len(indices), chunk_rows):
        idx = indices[start:start + chunk_rows]
        centered = g[idx].astype(np.float64) - mean
        variance += coefficients[idx] @ (centered * centered)
    # Constant teacher/student coordinates carry no fitted language direction.
    scale = np.sqrt(np.maximum(variance, 0.))
    scale[scale < 1e-6] = 1.
    gram = np.zeros((LID_DIM, LID_DIM), dtype=np.float64)
    cross = np.zeros((LID_DIM, HIDDEN_DIM), dtype=np.float64)
    for start in range(0, len(indices), chunk_rows):
        idx = indices[start:start + chunk_rows]
        standardized = (g[idx].astype(np.float64) - mean) / scale
        weighted = standardized * coefficients[idx, None]
        gram += standardized.T @ weighted
        cross += weighted.T @ (x[idx].astype(np.float64) - hidden_mean)
    if not all(np.isfinite(value).all() for value in (mean, scale, gram, cross)):
        raise FloatingPointError('Nonfinite weighted language covariance')
    return dict(mean=mean, scale=scale, gram=gram, cross=cross,
                real_rows=int(len(indices)), hidden_mean=hidden_mean)


def _state_from_moments(moments, ridge, alpha):
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Language ridge must be finite and positive')
    if not np.isfinite(alpha) or not 0 <= alpha <= .75:
        raise ValueError('Language alpha must lie in [0,0.75]')
    mapping = np.linalg.solve(moments['gram'] + ridge * np.eye(LID_DIM), moments['cross'])
    if not np.isfinite(mapping).all():
        raise FloatingPointError('Nonfinite language ridge solution')
    return dict(mean=moments['mean'].astype(np.float32).tolist(),
                scale=moments['scale'].astype(np.float32).tolist(),
                mapping=mapping.astype(np.float32).tolist(), alpha=float(alpha), ridge=float(ridge))


def fit_language_state(x, g, rows, ridge, alpha=.5, chunk_rows=4096):
    """Fit only real rows; centered correction preserves their weighted mean.

    The reference global mean is the balanced genuine-Train mean, not a mean
    estimated from fake, Dev, or test rows. No intercept or hidden mean is removed.
    """
    return _state_from_moments(_language_moments(x, g, rows, chunk_rows), float(ridge), float(alpha))


def _language_arrays(state):
    mean = np.asarray(state['mean'], dtype=np.float32)
    scale = np.asarray(state['scale'], dtype=np.float32)
    mapping = np.asarray(state['mapping'], dtype=np.float32)
    alpha = float(state['alpha'])
    ridge = float(state['ridge'])
    if mean.shape != (LID_DIM,) or scale.shape != (LID_DIM,) or mapping.shape != (LID_DIM, HIDDEN_DIM):
        raise ValueError('Language state dimensions must be 256 -> 512')
    if not all(np.isfinite(a).all() for a in (mean, scale, mapping)) or np.any(scale <= 0):
        raise ValueError('Language state must contain finite values and positive scales')
    if not np.isfinite(alpha) or not 0 <= alpha <= .75 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Language state needs alpha in [0,0.75] and positive ridge')
    return mean, scale, mapping, np.float32(alpha)


def _project(g, language_state, chunk_rows=4096):
    mean, scale, mapping, _ = _language_arrays(language_state)
    result = np.empty((len(g), HIDDEN_DIM), dtype=np.float32)
    for start in range(0, len(g), chunk_rows):
        stop = min(start + chunk_rows, len(g))
        result[start:stop] = ((g[start:stop] - mean) / scale) @ mapping
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite FP32 language projection')
    return result


def transform_features(x, g, language_state, chunk_rows=4096):
    """The precise FP32 transform used by fitting and exported-state prediction."""
    x, g = _features(x, g)
    if chunk_rows < 1:
        raise ValueError('Positive prediction chunk size required')
    if language_state is None:
        return x
    _, _, _, alpha = _language_arrays(language_state)
    if alpha == 0:
        return x
    if g is None:
        raise ValueError('Nonzero language subtraction needs continuous LID vectors')
    transformed = x - alpha * _project(g, language_state, chunk_rows)
    if not np.isfinite(transformed).all():
        raise FloatingPointError('Nonfinite FP32 debiased hidden vectors')
    return transformed


def logits_from_state(x, g=None, state=None, chunk_rows=4096):
    """Replay exported FP32 state; deployed LID comes from the internal student.

    An explicit g is supported for fitting tests and cached student projections;
    inference should omit it to use exactly the exported student parameters.
    """
    if state is None:
        raise ValueError('Classifier state is required')
    x, g = _features(x, g)
    if chunk_rows < 1:
        raise ValueError('Positive prediction chunk size required')
    weight, bias = np.asarray(state['weight'], dtype=np.float32), np.asarray(state['bias'], dtype=np.float32)
    if weight.shape != (2, HIDDEN_DIM) or bias.shape != (2,) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError('Classifier state requires finite [2,512] weights and [2] bias')
    language = state.get('language_state')
    if language is None:
        return predict(x, weight, bias, chunk_rows)
    _language_arrays(language)
    if float(language['alpha']) == 0:
        return predict(x, weight, bias, chunk_rows)
    student = state.get('student_state')
    if g is None and student is None:
        raise ValueError('Language candidate requires its fitted language student')
    result = np.empty((len(x), 2), dtype=np.float32)
    for start in range(0, len(x), chunk_rows):
        stop = min(start + chunk_rows, len(x))
        hidden = x[start:stop]
        language_vectors = predict_student(hidden, student) if g is None else g[start:stop]
        transformed = transform_features(hidden, language_vectors, language, chunk_rows)
        result[start:stop] = predict(transformed, weight, bias, chunk_rows)
    return result


def balanced_cross_entropy(rows, logits):
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (len(rows), 2) or not np.isfinite(logits).all():
        raise ValueError('Heldout CE requires finite [N,2] logits')
    labels = np.asarray([r['label'] for r in rows], dtype=np.float64)
    target = 1. - 2. * labels
    margin = logits[:, 0].astype(np.float64) - logits[:, 1]
    return float(source_coefficients(rows, 'group_balanced') @ np.logaddexp(0., -target * margin))


def _student_stats(predicted, teacher, rows, constant_mean=None, chunk_rows=4096):
    """Train-only fidelity evidence against a fitting-part constant predictor."""
    mass = source_coefficients(rows, 'group_balanced')
    predicted_mean = np.zeros(LID_DIM, dtype=np.float64)
    teacher_mean = np.zeros(LID_DIM, dtype=np.float64)
    cosine = 0.
    for start in range(0, len(rows), chunk_rows):
        stop = min(start + chunk_rows, len(rows))
        p = np.asarray(predicted[start:stop], dtype=np.float64)
        target = np.asarray(teacher[start:stop], dtype=np.float64)
        lengths = np.linalg.norm(target, axis=1, keepdims=True)
        if np.any(lengths < 1e-8):
            raise ValueError('Language teacher targets must be nonzero continuous vectors')
        target /= lengths
        predicted_mean += mass[start:stop] @ p
        teacher_mean += mass[start:stop] @ target
        p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
        cosine += float(mass[start:stop] @ np.einsum('nd,nd->n', p, target))
    variance = 0.
    for start in range(0, len(rows), chunk_rows):
        stop = min(start + chunk_rows, len(rows))
        delta = np.asarray(predicted[start:stop], dtype=np.float64) - predicted_mean
        variance += float(mass[start:stop] @ np.einsum('nd,nd->n', delta, delta))
    reference = teacher_mean if constant_mean is None else constant_mean
    length = float(np.linalg.norm(reference))
    constant_cosine = float(teacher_mean @ (reference / length)) if length >= 1e-8 else None
    diagnostics = dict(weighted_cosine=cosine, constant_fit_teacher_mean_cosine=constant_cosine,
                       weighted_predicted_variance=variance, constant_reference_norm=length,
                       effectively_constant=bool(variance <= 1e-10), variance_rejection_threshold=1e-10,
                       note='Source/group-balanced Train-only diagnostics; constant reference is fitted on fitting Train only.')
    if not np.isfinite([cosine, variance, length]).all():
        raise FloatingPointError('Nonfinite language student fidelity diagnostics')
    return diagnostics, teacher_mean


def _guardrail_config(cfg):
    result = dict(min_gain=float(cfg.get('min_gain', .002)),
                  max_clean_drop=float(cfg.get('max_clean_drop', .001)),
                  max_noisy_fake_recall_drop=float(cfg.get('max_noisy_fake_recall_drop', .005)),
                  min_en_real_gain=float(cfg.get('min_en_real_gain', .005)),
                  max_real_recall_drop=float(cfg.get('max_real_recall_drop', .005)))
    if any(not np.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError('Selection tolerances must be finite and nonnegative')
    if (result['min_gain'] < .002 or result['min_en_real_gain'] < .005
            or result['max_clean_drop'] > .001 or result['max_noisy_fake_recall_drop'] > .005
            or result['max_real_recall_drop'] > .005):
        raise ValueError('V3.7 fixed deployment guardrails may only be made stricter')
    return result


def select_candidate(baseline, candidates, cfg):
    """Compare exactly two predeclared candidates against all fixed Dev guards."""
    if not baseline.get('complete'):
        raise ValueError('Baseline Dev needs both classes in all three conditions')
    limits = _guardrail_config(cfg)
    selected, best = 'baseline', baseline['weighted_f1']
    for candidate in candidates:
        issues = []
        metrics = candidate.get('metrics')
        if candidate.get('status') != 'converged':
            issues.append('solver_not_converged')
        if not metrics or not metrics.get('complete'):
            issues.append('incomplete_dev')
        else:
            if metrics['weighted_f1'] < baseline['weighted_f1'] + limits['min_gain']:
                issues.append('weighted_gain_below_minimum')
            if metrics['noisy_f1'] < baseline['noisy_f1']:
                issues.append('noisy_below_baseline')
            if metrics['clean_f1'] < baseline['clean_f1'] - limits['max_clean_drop']:
                issues.append('clean_below_guardrail')
            en_reference, en_measured = [], []
            for condition in RECALL_GROUPS:
                for language in ('en', 'zh'):
                    name = condition + '/' + language
                    reference = baseline['groups'].get(name, {}).get('recall', [None, None])
                    measured = metrics['groups'].get(name, {}).get('recall', [None, None])
                    if reference[1] is None or measured[1] is None:
                        issues.append(name + '_real_recall_unavailable')
                    else:
                        if measured[1] < reference[1] - limits['max_real_recall_drop']:
                            issues.append(name + '_real_recall_below_guardrail')
                        if language == 'en':
                            en_reference.append(reference[1]); en_measured.append(measured[1])
                    if language == 'en':
                        if reference[0] is None or measured[0] is None:
                            issues.append(name + '_fake_recall_unavailable')
                        elif measured[0] < reference[0] - limits['max_noisy_fake_recall_drop']:
                            issues.append(name + '_fake_recall_below_guardrail')
            if len(en_reference) != 3:
                issues.append('en_real_mean_recall_unavailable')
            elif float(np.mean(en_measured)) < float(np.mean(en_reference)) + limits['min_en_real_gain']:
                issues.append('en_real_mean_gain_below_minimum')
        candidate['guardrails'] = issues
        candidate['eligible'] = not issues
        if not issues and metrics['weighted_f1'] > best:
            selected, best = candidate['name'], metrics['weighted_f1']
    return selected


def _check_bundle(bundle, weight, bias, name, teacher=False):
    x, g = _features(bundle['x'], bundle.get('lid') if teacher else None, name=name)
    if teacher and g is None:
        raise ValueError('Train requires continuous [N,256] LID teacher targets')
    if len(bundle['rows']) != len(x):
        raise ValueError(name + ' row/feature count mismatch')
    logits = np.asarray(bundle['logits'])
    if logits.shape != (len(x), 2) or not np.isfinite(logits).all():
        raise ValueError(name + ' needs finite original [N,2] logits')
    if not np.allclose(predict(x, weight, bias), logits, rtol=2e-5, atol=1e-3):
        raise ValueError(name + ' original logits do not replay from cached classifier input')


def _grid(cfg, name, default, upper=None):
    values = sorted(set(float(v) for v in cfg.get(name, default)))
    if not 1 <= len(values) <= (3 if name == 'alpha_grid' else 4):
        raise ValueError(name + ' must be a bounded, predeclared nonempty grid')
    if any(not np.isfinite(v) or v <= 0 or (upper is not None and v > upper) for v in values):
        raise ValueError(name + ' requires positive finite values' + (f' <= {upper}' if upper else ''))
    return values


def _take(bundle, indices):
    return dict(x=bundle['x'][indices], lid=bundle['lid'][indices],
                rows=[bundle['rows'][int(i)] for i in indices])


def _trial(x, tune_x, fit_rows, tune_rows, weight, bias, regularization, cfg, name):
    fitted = _solve(x, fit_rows, weight, bias, 'group_balanced', regularization, cfg, name)
    if fitted['status'] == 'converged':
        logits = predict(tune_x, **fitted['patch'], chunk_rows=int(cfg.get('fit_chunk_rows', 4096)))
        fitted['train_holdout_balanced_ce'] = balanced_cross_entropy(tune_rows, logits)
        fitted['metrics'] = evaluate(tune_rows, logits, train_proxy=True)
    fitted.pop('patch', None)
    return fitted


def _choose(trials):
    eligible = [trial for trial in trials if trial['status'] == 'converged'
                and np.isfinite(trial['train_holdout_balanced_ce'])]
    if not eligible:
        return None
    # Equal CE deterministically prefers less subtraction, larger ridge/anchor.
    return min(eligible, key=lambda t: (t['train_holdout_balanced_ce'], t.get('alpha', 0.),
                                       -t.get('ridge', 0.), -t['regularization']))


def save_fit_report(out, result):
    """Atomically save diagnostics suitable for the downloadable report archive.

    Deployment parameters stay in the in-memory result and the separate patch
    artifact. Recursively remove their containers and parameter fields, including
    any nested audit copies, without modifying the caller's result.
    """
    if out is None:
        return None
    private_keys = frozenset({'selected_patch', 'selected_state', 'patch',
                             'language_state', 'student_state', 'weight', 'bias',
                             'input_mean', 'input_scale', 'w1', 'b1', 'w2', 'b2',
                             'mapping'})

    def diagnostics(value):
        if isinstance(value, dict):
            return {key: diagnostics(item) for key, item in value.items() if key not in private_keys}
        if isinstance(value, (list, tuple)):
            return [diagnostics(item) for item in value]
        return value

    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'fit_report.json'
    temporary = directory / 'fit_report.json.tmp'
    temporary.write_text(json.dumps(diagnostics(result), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(target)
    return target


def fit_candidates(train_bundle, dev_bundle, weight, bias, cfg, out=None):
    """Select hyperparameters on grouped Train 80/20, then refit on full Train.

    Dev participates solely in the final two-candidate deployment guards. Every
    exported state is JSON-safe and replays the very same FP32 prediction path.
    """
    weight, bias = _numpy(weight).astype(np.float32), _numpy(bias).astype(np.float32)
    if weight.shape != (2, HIDDEN_DIM) or bias.shape != (2,) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError('Expected finite final Linear [2,512] weights and [2] bias')
    _check_bundle(train_bundle, weight, bias, 'Train', teacher=True)
    _check_bundle(dev_bundle, weight, bias, 'Dev')
    regularizations = _grid(cfg, 'lambda_grid', [.1, 1.])
    alphas = _grid(cfg, 'alpha_grid', [.25, .5, .75], upper=.75)
    ridges = _grid(cfg, 'ridge_grid', [.1, 1.])
    limits = _guardrail_config(cfg)
    chunk = int(cfg.get('fit_chunk_rows', 4096))
    if chunk < 1:
        raise ValueError('Positive fitting chunk size required')
    fit_idx, tune_idx, split = split_sources(train_bundle['rows'], float(cfg.get('holdout_fraction', .2)), int(cfg.get('seed', 3701)))
    fit, tune = _take(train_bundle, fit_idx), _take(train_bundle, tune_idx)
    baseline = evaluate(dev_bundle['rows'], dev_bundle['logits'])
    if not baseline['complete']:
        raise ValueError('Baseline Dev needs both classes in all three conditions')
    tune_baseline = evaluate(tune['rows'], np.asarray(train_bundle['logits'])[tune_idx], train_proxy=True)
    if not tune_baseline['complete']:
        raise ValueError('Train holdout requires actual Online and both noisy pools with both labels')
    score_files = {'baseline': _save_dev_scores(out, 'baseline', dev_bundle['rows'], dev_bundle['logits'])}
    candidates = []

    control_trials = [_trial(fit['x'], tune['x'], fit['rows'], tune['rows'], weight, bias,
                            lam, cfg, 'V3.7 head-only Train-holdout') for lam in regularizations]
    chosen = _choose(control_trials)
    if chosen is None:
        control = dict(status='rejected', reason='No head-only regularization converged on Train holdout')
    else:
        control = _solve(train_bundle['x'], train_bundle['rows'], weight, bias, 'group_balanced',
                         chosen['regularization'], cfg, 'V3.7 head-only all-Train refit')
        control['selected_regularization'] = chosen['regularization']
        if control['status'] == 'converged':
            control['patch'].update(language_state=None, student_state=None)
    control.update(name=CANDIDATES[0], tag=CANDIDATES[0], metrics=None, tuning=control_trials, alpha=0.)
    candidates.append(control)

    language_trials = []
    student_audit = {}
    try:
        print('[Phase] V3.7 fitting language student on grouped fitting Train only', flush=True)
        fit_student_state = fit_student(fit['x'], fit['lid'], fit['rows'], cfg)
        student_audit['holdout_fit'] = fit_student_state.get('diagnostics', {})
        fit_g = predict_student(fit['x'], fit_student_state)
        tune_g = predict_student(tune['x'], fit_student_state)
        _features(fit['x'], fit_g, name='Fitting student')
        _features(tune['x'], tune_g, name='Holdout student')
        fitting_stats, teacher_mean = _student_stats(fit_g, fit['lid'], fit['rows'], chunk_rows=chunk)
        holdout_stats, _ = _student_stats(tune_g, tune['lid'], tune['rows'], teacher_mean, chunk)
        student_audit.update(fitting_fidelity=fitting_stats, holdout_fidelity=holdout_stats)
        if fitting_stats['effectively_constant'] or holdout_stats['effectively_constant']:
            raise FloatingPointError('Language student predictions are effectively constant on grouped Train fitting/holdout')
        moments = _language_moments(fit['x'], fit_g, fit['rows'], chunk)
        for ridge in ridges:
            language = _state_from_moments(moments, ridge, alphas[0])
            fit_projection, tune_projection = _project(fit_g, language, chunk), _project(tune_g, language, chunk)
            for alpha in alphas:
                fit_x = fit['x'] - np.float32(alpha) * fit_projection
                tune_x = tune['x'] - np.float32(alpha) * tune_projection
                for lam in regularizations:
                    trial = _trial(fit_x, tune_x, fit['rows'], tune['rows'], weight, bias, lam, cfg,
                                   f'V3.7 language Train-holdout alpha={alpha}, ridge={ridge}')
                    trial.update(alpha=alpha, ridge=ridge)
                    language_trials.append(trial)
                del fit_x, tune_x
            del fit_projection, tune_projection
        chosen = _choose(language_trials)
        del moments, fit_g, tune_g, fit_student_state
        if chosen is None:
            language_candidate = dict(status='rejected', reason='No language grid setting converged on Train holdout')
        else:
            print('[Phase] V3.7 refitting language student and real-only map on full Train', flush=True)
            student_state = fit_student(train_bundle['x'], train_bundle['lid'], train_bundle['rows'], cfg)
            student_audit['all_train_refit'] = student_state.get('diagnostics', {})
            train_g = predict_student(train_bundle['x'], student_state)
            _features(train_bundle['x'], train_g, name='Full Train student')
            full_stats, _ = _student_stats(train_g, train_bundle['lid'], train_bundle['rows'], chunk_rows=chunk)
            student_audit['all_train_fidelity'] = full_stats
            if full_stats['effectively_constant']:
                raise FloatingPointError('Full Train refitted language student predictions are effectively constant')
            language = fit_language_state(train_bundle['x'], train_g, train_bundle['rows'],
                                          chosen['ridge'], chosen['alpha'], chunk)
            train_x = transform_features(train_bundle['x'], train_g, language, chunk)
            language_candidate = _solve(train_x, train_bundle['rows'], weight, bias, 'group_balanced',
                                        chosen['regularization'], cfg, 'V3.7 language all-Train refit')
            language_candidate.update(selected_regularization=chosen['regularization'],
                                      selected_alpha=chosen['alpha'], selected_ridge=chosen['ridge'])
            if language_candidate['status'] == 'converged':
                language_candidate['patch'].update(language_state=language, student_state=student_state)
            del train_x, train_g
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        language_candidate = dict(status='rejected', reason='Language fitting numerical failure: ' + str(exc))
    language_candidate.update(name=CANDIDATES[1], tag=CANDIDATES[1], metrics=None, tuning=language_trials)
    candidates.append(language_candidate)

    for candidate in candidates:
        if candidate['status'] == 'converged':
            try:
                logits = logits_from_state(dev_bundle['x'], state=candidate['patch'], chunk_rows=chunk)
                candidate['metrics'] = evaluate(dev_bundle['rows'], logits)
                score_files[candidate['name']] = _save_dev_scores(out, candidate['name'], dev_bundle['rows'], logits)
            except FloatingPointError as exc:
                candidate.update(status='rejected', reason=str(exc), metrics=None)
                candidate.pop('patch', None)
    selected = select_candidate(baseline, candidates, limits)
    original = dict(weight=weight.tolist(), bias=bias.tolist(), language_state=None, student_state=None)
    patch = next((c['patch'] for c in candidates if c['name'] == selected), original)
    counts = Counter(r['language'] + '/' + str(r['label']) for r in sources(train_bundle['rows']).values())
    outcome = {'baseline': 'Neither candidate passed all fixed Dev guards; exact baseline retained.',
               'head_only_control': 'Head-only control selected. This is not evidence of a language-debias improvement.',
               'language_debias': 'Language candidate selected by fixed Dev guards; selection does not establish a causal language mechanism.'}[selected]
    report = dict(schema='w2v_v37_language_distilled_classifier_v1', baseline=baseline, candidates=candidates,
                  selected=selected, selected_patch=patch, split=split, train_source_counts=dict(counts),
                  train_holdout_baseline=tune_baseline, student_audit=student_audit,
                  dev_score_files={k: v for k, v in score_files.items() if v is not None},
                  effective_fit_config=dict(lambda_grid=regularizations, alpha_grid=alphas, ridge_grid=ridges,
                  holdout_fraction=float(cfg.get('holdout_fraction', .2)), seed=int(cfg.get('seed', 3701)),
                  max_iterations=int(cfg.get('max_iterations', 100)), fit_chunk_rows=chunk, **limits),
                  language_fit_definition='Centered standardized weighted ridge from internal student LID (256) to final hidden (512), using genuine Train only; equal EN/ZH source mass; ordinary/noisy mass 0.5/0.5.',
                  regularization_definition='weighted mean CE + lambda/2 * (squared margin-weight displacement + squared margin-bias displacement), anchored to the original final Linear',
                  hyperparameter_selection='Minimum group/source-balanced CE on grouped Train holdout; no Dev fitting or tuning; full Train student/map/head refit afterwards.',
                  mean_preservation='The centered correction preserves the balanced genuine-Train hidden mean, up to FP32 rounding; no Dev/test mean is fitted.',
                  method_note='Teacher-distilled nonlinear approximation inspired by language orthogonalization; not a reproduction of the paper or an external-teacher inference ensemble.',
                  selection_note='Exactly two predeclared candidates on fixed Dev. Local selection has model-selection bias and is not an official score.',
                  outcome=outcome)
    save_fit_report(out, report)
    print('[Phase] V3.7 classifier selection: ' + selected + '. ' + outcome, flush=True)
    return report
