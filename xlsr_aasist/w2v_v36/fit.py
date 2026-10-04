"""Deterministic source-balanced linear-head adaptation on cached FP32 vectors.

Only the fake-minus-real margin is optimized. Shared logits are preserved, with
the final two-class weights rounded to FP32 for deployment. No Dev sample enters
the optimizer, source split, or regularization selection.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from live_progress import Phase, phase

from .metrics import evaluate, select_candidate


ARMS = ('class_balanced', 'group_balanced')
CONDITIONS = ('offline', 'online', 'noisy_a', 'noisy_b')


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def sources(rows):
    result, same_group = {}, {}
    for index, row in enumerate(rows):
        source, condition = row.get('source_id'), row.get('condition')
        label, language = row.get('label'), row.get('language')
        group = row.get('group_id', source)
        if not isinstance(source, str) or not source or condition not in CONDITIONS:
            raise ValueError('Each Train row needs canonical source_id and a supported condition')
        if not isinstance(group, str) or not group or label not in (0, 1) or language not in ('en', 'zh'):
            raise ValueError('Each Train row needs group_id, EN/ZH and fake=0/real=1')
        if row.get('view', 'full') != 'full':
            raise ValueError('Only one full recording per source/condition is accepted')
        identity = (language, label)
        if group in same_group and same_group[group] != identity:
            raise ValueError('A repeated original-audio group has conflicting language/label')
        same_group[group] = identity
        item = result.setdefault(source, dict(language=language, label=label, group_id=group, indices={}))
        if (item['language'], item['label'], item['group_id']) != (language, label, group):
            raise ValueError('Source views disagree on identity')
        if condition in item['indices']:
            raise ValueError('Duplicate full source condition: ' + source + '/' + condition)
        item['indices'][condition] = index
    if not result:
        raise ValueError('Empty Train sources')
    for item in result.values():
        if not {'offline', 'noisy_a', 'noisy_b'} <= set(item['indices']):
            raise ValueError('Each source requires Offline and both full noisy versions')
    return result


def source_coefficients(rows, arm):
    """One source budget: ordinary=0.5, noisy_a=noisy_b=0.25; sum=1."""
    if arm not in ARMS:
        raise ValueError('Unknown balancing arm')
    records = sources(rows)
    def key(item):
        return str(item['label']) if arm == 'class_balanced' else item['language'] + '/' + str(item['label'])
    counts = Counter(key(item) for item in records.values())
    expected = {'0', '1'} if arm == 'class_balanced' else {'en/0', 'en/1', 'zh/0', 'zh/1'}
    if set(counts) != expected:
        raise ValueError('All balance groups need at least one source')
    coefficients = np.zeros(len(rows), dtype=np.float64)
    for item in records.values():
        budget = 1. / (len(expected) * counts[key(item)])
        indices = item['indices']
        ordinary_count = 1 + int('online' in indices)
        for condition, index in indices.items():
            coefficients[index] = budget * (.25 if condition.startswith('noisy') else .5 / ordinary_count)
    if not np.isclose(coefficients.sum(), 1., rtol=0, atol=1e-12):
        raise AssertionError('Source coefficient mass must be one')
    return coefficients


def split_sources(rows, fraction=.2, seed=3601):
    """Group by original audio identity, then stratify EN/ZH x real/fake."""
    if not 0 < fraction < .5:
        raise ValueError('Train holdout fraction must lie between 0 and 0.5')
    records = sources(rows)
    strata = defaultdict(set)
    for item in records.values():
        strata[item['language'] + '/' + str(item['label'])].add(item['group_id'])
    if set(strata) != {'en/0', 'en/1', 'zh/0', 'zh/1'} or any(len(g) < 2 for g in strata.values()):
        raise ValueError('Each language/class needs at least two independent source groups')
    heldout = set()
    for name, groups in sorted(strata.items()):
        ordered = sorted(groups, key=lambda group: hashlib.sha256(f'{seed}\0{name}\0{group}'.encode()).hexdigest())
        count = max(1, min(len(ordered) - 1, int(round(len(ordered) * fraction))))
        heldout.update(ordered[:count])
    train_indices, tune_indices = [], []
    for item in records.values():
        (tune_indices if item['group_id'] in heldout else train_indices).extend(item['indices'].values())
    train_indices.sort(); tune_indices.sort()
    train_groups = {r['group_id'] for r in records.values()} - heldout
    audit = dict(seed=int(seed), holdout_fraction=float(fraction),
                 fit_source_count=sum(r['group_id'] not in heldout for r in records.values()),
                 tune_source_count=sum(r['group_id'] in heldout for r in records.values()),
                 fit_group_count=len(train_groups), tune_group_count=len(heldout),
                 fit_groups_sha256=hashlib.sha256('\n'.join(sorted(train_groups)).encode()).hexdigest(),
                 tune_groups_sha256=hashlib.sha256('\n'.join(sorted(heldout)).encode()).hexdigest(),
                 group_overlap=0, encoder_previously_saw_train=True,
                 note='Classifier holdout only; not an unseen-source generalization test for the encoder.')
    return np.asarray(train_indices), np.asarray(tune_indices), audit


def margin_objective(delta, x, labels, coefficients, anchor, regularization, chunk_rows=4096):
    """CE + lambda/2 * ||margin_parameters - original_margin||_2^2.

    The intercept receives the same penalty as each unscaled feature coefficient.
    Features stay FP32 on disk; a single chunk is promoted to FP64 for optimization.
    """
    parameters = anchor + delta
    gradient = regularization * delta.copy()
    value = .5 * regularization * float(delta @ delta)
    for start in range(0, len(labels), chunk_rows):
        stop = min(start + chunk_rows, len(labels))
        chunk = np.asarray(x[start:stop], dtype=np.float64)
        margin = chunk @ parameters[:-1] + parameters[-1]
        target = 1. - 2. * np.asarray(labels[start:stop], dtype=np.float64)
        coeff = coefficients[start:stop]
        value += float(coeff @ np.logaddexp(0., -target * margin))
        derivative = -coeff * target * expit(-target * margin)
        gradient[:-1] += chunk.T @ derivative
        gradient[-1] += derivative.sum()
    return value, gradient


def _solve(x, rows, weight, bias, arm, regularization, cfg, label):
    coefficients = source_coefficients(rows, arm)
    anchor = np.r_[weight[0].astype(np.float64) - weight[1], float(bias[0]) - float(bias[1])]
    labels = np.asarray([r['label'] for r in rows], dtype=np.int64)
    chunk = int(cfg.get('fit_chunk_rows', 4096))
    iterations = int(cfg.get('max_iterations', 100))
    if chunk < 1 or iterations < 1:
        raise ValueError('Positive optimizer iteration and chunk limits required')
    trace = []
    progress = Phase('V3.6 ' + label + ' solver (maximum iterations)', iterations)
    def objective(delta):
        value, grad = margin_objective(delta, x, labels, coefficients, anchor, regularization, chunk)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError('Nonfinite full-batch objective/gradient')
        return value, grad
    def callback(delta):
        # The optimizer already performed the full pass: progress adds no extra pass.
        trace.append(len(trace) + 1)
        progress.update(len(trace))
        if len(trace) == 1 or len(trace) % 10 == 0:
            print(f'[Phase] V3.6 {label}: solver iteration {len(trace)}/{iterations}', flush=True)
    print(f'[Phase] V3.6 {label}: {len(rows)} cached vectors, lambda={regularization}', flush=True)
    try:
        answer = minimize(objective, np.zeros(len(anchor)), method='L-BFGS-B', jac=True, callback=callback,
                          options=dict(maxiter=iterations, maxls=30, ftol=1e-10, gtol=1e-7))
    except FloatingPointError as exc:
        phase('V3.6 ' + label + ' rejected')
        print(f'[Phase] V3.6 {label}: rejected, {exc}', flush=True)
        return dict(status='rejected', reason=str(exc), regularization=float(regularization))
    if not answer.success or not np.isfinite(answer.x).all() or not np.isfinite(answer.fun):
        phase('V3.6 ' + label + ' rejected')
        print(f'[Phase] V3.6 {label}: rejected, {answer.message}', flush=True)
        return dict(status='rejected', reason=str(answer.message), iterations=int(answer.nit),
                    regularization=float(regularization))
    margin = anchor + answer.x
    shared_w = .5 * (weight[0].astype(np.float64) + weight[1])
    shared_b = .5 * (float(bias[0]) + float(bias[1]))
    fitted_w = np.stack([shared_w + .5 * margin[:-1], shared_w - .5 * margin[:-1]]).astype(np.float32)
    fitted_b = np.asarray([shared_b + .5 * margin[-1], shared_b - .5 * margin[-1]], dtype=np.float32)
    print(f'[Phase] V3.6 {label}: converged in {answer.nit} iterations, objective={answer.fun:.7f}', flush=True)
    phase('V3.6 ' + label + f' converged ({answer.nit} iterations)')
    return dict(status='converged', iterations=int(answer.nit), objective=float(answer.fun),
                margin_delta_l2=float(np.linalg.norm(answer.x)), regularization=float(regularization),
                patch=dict(weight=fitted_w.tolist(), bias=fitted_b.tolist()))


def predict(x, weight, bias, chunk_rows=4096):
    weight, bias = np.asarray(weight, dtype=np.float32), np.asarray(bias, dtype=np.float32)
    result = np.empty((len(x), 2), dtype=np.float32)
    for start in range(0, len(x), chunk_rows):
        stop = min(start + chunk_rows, len(x))
        result[start:stop] = np.asarray(x[start:stop], dtype=np.float32) @ weight.T + bias
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite FP32 deployment logits')
    return result


def _check_bundle(bundle, weight, bias, name):
    x, rows, logits = bundle['x'], bundle['rows'], bundle['logits']
    if x.shape != (len(rows), weight.shape[1]) or x.dtype != np.float32 or len(rows) == 0:
        raise ValueError(name + ' feature cache must be nonempty FP32 [N,D]')
    if np.asarray(logits).shape != (len(rows), 2):
        raise ValueError(name + ' needs original [N,2] logits')
    replay = predict(x, weight, bias)
    if not np.allclose(replay, logits, rtol=2e-5, atol=1e-3):
        raise ValueError(name + ' original logits do not replay from cached classifier input')


def _take(bundle, indices):
    return dict(x=bundle['x'][indices], rows=[bundle['rows'][int(i)] for i in indices])


def _save_dev_scores(out, tag, rows, logits):
    """Compact, pickle-free diagnostics; stores already computed logits once."""
    if out is None:
        return None
    directory = Path(out); directory.mkdir(parents=True, exist_ok=True)
    filename = 'dev_scores_' + tag + '.npz'
    target = directory / filename
    temporary = target.with_suffix(target.suffix + '.tmp')
    # Explicit Unicode dtypes prevent object arrays even when identifiers vary.
    source_ids = [str(r.get('source_id', r.get('id', ''))) for r in rows]
    ids = [str(r.get('id', r['condition'] + ':' + source_ids[i])) for i, r in enumerate(rows)]
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, ids=np.asarray(ids, dtype=np.str_),
                            source_ids=np.asarray(source_ids, dtype=np.str_),
                            language=np.asarray([r['language'] for r in rows], dtype=np.str_),
                            condition=np.asarray([r['condition'] for r in rows], dtype=np.str_),
                            labels=np.asarray([r['label'] for r in rows], dtype=np.int64),
                            logits=np.asarray(logits, dtype=np.float32))
    temporary.replace(target)
    return filename


def fit_candidates(train_bundle, dev_bundle, weight, bias, cfg, out):
    """Fit two predeclared arms; select lambda on Train and deploy using fixed Dev.

    Returns JSON-safe baseline/candidates/selected/selected_patch. A rejected or
    degraded candidate never replaces the exact supplied baseline weights.
    """
    weight, bias = _numpy(weight).astype(np.float32), _numpy(bias).astype(np.float32)
    if weight.ndim != 2 or weight.shape[0] != 2 or bias.shape != (2,) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError('Expected finite two-class Linear weights/bias')
    _check_bundle(train_bundle, weight, bias, 'Train')
    _check_bundle(dev_bundle, weight, bias, 'Dev')
    regularizations = sorted(set(float(v) for v in cfg.get('lambda_grid', [.1, 1.])), reverse=True)
    if not 1 <= len(regularizations) <= 4 or any(not np.isfinite(v) or v <= 0 for v in regularizations):
        raise ValueError('Use one to four predetermined positive regularization values')
    fit_idx, tune_idx, split = split_sources(train_bundle['rows'], float(cfg.get('holdout_fraction', .2)), int(cfg.get('seed', 3601)))
    fit, tune = _take(train_bundle, fit_idx), _take(train_bundle, tune_idx)
    baseline = evaluate(dev_bundle['rows'], dev_bundle['logits'])
    score_files = {'baseline': _save_dev_scores(out, 'baseline', dev_bundle['rows'], dev_bundle['logits'])}
    tune_baseline = evaluate(tune['rows'], np.asarray(train_bundle['logits'])[tune_idx], train_proxy=True)
    if not tune_baseline['complete']:
        raise ValueError('Train holdout cannot select lambda: actual Online and both noisy pools need both labels')
    candidates = []
    for arm in ARMS:
        trials = []
        for regularization in regularizations:
            fitted = _solve(fit['x'], fit['rows'], weight, bias, arm, regularization, cfg, arm + ' Train-holdout')
            if fitted['status'] == 'converged':
                z = predict(tune['x'], **fitted['patch'])
                fitted['metrics'] = evaluate(tune['rows'], z, train_proxy=True)
            # No large trial parameter arrays are needed after the Train-only choice.
            fitted.pop('patch', None)
            trials.append(fitted)
        eligible = [trial for trial in trials if trial['status'] == 'converged' and trial['metrics']['complete']]
        if not eligible:
            candidates.append(dict(name=arm, tag=arm, status='rejected', metrics=None,
                                   reason='No regularization converged on Train holdout', tuning=trials))
            continue
        # Ties deterministically prefer the stronger anchor; Dev has no role here.
        chosen = max(eligible, key=lambda trial: (trial['metrics']['weighted_f1'], trial['regularization']))
        fitted = _solve(train_bundle['x'], train_bundle['rows'], weight, bias, arm,
                        chosen['regularization'], cfg, arm + ' all-Train refit')
        fitted.update(name=arm, tag=arm, metrics=None, tuning=trials, selected_regularization=chosen['regularization'])
        if fitted['status'] == 'converged':
            logits = predict(dev_bundle['x'], **fitted['patch'])
            fitted['metrics'] = evaluate(dev_bundle['rows'], logits)
            score_files[arm] = _save_dev_scores(out, arm, dev_bundle['rows'], logits)
        candidates.append(fitted)
    selected = select_candidate(baseline, candidates, cfg)
    original_patch = dict(weight=weight.tolist(), bias=bias.tolist())
    patch = next((c['patch'] for c in candidates if c['name'] == selected), original_patch)
    counts = Counter(r['language'] + '/' + str(r['label']) for r in sources(train_bundle['rows']).values())
    report = dict(schema='w2v_v36_frozen_classifier_v1', baseline=baseline, candidates=candidates,
                  selected=selected, selected_patch=patch, split=split, train_source_counts=dict(counts),
                  dev_score_files={name: filename for name, filename in score_files.items() if filename is not None},
                  train_holdout_baseline=tune_baseline, effective_fit_config=dict(lambda_grid=regularizations,
                  max_iterations=int(cfg.get('max_iterations', 100)), fit_chunk_rows=int(cfg.get('fit_chunk_rows', 4096)),
                  min_gain=float(cfg.get('min_gain', .002)), max_clean_drop=float(cfg.get('max_clean_drop', .001)),
                  max_noisy_fake_recall_drop=float(cfg.get('max_noisy_fake_recall_drop', .005))),
                  regularization_definition='weighted mean CE + lambda/2 * (squared margin-weight displacement + squared margin-bias displacement)',
                  selection_note='Two predeclared arms selected on this fixed Dev. Scores have model-selection bias; no guarantee of official Weighted 97.')
    if out is not None:
        directory = Path(out); directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / 'fit_report.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(directory / 'fit_report.json')
    print('[Phase] V3.6 classifier selection: ' + selected, flush=True)
    return report
