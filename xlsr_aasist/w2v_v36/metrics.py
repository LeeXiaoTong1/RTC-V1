"""Fixed-threshold, pooled Online/Seen/Heldout metrics for frozen-head trials."""
from collections import defaultdict

import numpy as np


def _auc(labels, scores):
    """Mann-Whitney AUC; fake is positive and tied scores receive half credit."""
    positive = np.asarray(labels) == 0
    npos = int(positive.sum())
    nneg = len(positive) - npos
    if not npos or not nneg:
        return None
    order = np.argsort(scores, kind='stable')
    values = np.asarray(scores)[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[start:end] = .5 * (start + 1 + end)
    rank_sum = ranks[positive[order]].sum()
    return float((rank_sum - npos * (npos + 1) / 2) / (npos * nneg))


def _group(labels, margins):
    labels = np.asarray(labels, dtype=np.int64)
    pred = np.where(margins >= 0., 0, 1)
    counts = np.bincount(labels * 2 + pred, minlength=4).reshape(2, 2)
    support = counts.sum(1)
    denom = support + counts.sum(0)
    f1 = np.divide(2 * np.diag(counts), denom, out=np.zeros(2, dtype=float), where=denom > 0)
    recall = [float(counts[i, i] / support[i]) if support[i] else None for i in range(2)]
    return dict(count=int(len(labels)), confusion=counts.tolist(), class_counts=support.tolist(),
                macro_f1=float(f1.mean()), f1=f1.tolist(), recall=recall,
                auc=_auc(labels, margins))


def evaluate(rows, logits, *, train_proxy=False):
    """Both classes must occur in Online and each noisy pool to score a candidate.

    No missing Online recordings are replaced by Offline. Train proxies map the
    existing noisy_a/noisy_b views to the two pools; these are *not* unseen RTC.
    """
    logits = np.asarray(logits)
    if logits.shape != (len(rows), 2) or not np.isfinite(logits).all():
        raise ValueError('Expected finite [rows, 2] logits')
    buckets = defaultdict(list)
    aliases = {'clean': 'online'}
    if train_proxy:
        aliases.update(noisy_a='seen', noisy_b='heldout')
    for i, row in enumerate(rows):
        condition = aliases.get(row['condition'], row['condition'])
        language, label = row['language'], row['label']
        if condition not in ('offline', 'online', 'seen', 'heldout'):
            raise ValueError('Unsupported metric condition: ' + str(condition))
        if language not in ('en', 'zh') or label not in (0, 1):
            raise ValueError('Expected EN/ZH and fake=0/real=1')
        buckets[condition].append(i)
        buckets[condition + '/' + language].append(i)
    margin = logits[:, 0].astype(np.float64) - logits[:, 1]
    groups = {name: _group([rows[i]['label'] for i in indices], margin[indices])
              for name, indices in buckets.items()}
    complete = all(name in groups and min(groups[name]['class_counts']) > 0
                   for name in ('online', 'seen', 'heldout'))
    values = {key: groups[name]['macro_f1'] if name in groups else None
              for key, name in [('clean_f1', 'online'), ('seen_f1', 'seen'), ('heldout_f1', 'heldout')]}
    noisy = .5 * (values['seen_f1'] + values['heldout_f1']) if complete else None
    weighted = .3 * values['clean_f1'] + .7 * noisy if complete else None
    return dict(**values, noisy_f1=noisy, weighted_f1=weighted, complete=complete, groups=groups,
                threshold=.5, metric_schema='v36_full_online_pooled_conditions_v1',
                metric_note=('Train-source holdout proxy, encoder previously saw official Train; not an '
                             'independent unseen-source test. Noisy pools use training conditions.' if train_proxy else
                             'Fixed full Dev, pooled Online Clean and equal Seen/Heldout Noisy, '
                             'weighted 0.3/0.7. Local model selection, not an official platform score.'))


def select_candidate(baseline, candidates, cfg):
    """Conservative Dev selection; baseline always remains the deployment fallback."""
    if not baseline.get('complete'):
        raise ValueError('Baseline Dev needs both classes in all three conditions')
    gain = float(cfg.get('min_gain', .002))
    clean_drop = float(cfg.get('max_clean_drop', .001))
    fake_drop = float(cfg.get('max_noisy_fake_recall_drop', .005))
    if any(not np.isfinite(v) or v < 0 for v in (gain, clean_drop, fake_drop)):
        raise ValueError('Selection tolerances must be finite and nonnegative')
    selected, best = 'baseline', baseline['weighted_f1']
    for candidate in candidates:
        issues = []
        metrics = candidate.get('metrics')
        if candidate.get('status') != 'converged':
            issues.append('solver_not_converged')
        if not metrics or not metrics.get('complete'):
            issues.append('incomplete_dev')
        else:
            if metrics['weighted_f1'] < baseline['weighted_f1'] + gain:
                issues.append('weighted_gain_below_minimum')
            if metrics['noisy_f1'] < baseline['noisy_f1']:
                issues.append('noisy_below_baseline')
            if metrics['clean_f1'] < baseline['clean_f1'] - clean_drop:
                issues.append('clean_below_guardrail')
            for name in ('seen/en', 'heldout/en'):
                reference = baseline['groups'].get(name, {}).get('recall', [None])[0]
                measured = metrics['groups'].get(name, {}).get('recall', [None])[0]
                if reference is None or measured is None:
                    issues.append(name + '_fake_recall_unavailable')
                elif measured < reference - fake_drop:
                    issues.append(name + '_fake_recall_below_guardrail')
        candidate['guardrails'] = issues
        candidate['eligible'] = not issues
        if not issues and metrics['weighted_f1'] > best:
            selected, best = candidate['name'], metrics['weighted_f1']
    return selected
