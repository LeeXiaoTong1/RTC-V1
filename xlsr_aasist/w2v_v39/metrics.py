"""Separate calibration shifts from rank improvement; protect both languages."""
from collections import defaultdict

import numpy as np

from w2v_v36.metrics import evaluate

GROUPS = tuple(c + '/' + lang for c in ('online', 'seen', 'heldout') for lang in ('en', 'zh'))


def matched_real_recall(labels, margins, target=.99):
    """Largest deterministic threshold retaining >= target fake recall, ties intact.

    This is a ROC diagnostic only. It NEVER supplies a deployment threshold.
    """
    labels, margins = np.asarray(labels), np.asarray(margins, dtype=np.float64)
    fake, real = np.sort(margins[labels == 0]), margins[labels == 1]
    if not len(fake) or not len(real) or not 0 < target <= 1 or not np.isfinite(margins).all():
        raise ValueError('Finite scores and both classes required for matched recall')
    allowed_misses = int(np.floor((1. - target) * len(fake) + 1e-9))
    threshold = fake[min(allowed_misses, len(fake) - 1)]
    return dict(real_recall=float(np.mean(real < threshold)),
                actual_fake_recall=float(np.mean(fake >= threshold)), target_fake_recall=float(target))


def measure(rows, logits, *, train_proxy=False, target=.99):
    result = evaluate(rows, logits, train_proxy=train_proxy)
    groups = defaultdict(list)
    aliases = {'noisy_a': 'seen', 'noisy_b': 'heldout'} if train_proxy else {}
    for i, row in enumerate(rows):
        condition = aliases.get(row['condition'], row['condition'])
        groups[condition + '/' + row['language']].append(i)
    margins = np.asarray(logits, dtype=np.float64)[:, 0] - np.asarray(logits)[:, 1]
    result['matched'] = {}
    for name, ids in groups.items():
        labels = [rows[i]['label'] for i in ids]
        if set(labels) == {0, 1}:
            result['matched'][name] = matched_real_recall(labels, margins[ids], target)
    return result


def selection(baseline, candidates, cfg):
    by_name = {c['name']: c for c in candidates}
    best, selected = baseline['weighted_f1'], 'baseline'
    if not baseline['complete'] or any(g not in baseline['matched'] for g in GROUPS):
        raise ValueError('Fixed Dev requires both classes in EN/ZH and all three conditions')
    for item in candidates:
        reasons, attribution, value = [], [], item.get('metrics')
        if item.get('status') != 'fitted' or not value or not value.get('complete'):
            reasons.append(item.get('reason', 'fit_or_replay_failed'))
        else:
            if value['weighted_f1'] < baseline['weighted_f1'] + cfg['min_gain']:
                reasons.append('weighted_gain_below_minimum')
            if value['noisy_f1'] < baseline['noisy_f1']:
                reasons.append('noisy_below_baseline')
            if value['clean_f1'] < baseline['clean_f1'] - cfg['max_clean_drop']:
                reasons.append('clean_below_guard')
            for name in GROUPS:
                base, current = baseline['groups'].get(name), value['groups'].get(name)
                if not current or min(current['class_counts']) < 1:
                    reasons.append(name + '_incomplete')
                    continue
                if current['recall'][0] < base['recall'][0] - cfg['max_fake_drop']:
                    reasons.append(name + '_fake_recall_drop')
                if current['recall'][1] < base['recall'][1] - cfg['max_real_drop']:
                    reasons.append(name + '_real_recall_drop')
                if current['auc'] < base['auc'] - cfg['max_auc_drop']:
                    reasons.append(name + '_auc_drop')
                a = baseline['matched'][name]['real_recall']
                b = value.get('matched', {}).get(name, {}).get('real_recall', -1.)
                if b < a - cfg['max_matched_real_drop']:
                    reasons.append(name + '_matched_fake_recall_real_drop')
            if item['name'] == 'language_residual' and all(
                    g in value['matched'] and g in value['groups'] and min(value['groups'][g]['class_counts']) > 0
                    for g in GROUPS):
                if not item.get('student_qualified'):
                    reasons.append('student_not_qualified')
                en = ('online/en', 'seen/en', 'heldout/en')
                gain = np.mean([value['groups'][g]['recall'][1] - baseline['groups'][g]['recall'][1] for g in en])
                if gain < cfg['min_en_real_gain']:
                    reasons.append('en_real_gain_below_minimum')
                for control in ('calibration', 'residual_control'):
                    other = by_name.get(control, {})
                    if other.get('status') != 'fitted' or not other.get('metrics'):
                        attribution.append(control + '_unavailable_for_attribution')
                    elif value['weighted_f1'] < other['metrics']['weighted_f1'] + cfg['min_language_control_gain']:
                        attribution.append('no_weighted_gain_over_' + control)
                noisy = ('seen/en', 'heldout/en')
                rank_gain = np.mean([value['matched'][g]['real_recall'] - baseline['matched'][g]['real_recall'] for g in noisy])
                if rank_gain < cfg['min_ranking_gain']:
                    attribution.append('no_en_noisy_rank_gain_at_matched_fake_recall')
        item['guardrails'], item['eligible'] = reasons, not reasons
        item['language_attribution_reasons'] = attribution
        item['language_specific_evidence'] = (item['name'] == 'language_residual'
                                              and not reasons and not attribution)
        if not reasons and value['weighted_f1'] > best:
            best, selected = value['weighted_f1'], item['name']
    return selected


def change_audit(rows, original, candidate):
    old, new = np.asarray(original).argmax(1), np.asarray(candidate).argmax(1)
    groups = defaultdict(lambda: dict(real_rescued=0, fake_rescued=0, new_real_errors=0, new_fake_errors=0))
    for row, before, after in zip(rows, old, new):
        group = groups[row['condition'] + '/' + row['language']]
        if before == after:
            continue
        label = row['label']
        name = ('fake' if label == 0 else 'real') + '_rescued' if after == label else 'new_' + ('fake' if label == 0 else 'real') + '_errors'
        group[name] += 1
    return dict(groups)
