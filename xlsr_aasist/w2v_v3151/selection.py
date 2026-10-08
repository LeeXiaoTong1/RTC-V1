"""Separate historical Weighted from broader guarded selection; retain the parent."""
import math

import numpy as np

from w2v_v39.metrics import GROUPS


DEFAULTS = dict(max_weighted_drop=.001, max_clean_drop=.003, max_noisy_drop=.001,
    max_fake_drop=.005, max_real_drop=.015, max_auc_drop=.002, max_matched_real_drop=.02,
    max_panel_drop=.005, max_panel_auc_drop=.005, max_panel_family_drop=.03,
    max_panel_family_auc_drop=.03, panel_family_stable_tolerance=.01,
    min_panel_f1_gain=.002, min_panel_auc_gain=.001,
    min_panel_rank_gain=.01, min_fixed_rank_gain=.005, min_fixed_auc_gain=.0005,
    min_panel_language_class_support=25, max_panel_language_recall_drop=.04,
    max_panel_language_auc_drop=.01)


def acceptance(baseline, metrics, cfg, base_panel=None, panel=None):
    """Safety on historical Dev + broader improvement + evidence beyond a shift.

    Fractions, not percentages. The bounded panel is noisy: per-family rank
    improvements are not demanded, and matched99 is complemented by AUC.
    This is a deterministic selection heuristic, not a significance test.
    """
    limits = dict(DEFAULTS)
    limits.update({key: cfg[key] for key in DEFAULTS if key in cfg})
    if any(not np.isfinite(x) or x < 0 for x in limits.values()):
        raise ValueError('Selection tolerances must be finite and nonnegative')
    if not baseline.get('complete') or not metrics.get('complete'):
        return False, ['incomplete_fixed_dev']
    reasons = []
    for name, tolerance in (('weighted_f1', 'max_weighted_drop'), ('clean_f1', 'max_clean_drop'),
                            ('noisy_f1', 'max_noisy_drop')):
        if not np.isfinite(metrics.get(name, float('nan'))) or metrics[name] < baseline[name]-limits[tolerance]:
            reasons.append(name+'_below_guard')
    fixed_auc, fixed_rank = [], []
    for name in GROUPS:
        a, b = baseline['groups'].get(name), metrics['groups'].get(name)
        ma, mb = baseline.get('matched', {}).get(name), metrics.get('matched', {}).get(name)
        if not a or not b or min(b.get('class_counts', [0])) < 1 or not ma or not mb:
            reasons.append(name+'_incomplete')
            continue
        required = (b['recall'][0], b['recall'][1], b['auc'], mb['real_recall'])
        if any(x is None or not np.isfinite(x) for x in required):
            reasons.append(name+'_nonfinite')
            continue
        for label in (0, 1):
            if b['recall'][label] < a['recall'][label]-limits['max_fake_drop' if label == 0 else 'max_real_drop']:
                reasons.append(name+('_fake_recall_drop' if label == 0 else '_real_recall_drop'))
        if b['auc'] < a['auc']-limits['max_auc_drop']:
            reasons.append(name+'_auc_drop')
        if mb['real_recall'] < ma['real_recall']-limits['max_matched_real_drop']:
            reasons.append(name+'_matched_recall_drop')
        if name in ('seen/en', 'heldout/en'):
            fixed_auc.append(b['auc']-a['auc'])
            fixed_rank.append(mb['real_recall']-ma['real_recall'])
    if base_panel is None or panel is None:
        reasons.append('tune_panel_not_measured')
        return False, reasons
    if base_panel.get('partition') != 'tune' or panel.get('partition') != 'tune':
        raise ValueError('The final-only audit partition cannot be used for model selection')
    if not base_panel.get('complete') or not panel.get('complete'):
        reasons.append('tune_panel_incomplete')
        return False, reasons
    if (base_panel['views'] != panel['views'] or set(base_panel['families']) != set(panel['families'])):
        raise ValueError('Baseline/candidate panel support changed')
    if any(not np.isfinite(panel.get(key, float('nan'))) for key in ('macro_f1', 'macro_auc', 'macro_matched_real')):
        return False, reasons+['tune_panel_nonfinite']
    f1_gain = panel['macro_f1']-base_panel['macro_f1']
    auc_gain = panel['macro_auc']-base_panel['macro_auc']
    rank_gain = panel['macro_matched_real']-base_panel['macro_matched_real']
    if f1_gain < -limits['max_panel_drop']:
        reasons.append('tune_panel_f1_regression')
    if auc_gain < -limits['max_panel_auc_drop']:
        reasons.append('tune_panel_auc_regression')
    stable = 0
    for name, b in panel['families'].items():
        a = base_panel['families'][name]
        if a['class_counts'] != b['class_counts']:
            raise ValueError('Mechanism class support changed')
        if b['macro_f1'] < a['macro_f1']-limits['max_panel_family_drop']:
            reasons.append(name+'_panel_f1_collapse')
        if b['auc'] < a['auc']-limits['max_panel_family_auc_drop']:
            reasons.append(name+'_panel_auc_collapse')
        if b['macro_f1'] >= a['macro_f1']-limits['panel_family_stable_tolerance']:
            stable += 1
    if stable < math.ceil(len(panel['families'])/2):
        reasons.append('majority_panel_mechanisms_regressed')
    # Default 256-view selection panel has 64 examples in each language/class:
    # one recall change is 1/64=1.5625pp. A 4pp allowance admits two such errors,
    # not three. Below 25/class these small groups remain diagnostics only;
    # neither tiny-group matched99 nor its quantized changes is a hard guard.
    for language in ('en', 'zh'):
        a = base_panel.get('groups', {}).get(language)
        b = panel.get('groups', {}).get(language)
        if not a or not b or a['class_counts'] != b['class_counts']:
            reasons.append(language+'_panel_language_support_missing_or_changed')
            continue
        if min(b['class_counts']) < limits['min_panel_language_class_support']:
            continue
        if any(x is None or not np.isfinite(x) for x in (*b['recall'], b['auc'])):
            reasons.append(language+'_panel_language_nonfinite')
            continue
        for label in (0, 1):
            tolerance = max(limits['max_panel_language_recall_drop'], 1./b['class_counts'][label])
            if b['recall'][label] < a['recall'][label]-tolerance:
                reasons.append(language+('_panel_fake_recall_collapse' if label == 0 else '_panel_real_recall_collapse'))
        if b['auc'] < a['auc']-limits['max_panel_language_auc_drop']:
            reasons.append(language+'_panel_language_auc_collapse')
    if not (f1_gain >= limits['min_panel_f1_gain'] or auc_gain >= limits['min_panel_auc_gain']):
        reasons.append('no_broader_panel_improvement')
    ranking_evidence = (auc_gain >= limits['min_panel_auc_gain'] or rank_gain >= limits['min_panel_rank_gain'] or
        (len(fixed_auc) == 2 and np.mean(fixed_auc) >= limits['min_fixed_auc_gain']) or
        (len(fixed_rank) == 2 and np.mean(fixed_rank) >= limits['min_fixed_rank_gain']))
    if not ranking_evidence:
        reasons.append('no_ranking_improvement_beyond_boundary_shift')
    return not reasons, reasons


def guarded_score(metrics, panel):
    """Only rank already-qualified candidates. This is NOT competition Weighted."""
    if not panel or panel.get('partition') != 'tune' or not panel.get('complete'):
        raise ValueError('Guarded ranking requires the complete selection panel')
    return .5*float(metrics['weighted_f1'])+.5*float(panel['macro_f1'])


def update_selections(selections, scores, candidates, tag, metrics, eligible, candidate, panel):
    promoted = []
    values = dict(best_weighted=float(metrics['weighted_f1']), best_guarded=guarded_score(metrics, panel))
    for kind in ('best_weighted', 'best_guarded'):
        if (kind == 'best_weighted' or eligible) and values[kind] > scores[kind]:
            selections[kind], scores[kind] = tag, values[kind]
            candidates[tag] = candidate
            promoted.append(kind)
    keep = set(selections.values())
    candidates = {key: value for key, value in candidates.items() if key in keep}
    return candidates, promoted
