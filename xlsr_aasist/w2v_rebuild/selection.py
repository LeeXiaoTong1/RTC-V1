"""Development-set guardrails for a new adaptation run, not test-set guarantees."""


def guard_metrics(dev):
    metrics = {}
    for kind in ('online', 'offline'):
        metrics[kind + '_real_recall'] = dev[kind]['recall'][1]
    for kind in ('online', 'seen', 'heldout'):
        metrics[kind + '_f1'] = dev[kind]['macro_f1']
    for kind in ('seen', 'heldout'):
        bands = dev[kind]['bands']
        metrics[kind + '_real_recall'] = sum(b['recall'][1] for b in bands) / len(bands)
    return metrics


def candidate_decision(dev, best_key, baseline=None):
    """Keep the original best unless the candidate improves and passes all floors."""
    key = (dev['robust_f1'], -dev['robust_ce'])
    reasons, deltas = [], {}
    if key <= tuple(best_key):
        reasons.append('robust_score_not_improved')
    if baseline is not None:
        if dev['robust_f1'] <= baseline['robust_f1'] + 1e-12:
            reasons.append('robust_f1_not_above_baseline')
        current, anchor = guard_metrics(dev), guard_metrics(baseline)
        deltas = {name: current[name] - anchor[name] for name in anchor}
        reasons.extend(name + '_below_baseline' for name, delta in deltas.items() if delta < -1e-12)
        # Preserve band-level evidence even though selection uses condition averages.
        for kind in ('seen', 'heldout'):
            for i, (a, b) in enumerate(zip(dev[kind]['bands'], baseline[kind]['bands'])):
                deltas[f'{kind}_band_{i}_real_recall'] = a['recall'][1] - b['recall'][1]
    return not reasons, {'accepted': not reasons, 'reasons': reasons, 'baseline_deltas': deltas}
