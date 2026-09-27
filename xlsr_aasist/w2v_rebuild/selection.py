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


def noisy_metrics(dev):
    return {'noisy_f1': (dev['seen']['macro_f1'] + dev['heldout']['macro_f1']) / 2,
            'noisy_real_recall': sum(b['recall'][1] for k in ('seen', 'heldout')
                                     for b in dev[k]['bands']) / 8}


def candidate_decision(dev, best_key, baseline=None, policy='strict'):
    """Keep the original best unless the candidate improves and passes all floors."""
    if policy not in ('strict', 'targeted'):
        raise ValueError('Unknown selection policy')
    key = (dev['robust_f1'], -dev['robust_ce'])
    reasons, deltas = [], {}
    if key <= tuple(best_key):
        reasons.append('robust_score_not_improved')
    if baseline is not None:
        if dev['robust_f1'] <= baseline['robust_f1'] + 1e-12:
            reasons.append('robust_f1_not_above_baseline')
        current, anchor = guard_metrics(dev), guard_metrics(baseline)
        deltas = {name: current[name] - anchor[name] for name in anchor}
        if policy == 'targeted':
            current.update(noisy_metrics(dev))
            anchor.update(noisy_metrics(baseline))
            deltas.update({name: current[name] - anchor[name] for name in noisy_metrics(dev)})
            floors = ('online_real_recall', 'noisy_real_recall', 'noisy_f1')
        else:
            floors = tuple(deltas)
        reasons.extend(name + '_below_baseline' for name in floors if deltas[name] < -1e-12)
        # Preserve band-level evidence even though selection uses condition averages.
        for kind in ('seen', 'heldout'):
            for i, (a, b) in enumerate(zip(dev[kind]['bands'], baseline[kind]['bands'])):
                deltas[f'{kind}_band_{i}_real_recall'] = a['recall'][1] - b['recall'][1]
    return not reasons, {'accepted': not reasons, 'reasons': reasons, 'baseline_deltas': deltas,
                         'policy': policy,
                         'warnings': [name + '_decreased' for name, delta in deltas.items() if delta < -1e-12]}
