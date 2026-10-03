"""Four source-group budgets and label-supervised difficult-condition risk."""
import math
import torch

CONDITIONS = ('offline', 'online', 'noisy_a', 'noisy_b')
GROUPS = ('en/0', 'en/1', 'zh/0', 'zh/1')


def group_weights(counts):
    """One global coefficient per independent language/label source, never view."""
    if set(counts) != set(GROUPS) or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                                        for v in counts.values()):
        raise ValueError('All four language/label groups require positive unique-source counts')
    total = sum(counts.values())
    return {key: total / (4 * counts[key]) for key in GROUPS}


def source_components(examples, weights, device='cpu', source_denominator=None,
                      offline_weight=.1, online_weight=.3, noisy_weight=.6):
    """Validate one full view per condition and preserve one budget per source.

    The fixed logical-source denominator is retained for the last partial batch.
    Combined with N/(4*n_group), a complete source traversal gives every group
    exactly the same coefficient mass. There is no batch class normalization.
    """
    budgets = dict(offline=offline_weight, online=online_weight, noisy=noisy_weight)
    if any(not math.isfinite(v) or v <= 0 for v in budgets.values()):
        raise ValueError('Positive finite condition budgets required')
    if not math.isclose(sum(budgets.values()), 1., abs_tol=1e-8):
        raise ValueError('Condition budgets must sum to one')
    if set(weights) != set(GROUPS) or any(not math.isfinite(float(v)) or float(v) <= 0
                                        for v in weights.values()):
        raise ValueError('Four finite positive source-group weights required')
    sources = {}
    for index, row in enumerate(examples):
        key, condition = row.get('source_id'), row.get('condition')
        if not isinstance(key, str) or not key or condition not in CONDITIONS:
            raise ValueError('Each view needs a canonical source and supported condition')
        if row.get('view') != 'full' or row.get('language') not in ('en', 'zh') or row.get('label') not in (0, 1):
            raise ValueError('Only full EN/ZH views with fake=0 or real=1 are supported')
        if bool(row.get('noisy', condition.startswith('noisy'))) != condition.startswith('noisy'):
            raise ValueError('Noisy flag disagrees with condition')
        if 'view_weight' in row and float(row['view_weight']) != 1.:
            raise ValueError('V3.5 has one complete view, without short-view weighting')
        group = row['language'] + '/' + str(row['label'])
        source = sources.setdefault(key, dict(group=group, label=row['label'], conditions={}))
        if source['group'] != group or condition in source['conditions']:
            raise ValueError('Source views disagree or duplicate a condition')
        source['conditions'][condition] = index
    if not sources:
        raise ValueError('A nonempty source batch is required')
    denominator = len(sources) if source_denominator is None else source_denominator
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator < len(sources):
        raise ValueError('Logical source denominator must cover the complete source batch')
    for source in sources.values():
        if not {'offline', 'noisy_a', 'noisy_b'} <= set(source['conditions']):
            raise ValueError('Each source requires Offline and both full noisy versions')
        available = 1. if 'online' in source['conditions'] else offline_weight + noisy_weight
        source['coefficient'] = float(weights[source['group']]) / denominator
        source['budgets'] = {key: value / available for key, value in budgets.items()
                             if key != 'online' or 'online' in source['conditions']}
    return dict(sources=sources, source_count=len(sources), denominator=denominator,
                labels=torch.tensor([e['label'] for e in examples], dtype=torch.long, device=device))


def objective(losses, components, source_keys=None):
    """0.5 * mean + 0.5 * max gives 75/25 gradients, and 50/50 at ties."""
    chosen = list(components['sources']) if source_keys is None else list(source_keys)
    total = None
    stats = {}
    for key in chosen:
        source = components['sources'][key]
        indices, budget, scale = source['conditions'], source['budgets'], source['coefficient']
        ordinary = budget['offline'] * losses[indices['offline']]
        if 'online' in indices:
            ordinary = ordinary + budget['online'] * losses[indices['online']]
        pair = torch.stack([losses[indices['noisy_a']], losses[indices['noisy_b']]])
        # amax, unlike max(dim), splits a tie's subgradient equally.
        noisy = budget['noisy'] * (.5 * pair.mean() + .5 * pair.amax())
        term = scale * (ordinary + noisy)
        total = term if total is None else total + term
        for name, value in (('ordinary_ce', scale * ordinary), ('noisy_ce', scale * noisy),
                            ('noisy_mean', scale * budget['noisy'] * pair.mean()),
                            ('noisy_max', scale * budget['noisy'] * pair.amax()),
                            ('group_' + source['group'], term)):
            stats[name] = stats.get(name, 0.) + value.detach()
    if total is None:
        raise ValueError('At least one source is required in a gradient chunk')
    return total, stats
