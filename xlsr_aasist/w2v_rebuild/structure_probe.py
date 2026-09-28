"""Fixed, Train-only diagnostic probes; never a submission or a score ensemble.

All hyperparameters and random projections are fixed before reading Dev labels.
Features stay in memory. Source-balanced weights prevent a source with more
cached views from acquiring more training weight.
"""
from collections import Counter, defaultdict
import math
import numpy as np
import torch
from torch.nn import functional as F


DIMENSION = 64
RIDGE = .1


def project(values, name, dimension=DIMENSION):
    import hashlib
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite descriptor matrix')
    if x.shape[1] == dimension:
        return x
    seed = int.from_bytes(hashlib.sha256(('structure-probe-v1:'+name).encode()).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((x.shape[1], dimension)) / math.sqrt(dimension)
    return x @ matrix


def physical_descriptors(waves):
    """Log-spaced spectral bands, NOT learned SSL dimensions interpreted as Hz.

    Subtract frame-wise mean log energy, then measure temporal differences.
    Gain/static-channel cancellation is only an approximation; additive noise,
    nonlinear enhancement and silence violate it. Both candidates and pooling
    controls are projected to the same fixed dimension by the caller.
    """
    x = waves.detach().float().cpu()
    spectrum = torch.stft(x, n_fft=512, hop_length=160, win_length=400,
                          window=torch.hann_window(400), return_complex=True).abs().square()
    # 32 nonempty frequency bands spanning positive FFT bins.
    edges = np.rint(np.geomspace(1, 257, 33)).astype(int)
    for i in range(1, len(edges)):
        edges[i] = min(max(edges[i], edges[i-1]+1), 257-(32-i))
    bands = torch.stack([spectrum[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])], -1)
    logband = torch.log(bands.clamp_min(1e-10))
    relative = logband - logband.mean(-1, keepdim=True)
    dynamics = torch.cat([(relative[:, lag:] - relative[:, :-lag]).square().mean(1).sqrt()
                          for lag in (1, 2, 4, 8)], -1)
    pooled = torch.cat((logband.mean(1), logband.std(1, unbiased=False)), -1)
    return dynamics.numpy(), pooled.numpy()


def source_weights(rows):
    counts = Counter((r['split'], r['source_id']) for r in rows)
    weights = np.array([1. / counts[(r['split'], r['source_id'])] for r in rows])
    return weights / weights.sum()


def fit_probe(train_x, train_y, train_rows, ridge=RIDGE):
    """Full-batch convex logistic regression, fixed L2=.1, no Dev tuning."""
    x = np.asarray(train_x, np.float64)
    y = np.asarray(train_y)
    if set(y.tolist()) != {0, 1} or len(train_rows) != len(x):
        raise ValueError('Probe requires both truth classes and matching source metadata')
    if any(r['split'] != 'train' for r in train_rows):
        raise ValueError('Only official Train may fit the diagnostic probe')
    weights = source_weights(train_rows)
    mean = np.sum(x * weights[:, None], axis=0)
    scale = np.sqrt(np.sum((x-mean)**2 * weights[:, None], axis=0)).clip(1e-4)
    xx = torch.from_numpy(np.clip((x-mean)/scale, -20., 20.))
    yy = torch.from_numpy((y == 0).astype(np.float64))  # fake probability throughout
    ww = torch.from_numpy(weights)
    theta = torch.zeros(xx.shape[1]+1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([theta], lr=1., max_iter=100, tolerance_grad=1e-8,
                                 tolerance_change=1e-10, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        logits = xx @ theta[:-1] + theta[-1]
        loss = (F.binary_cross_entropy_with_logits(logits, yy, reduction='none') * ww).sum()
        loss = loss + ridge * theta[:-1].square().sum()/2
        loss.backward()
        return loss
    optimizer.step(closure)
    if not torch.isfinite(theta).all():
        raise FloatingPointError('Non-finite diagnostic probe')
    return {'mean': mean, 'scale': scale, 'theta': theta.detach().numpy(), 'ridge': ridge}


def predict_probe(probe, values):
    x = np.clip((np.asarray(values)-probe['mean'])/probe['scale'], -20., 20.)
    logits = np.clip(x @ probe['theta'][:-1] + probe['theta'][-1], -50., 50.)
    return 1/(1+np.exp(-logits))


def metrics(labels, pfake):
    y, p = np.asarray(labels, int), np.asarray(pfake, float)
    if not len(y):
        return {'count': 0, 'macro_f1': None, 'auc': None, 'recall_fake': None, 'recall_real': None}
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid fake probabilities')
    pred = (p < .5).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (y, pred), 1)
    scores, recalls = [], []
    for cls in (0, 1):
        tp = int(cm[cls, cls]); denominator = int(cm[cls].sum()+cm[:, cls].sum())
        scores.append(2*tp/denominator if denominator else 0.)
        recalls.append(tp/int(cm[cls].sum()) if cm[cls].sum() else None)
    positive = y == 0
    count = int(positive.sum())
    auc = None
    if 0 < count < len(y):
        order = np.argsort(p, kind='stable'); sorted_p = p[order]
        ranks = np.empty(len(y), dtype=float)
        start = 0
        while start < len(y):
            end = start+1
            while end < len(y) and sorted_p[end] == sorted_p[start]:
                end += 1
            ranks[order[start:end]] = (start+1+end)/2
            start = end
        auc = float((ranks[positive].sum()-count*(count+1)/2)/(count*(len(y)-count)))
    return {'count': len(y), 'macro_f1': float(np.mean(scores)), 'auc': auc,
            'recall_fake': recalls[0], 'recall_real': recalls[1], 'confusion': cm.tolist()}


def grouped_metrics(rows, probabilities):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row['split'] != 'dev':
            continue
        condition = row['condition']
        groups[('condition', condition)].append(index)
        groups[('language_condition', row['language']+'/'+condition)].append(index)
        groups[('family', condition+'/'+row['family'])].append(index)
        groups[('class_language', condition+'/'+row['language']+'/'+str(row['label']))].append(index)
        if row['band'] >= 0:
            groups[('band', condition+'/'+str(row['band']))].append(index)
            groups[('family_band', condition+'/'+row['family']+'/'+str(row['band']))].append(index)
            groups[('condition', 'all_noisy')].append(index)
    report = []
    for (kind, name), indices in sorted(groups.items()):
        labels = [rows[i]['label'] for i in indices]
        for descriptor, p in probabilities.items():
            report.append({'group_kind': kind, 'group': name, 'descriptor': descriptor,
                           **metrics(labels, np.asarray(p)[indices])})
    return report


def paired_transitions(rows, probabilities):
    clean = {(r['split'], r['source_id']): i for i, r in enumerate(rows) if r['condition'] == 'clean'}
    result = []
    for name, values in probabilities.items():
        groups = defaultdict(lambda: Counter())
        for i, r in enumerate(rows):
            if r['split'] != 'dev' or r['band'] < 0:
                continue
            j = clean[(r['split'], r['source_id'])]
            before = int(values[j] < .5) == r['label']
            after = int(values[i] < .5) == r['label']
            for group in ('all', r['language']+'/'+str(r['label']), r['condition']+'/'+r['family']):
                counter = groups[group]
                counter['views'] += 1
                counter['clean_correct'] += before
                counter['clean_correct_to_noisy_error'] += before and not after
                counter['clean_error_to_noisy_correct'] += not before and after
                counter['both_wrong'] += not before and not after
        result.extend({'descriptor': name, 'group': group, **dict(value)} for group, value in sorted(groups.items()))
    return result


def invariance(rows, features, valid):
    train_clean = np.array([r['split'] == 'train' and r['condition'] == 'clean' for r in rows])
    clean = {(r['split'], r['source_id']): i for i, r in enumerate(rows) if r['condition'] == 'clean'}
    results = []
    for name, x in features.items():
        mean, scale = x[train_clean].mean(0), x[train_clean].std(0).clip(1e-4)
        z = np.clip((x-mean)/scale, -20, 20)
        z /= np.linalg.norm(z, axis=1, keepdims=True).clip(1e-10)
        groups = defaultdict(list)
        invalid = Counter()
        for i, r in enumerate(rows):
            if r['split'] != 'dev' or r['band'] < 0:
                continue
            j = clean[(r['split'], r['source_id'])]
            for group in ('all_noisy', r['condition']+'/'+r['family'], r['language']+'/'+str(r['label'])):
                if not valid[name][i] or not valid[name][j]:
                    invalid[group] += 1
                else:
                    groups[group].append(float(z[i] @ z[j]))
        for group in sorted(set(groups) | set(invalid)):
            values = groups[group]
            results.append({'descriptor': name, 'group': group, 'valid_pairs': len(values),
                            'invalid_pairs': invalid[group], 'mean_cosine': float(np.mean(values)) if values else None,
                            'median_cosine': float(np.median(values)) if values else None})
    return results


def source_bootstrap(rows, probabilities, repetitions=300, seed=991):
    """Paired source-cluster intervals; all views of a source travel together."""
    grouped = defaultdict(dict)
    for index, row in enumerate(rows):
        if row['split'] == 'dev' and row['band'] >= 0:
            grouped[(row['language'], row['label'])].setdefault(row['source_id'], []).append(index)
    comparisons = [('readout+structure', 'readout+pooled'),
                   ('structure', 'pooled'), ('readout+spectral_dynamic', 'readout+spectral_pool')]
    rng = np.random.default_rng(seed)
    draws = {pair: [] for pair in comparisons}
    ys = np.array([row['label'] for row in rows])
    for _ in range(repetitions):
        indices = []
        for sources in grouped.values():
            units = list(sources.values())
            for choice in rng.integers(0, len(units), size=len(units)):
                indices.extend(units[choice])
        for a, b in comparisons:
            delta = metrics(ys[indices], np.asarray(probabilities[a])[indices])['macro_f1'] - metrics(ys[indices], np.asarray(probabilities[b])[indices])['macro_f1']
            draws[(a, b)].append(delta)
    result = []
    all_indices = [i for g in grouped.values() for unit in g.values() for i in unit]
    for (a, b), values in draws.items():
        delta = metrics(ys[all_indices], np.asarray(probabilities[a])[all_indices])['macro_f1'] - metrics(ys[all_indices], np.asarray(probabilities[b])[all_indices])['macro_f1']
        result.append({'candidate': a, 'control': b, 'delta_macro_f1': delta,
                       'ci95': np.quantile(values, [.025, .975]).tolist(), 'replicates': repetitions,
                       'unit': 'original source; stratified language and class; correlated views kept together'})
    return result


def analyze(rows, features, valid, baseline_probabilities, bootstrap=300):
    train = np.array([r['split'] == 'train' for r in rows])
    train_rows = [r for r in rows if r['split'] == 'train']
    labels = np.array([r['label'] for r in rows])
    normalized = {name: project(x, name) for name, x in features.items()}
    combined = dict(normalized)
    for name in ('structure', 'pooled', 'spectral_dynamic', 'spectral_pool'):
        combined['readout+'+name] = np.concatenate((normalized['readout'], normalized[name]), axis=1)
    probabilities = {'original_head': np.asarray(baseline_probabilities)}
    for name, values in combined.items():
        probe = fit_probe(values[train], labels[train], train_rows)
        probabilities[name] = predict_probe(probe, values)
    grouped = grouped_metrics(rows, probabilities)
    intervals = source_bootstrap(rows, probabilities, repetitions=bootstrap)
    dev_valid = np.array([r['split'] == 'dev' for r in rows])
    fraction = float(np.asarray(valid['structure'])[dev_valid].mean())
    # A screening decision, not a performance guarantee or permission to train.
    evidence = any(item['ci95'][0] > 0 for item in intervals[:2])
    clearly_worse = all(item['ci95'][1] < 0 for item in intervals[:2])
    recommendation = 'inconclusive'
    if fraction < .8 or clearly_worse:
        recommendation = 'reject'
    elif evidence:
        recommendation = 'ready_for_review'
    return {'recommendation': recommendation,
            'recommendation_rule': 'Review candidate only if one local-structure matched-control paired CI excludes zero positively; reject if valid fraction <80% or both local comparisons negative; otherwise inconclusive. Review still required; no automatic training.',
            'structure_valid_fraction_dev': fraction,
            'probe': {'train_only': True, 'fixed_ridge': RIDGE, 'dimensions': DIMENSION,
                      'combined_dimensions': 2*DIMENSION, 'threshold_pfake': .5,
                      'weights': 'equal per original Train source; four groups sampled equally',
                      'normalization': 'Train-only weighted mean/std; clipped at 20 standard deviations',
                      'warning': 'Diagnostic probe combinations are not a submission or an approved model architecture.'},
            'metrics': grouped, 'invariance': invariance(rows, normalized, valid),
            'paired_transitions': paired_transitions(rows, probabilities), 'paired_intervals': intervals}, probabilities
