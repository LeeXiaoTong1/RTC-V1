"""Historical fixed-Dev F1 is unchanged; ranking diagnostics use fake as positive."""
from collections import defaultdict

import numpy as np

from w2v_v36.metrics import _group
from w2v_v39.metrics import GROUPS, matched_real_recall, measure as historical_measure


def ap_eer(labels, scores):
    """Tie-grouped AP and linearly interpolated ROC EER (no threshold fitting)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Expected equally long finite one-dimensional labels/scores')
    if not np.isin(labels, (0, 1)).all():
        raise ValueError('Expected fake=0, real=1')
    positive = labels == 0
    p, n = int(positive.sum()), int((~positive).sum())
    if not p or not n:
        return dict(ap=None, eer=None)
    order = np.argsort(-scores, kind='stable')
    values, positive = scores[order], positive[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values)-1]
    tp, fp = np.r_[0, np.cumsum(positive)[ends]], np.r_[0, np.cumsum(~positive)[ends]]
    tpr, fpr = tp / p, fp / n
    ap = float(np.sum(np.diff(tpr) * tp[1:] / (tp[1:] + fp[1:])))
    difference = fpr - (1-tpr)
    hi = int(np.flatnonzero(difference >= 0)[0])
    lo = max(0, hi-1)
    ratio = 0. if hi == lo else -difference[lo] / (difference[hi]-difference[lo])
    return dict(ap=ap, eer=float(fpr[lo] + ratio*(fpr[hi]-fpr[lo])))


def binary_metrics(labels, margins, target=.99):
    labels, margins = np.asarray(labels), np.asarray(margins, dtype=np.float64)
    extra = ap_eer(labels, margins)
    value = _group(labels, margins)
    value.update(extra)
    value['matched'] = (matched_real_recall(labels, margins, target)
                        if set(labels.tolist()) == {0, 1} else None)
    value['matched_resolution_note'] = ('fewer than 100 fake examples: a 99% target may require '
        '100% empirical fake recall; use as a diagnostic, not a fitted deployment threshold'
        if value['class_counts'][0] < 100 else None)
    return value


def measure(rows, logits, *, target=.99, train_proxy=False):
    value = historical_measure(rows, logits, target=target, train_proxy=train_proxy)
    aliases = {'clean': 'online'}
    if train_proxy:
        aliases.update(noisy_a='seen', noisy_b='heldout')
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        condition = aliases.get(row['condition'], row['condition'])
        buckets[condition].append(i)
        buckets[condition+'/'+row['language']].append(i)
    z = np.asarray(logits, dtype=np.float64)
    margins = z[:, 0]-z[:, 1]
    for name, indices in buckets.items():
        value['groups'][name].update(ap_eer([rows[i]['label'] for i in indices], margins[indices]))
    value['ranking_positive_class'] = 'fake=0; AP is average precision; EER interpolates ROC'
    return value


def print_metrics(tag, value, eligible=None, reasons=(), panel=None):
    def percent(x):
        return 'NA' if x is None else f'{100*x:.3f}'
    print(f'\n[Dev] V3.16 {tag} Clean={percent(value["clean_f1"])} '
          f'Noisy={percent(value["noisy_f1"])} Weighted={percent(value["weighted_f1"])}', flush=True)
    for name in GROUPS:
        group = value['groups'][name]
        print(f'  {name} Recall(fake/real)={percent(group["recall"][0])}/{percent(group["recall"][1])} '
              f'F1={percent(group["macro_f1"])} AP={percent(group["ap"])} '
              f'AUC={percent(group["auc"])} EER={percent(group["eer"])} '
              f'real@99%fake={percent(value["matched"][name]["real_recall"])}', flush=True)
    if panel is not None:
        print(f'  Bounded {panel["partition"]} panel: macro-mechanism F1={percent(panel["macro_f1"])} '
              f'AUC={percent(panel["macro_auc"])}; sources={panel["views"]}; excluded from Weighted', flush=True)
        for name, group in panel['families'].items():
            print(f'    {name} F1={percent(group["macro_f1"])} AP={percent(group["ap"])} '
                  f'AUC={percent(group["auc"])} EER={percent(group["eer"])} '
                  f'Recall(fake/real)={percent(group["recall"][0])}/{percent(group["recall"][1])}', flush=True)
    if eligible is not None:
        print('  guarded_eligible='+str(eligible)+'; reasons='+','.join(reasons), flush=True)
