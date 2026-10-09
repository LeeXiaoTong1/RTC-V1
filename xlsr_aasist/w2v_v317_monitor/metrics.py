"""Condition/language metrics with explicit support and comparable CE."""
from collections import defaultdict
import math
import numpy as np
from w2v_v316_tfcl.metrics import binary_metrics


def summarize(rows, logits):
    z = np.asarray(logits, dtype=np.float64)
    if not rows or z.shape != (len(rows), 2) or not np.isfinite(z).all():
        raise ValueError('Expected nonempty aligned finite two-class logits')
    labels = np.asarray([r['label'] for r in rows])
    if not np.isin(labels, (0, 1)).all():
        raise ValueError('Expected fake=0 / real=1')
    shifted = z - z.max(1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(rows)), labels]
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        if row['language'] not in ('en', 'zh'):
            raise ValueError('Expected dataset language tag en/zh')
        buckets[row['condition'] + '/' + row['language']].append(i)
    groups = {}
    for name, ix in sorted(buckets.items()):
        metric = binary_metrics(labels[ix], z[ix, 0] - z[ix, 1])
        class_ce = [float(ce[[i for i in ix if labels[i] == c]].mean())
                    if any(labels[i] == c for i in ix) else None for c in (0, 1)]
        metric.update(ce=float(ce[ix].mean()), class_ce=class_ce,
                      balanced_ce=float(np.mean(class_ce)) if None not in class_ce else None,
                      balanced_recall=float(np.mean(metric['recall'])) if None not in metric['recall'] else None)
        groups[name] = metric
    conditions = {}
    for name in sorted({r['condition'] for r in rows}):
        cells = [groups.get(name + '/' + lang) for lang in ('en', 'zh')]
        complete = all(c is not None and min(c['class_counts']) > 0 for c in cells)
        conditions[name] = dict(complete=complete,
            balanced_ce=float(np.mean([c['balanced_ce'] for c in cells])) if complete else None,
            balanced_recall=float(np.mean([c['balanced_recall'] for c in cells])) if complete else None,
            macro_auc=float(np.mean([c['auc'] for c in cells])) if complete else None)
    return dict(groups=groups, conditions=conditions, count=len(rows),
                threshold=.5, positive_class='fake=0', ranking_note='AP depends on class prevalence; sample counts are shown')


def fit_state(epochs):
    """Describe trends, never assert a causal diagnosis from one epoch."""
    complete = [e for e in epochs if e.get('train') and e.get('dev')]
    if not complete:
        return dict(code='unavailable', text='尚无完整固定 Train/Dev 复测，暂不判断拟合趋势。')
    current = complete[-1]
    a = current['train']['conditions'].get('online', {})
    b = current['dev']['conditions'].get('online', {})
    keys = ('balanced_ce', 'balanced_recall')
    if any(a.get(k) is None or b.get(k) is None for k in keys):
        return dict(code='incomplete', text='Online 四组不完整，暂不判断拟合趋势。')
    gaps = dict(recall_gap_pp=100*(a['balanced_recall']-b['balanced_recall']),
                ce_gap=b['balanced_ce']-a['balanced_ce'])
    if len(complete) < 2:
        return dict(code='one_epoch', text='只有一轮：已有 Train/Dev 差距可观察，尚不能判断过拟合趋势。', **gaps)
    previous = complete[-2]
    pa = previous['train']['conditions']['online']; pb = previous['dev']['conditions']['online']
    dt = a['balanced_ce']-pa['balanced_ce']; dd = b['balanced_ce']-pb['balanced_ce']
    if dt < -1e-6 and dd > 1e-6:
        code, text = 'overfitting_risk', '过拟合或域偏移风险：Train CE 下降、Dev CE 上升；需结合 Noisy AUC 与后续轮次。'
    elif dd < -1e-6:
        code, text = 'dev_ce_improving', '固定 Dev Online CE 改善；仍需一起检查 Noisy、AUC 和真假召回。'
    else:
        code, text = 'mixed_or_flat', '拟合趋势混合或停滞，不能仅凭训练 loss 下降判断泛化改善。'
    return dict(code=code, text=text, train_ce_change=dt, dev_ce_change=dd, **gaps)


def percent(value):
    return 'NA' if value is None or not math.isfinite(value) else f'{100*value:.3f}'


def table_lines(title, result):
    lines = [title, '  condition/lang       n(fake/real)  Recall(fake/real)%   F1%     AP%     AUC%    EER%    CE(bal)']
    for name, item in result['groups'].items():
        support = '/'.join(map(str, item['class_counts']))
        recall = '/'.join(percent(v) for v in item['recall'])
        ce = 'NA' if item['balanced_ce'] is None else f'{item["balanced_ce"]:.4f}'
        lines.append(f'  {name:20s} {support:12s} {recall:20s} ' +
            ' '.join(f'{percent(item[k]):7s}' for k in ('macro_f1','ap','auc','eer')) + ' ' + ce)
    return lines
