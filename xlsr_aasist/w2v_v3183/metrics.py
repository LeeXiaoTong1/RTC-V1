"""Fixed-threshold metrics and matched-recall diagnostics are reported separately."""
from collections import defaultdict
import numpy as np
from w2v_v317_monitor.metrics import summarize as base_summary, fit_state, percent
from w2v_v316_tfcl.metrics import measure as base_measure
from w2v_v39.metrics import matched_real_recall


def add_details(rows, logits, groups):
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        buckets[row['condition']].append(i)
        buckets[row['condition']+'/'+row['language']].append(i)
    z = np.asarray(logits, dtype=np.float64)
    margin = z[:,0]-z[:,1]
    for name, metric in groups.items():
        ix = buckets[name]
        labels = np.array([rows[i]['label'] for i in ix])
        confusion = np.array(metric['confusion'])
        metric['precision'] = np.divide(np.diag(confusion), confusion.sum(0),
            out=np.zeros(2, dtype=float), where=confusion.sum(0)>0).tolist()
        metric['matched_95'] = matched_real_recall(labels, margin[ix], .95) if set(labels)=={0,1} else None
        metric['matched_99'] = matched_real_recall(labels, margin[ix], .99) if set(labels)=={0,1} else None
    return groups


def summarize(rows, logits):
    value = base_summary(rows, logits)
    add_details(rows, logits, value['groups'])
    value['matched_note'] = 'Group-specific ROC diagnostics only; never deployment thresholds.'
    return value


def measure(rows, logits):
    value = base_measure(rows, logits)
    add_details(rows, logits, value['groups'])
    return value


def table_lines(title, result):
    lines = [title, '  condition/lang n(fake/real) Recall(fake/real)% Precision(fake/real)% F1% AUC% EER% Real@95F% Real@99F%']
    for name, item in result['groups'].items():
        support = '/'.join(map(str,item['class_counts']))
        recall = '/'.join(percent(v) for v in item['recall'])
        precision = '/'.join(percent(v) for v in item['precision'])
        matched = [item[k]['real_recall'] if item[k] else None for k in ('matched_95','matched_99')]
        lines.append(f'  {name:16s} {support:12s} {recall:20s} {precision:20s} '+
                     ' '.join(percent(v) for v in [item['macro_f1'],item['auc'],item['eer'],*matched]))
    return lines
