"""Paired Dev reports, kept outside the training engine's source-hash namespace."""
import csv
import json
import math
from pathlib import Path

import numpy as np


CONDITIONS = ('online', 'offline', 'seen', 'heldout')
META_FIELDS = ('condition', 'source_id', 'band', 'label', 'audio_path', 'snr_db',
               'processing_family', 'source_directory', 'processing_json', 'rtc_json', 'mix_id')
OPTIONAL_META_FIELDS = ('noise_json', 'source_samples', 'output_samples_before_crop',
                        'source_sha256', 'audio_size', 'audio_mtime_ns')


def auc_fake(labels, scores):
    """Mann-Whitney AUC with average ranks for ties; fake is the positive class."""
    y, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    positive = y == 0
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if not npos or not nneg:
        return None
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    stops = np.r_[starts[1:], len(y)]
    ranks = np.empty(len(y), dtype=np.float64)
    ranks[order] = np.repeat((starts + stops + 1) / 2, stops - starts)
    return float((ranks[positive].sum() - npos*(npos+1)/2)/(npos*nneg))


def _arrays(scores):
    z = np.asarray([[s['logit_fake'], s['logit_real']] for s in scores], dtype=np.float64)
    p = np.asarray([s['pfake'] for s in scores], dtype=np.float64)
    return z, p, (p < .5).astype(np.int64)


def metrics(labels, scores):
    """Match core.Metrics; one-class diagnostic groups have undefined macro metrics."""
    y = np.asarray(labels, dtype=np.int64)
    z, p, predicted = _arrays(scores)
    cm = np.bincount(2*y + predicted, minlength=4).reshape(2, 2)
    count = cm.sum(axis=1)
    ce = np.logaddexp(z[:, 0], z[:, 1]) - z[np.arange(len(y)), y]
    both = bool(np.all(count > 0))
    f1 = 2*np.diag(cm)/np.maximum(count + cm.sum(axis=0), 1)
    return {
        'count': int(len(y)), 'class_count': count.tolist(), 'confusion': cm.tolist(),
        'macro_f1': float(f1.mean()) if both else None,
        'balanced_ce': float(np.mean([ce[y == c].mean() for c in (0, 1)])) if both else None,
        'accuracy': float(np.trace(cm)/len(y)),
        'recall': [float(cm[c, c]/count[c]) if count[c] else None for c in (0, 1)],
        'fake_prediction_fraction': float((predicted == 0).mean()),
        'mean_fake_score_by_class': [float(p[y == c].mean()) if count[c] else None for c in (0, 1)],
        # The raw binary logit difference preserves ranking when FP32 softmax
        # probabilities have saturated to 0 or 1. Classification still uses p.
        'auc_fake': auc_fake(y, z[:, 0]-z[:, 1]),
    }


def _validate(records, baseline, candidate):
    if not records or len(records) != len(baseline) or len(records) != len(candidate):
        raise ValueError('Empty or misaligned records and predictions')
    keys, labels, noisy_sets = set(), {}, {}
    conditions = set()
    for i, r in enumerate(records):
        if any(k not in r for k in META_FIELDS):
            raise ValueError('Missing record metadata at index %d' % i)
        condition, source, band, label = (r[k] for k in ('condition', 'source_id', 'band', 'label'))
        if condition not in CONDITIONS or not isinstance(source, str) or not source:
            raise ValueError('Invalid condition or source ID')
        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError('Expected fake=0 or real=1')
        if band not in ((-1,) if condition in ('online', 'offline') else range(4)):
            raise ValueError('Invalid condition/band combination')
        key = (condition, source, band)
        if key in keys:
            raise ValueError('Duplicate condition/source/band: %r' % (key,))
        keys.add(key)
        if source in labels and labels[source] != label:
            raise ValueError('Inconsistent labels for source %s' % source)
        labels[source] = label
        conditions.add(condition)
        if condition in ('seen', 'heldout'):
            noisy_sets.setdefault((condition, band), set()).add(source)
        snr = r['snr_db']
        if snr not in ('', None) and not math.isfinite(float(snr)):
            raise ValueError('Non-finite SNR')
        for scores in (baseline, candidate):
            s = scores[i]
            values = [float(s[k]) for k in ('logit_fake', 'logit_real', 'pfake', 'margin')]
            if not all(math.isfinite(x) for x in values) or not 0 <= values[2] <= 1:
                raise ValueError('Invalid/non-finite prediction at index %d' % i)
            margin = values[0] - values[1]
            expected = float(np.exp(-np.logaddexp(0., -margin)))
            if not math.isclose(values[3], margin, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError('Margin is not logit_fake - logit_real')
            if not math.isclose(values[2], expected, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError('Probability does not match logits')
    if conditions != set(CONDITIONS) or len(noisy_sets) != 8:
        raise ValueError('Require Online, Offline, and all four Seen/Heldout bands')
    reference = noisy_sets[('seen', 0)]
    if any(s != reference for s in noisy_sets.values()):
        raise ValueError('Noisy conditions/bands must cover identical source IDs')
    for condition in CONDITIONS:
        if {r['label'] for r in records if r['condition'] == condition} != {0, 1}:
            raise ValueError('Both classes required in each validation condition')


def _condition_metrics(records, scores):
    out = {}
    for condition in CONDITIONS:
        indices = [i for i, r in enumerate(records) if r['condition'] == condition]
        if condition in ('online', 'offline'):
            out[condition] = metrics([records[i]['label'] for i in indices], [scores[i] for i in indices])
        else:
            bands = []
            for band in range(4):
                ix = [i for i in indices if records[i]['band'] == band]
                bands.append(metrics([records[i]['label'] for i in ix], [scores[i] for i in ix]))
            out[condition] = {
                'aggregation': 'unweighted mean of four band metrics (not pooled F1)',
                'bands': bands,
                'macro_f1': float(np.mean([b['macro_f1'] for b in bands])),
                'balanced_ce': float(np.mean([b['balanced_ce'] for b in bands])),
                'recall': np.mean([b['recall'] for b in bands], axis=0).tolist(),
                'auc_fake': float(np.mean([b['auc_fake'] for b in bands])),
            }
    for key, metric in [('robust_f1', 'macro_f1'), ('robust_ce', 'balanced_ce')]:
        out[key] = .3*out['online'][metric] + .35*out['seen'][metric] + .35*out['heldout'][metric]
    out['noisy_f1'] = (out['seen']['macro_f1'] + out['heldout']['macro_f1'])/2
    out['noisy_real_recall'] = (out['seen']['recall'][1] + out['heldout']['recall'][1])/2
    return out


def _transitions(labels, old, new):
    y = np.asarray(labels)
    _, po, bo = _arrays(old)
    _, pn, bn = _arrays(new)
    result = {}
    for truth, name in ((0, 'fake'), (1, 'real')):
        selected = y == truth
        before, after = bo == y, bn == y
        result[name] = {'count': int(selected.sum())}
        for key, mask in {
            'rescued': ~before & after, 'new_errors': before & ~after,
            'persistent_errors': ~before & ~after, 'both_correct': before & after,
            'new_errors_candidate_near_boundary': before & ~after & (pn >= .45) & (pn <= .55),
            'persistent_candidate_near_boundary': ~before & ~after & (pn >= .45) & (pn <= .55),
            'baseline_high_confidence_wrong': ~before & ((po >= .9) if truth else (po <= .1)),
            'candidate_high_confidence_wrong': ~after & ((pn >= .9) if truth else (pn <= .1)),
        }.items():
            result[name][key] = int((selected & mask).sum())
    return result


def _bootstrap_samples(labels, baseline_pred, candidate_pred, baseline_ce, candidate_ce, repeats, rng):
    """Resample source rows jointly across all columns; class counts remain fixed."""
    y = np.asarray(labels, dtype=np.int64)
    classes = [np.flatnonzero(y == c) for c in (0, 1)]
    views = baseline_pred.shape[1]
    encoded = []
    for pred in (baseline_pred, candidate_pred):
        code = 2*y[:, None] + pred
        encoded.append(np.eye(4, dtype=np.float64)[code].reshape(len(y), views*4))
    samples = {k: np.empty((repeats, views)) for k in ('macro_f1', 'real_recall', 'fake_recall', 'balanced_ce')}
    ce_delta = candidate_ce-baseline_ce
    for draw in range(repeats):
        chosen = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in classes])
        weights = np.bincount(chosen, minlength=len(y)).astype(np.float64)
        cm = [np.sum(e*weights[:, None], axis=0).reshape(views, 2, 2) for e in encoded]
        f1, recalls = [], []
        for m in cm:
            diagonal = np.diagonal(m, axis1=1, axis2=2)
            f1.append((2*diagonal/np.maximum(m.sum(axis=1)+m.sum(axis=2), 1)).mean(axis=1))
            recalls.append(diagonal/m.sum(axis=2))
        samples['macro_f1'][draw] = f1[1] - f1[0]
        samples['fake_recall'][draw] = recalls[1][:, 0] - recalls[0][:, 0]
        samples['real_recall'][draw] = recalls[1][:, 1] - recalls[0][:, 1]
        samples['balanced_ce'][draw] = sum(np.sum(ce_delta[c]*weights[c, None], axis=0)/len(c) for c in classes)/2
    return samples


def _bootstrap(records, baseline, candidate, repeats, seed):
    result = {'replicates': repeats, 'seed': seed, 'confidence': .95,
              'unit': 'class-stratified source; same noisy source draw for all eight views',
              'scope': 'Exploratory paired Dev intervals, not independent test evidence',
              'robust_interval': None,
              'robust_interval_reason': 'Online/noisy source dependence is not established; no cross-domain independence assumption.'}
    if not repeats:
        result['intervals'] = {}
        return result
    rng = np.random.default_rng(seed)
    intervals = {}
    for kind in ('online', 'offline', 'noisy'):
        conditions = (kind,) if kind != 'noisy' else ('seen', 'heldout')
        views = [(kind, -1)] if kind != 'noisy' else [(c, b) for c in conditions for b in range(4)]
        lookup = {(r['source_id'], r['condition'], r['band']): i for i, r in enumerate(records) if r['condition'] in conditions}
        sources = sorted({k[0] for k in lookup})
        ix = np.asarray([[lookup[(s, c, b)] for c, b in views] for s in sources])
        labels = np.asarray([records[i]['label'] for i in ix[:, 0]])
        preds, ces = [], []
        for scores in (baseline, candidate):
            z, _, pred = _arrays([scores[i] for i in ix.ravel()])
            y = np.repeat(labels, len(views))
            preds.append(pred.reshape(ix.shape))
            ces.append((np.logaddexp(z[:, 0], z[:, 1])-z[np.arange(len(z)), y]).reshape(ix.shape))
        samples = _bootstrap_samples(labels, preds[0], preds[1], ces[0], ces[1], repeats, rng)
        selections = [(kind, slice(None))] if kind != 'noisy' else [('seen', slice(0, 4)), ('heldout', slice(4, 8)), ('noisy', slice(None))]
        for name, columns in selections:
            intervals[name] = {}
            for metric, values in samples.items():
                a = values[:, columns].mean(axis=1)
                intervals[name][metric] = {'lower': float(np.quantile(a, .025)), 'upper': float(np.quantile(a, .975))}
    result['intervals'] = intervals
    return result


def _safe_csv(value):
    # Spreadsheet formula injection protection applies to strings, not numeric cells.
    if isinstance(value, str) and value.lstrip(' \t\r\n').startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def _write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _safe_csv(v) for k, v in row.items()})


def build_report(records, baseline, candidate, out_dir, bootstrap=400, seed=20260928):
    """Write small reports and per-view predictions; never copy audio or checkpoints."""
    _validate(records, baseline, candidate)
    if isinstance(bootstrap, bool) or int(bootstrap) != bootstrap or bootstrap < 0:
        raise ValueError('bootstrap must be a nonnegative integer')
    # Canonical order makes saved rows and bootstrap draws independent of input batching.
    indices = sorted(range(len(records)), key=lambda i: (CONDITIONS.index(records[i]['condition']), records[i]['band'], records[i]['source_id']))
    records = [records[i] for i in indices]
    baseline, candidate = ([scores[i] for i in indices] for scores in (baseline, candidate))
    old, new = _condition_metrics(records, baseline), _condition_metrics(records, candidate)
    summary = {'schema': 1, 'threshold': .5, 'tie_prediction': 'fake', 'labels': {'fake': 0, 'real': 1},
               'auc_score': 'logit_fake - logit_real (before probability saturation)',
               'baseline': old, 'candidate': new, 'delta': {}, 'transitions': {},
               'notes': ['All metrics use unchanged threshold 0.5; candidate minus baseline deltas.',
                         'Noisy metrics average four bands per condition, not pooled F1.',
                         'Path directories describe filesystem layout, not verified speakers or recording devices.',
                         'No threshold tuning, weight changes, audio copies or checkpoint copies.',
                         'Dev data informed training selection; this report cannot establish independent generalization or a root cause.']}
    for condition in CONDITIONS:
        summary['delta'][condition] = {k: new[condition][k]-old[condition][k] for k in ('macro_f1', 'balanced_ce', 'auc_fake')}
        summary['delta'][condition]['recall'] = [n-o for n, o in zip(new[condition]['recall'], old[condition]['recall'])]
        ix = [i for i, r in enumerate(records) if r['condition'] == condition]
        summary['transitions'][condition] = _transitions([records[i]['label'] for i in ix], [baseline[i] for i in ix], [candidate[i] for i in ix])
    for k in ('robust_f1', 'robust_ce', 'noisy_f1', 'noisy_real_recall'):
        summary['delta'][k] = new[k]-old[k]
    summary['bootstrap'] = _bootstrap(records, baseline, candidate, int(bootstrap), seed)
    prediction_rows = []
    meta_fields = META_FIELDS + tuple(k for k in OPTIONAL_META_FIELDS if any(k in r for r in records))
    for r, b, c in zip(records, baseline, candidate):
        row = {k: r.get(k, '') for k in meta_fields}
        truth = r['label']
        correct = []
        for name, score in (('baseline', b), ('candidate', c)):
            pred = int(score['pfake'] < .5)
            correct.append(pred == truth)
            row.update({name+'_'+k: score[k] for k in ('logit_fake', 'logit_real', 'pfake', 'margin')})
            row.update({name+'_prediction': pred, name+'_correct': pred == truth,
                        name+'_ce': float(np.logaddexp(score['logit_fake'], score['logit_real'])-score['logit_fake' if truth == 0 else 'logit_real']),
                        name+'_near_boundary': .45 <= score['pfake'] <= .55,
                        name+'_high_confidence_wrong': pred != truth and (score['pfake'] >= .9 if truth else score['pfake'] <= .1)})
        row['transition'] = ('both_correct' if correct[1] else 'new_error') if correct[0] else ('rescued' if correct[1] else 'persistent_error')
        row['delta_pfake'] = c['pfake']-b['pfake']
        prediction_rows.append(row)
    group_indices = {}
    for i, r in enumerate(records):
        groups = [('condition', r['condition'], '', ''),
                  ('condition_band', r['condition'], r['band'], ''),
                  ('processing_family', r['condition'], '', r['processing_family'] or '(unspecified)'),
                  ('source_directory', r['condition'], '', r['source_directory'] or '(root)')]
        for group in groups:
            group_indices.setdefault(group, []).append(i)
    group_rows = []
    for (kind, condition, band, value), ix in sorted(group_indices.items(), key=lambda x: tuple(map(str, x[0]))):
        labels = [records[i]['label'] for i in ix]
        scores = [[s[i] for i in ix] for s in (baseline, candidate)]
        ms = [metrics(labels, s) for s in scores]
        row = {'group_type': kind, 'condition': condition, 'band': band, 'group_value': value,
               'count': len(ix), 'fake_count': labels.count(0), 'real_count': labels.count(1),
               'unique_source_count': len({records[i]['source_id'] for i in ix}),
               'metric_aggregation': 'pooled within this diagnostic group'}
        for name, m in zip(('baseline', 'candidate'), ms):
            row.update({name+'_'+k: m[k] for k in ('macro_f1', 'balanced_ce', 'auc_fake')})
            row.update({name+'_fake_recall': m['recall'][0], name+'_real_recall': m['recall'][1]})
        for k in ('macro_f1', 'balanced_ce', 'auc_fake', 'fake_recall', 'real_recall'):
            a, b = row['baseline_'+k], row['candidate_'+k]
            row['delta_'+k] = None if a is None or b is None else b-a
        transitions = _transitions(labels, *scores)
        for label, counts in transitions.items():
            row.update({label+'_'+k: v for k, v in counts.items() if k != 'count'})
        group_rows.append(row)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(prediction_rows[0])
    _write_csv(out/'predictions.csv', fields, prediction_rows)
    _write_csv(out/'changed_errors.csv', fields, [r for r in prediction_rows if r['transition'] in ('rescued', 'new_error')])
    _write_csv(out/'persistent_errors.csv', fields, [r for r in prediction_rows if r['transition'] == 'persistent_error'])
    _write_csv(out/'groups.csv', list(group_rows[0]), group_rows)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    _write_markdown(out/'report.md', summary)
    return summary


def _write_markdown(path, summary):
    old, new = summary['baseline'], summary['candidate']
    lines = ['# 原 best 与第一轮候选：逐样本 Dev 对比', '',
             '固定阈值 0.5，恰好等于阈值判为 fake；标签 fake=0、real=1。所有变化均为候选减原 best。', '',
             '| 条件 | 原 F1 | 候选 F1 | ΔF1（百分点） | 原 real recall | 候选 real recall | Δreal（百分点） |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        a, b = old[condition], new[condition]
        lines.append('| %s | %.3f | %.3f | %+.3f | %.3f | %.3f | %+.3f |' % (
            condition, 100*a['macro_f1'], 100*b['macro_f1'], 100*(b['macro_f1']-a['macro_f1']),
            100*a['recall'][1], 100*b['recall'][1], 100*(b['recall'][1]-a['recall'][1])))
    lines += ['', 'Seen/Heldout 分别对四个噪声区间等权平均；不是把所有视图合并后计算 F1。',
              'RobustF1 = 0.3 × Online + 0.35 × Seen + 0.35 × Heldout：%.3f%% → %.3f%%（%+.3f 个百分点）。' %
              (100*old['robust_f1'], 100*new['robust_f1'], 100*(new['robust_f1']-old['robust_f1'])), '',
              '| 条件 | 原 Balanced CE | 候选 Balanced CE | 原 AUC | 候选 AUC |',
              '|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        a, b = old[condition], new[condition]
        lines.append('| %s | %.6f | %.6f | %.6f | %.6f |' %
                     (condition, a['balanced_ce'], b['balanced_ce'], a['auc_fake'], b['auc_fake']))
    lines += ['', 'AUC 以 fake 为正类，按 logit_fake − logit_real 排名，避免概率饱和产生人为并列；原始分数相同按并列排名处理。分类仍严格使用 p(fake) 与 0.5 比较。CE 越低越好，AUC 越高越好；二者不直接等同于固定阈值下的 real recall。', '',
              '## 误判转换', '',
              '| 条件 | 真实类别 | 救回 | 新增误判 | 持续误判 | 新增误判中靠近边界 | 候选高置信度误判 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for condition, labels in summary['transitions'].items():
        for label, c in labels.items():
            lines.append('| %s | %s | %d | %d | %d | %d | %d |' % (condition, label, c['rescued'], c['new_errors'], c['persistent_errors'], c['new_errors_candidate_near_boundary'], c['candidate_high_confidence_wrong']))
    lines += ['', '边界附近定义为 p(fake) ∈ [0.45, 0.55]。高置信度误判：real 的 p(fake) ≥ 0.9，或 fake 的 p(fake) ≤ 0.1。',
              'Seen/Heldout 的误判数量按“源音频 × 噪声区间”视图统计，同一源可能出现四次；不可当作独立说话人数量。', '',
              '## 成对重采样的不确定性', '',
              '以下为 Dev 上的探索性 95% 区间。按真假分层抽取源音频；同一 noisy 源的八个视图始终一起抽取，两个模型使用同一次抽样。',
              'Online/Offline 分别按各自源音频抽样。未确认 Online 与 noisy 的依赖关系，因此不计算 RobustF1 的联合区间。', '',
              '| 条件 | ΔF1 区间（百分点） | Δreal recall 区间（百分点） |',
              '|---|---:|---:|']
    for condition, intervals in summary['bootstrap']['intervals'].items():
        f, r = intervals['macro_f1'], intervals['real_recall']
        lines.append('| %s | [%+.3f, %+.3f] | [%+.3f, %+.3f] |' % (condition, 100*f['lower'], 100*f['upper'], 100*r['lower'], 100*r['upper']))
    if not summary['bootstrap']['replicates']:
        lines.append('本次关闭了重采样。')
    lines += ['', '## 文件与解读限制', '',
              '- `predictions.csv`：所有样本的路径、条件、真假标签、两个模型的 logits/概率/误判状态。',
              '- `changed_errors.csv`：救回与新增误判；`persistent_errors.csv`：两模型持续误判。',
              '- `groups.csv`：按条件、噪声区间、处理算法、路径目录分组的指标与误判转换。目录只是文件路径，不能视为已核实的说话人或设备身份。',
              '- `summary.json`：完整汇总、每个噪声区间指标和置信区间。CSV 使用 UTF-8 BOM；文本公式字符已转义。',
              '- 分组 CSV 的 F1 为组内合并统计，不能替代上方四区间平均的正式 Dev 指标。样本少或只有一个类别的组不足以比较宏平均 F1/AUC；相应单元格留空。',
              '- Dev 已参与模型选择，这不是独立测试集上的提升证明。边界错误或高置信度错误是诊断线索，不能单凭本报告确认校准、过拟合或特定处理算法为根因。',
              '- 本报告没有搜索阈值、更新权重，也不包含音频或 checkpoint。', '']
    path.write_text('\n'.join(lines), encoding='utf-8')
