"""Read-only analysis of original V3.18.10 Select/Audit prediction artifacts.

No new thresholds are fitted on Dev, and no Progress/Eval inputs are accepted.
"""
import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

SELECT_RE = re.compile(r'^selection_scores_epoch_(\d+)_step_(\d+)\.npz$')
REQUIRED = {'logits', 'ids', 'canonical_source', 'language', 'label', 'panel', 'metadata'}


def _json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True,
                                     allow_nan=False), encoding='utf8')


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf8', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0])); out.writeheader(); out.writerows(rows)


def _load(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as obj:
        if not REQUIRED.issubset(obj.files):
            raise ValueError('Missing required V3.18.10 scores fields')
        a = {k: np.asarray(obj[k]).copy() for k in REQUIRED if k != 'metadata'}
        meta = json.loads(str(obj['metadata'].item()))
    n = len(a['ids'])
    if a['logits'].shape != (n, 2) or any(len(a[k]) != n for k in REQUIRED - {'metadata','logits'}):
        raise ValueError('Scores shape/coverage mismatch')
    if not np.isfinite(a['logits']).all() or not set(np.unique(a['label'])).issubset({0, 1}):
        raise ValueError('Invalid logits/labels')
    if len(set(zip(a['ids'].tolist(), a['panel'].tolist()))) != n:
        raise ValueError('Duplicate (ID, panel) pairs in scores')
    a['metadata'] = meta
    return a


def load_artifacts(path):
    p = Path(path)
    if p.is_file() and p.suffix.lower() == '.zip':
        with zipfile.ZipFile(p) as z:
            files = {Path(n).name: z.read(n) for n in z.namelist()
                     if SELECT_RE.match(Path(n).name) or Path(n).name == 'locked_audit_scores.npz'}
    elif p.is_dir():
        files = {f.name: f.read_bytes() for f in p.glob('selection_scores_epoch_*_step_*.npz')}
        audit = p / 'locked_audit_scores.npz'
        if audit.is_file(): files[audit.name] = audit.read_bytes()
    else:
        raise ValueError('Pass a V3.18.10 run directory or report ZIP')
    epochs = []
    for name, blob in files.items():
        m = SELECT_RE.fullmatch(name)
        if m:
            data = _load(blob)
            if data['metadata'].get('split') != 'select':
                raise ValueError('Not a Select scores artifact: ' + name)
            epochs.append((int(m.group(1)), int(m.group(2)), data))
    epochs.sort(key=lambda item: item[:2])
    if not epochs: raise ValueError('No V3.18.10 per-epoch Select logits found')
    ref = epochs[0][2]
    for _, _, d in epochs[1:]:
        for key in ('ids', 'canonical_source', 'label', 'panel', 'language'):
            if not np.array_equal(ref[key], d[key]):
                raise ValueError('Epoch Select identity/order differs at ' + key)
        for key in ('protocol_hash', 'protocol_content_hash'):
            if d['metadata'].get(key) != ref['metadata'].get(key):
                raise ValueError('Protocol hash changed across epochs')
    audit = _load(files['locked_audit_scores.npz']) if 'locked_audit_scores.npz' in files else None
    if audit is not None:
        if audit['metadata'].get('split') != 'audit':
            raise ValueError('Not an Audit artifact')
        if audit['metadata'].get('protocol_hash') != ref['metadata'].get('protocol_hash'):
            raise ValueError('Audit/Select protocol hash differs')
        if set(audit['canonical_source']).intersection(set(ref['canonical_source'])):
            raise ValueError('Audit and Select sources overlap')
    return epochs, audit


def get_margin(data):
    return np.asarray(data['logits'][:, 0] - data['logits'][:, 1], dtype=np.float64)


def metrics(data, mask, bias=0.):
    y = data['label'][mask].astype(int)
    score = get_margin(data)[mask] + bias
    out = {'n': int(len(y)), 'real_n': int(sum(y==1)), 'fake_n': int(sum(y==0)),
           'bias': float(bias)}
    if len(y) == 0: return out
    pred = (score < 0).astype(int)  # fake=0 / real=1; positive margin => fake
    out['real_recall'] = float(np.mean(pred[y==1] == 1)) if any(y==1) else None
    out['fake_recall'] = float(np.mean(pred[y==0] == 0)) if any(y==0) else None
    out['real_false_positives'] = int(np.sum((y==1) & (pred==0)))
    out['fake_false_negatives'] = int(np.sum((y==0) & (pred==1)))
    out['confusion_actual_fake_real'] = [[int(np.sum((y==i)&(pred==j))) for j in (0,1)] for i in (0,1)]
    f = []
    for klass in (0,1):
        tp = np.sum((y==klass)&(pred==klass))
        fp = np.sum((y!=klass)&(pred==klass))
        fn = np.sum((y==klass)&(pred!=klass))
        f.append(float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn)>0 else 0.)
    out['macro_f1'] = float(np.mean(f))
    if len(set(y.tolist())) == 2:
        out['auc'] = float(roc_auc_score((y==0).astype(int), score))
        fpr,tpr,_ = roc_curve((y==0).astype(int), score, drop_intermediate=False)
        fnr=1-tpr; k=np.argmin(np.abs(fpr-fnr))
        out['eer_approx'] = float((fpr[k]+fnr[k])/2)
    for name, value in (('real',score[y==1]),('fake',score[y==0])):
        out[name+'_margin_quantiles'] = [float(x) for x in np.quantile(value,[0,.1,.25,.5,.75,.9,1])] if len(value) else None
    return out


def _groups(data):
    for panel in sorted(set(data['panel'].tolist())):
        m = data['panel'] == panel
        yield panel, m
        for language in ('en','zh'):
            q = m & (data['language']==language)
            if q.any(): yield panel+'/'+language, q


def _same_source(data, panel, language, label):
    indices=np.where((data['panel']==panel) & (data['language']==language) & (data['label']==label))[0]
    result=defaultdict(list)
    margins=get_margin(data)
    for i in indices:
        result[str(data['canonical_source'][i])].append(float(margins[i]))
    return {k:float(np.mean(v)) for k,v in result.items()}


def paired_conditions(data, output_dir):
    rows=[]
    for language in ('en','zh'):
        for label in (0,1):
            base=_same_source(data,'official_online',language,label)
            for condition in ('regular_noisy','heldout_noisy'):
                target=_same_source(data,condition,language,label)
                for key in sorted(base.keys() & target.keys()):
                    clean,noisy=base[key],target[key]
                    rows.append(dict(language=language,label=int(label),canonical_source=key,
                         condition=condition,online_margin=clean,noisy_margin=noisy,
                         shift=noisy-clean,online_error=int((clean>=0)!=(label==0)),
                         noisy_error=int((noisy>=0)!=(label==0))))
    _csv(output_dir/'paired_sources.csv',rows)
    result={}
    for language in ('en','zh'):
        for label in (0,1):
            for condition in ('regular_noisy','heldout_noisy'):
                q=[r for r in rows if r['language']==language and r['label']==label and r['condition']==condition]
                counts=Counter(('wrong' if r['online_error'] else 'correct',
                                'wrong' if r['noisy_error'] else 'correct') for r in q)
                result[f'{language}/{label}/{condition}']={
                    'paired_sources':len(q),
                    'both_wrong':counts['wrong','wrong'], 'online_only_wrong':counts['wrong','correct'],
                    'noisy_only_wrong':counts['correct','wrong'], 'both_correct':counts['correct','correct'],
                    'median_noisy_minus_online_margin':float(np.median([r['shift'] for r in q])) if q else None}
    _json(output_dir/'paired_summary.json',result)
    return result


def epoch_drift(epochs, output_dir):
    first,last=epochs[0][2],epochs[-1][2]
    ref=epochs[0][2]
    rows=[]
    for epoch,step,d in epochs:
        for name,mask in _groups(d):
            met=metrics(d,mask)
            rows.append(dict(epoch=epoch,step=step,group=name,
                             n=met['n'],real_recall=met.get('real_recall'),fake_recall=met.get('fake_recall'),
                             macro_f1=met.get('macro_f1'),auc=met.get('auc'),eer_approx=met.get('eer_approx'),
                             real_fp=met.get('real_false_positives'),fake_fn=met.get('fake_false_negatives')))
    _csv(output_dir/'epoch_metrics.csv',rows)
    # Compare earliest/later available epoch predictions at identical sources.
    wanted = next((d for e,_,d in epochs if e==6), first)
    flip=[]
    for panel,mask in _groups(ref):
        if '/' not in panel: continue
        for label in (0,1):
            q=mask & (ref['label']==label)
            m0=get_margin(wanted)[q]; m1=get_margin(last)[q]
            wrong0=(m0>=0)!=(label==0);wrong1=(m1>=0)!=(label==0)
            flip.append(dict(group=panel,label=label,n=int(q.sum()),
                earlier_epoch=next((e for e,_,d in epochs if d is wanted),epochs[0][0]),
                later_epoch=epochs[-1][0],both_wrong=int(np.sum(wrong0 & wrong1)),
                newly_wrong=int(np.sum(~wrong0 & wrong1)),recovered=int(np.sum(wrong0 & ~wrong1)),
                both_correct=int(np.sum(~wrong0 & ~wrong1)),
                earlier_margin_median=float(np.median(m0)) if len(m0) else None,
                later_margin_median=float(np.median(m1)) if len(m1) else None))
    _json(output_dir/'epoch_flips.json',flip)
    return rows,flip


def plot_scores(epochs, audit, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    last=epochs[-1][2]
    fig,ax=plt.subplots(figsize=(9,4))
    for label in (0,1):
        q=(last['panel']=='official_online') & (last['language']=='en') & (last['label']==label)
        ax.hist(np.clip(get_margin(last)[q],-12,12),bins=80,alpha=.5,
                label='EN '+('Fake' if label==0 else 'Real'),density=True)
    ax.axvline(0,linestyle='--',color='black',label='Decision margin 0')
    ax.set(xlabel='Logit margin: fake - real',ylabel='Density',title='Select Online EN (descriptive, no threshold fit)')
    ax.legend();fig.tight_layout();fig.savefig(output_dir/'en_online_margins.png',dpi=160);plt.close(fig)


def analyze(path, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    epochs,audit=load_artifacts(path)
    trends,flips=epoch_drift(epochs,out)
    sel=epochs[-1][2]
    pairs=paired_conditions(sel,out)
    metrics_by_group={name:metrics(sel,m) for name,m in _groups(sel)}
    if audit is not None:
        metrics_by_group.update({'audit:'+name:metrics(audit,m) for name,m in _groups(audit)})
        paired_conditions(audit,out/'audit') if False else None
    _json(out/'score_summary.json',dict(
        input=str(path),epochs=[e for e,_,_ in epochs],last_select=metrics_by_group,
        paired_source_summary=pairs,
        interpretation='Descriptive analysis only. No test labels, no Dev-fitted calibration, and no model updates.',
        caveats=['AUC/EER are not fixed-threshold F1','Historical Dev/Audit has been seen during development',
                 'Epoch drift needs no epoch checkpoints; feature-level epoch drift does require saved epoch weights']))
    plot_scores(epochs,audit,out)
    return {'epochs':len(epochs),'select_rows':len(sel['ids']),
            'audit_rows':len(audit['ids']) if audit is not None else 0,
            'output':str(out)}
