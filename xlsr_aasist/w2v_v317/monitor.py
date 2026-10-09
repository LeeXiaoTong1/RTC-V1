"""Small fixed Train replay and group-matched generalization diagnostics."""
from collections import defaultdict
import hashlib
import numpy as np
from w2v_v316_tfcl.metrics import binary_metrics


def probe_rows(rows,cfg):
    cells=defaultdict(list)
    for r in rows:
        if r['split']=='train' and r['condition']=='online': cells[(r['language'],r['label'])].append(r)
    if set(cells)!={('en',0),('en',1),('zh',0),('zh',1)}:raise ValueError('Train probe requires all four official Online groups')
    result=[]
    for key,values in sorted(cells.items()):
        ordered=sorted(values,key=lambda r:hashlib.sha256(f'{cfg["seed"]}/fixed-train-probe/{r["group_id"]}'.encode()).digest())
        result.extend(ordered[:cfg['train_probe_per_group']])
    return result


def diagnostics(rows,logits):
    z=np.asarray(logits,dtype=np.float64); labels=np.asarray([r['label'] for r in rows])
    if z.shape!=(len(rows),2) or not np.isfinite(z).all():raise ValueError('Invalid diagnostic logits')
    shifted=z-z.max(1,keepdims=True);lse=np.log(np.exp(shifted).sum(1));p=np.exp(shifted-lse[:,None])
    ce=lse-shifted[np.arange(len(rows)),labels]
    predicted=np.where(z[:,0]>=z[:,1],0,1)  # Matches P(fake)>=0.5 tie policy.
    cells=defaultdict(list)
    for i,r in enumerate(rows):cells[(r['condition'],r['language'],r['label'])].append(i)
    groups={f'{c}/{l}/{y}':dict(count=len(ix),ce=float(ce[ix].mean()),
        recall=float((predicted[ix]==labels[ix]).mean()),mean_true_probability=float(p[ix,labels[ix]].mean()))
        for (c,l,y),ix in sorted(cells.items())}
    by_condition={}
    for condition in sorted({r['condition'] for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['condition']==condition]
        metric=binary_metrics(labels[ix],z[ix,0]-z[ix,1])
        values=[g for k,g in groups.items() if k.startswith(condition+'/')]
        by_condition[condition]=dict(metric,balanced_ce=float(np.mean([g['ce'] for g in values])),
            balanced_recall=float(np.mean([g['recall'] for g in values])))
    return dict(groups=groups,conditions=by_condition,sample_count=len(rows),
        scope='eval-mode fixed replay; Train probe is seen training data, not a held-out test')
