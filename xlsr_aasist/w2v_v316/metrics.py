"""Same official-style local F1 proxy, with tie-safe AP/AUC/EER diagnostics."""
from collections import defaultdict
import numpy as np
from w2v_v39.metrics import measure as prior_measure, GROUPS


def ap_eer(labels,scores):
    labels,scores = np.asarray(labels),np.asarray(scores,dtype=np.float64)
    if len(labels)!=len(scores) or not np.isfinite(scores).all(): raise ValueError('Invalid scores')
    positives = labels==0
    p,n = int(positives.sum()),int((~positives).sum())
    if not p or not n: return dict(ap=None,eer=None)
    order = np.argsort(-scores,kind='stable'); s=scores[order]; y=positives[order]
    ends = np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp = np.r_[0,np.cumsum(y)[ends]]; fp=np.r_[0,np.cumsum(~y)[ends]]
    tpr,fpr = tp/p,fp/n
    precision = tp[1:]/(tp[1:]+fp[1:])
    ap = float(np.sum(np.diff(tpr)*precision))
    fnr = 1-tpr; difference=fpr-fnr
    hi = int(np.flatnonzero(difference>=0)[0]); lo=max(0,hi-1)
    ratio = 0. if hi==lo else -difference[lo]/(difference[hi]-difference[lo])
    eer = float(fpr[lo]+ratio*(fpr[hi]-fpr[lo]))
    return dict(ap=ap,eer=eer)


def measure(rows,logits,target=.99):
    value = prior_measure(rows,logits,target=target)
    buckets = defaultdict(list)
    for i,r in enumerate(rows):
        buckets[r['condition']].append(i); buckets[r['condition']+'/'+r['language']].append(i)
    z=np.asarray(logits); margins=z[:,0].astype(float)-z[:,1]
    for name,indices in buckets.items():
        value['groups'][name].update(ap_eer([rows[i]['label'] for i in indices],margins[indices]))
    value['ranking_positive_class']='fake=0; AP is average precision, EER uses interpolated ROC'
    return value


def print_metrics(tag,value):
    print(f'\n[Dev] V3.16 {tag} Clean={100*value["clean_f1"]:.3f} Noisy={100*value["noisy_f1"]:.3f} Weighted={100*value["weighted_f1"]:.3f}',flush=True)
    def percent(v): return 'NA' if v is None else f'{100*v:.3f}'
    for name in GROUPS:
        g=value['groups'][name]
        print(f'  {name} Recall(fake/real)={percent(g["recall"][0])}/{percent(g["recall"][1])} '
            f'F1={percent(g["macro_f1"])} AP={percent(g["ap"])} AUC={percent(g["auc"])} EER={percent(g["eer"])} '
            f'real@99%fake={percent(value["matched"][name]["real_recall"])}',flush=True)
