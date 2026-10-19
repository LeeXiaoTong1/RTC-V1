"""Frozen V3.18.10 feature probes; no original detector parameters are updated.

Train-only source-grouped model selection. Dev Select and historical Audit are
used solely to describe transfer; no teacher, test, or Progress samples enter.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .scores import _json, metrics, _groups


def _stable(seed, kind, source):
    return int(hashlib.sha256(f'{seed}|{kind}|{source}'.encode()).hexdigest()[:16],16)


def subset_tickets(rows, tickets, *, limit, seed, split):
    """Select source groups, keeping their existing panel/role views together."""
    strata=defaultdict(set)
    for row in rows:
        strata[row['language'],int(row['label'])].add(str(row['canonical_source']))
    chosen=set()
    for cell, sources in strata.items():
        selection=sorted(sources,key=lambda s:(_stable(seed,'probe_'+split,s),s))[:limit]
        chosen.update(selection)
    selected=[(r,t) for r,t in zip(rows,tickets) if str(r['canonical_source']) in chosen]
    if not selected: raise ValueError('No diagnostic audio selected')
    return [r for r,_ in selected],[t for _,t in selected]


def train_tickets(rows, limit, seed, noisy=True):
    strata=defaultdict(lambda:defaultdict(list))
    for index,row in enumerate(rows):
        if row.get('condition') not in ('offline','online'):
            continue
        strata[(row['language'],int(row['label']))][str(row['canonical_source'])].append((index,row))
    tickets=[]
    for cell,by_source in sorted(strata.items()):
        origins=sorted(by_source,key=lambda s:(_stable(seed,'train',s),s))[:limit]
        for origin in origins:
            members=by_source[origin]
            for kind,condition in (('online','online'),('offline','offline')):
                if kind=='offline' and not noisy:continue
                choices=sorted(((i,r) for i,r in members if r['condition']==condition),key=lambda z:z[1]['id'])
                if not choices:continue
                i,r=choices[0]
                ticket=dict(index=i,kind=kind,epoch=0,step=0,canonical_source=origin,
                    occurrence=f'v31811-fixed-train:{kind}:{origin}',probe=True,
                    base_variant=0,base_family='original',base_generation=0,base_dynamic=False,
                    base_view_id='v31811-original')
                if kind=='offline':
                    family=('webrtc','ffmpeg','light')[_stable(seed,'family',origin)%3]
                    scene=('office','coffee','rain','footsteps','keyboard','echo')[_stable(seed,'scene',origin)%6]
                    ticket.update(domain_override={'family':family,'scene':scene},
                         nuisance_seed=_stable(seed,'nuisance',origin)%2**31,
                         condition_rank=0,noise_recipe_view=0)
                tickets.append(ticket)
    if not tickets: raise ValueError('Training source registry contains no usable original Online/Offline recordings')
    return tickets


def _pooled(x):
    import torch
    if x.ndim != 2 or len(x)<1:
        raise ValueError('Bad feature shape')
    v=x.detach().float()
    return torch.cat((v.mean(0),v.var(0,unbiased=False).add(1e-8).sqrt())).cpu().numpy()


def _extract(model, waves):
    """One student encoding plus one normal head pass, with bounded reductions.

    Capture the exact FINAL, native-normalized SSL feature at the existing
    projection call (no second SSL forward); capture the 384d head input with
    the existing out_layer pre-hook (no second AASIST forward).
    """
    from w2v_v31810.model import Detector
    import torch
    if not isinstance(model,Detector):
        raise TypeError('Expected production V3.18.10 Detector, not a substitute encoder')
    fusion=model.fusion
    last_idx=fusion.layers[-1]
    original=fusion.project_layer
    shapes=[(len(w)-400)//320+1 for w in waves]
    captured_last=[]; captured_head=[]
    def intercept(hidden,index,weights=None):
        if index==last_idx:
            if hidden.ndim!=3 or hidden.shape[0]!=len(waves) or hidden.shape[2]!=model.cfg['encoder_dim']:
                raise ValueError('Native final encoder frame geometry mismatch')
            captured_last.extend(_pooled(hidden[i,:n]) for i,n in enumerate(shapes))
        return original(hidden,index,weights)
    def head_hook(_module, args):
        tensor=args[0]
        if tensor.ndim!=2 or tensor.shape[1]!=384 or tensor.shape[0]!=len(waves):
            raise ValueError('Classifier-input hook shape mismatch')
        captured_head.extend(row.detach().float().cpu().numpy() for row in tensor)
    hook=model.head.aasist.out_layer.register_forward_pre_hook(head_hook)
    fusion.project_layer=intercept
    try:
        with torch.inference_mode():
            encoded=model.encode(waves)
            fused=[_pooled(seq) for seq in encoded]
            logits=model.classify_sequences(encoded).detach().float().cpu().numpy()
    finally:
        fusion.project_layer=original
        hook.remove()
    if len(captured_last)!=len(waves) or len(captured_head)!=len(waves):
        raise RuntimeError('No/duplicate model feature hook: cannot silently substitute features')
    return dict(ssl_last=np.stack(captured_last),fused=np.stack(fused),preclassifier=np.stack(captured_head),logits=logits)


def _collect(model, cfg, rows, tickets, stage):
    from w2v_v31810.data import Waves
    from w2v_v31810.validation_data import FixedWaves
    from w2v_v31810.data import loader, close, microbatches
    local_cfg=dict(cfg,workers=int(cfg.get('eval_workers',2)),prefetch_factor=1)
    if stage!='train':
        assert stage in ('select','audit')
        local_cfg['scene_noise_records']={'dev':cfg['_probe_scene_noise'][stage]}
    dataset = Waves(rows,local_cfg,'train') if stage=='train' else FixedWaves(rows,local_cfg,'dev')
    # Tickets refer to the subset's own row indices, never the full protocol.
    ordered=[]
    for i,ticket in enumerate(tickets):
        t=dict(ticket,index=i)
        ordered.append(t)
    chunks=[ordered[i:i+int(cfg['eval_batch'])] for i in range(0,len(ordered),int(cfg['eval_batch']))]
    batches=loader(dataset,local_cfg,batch_sampler=chunks)
    result={key:[] for key in ('ssl_last','fused','preclassifier','logits')};meta=[]
    try:
        for units in batches:
            for micro in microbatches(units,int(cfg['microbatch']),int(cfg['frame_budget'])):
                examples=[r for unit in micro for r in unit]
                values=_extract(model,[r['wave'] for r in examples])
                for k in result:result[k].append(values[k])
                for r in examples:
                    meta.append(dict(id=r['id'],canonical_source=r['canonical_source'],language=r['language'],
                                     label=int(r['label']),panel=(r.get('validation_panel') if stage!='train'
                                        else ('official_online' if r['role']=='online' else 'regular_noisy'))))
    finally:
        close(batches)
    if len(meta)!=len(tickets):
        raise ValueError('Diagnostic features dropped an audio view')
    result={k:np.concatenate(v,axis=0) for k,v in result.items()}
    result.update({k:np.asarray([r[k] for r in meta]) for k in ('id','canonical_source','language','label','panel')})
    for k in ('ssl_last','fused','preclassifier','logits'):
        if not np.isfinite(result[k]).all():raise ValueError('Nonfinite diagnostic features')
    return result


def _group_holdout(groups,seed,fraction=.2):
    # Source-stratified split: multiple views of a source never cross boundary.
    s=sorted(set(groups),key=lambda g:(_stable(seed,'train_holdout',g),g))
    hold=set(s[:max(1,round(len(s)*fraction))]);mask=np.asarray([g not in hold for g in groups],bool)
    return mask


def _balanced_intercept(margin,label,language,panel,mask):
    # Convex train-only intercept fit, leaving original logit slope fixed.
    q=np.flatnonzero(mask)
    d=margin[q]; y=(label[q]==0).astype(float)
    keys=list(zip(panel[q],language[q],label[q]))
    counts=defaultdict(int)
    for key in keys: counts[key]+=1
    w=np.asarray([1.0/counts[key] for key in keys]);w/=w.sum()
    lo,hi=-60.,60.
    for _ in range(90):
        mid=(lo+hi)/2
        grad=np.sum(w*(expit(d+mid)-y))
        if grad>0:hi=mid
        else:lo=mid
    return float((lo+hi)/2)


def _to_score_dataset(features):
    return dict(logits=features['logits'],label=features['label'],panel=features['panel'],
                language=features['language'],canonical_source=features['canonical_source'])


def _report_group(data,bias=0.):
    return {name:metrics(data,mask,bias) for name,mask in _groups(data)}


def _calibration(train, select, audit, seed):
    data=_to_score_dataset(train);sel=_to_score_dataset(select)
    masks=_group_holdout(data['canonical_source'],seed)
    for target in (0,1):
        if not any(data['label'][masks]==target) or not any(data['label'][~masks]==target):
            raise ValueError('Train source holdout is not label-stratified; cannot calibrate')
    margin=data['logits'][:,0]-data['logits'][:,1]
    bias=_balanced_intercept(margin,data['label'],data['language'],data['panel'],masks)
    train_held={k:v[~masks] for k,v in data.items()}
    out=dict(bias=bias,fit='Train source-only weighted logistic intercept, no temperature or Dev tuning',
        train_holdout=_report_group(train_held,bias),
        select=_report_group(sel,bias),
        select_without_bias=_report_group(sel,0.))
    if audit is not None:
        out['historical_audit_descriptive']=_report_group(_to_score_dataset(audit),bias)
    return out


def _fit_one(name, train, select, audit, seed):
    features=train[name];targets=(train['label']==0).astype(int)
    groups=train['canonical_source']; cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed)
    candidates=(.01,.1,1.)
    choices=[]
    for c in candidates:
        costs=[]
        for a,b in cv.split(features,targets,groups):
            if len(np.unique(targets[a]))!=2 or len(np.unique(targets[b]))!=2:
                raise ValueError('Train-only grouped folds lost one of the labels')
            clf=make_pipeline(StandardScaler(), LogisticRegression(C=c,class_weight='balanced',
                solver='liblinear',max_iter=1500,random_state=seed))
            clf.fit(features[a],targets[a])
            prob=clf.predict_proba(features[b])[:,1]
            costs.append(log_loss(targets[b],np.clip(prob,1e-6,1-1e-6),labels=[0,1]))
        choices.append(dict(C=c,mean_train_cv_log_loss=float(np.mean(costs))))
    choice=min(choices,key=lambda item:(item['mean_train_cv_log_loss'],item['C']))
    clf=make_pipeline(StandardScaler(), LogisticRegression(C=choice['C'],class_weight='balanced',
        solver='liblinear',max_iter=1500,random_state=seed))
    clf.fit(features,targets)
    def eval_one(data):
        if data is None:return None
        score=clf.decision_function(data[name]).astype(float)
        # Wrap probe scores as binary logits, fake logit = score, real logit = 0.
        scores=_to_score_dataset(data);scores['logits']=np.column_stack((score,np.zeros_like(score)))
        return _report_group(scores)
    return dict(embedding=name,dim=int(features.shape[1]),train_samples=len(train['id']),
        best_C=choice['C'],cv_candidates=choices,train_fit=eval_one(train),
        select=eval_one(select),historical_audit_descriptive=eval_one(audit))


def run(run, out, max_train_per_class=64, max_dev_per_class=64, workers=2,
        checkpoint='best', include_audit=False):
    import torch
    from w2v_v31810.evaluate import selected
    from w2v_v31810.common import read_json, apply_partial, seed_all, digest
    from w2v_v31810.config import verify_inputs
    from w2v_v31810.model import load_model
    run=Path(run).resolve(); out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not run.is_dir():raise ValueError('Feature probes require original server run, not report ZIP')
    if max_train_per_class<3 or max_dev_per_class<3:
        raise ValueError('Each group must have at least 3 requested sources')
    original,weights,identity=selected(run,checkpoint)
    # Code/model provenance checked before any costly GPU inference.
    verify_inputs(original,training=False)
    cfg=dict(original, workers=workers,eval_workers=workers)
    seed_all(cfg['seed'])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Production Omni 3B feature extraction requires CUDA BF16')
    train=read_json(run/'train_rows.json');protocol=read_json(run/'validation_protocol.json')
    if original['validation_protocol_hash'] != __import__('w2v_v31810.common',fromlist=['fingerprint']).fingerprint(protocol):
        raise ValueError('Stored validation protocol differs from checkpoint')
    model=load_model(cfg);apply_partial(model,weights);del weights
    model.set_phase(True);model.requires_grad_(False);model.eval()
    cfg['_probe_scene_noise']={s:protocol[s]['scene_noise_records'] for s in ('select','audit')}
    tickets=train_tickets(train,max_train_per_class,cfg['seed'])
    train_data=_collect(model,cfg,train,tickets,'train')
    train_path=out/'train_features.npz';np.savez_compressed(train_path,**train_data)
    print('V31811_TRAIN_FEATURES='+str(train_path),flush=True)
    def collect(split):
        part=protocol[split];sub,t=subset_tickets(part['rows'],part['tickets'],
            limit=max_dev_per_class,seed=cfg['seed'],split=split)
        data=_collect(model,cfg,sub,t,split)
        path=out/(split+'_features.npz');np.savez_compressed(path,**data)
        print('V31811_'+split.upper()+'_FEATURES='+str(path),flush=True)
        return data
    select_data=collect('select')
    audit_data=collect('audit') if include_audit else None
    report=dict(source_run=str(run),checkpoint=identity['selected'],checkpoint_hash=digest(run/'last.pt'),
        source_registry=cfg.get('source_registry'),source_training_ids=len(set(train_data['canonical_source'])),
        observed_select_ids=len(set(select_data['canonical_source'])),
        historical_audit_included=include_audit,
        caveat='Pooled linear probes are a restricted readout; a failed probe does not prove lack of nonlinear/spatiotemporal evidence',
        fixed_classifier_select=_report_group(_to_score_dataset(select_data)),
        calibration=_calibration(train_data,select_data,audit_data,cfg['seed']),
        probes={})
    for name in ('ssl_last','fused','preclassifier'):
        print('V31811_PROBE_FIT='+name,flush=True)
        report['probes'][name]=_fit_one(name,train_data,select_data,audit_data,cfg['seed'])
    _json(out/'feature_probe_summary.json',report)
    return dict(artifacts=['train_features.npz','select_features.npz','feature_probe_summary.json']+
        (['audit_features.npz'] if include_audit else []),checkpoint=identity['selected'],
        result=str(out/'feature_probe_summary.json'))
