"""Read-only reuse of the released v1 feature review; no parent re-encoding."""
from datetime import datetime
from pathlib import Path
import os
import shutil
import numpy as np
import torch
from torch import nn
from w2v_v31810.common import ROOT, atomic_json, read_json, fingerprint, digest, verify_files
from . import SCHEMA

LEGACY_SCHEMA = 'rtc_v31812_frozen_global_residual_v1'
# Only the known release has the unchanged extraction contract audited here.
LEGACY_CODE = {
    '__init__.py':'74bb02c79bcf7de0e835ea0a435a0f75907d0c349ad46f4b58d0fff44f684815',
    'cache.py':'a6f494ced1fb6abedd14b51d8004b9b960a74a9aa59494509268516af4c38811',
    'evaluate.py':'38d98e1c24a602e5b1c9db3ef55640e16453c890c10b6ad3485438043b286798',
    'metrics.py':'f63df5550f6226f09909f57c9d90d9add8e2346623a7688359df73639fa5d793',
    'model.py':'382caab7dadcf2a0e333ec20dd4d73e69805aa2d9a18689ce40dc993956b546d',
    'partition.py':'758add1e20a28606779dca7442ed6ac4ea5f12e9ebbb40480e9281ccdffb5627',
    'training.py':'a96b3e18c7b9aab21d86fa8e8d764f98913f6167006253d42fd1f8acc33f6fb8',
    'workflow.py':'7dfbacb7bc9fb936c85c4e2852e9e4013f7553a229559ace890755067cd21f1e',
}


def immutable_copy(source, target):
    """Hard-link immutable snapshots on the same disk; copy on other filesystems."""
    target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)


def prepare_continuation(args, current_code):
    source=Path(args.continue_from).expanduser().resolve()
    old=read_json(source/'config.json');identity=fingerprint(old)
    if old.get('schema')!=LEGACY_SCHEMA or old.get('code')!=LEGACY_CODE:
        raise ValueError('Cache continuation requires the audited V3.18.12 v1 release; changed extractors are not interchangeable')
    done=read_json(source/'completed.json');summary=read_json(source/'summary.json')
    if (done.get('identity')!=identity or summary.get('identity')!=identity
            or done.get('summary_sha256')!=digest(source/'summary.json')
            or summary.get('stage')!='feature_review'
            or done.get('decision')!='stop_global_hypothesis_not_supported'):
        raise ValueError('Expected a completed v1 feature-review run, before any residual training')
    review=read_json(source/'feature_review.json')
    if fingerprint(review)!=fingerprint(summary['feature_review']):
        raise ValueError('Original feature review changed after completion')
    verify_files({str(source/k):v for k,v in old['artifacts'].items()})
    inherited={}
    for kind in ('global','local'):
        selected=read_json(source/f'probe_{kind}'/'selected.json')
        history=read_json(source/f'probe_{kind}'/'history.json')
        probe=review['probes'][kind]
        if (selected.get('identity')!=identity or selected.get('kind')!=kind
                or selected.get('epoch')!=probe['epoch']
                or fingerprint(selected['metrics'])!=fingerprint(probe['inner'])
                or not any(h['epoch']==probe['epoch'] and h['best'] for h in history)):
            raise ValueError('Original probe selection does not match its completed review')
        for name in (f'probe_{kind}/selected.json',f'probe_{kind}/history.json',f'probe_{kind}_select.npz'):
            inherited[name]=digest(source/name)
    for name in [f'train_{v}' for v in range(old['cache_views'])]+['train_inner','dev_select']:
        manifest=source/'cache'/name/'manifest.json'
        if not manifest.is_file() or read_json(manifest).get('identity')!=identity:
            raise ValueError('Original feature cache is missing or has a different identity: '+str(manifest))
    run=Path(args.run).expanduser().resolve() if args.run else ROOT/'exp'/('w2v_v31812_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_continued')
    if run==source or source in run.parents or (run.exists() and any(run.iterdir())):
        raise ValueError('Use a new empty run outside the original experiment')
    run.mkdir(parents=True,exist_ok=True)
    artifacts=dict(old['artifacts'])
    for name in artifacts:immutable_copy(source/name,run/name)
    for name,sha in inherited.items():
        immutable_copy(source/name,run/name);artifacts[name]=sha
    for name,target in (('config.json','origin_config.json'),('feature_review.json','feature_review_inherited.json')):
        immutable_copy(source/name,run/target);artifacts[target]=digest(run/target)
    cfg=dict(old,schema=SCHEMA,code=current_code,artifacts=artifacts,
             epochs=args.epochs,patience=args.patience,batch_size=args.batch_size,
             lr=args.lr,weight_decay=args.weight_decay,train_device=args.train_device,
             upload_report=args.upload_report,
             continued_from=dict(run=str(source),identity=identity,summary_sha256=done['summary_sha256']),
             continuation_policy='Reuse verified immutable evidence and completed probes; train fresh residuals only')
    atomic_json(run/'config.json',cfg)
    print('CONTINUED_FROM='+str(source),flush=True)
    print('Reusing completed features and both probes. Keep the original cache directory.',flush=True)
    return run,cfg


def load_classifier(run, cfg):
    """Load just the original 384 -> 2 weights, without constructing Omni/AASIST."""
    path=Path(run)/'parent.pt'
    if digest(path)!=cfg['artifacts']['parent.pt']:raise ValueError('Frozen parent snapshot changed')
    values=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    prefix='head.aasist.out_layer.'
    state={key:values[prefix+key] for key in ('weight','bias')}
    for key,shape in (('weight',(2,384)),('bias',(2,))):
        if state[key].shape!=shape or state[key].dtype!=torch.float32 or not torch.isfinite(state[key]).all():
            raise ValueError('Invalid original classifier tensor: '+key)
    classifier=nn.Linear(384,2)
    classifier.load_state_dict(state)
    return classifier.requires_grad_(False).eval()


def inherited_review(run, cfg, validation, selection, report):
    """Reconcile saved probe predictions with the verified cache and labels."""
    run=Path(run);review=read_json(run/'feature_review_inherited.json')
    for scope,data in (('inner',validation),('select',selection)):
        if fingerprint(report(data['rows'],data['logits']))!=fingerprint(review['baseline'][scope]):
            raise ValueError('Inherited baseline metrics do not match the completed cache')
    fields={'ids':'id','canonical_source':'canonical_source','language':'language','label':'label','panel':'validation_panel'}
    for kind in ('global','local'):
        path=run/f'probe_{kind}_select.npz'
        if digest(path)!=cfg['artifacts'][path.name]:raise ValueError('Inherited probe scores changed')
        with np.load(path,allow_pickle=False) as data:
            for field,key in fields.items():
                if not np.array_equal(data[field],np.asarray([r[key] for r in selection['rows']])):
                    raise ValueError('Inherited probe sources or labels changed: '+field)
            if not np.array_equal(data['baseline'],selection['logits']):
                raise ValueError('Inherited baseline logits changed')
            if fingerprint(report(selection['rows'],data['logits']))!=fingerprint(review['probes'][kind]['select']):
                raise ValueError('Inherited probe metrics do not match predictions')
    return review
