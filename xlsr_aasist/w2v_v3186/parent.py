"""Strict best-only migration; removing window ASP is not function-preserving."""
from pathlib import Path
import torch
from w2v_v3185.evaluate import selected
from w2v_v3183r2.parent import PARENT_ARTIFACTS, baseline_check as cached_baseline
from .common import read_json,digest,verify_files,partial_state,apply_partial,atomic_json


def inspect_parent(run,tag=None):
    run=Path(run).expanduser().resolve()
    cfg,weights,meta=selected(run,'best')
    if cfg.get('version')!='3.18.5' or cfg.get('variant')!='C3' or cfg.get('risk_coefficient')!=.25:
        raise ValueError('V3.18.6 requires a completed V3.18.5 C3 best')
    if cfg.get('pooling')!='asp' or tuple(cfg.get('view_factors',()))!=(5,5,1,3):
        raise ValueError('Parent must use V3.18.5 ASP and targeted 5/5/1/3 augmentation')
    if tag is not None and tag!=meta['selected']:raise ValueError('Expected parent '+tag+', found '+meta['selected'])
    verify_files({str(run/n):h for n,h in cfg['saved_manifests'].items()})
    files={str(run/n):(meta['checkpoint_sha256'] if n=='last.pt' else digest(run/n)) for n in PARENT_ARTIFACTS}
    return cfg,weights,dict(run=str(run),tag=meta['selected'],kind='best',parent_version='3.18.5',
        checkpoint_sha256=meta['checkpoint_sha256'],files=files)


def transfer(model,values):
    expected=partial_state(model)
    removed={key for key in values if key.startswith(('head.contribution.','head.attention_delta.'))}
    if set(values)-removed!=set(expected):
        raise ValueError('Parent tensor inventory mismatch; only external pooling may be removed')
    migrated={}
    for key,target in expected.items():
        value=values[key]
        if value.dtype!=target.dtype or not torch.isfinite(value).all():raise ValueError('Invalid parent tensor: '+key)
        if key=='head.aasist.out_layer.weight':
            if value.shape!=(2,320) or target.shape!=(2,160):raise ValueError('Classifier migration requires 320 -> 160')
            value=value[:,:160].contiguous()
        elif value.shape!=target.shape:raise ValueError('Parent shape mismatch: '+key)
        migrated[key]=value
    for key in removed:
        if not torch.isfinite(values[key]).all():raise ValueError('Invalid discarded parent tensor: '+key)
    apply_partial(model,migrated);model.set_phase(True)
    return dict(copied_tensor_count=len(migrated),removed_tensors=sorted(removed),
        classifier_initialization='first 160 parent mean columns and bias; std columns removed',
        bn_statistics_reset=[],optimizer_inherited=False,initial_function_preserved=False,
        exact_continuation=False,external_window_pooling=False)


def initialize(model,cfg):
    init=cfg['initialization'];verify_files(init['files'])
    old,values,meta=inspect_parent(init['run'],init['tag'])
    if meta['checkpoint_sha256']!=init['checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    audit=transfer(model,values)
    audit.update(mode='parent-transfer',parent_version=old['version'],parent_tag=meta['tag'],
        parent_checkpoint_sha256=meta['checkpoint_sha256'],new_fusion=model.fusion.describe(),
        note='Whole-utterance AASIST changes the function. Cached parent metrics are references, never candidate scores.')
    print('[Initialization] V3.18.5 best inherited; whole-utterance AASIST; fresh optimizer; initial function changed',flush=True)
    return audit


def baseline_check(model,cfg,run,dev,partition,infer,measure):
    result=cached_baseline(model,cfg,run,dev,partition,infer,measure)
    result.update(prediction_identity_expected=False,new_model_reproduces_parent=False,
        note='Cached V3.18.5 raw best scores only. Structural migration is not prediction-preserving; no parent fallback.')
    atomic_json(Path(run)/'parent_baseline.json',result)
    return result
