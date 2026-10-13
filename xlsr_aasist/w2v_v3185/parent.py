"""Transfer an R2 best into ASP while preserving its initial evaluation function."""
from pathlib import Path
import torch
from w2v_v3184.parent import inspect_parent as inspect_old
from w2v_v3183r2.parent import baseline_check as cached_baseline
from .common import verify_files,partial_state,apply_partial,atomic_json


def inspect_parent(run,tag=None):
    old,weights,meta=inspect_old(run,tag)
    if old['version']!='3.18.3-R2':
        raise ValueError('targeted augmentation requires the full-layer R2 best; seal a stopped R2 run first')
    return old,weights,meta


def transfer(model,values):
    expected=partial_state(model)
    added={k for k in expected if k.startswith('head.attention_delta.')}
    if set(values)!=set(expected)-added:raise ValueError('R2 tensor inventory mismatch')
    migrated={};extended='head.aasist.out_layer.weight'
    for key,value in values.items():
        target=expected[key]
        if value.dtype!=target.dtype or not torch.isfinite(value).all():raise ValueError('Invalid parent tensor: '+key)
        if added and key==extended:
            if value.shape!=(2,160) or target.shape!=(2,320):raise ValueError('Classifier shape mismatch')
            migrated[key]=torch.cat((value,torch.zeros_like(value)),dim=1)
        else:
            if value.shape!=target.shape:raise ValueError('Parent shape mismatch: '+key)
            migrated[key]=value
    # Reset only the new terminal attention projection, even for an injected
    # test factory; the other attention weights are newly initialized.
    for key in added:
        migrated[key]=torch.zeros_like(expected[key]) if '.2.' in key else expected[key]
    apply_partial(model,migrated);model.set_phase(False)
    return dict(copied_tensor_count=len(values),new_attention_tensors=sorted(added),
        extended_classifier=bool(added),bn_statistics_reset=[],optimizer_inherited=False,
        initial_function_preserved=True,all_partial_tensors_identical=not added)


def initialize(model,cfg):
    init=cfg['initialization'];verify_files(init['files'])
    old,values,meta=inspect_parent(init['run'],init['tag'])
    if meta['checkpoint_sha256']!=init['checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    result=transfer(model,values)
    result.update(mode='parent-transfer',parent_version=old['version'],parent_tag=meta['tag'],
        parent_checkpoint_sha256=meta['checkpoint_sha256'],pooling=cfg['pooling'],new_fusion=model.fusion.describe(),
        note='Mean classifier inherited; new std coefficients and terminal attention initialized to zero. Cached reference is not a new GPU measurement.')
    print(f'[Initialization] R2 best={meta["tag"]}; pooling={cfg["pooling"]}; initial function preserved; BN preserved; fresh optimizer',flush=True)
    return result


def baseline_check(model,cfg,run,dev,partition,infer,measure):
    result=cached_baseline(model,cfg,run,dev,partition,infer,measure)
    result.update(prediction_identity_expected=True,new_model_reproduces_parent=None,
        note='Cached R2 raw scores; initial function follows checked tensor migration. No repeated full Dev GPU inference.')
    atomic_json(Path(run)/'parent_baseline.json',result)
    return result
