"""Strict V3.18.3 best transfer; cached reference metrics are not an identity claim."""
from pathlib import Path
import numpy as np
import torch
from w2v_v3183.evaluate import selected
from .common import read_json, digest, verify_files, atomic_json, partial_state


PARENT_ARTIFACTS=('config.json','completed.json','last.pt','train_rows.json','dev_rows.json',
                  'dev_partition.json','train_probe_tickets.json','independent_panel.json')


def inspect_parent(run,tag=None):
    run=Path(run).expanduser().resolve()
    cfg,weights,meta=selected(run,'best')
    if cfg['version']!='3.18.3' or cfg['variant']!='C3' or not cfg['local_evidence'] or cfg['risk_coefficient']!=.25:
        raise ValueError('R2 requires a completed V3.18.3 C3 data/weight source')
    if tag is not None and meta['selected']!=tag:raise ValueError(f'Expected parent {tag}, found {meta["selected"]}')
    verify_files({str(run/n):h for n,h in cfg['saved_manifests'].items()})
    files={str(run/n):(meta['checkpoint_sha256'] if n=='last.pt' else digest(run/n)) for n in PARENT_ARTIFACTS}
    return cfg,weights,dict(run=str(run),tag=meta['selected'],kind='best',
                           checkpoint_sha256=meta['checkpoint_sha256'],files=files)


def initialize(model,cfg):
    init=cfg['initialization'];verify_files(init['files'])
    old,weights,meta=inspect_parent(init['run'],init['tag'])
    if meta['checkpoint_sha256']!=init['checkpoint_sha256']:raise ValueError('Parent changed')
    mode=init['mode']
    if mode not in ('parent-transfer','pretrained'):raise ValueError('Unknown initialization mode: '+mode)
    target=model.state_dict();expected=partial_state(model)
    source_fusion={'fusion.gate','fusion.mix_logits'} if old['fusion_layers'] else set()
    mapping={name:name for name in expected if name.startswith('head.') or name.endswith(('.lora_a','.lora_b'))}
    mapping.update({'head.aasist.LL.weight':'fusion.projection.weight',
                    'head.aasist.LL.bias':'fusion.projection.bias'})
    if set(weights)!=(set(mapping)|source_fusion):
        missing=sorted((set(mapping)|source_fusion)-set(weights))
        extra=sorted(set(weights)-(set(mapping)|source_fusion))
        raise ValueError(f'Parent tensor inventory mismatch: missing={missing}; extra={extra}')
    for source,destination in mapping.items():
        value=weights[source]
        if destination not in target or value.shape!=target[destination].shape or value.dtype!=target[destination].dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid parent transfer tensor: '+source+' -> '+destination)
    for name in source_fusion:
        value=weights[name]
        shape=() if name.endswith('.gate') else (len(old['fusion_layers']),)
        if value.shape!=shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Invalid discarded parent tensor: '+name)
    if mode=='parent-transfer':
        with torch.no_grad():
            for source,destination in mapping.items():target[destination].copy_(weights[source])
    reset=[]
    for name,module in model.head.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats();module.momentum=cfg['bn_momentum'];reset.append('head.'+name)
    model.set_phase(False)
    result=dict(mode=mode,parent_tag=meta['tag'],parent_checkpoint_sha256=meta['checkpoint_sha256'],
        copied_tensors=mapping if mode=='parent-transfer' else {},discarded_parent_fusion=sorted(source_fusion),
        bn_statistics_reset=reset,optimizer_inherited=False,old_model_prediction_identity=False,
        new_fusion=model.fusion.describe(),
        note='The normalized full-layer representation changes inputs; parent logits are a reference, not an expected initial prediction.')
    print(f'[Initialization] {mode}; copied={len(result["copied_tensors"])} tensors; reset BN={len(reset)}; parent best={meta["tag"]}',flush=True)
    return result


def baseline_check(model,cfg,run,dev,partition,infer,measure):
    # No GPU pass: these scores were produced by the selected completed parent.
    init=cfg['initialization'];parent=Path(init['run'])
    verify_files({str(parent/'last.pt'):init['checkpoint_sha256']})
    state=torch.load(parent/'last.pt',map_location='cpu',weights_only=True)
    if state['best_tag']!=init['tag']:raise ValueError('Parent best tag changed')
    tensor=state['best_logits']
    if not isinstance(tensor,torch.Tensor):raise ValueError('Parent cached best scores are missing')
    expected=tensor.numpy().copy();del state
    if expected.shape!=(len(dev),2) or not np.isfinite(expected).all():raise ValueError('Parent reference score shape/nonfinite mismatch')
    metrics=measure(dev,expected)
    ix=partition['select'];selected_metrics=measure([dev[i] for i in ix],expected[ix])
    result=dict(verified=True,reference_only=True,new_model_reproduces_parent=False,parent_tag=init['tag'],
        parent_checkpoint_sha256=init['checkpoint_sha256'],metrics=metrics,reference_metrics=metrics,
        selection_metrics=selected_metrics,rows=len(dev),fusion=model.fusion.describe(),
        note='Cached raw scores of the completed parent best. This is not an R2 candidate and no new-model identity is asserted.')
    atomic_json(Path(run)/'parent_baseline.json',result)
    np.savez_compressed(Path(run)/'parent_baseline_scores.npz',logits=expected)
    print(f'[Parent reference] {init["tag"]} cached full Weighted={100*metrics["weighted_f1"]:.3f}; selection={100*selected_metrics["weighted_f1"]:.3f}; no parent inference pass',flush=True)
    return result
