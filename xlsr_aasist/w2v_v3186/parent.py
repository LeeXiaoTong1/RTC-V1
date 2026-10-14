"""Restore the complete ASP parent and verify its actual zero-update predictions."""
from pathlib import Path
import numpy as np
import torch
from w2v_v3183r2.parent import PARENT_ARTIFACTS
from .common import read_json,digest,verify_files,partial_state,apply_partial,atomic_json
from .parent_discovery import validate_parent_config,RECIPE


def inspect_parent(run,tag=None):
    run=Path(run).expanduser().resolve()
    provenance=validate_parent_config(read_json(run/'config.json'))
    if provenance['parent_recipe']==RECIPE:
        from w2v_v3185.evaluate import selected
    else:
        from w2v_v3184_aug5.evaluate import selected
    # Each original loader verifies its own schema/config identity and hash.
    # Never relabel an AUG5 checkpoint as V3.18.5 to bypass these checks.
    cfg,weights,meta=selected(run,'best')
    if validate_parent_config(cfg)!=provenance:raise ValueError('Parent configuration changed during inspection')
    if tag is not None and tag!=meta['selected']:raise ValueError('Expected parent '+tag+', found '+meta['selected'])
    verify_files({str(run/n):h for n,h in cfg['saved_manifests'].items()})
    files={str(run/n):(meta['checkpoint_sha256'] if n=='last.pt' else digest(run/n)) for n in PARENT_ARTIFACTS}
    return cfg,weights,dict(run=str(run),tag=meta['selected'],kind='best',**provenance,
        checkpoint_sha256=meta['checkpoint_sha256'],files=files)


def transfer(model,values):
    expected=partial_state(model)
    if set(values)!=set(expected):
        missing=sorted(set(expected)-set(values));extra=sorted(set(values)-set(expected))
        raise ValueError(f'Parent tensor inventory mismatch: missing={missing}; extra={extra}')
    classifier='head.aasist.out_layer.weight'
    if classifier not in expected or expected[classifier].shape!=(2,320):
        raise ValueError('Restored ASP requires the complete 320-dimensional parent classifier')
    if not all(any(key.startswith(prefix) for key in expected)
               for prefix in ('head.contribution.','head.attention_delta.')):
        raise ValueError('Restored ASP requires parent contribution and attention pooling tensors')
    for key,target in expected.items():
        value=values[key]
        if not isinstance(value,torch.Tensor) or value.dtype!=target.dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid parent tensor: '+key)
        if value.shape!=target.shape:raise ValueError('Parent shape mismatch: '+key)
    # Validate every tensor before mutating the model. In particular, retain
    # learned std coefficients, attention/contribution parameters and BN buffers.
    apply_partial(model,values);model.set_phase(True)
    return dict(copied_tensor_count=len(values),removed_tensors=[],
        classifier_initialization='all 320 parent mean/std columns and bias copied exactly',
        bn_statistics_reset=[],optimizer_inherited=False,initial_function_preserved=True,
        all_partial_tensors_identical=True,exact_continuation=False,external_window_pooling=True)


def initialize(model,cfg):
    init=cfg['initialization'];verify_files(init['files'])
    old,values,meta=inspect_parent(init['run'],init['tag'])
    if meta['checkpoint_sha256']!=init['checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    audit=transfer(model,values)
    audit.update(mode='parent-transfer',parent_version=old['version'],parent_recipe=old['recipe'],
        parent_view_factors=meta['parent_view_factors'],training_view_factors=cfg['view_factors'],parent_tag=meta['tag'],
        parent_checkpoint_sha256=meta['checkpoint_sha256'],new_fusion=model.fusion.describe(),
        note='Complete parent window/ASP detector restored. Fresh optimizer and dual PCL change training; full Dev prediction identity is checked before updates.')
    print('[Initialization] V'+old['version']+' '+old['recipe']+' best restored exactly; window ASP and BN preserved; fresh optimizer',flush=True)
    return audit


def baseline_check(model,cfg,run,dev,partition,infer,measure):
    init=cfg['initialization'];parent=Path(init['run']);run=Path(run)
    verify_files({str(parent/'last.pt'):init['checkpoint_sha256']})
    if read_json(parent/'dev_rows.json')!=dev:
        raise ValueError('Zero-update Dev manifest/order differs from parent')
    if read_json(parent/'dev_partition.json')!=partition:
        raise ValueError('Zero-update Dev partition differs from parent')
    if not dev:raise ValueError('Zero-update Dev must not be empty')
    for name in ('select','calibration'):
        indices=partition[name]
        if not indices or any(type(i) is not int or i<0 or i>=len(dev) for i in indices) or len(set(indices))!=len(indices):
            raise ValueError('Invalid zero-update Dev partition: '+name)
    if set(partition['select'])&set(partition['calibration']) or set(partition['select'])|set(partition['calibration'])!=set(range(len(dev))):
        raise ValueError('Zero-update Dev partitions must cover each row exactly once')
    state=torch.load(parent/'last.pt',map_location='cpu',weights_only=True)
    if state['best_tag']!=init['tag']:raise ValueError('Parent best tag changed')
    tensor=state.get('best_logits')
    if not isinstance(tensor,torch.Tensor):raise ValueError('Parent cached best scores are missing')
    expected=tensor.numpy().copy();del tensor,state
    if expected.shape!=(len(dev),2) or not np.isfinite(expected).all():
        raise ValueError('Parent reference score shape/nonfinite mismatch')
    old=read_json(parent/'config.json')
    # Match the parent's physical batches to avoid mistaking BF16 packing
    # differences for migration errors. This is one pass of the restored model,
    # not a second model loaded on GPU and not a cached-score-only assertion.
    batch_keys=('microbatch','frame_budget','eval_batch')
    eval_cfg=dict(cfg,**{key:old[key] for key in batch_keys})
    print('[Parent verification] full fixed Dev inference before the first update',flush=True)
    actual,actual_rows=infer(model,dev,eval_cfg)
    actual=np.asarray(actual)
    if actual.shape!=expected.shape or not np.isfinite(actual).all():
        raise ValueError('Zero-update Dev score shape/nonfinite mismatch')
    if len(actual_rows)!=len(dev) or any(actual_rows[i].get('row_index')!=i or
        any(actual_rows[i].get(key)!=value for key,value in row.items()) for i,row in enumerate(dev)):
        raise ValueError('Zero-update Dev inference row identity/order differs')
    metrics=measure(dev,actual);reference=measure(dev,expected)
    subsets={}
    for name in ('select','calibration'):
        indices=partition[name];rows=[dev[i] for i in indices]
        subsets[name]=(measure(rows,actual[indices]),measure(rows,expected[indices]))
    # Same explicit tolerance used for the prior function-preserving V3.18.3
    # continuation. Weighted tolerance is an absolute fraction: .001 = .1 pp.
    rtol=.001;atol=.003;max_weighted_delta=.001
    deltas={'full':abs(metrics['weighted_f1']-reference['weighted_f1'])}
    deltas.update({name:abs(pair[0]['weighted_f1']-pair[1]['weighted_f1']) for name,pair in subsets.items()})
    error=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    verified=bool(np.allclose(actual,expected,rtol=rtol,atol=atol) and max(deltas.values())<=max_weighted_delta)
    result=dict(verified=verified,reference_only=False,prediction_identity_expected=True,
        new_model_reproduces_parent=verified,parent_tag=init['tag'],parent_checkpoint_sha256=init['checkpoint_sha256'],
        metrics=metrics,reference_metrics=reference,selection_metrics=subsets['select'][0],
        reference_selection_metrics=subsets['select'][1],calibration_metrics=subsets['calibration'][0],
        reference_calibration_metrics=subsets['calibration'][1],weighted_deltas=deltas,
        rows=len(dev),fusion=model.fusion.describe(),rtol=rtol,atol=atol,max_weighted_delta=max_weighted_delta,
        max_abs_logit_error=float(error.max()),mean_abs_logit_error=float(error.mean()),
        inference_batches={key:eval_cfg[key] for key in batch_keys},
        note='Actual zero-update full Dev replay of the restored window ASP model. Calibration partition metrics use raw logits; no calibration is fitted. Epoch 0 is diagnostic, never a trained candidate or parent fallback.')
    np.savez_compressed(run/'parent_baseline_scores.npz',logits=actual)
    np.savez_compressed(run/'parent_reference_scores.npz',logits=expected)
    atomic_json(run/'parent_baseline.json',result)
    if not verified:
        raise ValueError('Parent prediction reproduction failed; see parent_baseline.json. No training update made.')
    print(f'[Parent verified] {init["tag"]} full Weighted={100*metrics["weighted_f1"]:.3f}; selection={100*subsets["select"][0]["weighted_f1"]:.3f}; max logit error={error.max():.6g}',flush=True)
    return result
