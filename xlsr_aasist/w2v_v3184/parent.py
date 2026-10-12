"""Continue a completed R2 best exactly; explicitly migrate an older V3.18.3."""
from pathlib import Path
import torch
from .common import read_json, digest, verify_files, partial_state, apply_partial, atomic_json
from w2v_v3183r2.parent import PARENT_ARTIFACTS, baseline_check as cached_baseline


def inspect_parent(run, tag=None):
    run=Path(run).expanduser().resolve()
    version=read_json(run/'config.json').get('version')
    if version=='3.18.3-R2':
        from w2v_v3183r2.evaluate import selected
    elif version=='3.18.3':
        from w2v_v3183.evaluate import selected
    else:
        raise ValueError('V3.18.4 requires a completed V3.18.3-R2 (recommended) or V3.18.3 parent')
    cfg,weights,meta=selected(run,'best')
    if cfg['variant']!='C3' or not cfg['local_evidence'] or cfg['risk_coefficient']!=.25:
        raise ValueError('The parent must use C3 with local evidence')
    if tag is not None and tag!=meta['selected']:
        raise ValueError(f'Expected parent {tag}, found {meta["selected"]}')
    verify_files({str(run/n):h for n,h in cfg['saved_manifests'].items()})
    files={str(run/n):(meta['checkpoint_sha256'] if n=='last.pt' else digest(run/n)) for n in PARENT_ARTIFACTS}
    return cfg,weights,dict(run=str(run),tag=meta['selected'],kind='best',parent_version=version,
        exact_continuation=version=='3.18.3-R2',checkpoint_sha256=meta['checkpoint_sha256'],files=files)


def initialize(model,cfg):
    init=cfg['initialization'];verify_files(init['files'])
    old,values,meta=inspect_parent(init['run'],init['tag'])
    if meta['checkpoint_sha256']!=init['checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    if old['version']=='3.18.3':
        from w2v_v3183r2.parent import initialize as migrate
        audit=migrate(model,cfg)
        audit.update(parent_version=old['version'],exact_continuation=False)
        return audit
    expected=partial_state(model)
    if set(expected)!=set(values):raise ValueError('Parent tensor inventory mismatch')
    # Validate ALL tensors before changing any weight, including BN buffers.
    for key,value in values.items():
        if value.shape!=expected[key].shape or value.dtype!=expected[key].dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid parent tensor: '+key)
    apply_partial(model,values)
    model.set_phase(False)
    result=dict(mode='parent-transfer',parent_version=old['version'],parent_tag=meta['tag'],
        parent_checkpoint_sha256=meta['checkpoint_sha256'],exact_continuation=True,
        copied_tensor_count=len(values),bn_statistics_reset=[],optimizer_inherited=False,
        all_partial_tensors_identical=True,new_fusion=model.fusion.describe(),
        note='R2 model implementation, trainable weights and inference buffers preserved; optimizer restarted.')
    print(f'[Initialization] R2 best={meta["tag"]}; all {len(values)} tensors copied; BN preserved; fresh optimizer',flush=True)
    return result


def baseline_check(model,cfg,run,dev,partition,infer,measure):
    result=cached_baseline(model,cfg,run,dev,partition,infer,measure)
    result.update(prediction_identity_expected=cfg['initialization']['exact_continuation'],new_model_reproduces_parent=None,
        note='Cached raw parent best scores; full initial GPU inference is not repeated. Tensor identity is checked separately for R2 parents.')
    atomic_json(Path(run)/'parent_baseline.json',result)
    return result
