"""Strict continuation from a completed C3 detector; no partial-load fallback."""
from pathlib import Path
import numpy as np
import torch
from w2v_v318.evaluate import selected
from w2v_v318.common import apply_partial as apply_legacy
from .common import read_json, digest, verify_files, atomic_json


CONFIRMED_RUN='w2v_v318_20261010_135116_c6fa'
CONFIRMED_SHA='cb355866a76d9a1e959e3941747f600b2280c2f4244601c1d74fd198eea6149d'
PARENT_ARTIFACTS=('config.json','completed.json','last.pt','train_rows.json','dev_rows.json',
                  'dev_partition.json','train_probe_tickets.json','independent_panel.json')


def inspect_parent(run,tag='epoch_10_step_2417'):
    run=Path(run).expanduser().resolve()
    cfg,weights,meta=selected(run,'last')
    if cfg['version']!='3.18' or cfg['variant']!='C3' or not cfg['local_evidence'] or cfg['risk_coefficient']!=.25:
        raise ValueError('V3.18.3 continuation requires the completed V3.18 C3 model')
    if meta['selected']!=tag:raise ValueError(f'Expected parent {tag}, found {meta["selected"]}')
    if run.name==CONFIRMED_RUN and meta['checkpoint_sha256']!=CONFIRMED_SHA:
        raise ValueError('Confirmed epoch 10 checkpoint hash differs from the audited checkpoint')
    verify_files({str(run/n):h for n,h in cfg['saved_manifests'].items()})
    files={str(run/n):(meta['checkpoint_sha256'] if n=='last.pt' else digest(run/n)) for n in PARENT_ARTIFACTS}
    return cfg,weights,dict(run=str(run),tag=tag,checkpoint_sha256=meta['checkpoint_sha256'],files=files)


def initialize(model,cfg):
    init=cfg['initialization'];verify_files(init['files'])
    old,weights,meta=inspect_parent(init['run'],init['tag'])
    if meta['checkpoint_sha256']!=init['checkpoint_sha256']:raise ValueError('Parent changed')
    # This checks ALL old keys, including every BN buffer and LoRA tensor.
    # Only the separate zero-gate fusion module is new.
    apply_legacy(model,weights)
    if any(torch.count_nonzero(p).item() for p in model.fusion.parameters()):
        raise ValueError('New fusion must initialize at zero')
    model.set_phase(True)
    return old


def baseline_check(model,cfg,run,dev,partition,infer,measure):
    parent=Path(cfg['initialization']['run'])
    state=torch.load(parent/'last.pt',map_location='cpu',weights_only=True)
    expected=state['last_logits'].numpy();del state
    old=read_json(parent/'config.json')
    # Reproduce identical physical batches to avoid confusing BF16 batching
    # differences with a checkpoint initialization error.
    eval_cfg=dict(cfg,**{k:old[k] for k in ('microbatch','frame_budget','eval_batch')})
    actual,_=infer(model,dev,eval_cfg)
    if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('Parent baseline shape/nonfinite mismatch')
    metrics=measure(dev,actual);reference=measure(dev,expected)
    max_error=float(np.max(np.abs(actual-expected)))
    verified=bool(np.allclose(actual,expected,rtol=.001,atol=.003) and
                  abs(metrics['weighted_f1']-reference['weighted_f1'])<=.001)
    ix=partition['select'];selected_metrics=measure([dev[i] for i in ix],actual[ix])
    result=dict(verified=verified,parent_tag=cfg['initialization']['tag'],max_abs_logit_error=max_error,
        rtol=.001,atol=.003,max_weighted_delta=.001,metrics=metrics,reference_metrics=reference,
        selection_metrics=selected_metrics,rows=len(dev),fusion=model.fusion.describe(),
        note='Epoch 0 diagnostic only. Not a trained V3.18.3 candidate; parent is never silently substituted.')
    atomic_json(Path(run)/'parent_baseline.json',result)
    np.savez_compressed(Path(run)/'parent_baseline_scores.npz',logits=actual)
    if not verified:raise ValueError('Epoch 10 prediction reproduction failed; see parent_baseline.json. No training update made.')
    print(f'[Parent verified] {result["parent_tag"]} full Weighted={100*metrics["weighted_f1"]:.3f}; max logit error={max_error:.6g}',flush=True)
    return result
