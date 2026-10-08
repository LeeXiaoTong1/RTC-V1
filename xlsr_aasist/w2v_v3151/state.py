"""One atomic checkpoint: current state and at most two distinct selected models.

Selectors are aliases into this transaction, not duplicate full checkpoints.
The frozen prefix and V3.15 starting detector are referenced, never duplicated.
"""
from pathlib import Path

import torch

from w2v_v313.state import (identity, partial_state, apply_partial, to_cpu, tensor_bytes,
                            atomic_save, capture_rng, restore_rng)
from w2v_v39.common import digest, read_json

SCHEMA = 'rtc_v3151_partial_state_v1'
KINDS = ('best_guarded', 'best_weighted', 'last')


def load_resume(path, cfg):
    path = Path(path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('schema') != SCHEMA or state.get('identity') != identity(cfg):
        raise ValueError('V3.15.1 resume schema/configuration differs')
    if state.get('inference_only'):
        raise ValueError('Completed compact checkpoint is for inference only')
    if state.get('execution_plan_sha256') != digest(path.parent/'execution_plan.json'):
        raise ValueError('Recorded numerical execution plan differs')
    return state


def storage_budget(model, cfg, auxiliary=None):
    p = sum(v.numel()*v.element_size() for v in model.parameters() if v.requires_grad)
    # Worst case: current + Adam's two moments + two different selected states.
    # torch.save shares the current state when it is also selected.
    aux = sum(v.numel()*v.element_size() for v in auxiliary.parameters()) if auxiliary is not None else 0
    one = int((5*p+3*aux)*1.1) + cfg['disk_margin_bytes']
    new_peak = 2*one+cfg['rolling_cache_bytes']+256*1024**2
    if new_peak>cfg['maximum_new_peak_bytes']:
        raise OSError('V3.15.1 estimated new disk peak exceeds the configured 20 GiB cap')
    return dict(trainable_bytes=p, maximum_resume_estimate_bytes=one,
        maximum_atomic_peak_bytes=2*one, feature_tensor_bytes=0,
        rolling_cache_cap_bytes=cfg['rolling_cache_bytes'], new_frame_cache_bytes=0,
        frozen_prefix_copies=0, maximum_new_peak_bytes=new_peak,
        required_start_free_bytes=new_peak+cfg['free_reserve_bytes'])


def prune_candidates(candidates, selections):
    return {tag: value for tag, value in candidates.items() if tag in set(selections.values())}


def apply_candidate(model, candidate):
    if candidate is None:
        return
    if candidate['kind'] == 'head':
        if any(not bool(torch.isfinite(p).all()) for p in candidate['state'].values()):
            raise ValueError('Nonfinite selected head')
        model.head.classifier[-1].load_state_dict(candidate['state'], strict=True)
    elif candidate['kind'] == 'partial':
        apply_partial(model, candidate['state'])
    else:
        raise ValueError('Unknown V3.15.1 candidate format')


def load_selected(run, kind='best_guarded'):
    if kind not in KINDS:
        raise ValueError('Choose best_guarded, best_weighted, or last explicitly')
    run = Path(run).expanduser().resolve()
    cfg, done = read_json(run/'config.json'), read_json(run/'completed.json')
    filename = done.get('state_file')
    if filename not in ('last.pt', 'inference.pt'):
        raise ValueError('Unknown completed checkpoint path')
    path = run/filename
    if (done.get('version') != '3.15.1' or done.get('status') != 'complete'
            or path.is_symlink() or digest(path) != done['checkpoint_sha256']):
        raise ValueError('Completed V3.15.1 checkpoint hash differs')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state.get('schema') != SCHEMA or state.get('identity') != identity(cfg)
            or state['cursor'] != done['committed_updates'] or state['selections'] != done['selections']
            or state['last_tag'] != done['last_tag']
            or cfg['base_checkpoint_sha256'] != done['base_checkpoint_sha256']
            or cfg['starting_checkpoint_sha256'] != done['starting_checkpoint_sha256']):
        raise ValueError('Completed checkpoint provenance/cursor differs')
    for key in ('parent_checkpoint_sha256','parent_selected_tag','parent_selector'):
        if cfg.get(key) is None or done.get(key)!=cfg[key]:
            raise ValueError('Completed V3.15 parent provenance differs: '+key)
    if (state.get('execution_plan_sha256') != done.get('execution_plan_sha256')
            or state.get('execution_plan_sha256') != digest(run/'execution_plan.json')):
        raise ValueError('Completed numerical execution plan differs')
    if state['cursor'] > 0 and (not state['history'] or not state['history'][-1].get('committed')
            or state['history'][-1]['cursor'] != state['cursor'] or state['history'][-1]['tag'] != state['last_tag']):
        raise ValueError('LAST is not the matching committed validation state')
    if kind == 'last':
        tag = state['last_tag']
        candidate = dict(kind='partial', state=state['model'])
    else:
        tag = state['selections'][kind]
        candidate = state['candidates'].get(tag)
        if tag != 'starting_parent' and candidate is None:
            raise ValueError('Selected weights missing; refusing fallback')
    meta = dict(done, selected=tag, checkpoint_kind=kind, checkpoint_path=str(path),
                baseline_fallback=tag == 'starting_parent')
    return dict(config=cfg, candidate=candidate), meta
