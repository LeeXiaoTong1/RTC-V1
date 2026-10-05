"""One transactional resume file with current+selected partial weights.

No frozen encoder prefix is copied. best.pt is materialized only at completion.
The previous last.pt survives failed writes; the selected state is inside the
same transaction, so an interrupted best export cannot change model selection.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from w2v_v39.common import digest, read_json

SCHEMA = 'rtc_v310_partial_state_v1'


def identity(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def partial_state(model):
    return {k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}


def apply_partial(model, state):
    expected = {k:p for k,p in model.named_parameters() if p.requires_grad}
    if set(expected) != set(state):
        raise ValueError('Partial checkpoint trainable parameter keys differ')
    with torch.no_grad():
        for name, parameter in expected.items():
            value = state[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid partial tensor: ' + name)
            parameter.copy_(value)


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:to_cpu(v) for k,v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return value


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list,tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def storage_budget(model, adversary, cfg):
    detector = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    critic = sum(p.numel()*p.element_size() for p in adversary.parameters())
    # Current parameters + two Adam moments, selected detector parameters, metadata.
    resume = int((3*(detector+critic) + detector) * 1.1) + cfg['disk_margin_bytes']
    return dict(trainable_detector_bytes=detector, adversary_bytes=critic,
                estimated_last_bytes=resume, estimated_best_bytes=int(detector*1.1)+cfg['disk_margin_bytes'],
                peak_new_run_bytes=2*resume, new_audio_bytes=0, new_frame_cache_bytes=0)


def atomic_save(path, value, margin=64*1024**2):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    # Only this owned temporary file may be removed, never an input cache or old best.
    temporary.unlink(missing_ok=True)
    needed = int(tensor_bytes(value) * 1.1) + margin
    if shutil.disk_usage(path.parent).free < needed:
        raise OSError(f'Need {needed/1024**3:.2f} GiB free for atomic partial checkpoint; previous last.pt retained')
    try:
        with temporary.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng():
    state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python=random.getstate(), numpy=(state[0], state[1].tolist(), state[2], state[3], state[4]))


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])
    a,b,c,d,e = state['numpy']
    np.random.set_state((a,np.asarray(b,dtype=np.uint32),c,d,e))


def load_resume(path, cfg):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('schema') != SCHEMA or state.get('identity') != identity(cfg):
        raise ValueError('Resume configuration or state schema changed')
    return state


def load_selected(run):
    run = Path(run).expanduser().resolve()
    done = read_json(run / 'completed.json')
    if done.get('version') != '3.10' or done.get('status') != 'complete' or digest(run/'best.pt') != done['checkpoint_sha256']:
        raise ValueError('Completed V3.10 best.pt identity required')
    checkpoint = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    if (checkpoint.get('schema') != SCHEMA or checkpoint.get('identity') != identity(cfg)
            or checkpoint.get('selected') != done['selected']
            or cfg['base_checkpoint_sha256'] != done['base_checkpoint_sha256']
            or bool(checkpoint.get('model') is None) != done['baseline_fallback']):
        raise ValueError('Selected partial checkpoint metadata differs')
    return checkpoint, done
