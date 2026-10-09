"""Small artifacts, deterministic identities, and bounded checkpoint storage."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from w2v_v39.common import ROOT, read_json, digest, atomic_json, verify_files
from w2v_v313.state import atomic_save, to_cpu, capture_rng, restore_rng

GROUPS = (('en', 0), ('en', 1), ('zh', 0), ('zh', 1))
SCHEMA = 'rtc_v318_omni1b_aasist_v1'


def schema_for(cfg):
    arch=cfg.get('omni_arch','1b')
    if arch not in ('1b','3b'):raise ValueError('Unsupported V3.18 frontend')
    return 'rtc_v318_omni'+arch+'_aasist_v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def seed_for(*parts):
    return int(fingerprint(parts)[:15], 16)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def observation(model):
    rng = capture_rng()
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for m, mode in modes:
            m.training = mode
        restore_rng(rng)


def partial_state(model):
    # ALL head buffers (especially BatchNorm running stats) are part of inference.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith('head.') or k.endswith(('.lora_a', '.lora_b'))}


def apply_partial(model, values):
    expected = partial_state(model)
    if set(expected) != set(values):
        raise ValueError('V3.18 partial keys differ; head buffers and all LoRA weights required')
    target = model.state_dict()
    with torch.no_grad():
        for name, value in values.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
                raise ValueError('Invalid partial tensor: ' + name)
            target[name].copy_(value)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')


def require_finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite ' + name)
