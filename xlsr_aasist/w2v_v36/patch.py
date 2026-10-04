"""Tiny deployable final-layer patch pinned to immutable full base weights."""
from pathlib import Path
import os
import shutil
import torch
from torch import nn

from w2v_aasist.runtime import sha256
from w2v_v3.model import Detector
from . import SCHEMA


def final_layer(model):
    layer = model.head.classifier[-1]
    if not isinstance(layer, nn.Linear) or layer.out_features != 2 or layer.bias is None:
        raise ValueError('Expected the existing binary final Linear classifier')
    return layer


def load_base(cfg, device='cpu'):
    path = Path(cfg['base_checkpoint'])
    if sha256(path) != cfg['base_checkpoint_sha256']:
        raise ValueError('Protected base checkpoint SHA256 differs')
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if state.get('kind') != 'weights' or state.get('tag') != cfg['base_tag']:
        raise ValueError('Base weight type/tag differs')
    hashes = {v for k, v in state.get('data_fingerprints', {}).items()
              if Path(k).name == 'preprocessor_config.json'}
    if len(hashes) != 1 or sha256(Path(cfg['ssl_path'])/'preprocessor_config.json') not in hashes:
        raise ValueError('Feature extractor differs from the submitted best')
    model = Detector.from_checkpoint(state, checkpointing=False)
    model.requires_grad_(False)
    return model.to(device).eval()


def apply_patch(model, patch):
    if patch.get('schema') != SCHEMA or patch.get('kind') != 'classifier_patch':
        raise ValueError('Expected a V3.6 classifier patch')
    layer = final_layer(model)
    w = torch.as_tensor(patch['weight'], dtype=torch.float32)
    b = torch.as_tensor(patch['bias'], dtype=torch.float32)
    if w.shape != layer.weight.shape or b.shape != layer.bias.shape:
        raise ValueError('Patch classifier dimensions differ')
    if not torch.isfinite(w).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite classifier patch')
    with torch.no_grad():
        layer.weight.copy_(w.to(layer.weight.device))
        layer.bias.copy_(b.to(layer.bias.device))
    return model.eval()


def save_patch(path, cfg, result):
    chosen = result['selected_patch']
    state = dict(schema=SCHEMA, kind='classifier_patch', tag=result['selected'],
        weight=torch.as_tensor(chosen['weight'], dtype=torch.float32),
        bias=torch.as_tensor(chosen['bias'], dtype=torch.float32),
        base_checkpoint=cfg['base_checkpoint'], base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
        base_tag=cfg['base_tag'], config=cfg, baseline_fallback=result['selected']=='baseline',
        score='P(fake)', threshold=.5, input_policy='full utterance', eval_amp='none')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A tiny patch does not need the legacy 512 MiB full-checkpoint reserve.
    if shutil.disk_usage(path.parent).free < 8*1024**2:
        raise OSError('Need 8 MiB free for atomic classifier patch and diagnostics')
    temporary = path.with_name(path.name+'.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return state


def load_selected(run):
    import json
    run = Path(run).expanduser().resolve()
    done = json.loads((run/'completed.json').read_text(encoding='utf-8'))
    path = run/'best_patch.pt'
    if done.get('version') != '3.6' or sha256(path) != done.get('patch_sha256'):
        raise ValueError('A completed V3.6 run with a matching selected patch is required')
    patch = torch.load(path, map_location='cpu', weights_only=True)
    if (patch.get('schema') != SCHEMA or patch.get('kind') != 'classifier_patch'
            or patch.get('tag') != done.get('selected')
            or patch.get('base_checkpoint_sha256') != done.get('base_checkpoint_sha256')
            or patch.get('threshold') != .5 or patch.get('score') != 'P(fake)'):
        raise ValueError('Selected patch metadata differs')
    cfg = patch['config']
    if (cfg.get('base_checkpoint_sha256') != patch['base_checkpoint_sha256']
            or cfg.get('base_checkpoint') != patch['base_checkpoint'] or cfg.get('base_tag') != patch['base_tag']):
        raise ValueError('Patch config/base binding differs')
    return patch, done
