"""Deploy one frozen detector plus its selected compact correction."""
from pathlib import Path

import torch

from .common import digest, read_json, save_small
from .model import ResidualClassifier

SCHEMA = 'rtc_w2v_v38_bounded_residual_v1'


def save_patch(path, cfg, selected, spec):
    module = ResidualClassifier.restore(spec)
    if module.arm != selected:
        raise ValueError('Selected arm and model specification disagree')
    patch = dict(schema=SCHEMA, kind='bounded_residual_patch', tag=selected, spec=spec,
        config=cfg, base_checkpoint=cfg['base_checkpoint'],
        base_checkpoint_sha256=cfg['base_checkpoint_sha256'], base_tag=cfg['base_tag'],
        baseline_fallback=selected == 'baseline', language_debias_applied=selected == 'language_residual',
        score='P(fake)', threshold=.5, input_policy='full utterance', eval_amp='none',
        external_teacher_at_inference=False)
    save_small(path, patch)
    return patch


def load_selected(run):
    run = Path(run).expanduser().resolve()
    done = read_json(run / 'completed.json')
    if done.get('version') != '3.8' or digest(run / 'best_patch.pt') != done.get('patch_sha256'):
        raise ValueError('A completed V3.8 run and matching best_patch.pt are required')
    patch = torch.load(run / 'best_patch.pt', map_location='cpu', weights_only=True)
    if (patch.get('schema') != SCHEMA or patch.get('kind') != 'bounded_residual_patch'
            or patch.get('tag') != done.get('selected') or patch['spec']['arm'] != done.get('selected')
            or patch.get('score') != 'P(fake)' or patch.get('threshold') != .5
            or patch.get('base_checkpoint_sha256') != done.get('base_checkpoint_sha256')
            or patch.get('external_teacher_at_inference') is not False
            or patch.get('baseline_fallback') != (done['selected'] == 'baseline')
            or patch.get('language_debias_applied') != (done['selected'] == 'language_residual')):
        raise ValueError('Selected V3.8 patch metadata differs')
    if any(patch['config'].get(k) != patch.get(k) for k in ('base_checkpoint', 'base_checkpoint_sha256', 'base_tag')):
        raise ValueError('Patch base/config identity differs')
    ResidualClassifier.restore(patch['spec'])
    return patch, done


def apply_patch(model, patch):
    if patch.get('schema') != SCHEMA:
        raise ValueError('Expected V3.8 patch')
    original = model.head.classifier[-1]
    module = ResidualClassifier.restore(patch['spec']).to(original.weight.device)
    if (original.weight.shape != module.weight.shape or not torch.equal(original.weight, module.weight)
            or not torch.equal(original.bias, module.bias)):
        raise ValueError('Original classifier differs; refusing to stack or apply to another base')
    if module.arm != 'baseline':
        model.head.classifier[-1] = module
    return model.eval().requires_grad_(False)
