"""A single detector with a small internal language correction branch."""
import json
import os
from pathlib import Path
import shutil

import torch
from torch import nn
from torch.nn import functional as F

from w2v_aasist.runtime import sha256
from w2v_v36.patch import load_base, final_layer
from .student import LanguageStudent

SCHEMA = 'rtc_w2v_v37_language_patch_v1'


class LanguageCorrectedClassifier(nn.Module):
    """Consume one detector representation; no external model or metadata."""
    def __init__(self, state, feature_dim):
        super().__init__()
        weight = torch.as_tensor(state['weight'],dtype=torch.float32)
        bias = torch.as_tensor(state['bias'],dtype=torch.float32)
        if weight.shape != (2,feature_dim) or bias.shape != (2,) or not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):
            raise ValueError('Invalid V3.7 binary classifier')
        self.register_buffer('weight',weight)
        self.register_buffer('bias',bias)
        language = state.get('language_state')
        student = state.get('student_state')
        if (language is None) != (student is None):
            raise ValueError('Language correction and internal student must be present together')
        self.student = LanguageStudent(student) if language is not None else None
        self.alpha = 0.
        if language is not None:
            self.alpha = float(language['alpha'])
            if not 0 < self.alpha <= .75:
                raise ValueError('Language correction alpha must lie in (0,.75]')
            dim = self.student.b2.numel()
            for name in ('mean','scale','mapping'):
                tensor = torch.as_tensor(language[name],dtype=torch.float32)
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError('Nonfinite language projection')
                self.register_buffer('language_'+name,tensor)
            if (self.language_mean.shape != (dim,) or self.language_scale.shape != (dim,)
                    or self.language_mapping.shape != (dim,feature_dim)
                    or not bool((self.language_scale > 0).all())
                    or self.student.input_mean.numel()!=feature_dim):
                raise ValueError('Language correction dimensions/scales differ')

    def forward(self, hidden):
        x = hidden.float()
        if self.student is not None:
            language = self.student(x)
            predicted = ((language-self.language_mean)/self.language_scale) @ self.language_mapping
            x = x-self.alpha*predicted
        return F.linear(x,self.weight,self.bias)


def apply_patch(model, patch):
    if patch.get('schema') != SCHEMA or patch.get('kind') != 'language_classifier_patch':
        raise ValueError('Expected V3.7 language classifier patch')
    original = final_layer(model)
    correction = LanguageCorrectedClassifier(patch,original.in_features).to(original.weight.device)
    if correction.student is None:
        # Exact original path for baseline/control, including Linear execution.
        with torch.no_grad():
            original.weight.copy_(correction.weight)
            original.bias.copy_(correction.bias)
    else:
        model.head.classifier[-1] = correction
    return model.eval().requires_grad_(False)


def save_patch(path, cfg, result):
    chosen = result['selected_patch']
    state = dict(schema=SCHEMA,kind='language_classifier_patch',tag=result['selected'],
        weight=torch.as_tensor(chosen['weight'],dtype=torch.float32),
        bias=torch.as_tensor(chosen['bias'],dtype=torch.float32),
        language_state=chosen.get('language_state'),student_state=chosen.get('student_state'),
        base_checkpoint=cfg['base_checkpoint'],base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
        base_tag=cfg['base_tag'],config=cfg,baseline_fallback=result['selected']=='baseline',
        language_debias_applied=chosen.get('language_state') is not None,
        score='P(fake)',threshold=.5,input_policy='full utterance',eval_amp='none',
        external_teacher_at_inference=False)
    LanguageCorrectedClassifier(state,len(chosen['weight'][0]))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(path.parent).free < 16*1024**2:
        raise OSError('Need 16 MiB free for the small atomic V3.7 patch')
    temporary=path.with_name(path.name+'.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    return state


def load_selected(run):
    run=Path(run).expanduser().resolve()
    done=json.loads((run/'completed.json').read_text(encoding='utf-8'))
    path=run/'best_patch.pt'
    if done.get('version')!='3.7' or sha256(path)!=done.get('patch_sha256'):
        raise ValueError('A completed V3.7 run with matching selected patch is required')
    patch=torch.load(path,map_location='cpu',weights_only=True)
    if (patch.get('schema')!=SCHEMA or patch.get('kind')!='language_classifier_patch'
            or patch.get('tag')!=done.get('selected')
            or patch.get('base_checkpoint_sha256')!=done.get('base_checkpoint_sha256')
            or patch.get('threshold')!=.5 or patch.get('score')!='P(fake)'
            or patch.get('external_teacher_at_inference') is not False
            or patch.get('baseline_fallback')!=(done.get('selected')=='baseline')):
        raise ValueError('V3.7 selected patch metadata differs')
    cfg=patch['config']
    if any(cfg.get(k)!=patch.get(k) for k in ('base_checkpoint','base_checkpoint_sha256','base_tag')):
        raise ValueError('Patch config/base binding differs')
    LanguageCorrectedClassifier(patch,patch['weight'].shape[1])
    return patch,done
