"""Zero-start repair with enough range to reverse a confident wrong decision.

For margin m, delta=(cap+abs(m))*tanh(u). The cap is a finite buffer beyond
the original decision boundary, not a fixed ceiling that makes errors unreachable.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import cpu_state


class CenteredStudent(nn.Module):
    def __init__(self, feature_dim, target_dim, hidden, mean, scale):
        super().__init__()
        self.register_buffer('input_mean', torch.as_tensor(mean).float().clone())
        self.register_buffer('input_scale', torch.as_tensor(scale).float().clone())
        self.hidden = nn.Linear(feature_dim, hidden)
        self.output = nn.Linear(hidden, target_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = ((x.float() - self.input_mean) / self.input_scale).clamp(-8, 8)
        return self.output(torch.tanh(self.hidden(x)))

    def spec(self):
        return dict(feature_dim=self.hidden.in_features, target_dim=self.output.out_features,
                    hidden=self.hidden.out_features, state=cpu_state(self))

    @classmethod
    def restore(cls, spec):
        state = spec['state']
        model = cls(spec['feature_dim'], spec['target_dim'], spec['hidden'],
                    state['input_mean'], state['input_scale'])
        model.load_state_dict(state, strict=True)
        return model.eval().requires_grad_(False)


class ResidualClassifier(nn.Module):
    def __init__(self, weight, bias, *, arm='baseline', mean=None, scale=None,
                 hidden=64, cap=2., student=None):
        super().__init__()
        weight, bias = torch.as_tensor(weight).float(), torch.as_tensor(bias).float()
        if weight.ndim != 2 or weight.shape[0] != 2 or bias.shape != (2,):
            raise ValueError('A binary frozen classifier is required')
        if arm not in ('baseline', 'calibration', 'residual_control', 'language_residual'):
            raise ValueError('Unknown V3.9 arm')
        if not math.isfinite(cap) or not 0 < cap <= 2.:
            raise ValueError('Boundary-crossing buffer must be in (0, 2] logit units')
        if (student is not None) != (arm == 'language_residual'):
            raise ValueError('Only the language residual uses an internal student')
        self.arm, self.cap, self.hidden_width = arm, float(cap), int(hidden)
        self.register_buffer('weight', weight.clone())
        self.register_buffer('bias', bias.clone())
        self.student = student
        if student is not None:
            student.eval().requires_grad_(False)
        if arm == 'calibration':
            self.raw_scale = nn.Parameter(torch.zeros(()))
            self.raw_shift = nn.Parameter(torch.zeros(()))
        elif arm != 'baseline':
            self.register_buffer('input_mean', torch.as_tensor(mean).float().clone())
            self.register_buffer('input_scale', torch.as_tensor(scale).float().clone())
            if self.input_mean.shape != weight.shape[1:] or self.input_scale.shape != weight.shape[1:]:
                raise ValueError('Input normalization dimensions differ')
            dim = weight.shape[1] + (student.output.out_features if student else 0)
            self.hidden = nn.Linear(dim, hidden)
            self.output = nn.Linear(hidden, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        self.validate()

    def validate(self):
        if any(not bool(torch.isfinite(v).all()) for v in self.state_dict().values()):
            raise ValueError('Nonfinite residual parameters')
        for key, value in self.named_buffers():
            if key.endswith('input_scale') and not bool((value > 0).all()):
                raise ValueError('Input scale must be positive')

    def adjustment(self, x, margin, context=None):
        if self.arm == 'baseline':
            return torch.zeros_like(margin)
        if self.arm == 'calibration':
            # Strictly positive global scale preserves every score ordering.
            scale = torch.exp(.7 * torch.tanh(self.raw_scale))
            return (scale - 1.) * margin + self.cap * torch.tanh(self.raw_shift)
        value = ((x.float() - self.input_mean) / self.input_scale).clamp(-8, 8)
        if self.student is not None:
            context = self.student(x) if context is None else context
            context = (context * math.sqrt(self.student.output.out_features)).clamp(-8, 8)
            value = torch.cat((value, context), dim=-1)
        fraction = torch.tanh(self.output(torch.tanh(self.hidden(value))).squeeze(-1))
        return (self.cap + margin.detach().abs()) * fraction

    def forward(self, x):
        logits = F.linear(x.float(), self.weight, self.bias)
        if self.arm == 'baseline':
            return logits
        delta = self.adjustment(x, logits[:, 0] - logits[:, 1])
        return logits + torch.stack((.5 * delta, -.5 * delta), dim=-1)

    def spec(self):
        return dict(arm=self.arm, cap=self.cap, hidden=self.hidden_width,
                    correction_rule='adaptive_margin_v1',
                    student=self.student.spec() if self.student is not None else None,
                    state=cpu_state(self))

    @classmethod
    def restore(cls, spec):
        if spec.get('correction_rule') != 'adaptive_margin_v1':
            raise ValueError('V3.9 requires an adaptive-margin specification; cannot reinterpret V3.8')
        state = spec['state']
        student = CenteredStudent.restore(spec['student']) if spec.get('student') else None
        module = cls(state['weight'], state['bias'], arm=spec['arm'],
                     mean=state.get('input_mean'), scale=state.get('input_scale'),
                     hidden=spec['hidden'], cap=spec['cap'], student=student)
        module.load_state_dict(state, strict=True)
        module.validate()
        return module.eval().requires_grad_(False)
