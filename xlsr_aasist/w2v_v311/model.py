"""Train the detector representation, not a language-conditioned score offset."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from w2v_v3.model import Detector
from w2v_v32.model import install_runtime
from w2v_v39.common import verify_files

CONDITIONS = ('offline', 'online', 'noisy_a', 'noisy_b')


class FeatureClassifier(nn.Module):
    """Preserve old parameter names and start at exactly the original function."""
    def __init__(self, original, hidden):
        super().__init__()
        for i, module in enumerate(original):
            self.add_module(str(i), module)
        width = original[0].in_features
        self.adapter = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                     nn.GELU(), nn.Linear(hidden, width))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.features = None

    def __getitem__(self, i):
        return self._modules[str(i if i >= 0 else 3 + i)]

    def forward(self, stats):
        corrected = stats + self.adapter(stats)
        self.features = self[1](self[0](corrected))
        return self[2](self.features)


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = float(strength)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class ProtectedReverse(torch.autograd.Function):
    """Remove only the reversal component that opposes real/fake supervision.

    For a real sample the binary CE gradient at the final representation is a
    nonnegative multiple of w_fake-w_real. A negative dot product with this
    direction is a first-order conflict. This is a feature-space approximation,
    NOT parameter-space PCGrad or a guarantee about Adam's shared-weight update.
    The adversary still receives its ordinary minimization gradient.
    """
    @staticmethod
    def forward(ctx, x, strength, direction, protect, audit):
        ctx.strength, ctx.protect, ctx.audit = float(strength), protect, audit
        ctx.save_for_backward(F.normalize(direction.detach().float(), dim=-1, eps=1e-12))
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        (unit,) = ctx.saved_tensors
        raw = -ctx.strength * gradient.float()
        dot = (raw * unit).sum(-1, keepdim=True)
        conflict = dot.clamp_max(0) * unit
        corrected = raw - conflict if ctx.protect else raw
        if ctx.audit is not None:
            active = raw.square().sum(-1) > 0
            ctx.audit.append(torch.stack((active.sum(), ((dot[:, 0] < 0) & active).sum(),
                raw.square().sum(), (raw-corrected).square().sum())).detach())
        return corrected.to(gradient.dtype), None, None, None, None


class LanguageAdversary(nn.Module):
    """A separate EN/ZH classifier for each communication condition."""
    def __init__(self, width, hidden=128):
        super().__init__()
        self.heads = nn.ModuleDict({condition: nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 2))
            for condition in CONDITIONS})

    def loss(self, features, rows, weights, strength, direction=None, protect=True, audit=None):
        # weights were normalized over the WHOLE logical batch, before microbatching.
        total = features.float().sum() * 0.
        for condition in CONDITIONS:
            indices = [i for i, row in enumerate(rows)
                       if row['label'] == 1 and row['condition'] == condition and weights[i] > 0]
            if indices:
                h = (Reverse.apply(features[indices], strength) if direction is None else
                     ProtectedReverse.apply(features[indices], strength, direction, protect, audit))
                logits = self.heads[condition](h)
                labels = torch.tensor([int(rows[i]['language'] == 'zh') for i in indices], device=h.device)
                w = torch.tensor([weights[i] for i in indices], device=h.device, dtype=torch.float32)
                total = total + (F.cross_entropy(logits.float(), labels, reduction='none') * w).sum()
        return total


def load_model(cfg, device=None, training=True):
    verify_files({cfg['base_checkpoint']: cfg['base_checkpoint_sha256']})
    state = torch.load(cfg['base_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    if state.get('kind') != 'weights' or state.get('tag') != cfg['base_tag']:
        raise ValueError('The protected base kind/tag changed')
    processor = str(Path(cfg['ssl_path']) / 'preprocessor_config.json')
    recorded = state.get('data_fingerprints', {}).get(processor)
    if recorded:
        verify_files({processor: recorded})
    model = Detector.from_checkpoint(state, checkpointing=training and cfg['checkpointing'])
    model = install_runtime(model)
    model.head.classifier = FeatureClassifier(model.head.classifier, cfg['adapter_hidden'])
    model.configure_trainable_layers(cfg['trainable_layers'])
    return model.to(device or cfg['device']).train(training)


def optimizer_for(model, adversary, cfg):
    groups = []
    count = len(model.backbone.encoder.layers)
    for index, layer in enumerate(model.backbone.encoder.layers):
        parameters = [p for p in layer.parameters() if p.requires_grad]
        if parameters:
            lr = cfg['encoder_lr'] * cfg['layer_decay'] ** (count - 1 - index)
            groups.append(dict(params=parameters, lr=lr, initial_lr=lr, name=f'encoder_{index}'))
    adapter = list(model.head.classifier.adapter.parameters())
    adapter_ids = {id(p) for p in adapter}
    groups.extend([
        dict(params=[p for p in model.head.parameters() if id(p) not in adapter_ids],
             lr=cfg['head_lr'], initial_lr=cfg['head_lr'], name='multiconv'),
        dict(params=adapter, lr=cfg['adapter_lr'], initial_lr=cfg['adapter_lr'], name='feature_adapter'),
        dict(params=list(adversary.parameters()), lr=cfg['adversary_lr'],
             initial_lr=cfg['adversary_lr'], name='language_adversary')])
    flattened = [p for group in groups for p in group['params']]
    expected = [p for module in (model, adversary) for p in module.parameters() if p.requires_grad]
    if len({id(p) for p in flattened}) != len(flattened) or {id(p) for p in flattened} != {id(p) for p in expected}:
        raise ValueError('Optimizer must cover every trainable parameter exactly once')
    return torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
