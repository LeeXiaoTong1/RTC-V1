"""Input-conditioned layer mixture, retaining the parent's projected features."""
from contextlib import contextmanager
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from w2v_v3183r2.fusion import LayerFusion


def acoustic_summary(features, lengths):
    """Per-recording population mean/std; padded frames never contribute."""
    if features.ndim != 3 or len(lengths) != features.shape[0]:
        raise ValueError('Expected batch/time/channel acoustic features and lengths')
    if not lengths or any(type(n) is not int or n < 1 or n > features.shape[1] for n in lengths):
        raise ValueError('Invalid acoustic feature lengths')
    with torch.autocast(features.device.type, enabled=False):
        x = features.detach().float()
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        count = torch.tensor(lengths, device=x.device, dtype=torch.float32)[:, None]
        mean = x.masked_fill(~mask[..., None], 0.).sum(1) / count
        centered = (x - mean[:, None]).masked_fill(~mask[..., None], 0.)
        variance = centered.square().sum(1) / count
        return torch.cat((mean, variance.clamp_min(0.).sqrt()), -1)


class AdaptiveLayerFusion(LayerFusion):
    def __init__(self, base, *, width=128, mode='dynamic', seed=53001):
        nn.Module.__init__(self)
        if mode not in ('dynamic', 'static') or type(width) is not int or width < 1:
            raise ValueError('Invalid adaptive fusion configuration')
        self.layers, self.hidden, self.output_dim = base.layers, base.hidden, base.output_dim
        self.checkpointing = base.checkpointing
        # Reuse every original tensor, including the trained SwiGLU and logits.
        self.input_norm, self.layer_logits = base.input_norm, base.layer_logits
        self.projection, self.gating = base.projection, base.gating
        with torch.random.fork_rng(devices=[]):
            # Seed only CPU initialization; leave CUDA/dropout RNG untouched.
            torch.random.default_generator.manual_seed(seed)
            self.conditioner = nn.Sequential(nn.LayerNorm(2*self.hidden),
                nn.Linear(2*self.hidden, width), nn.SiLU(), nn.Linear(width, len(self.layers)))
            nn.init.zeros_(self.conditioner[-1].weight)
            nn.init.zeros_(self.conditioner[-1].bias)
        self.conditioner.requires_grad_(mode == 'dynamic')
        self.mode = mode
        self._initial_conditioner = {k: v.detach().cpu().clone() for k,v in self.conditioner.state_dict().items()}
        self._fixed_weights = None
        self._collect = False
        self.reset_observations()

    def reset_observations(self, enabled=False):
        self._collect = enabled
        self._observations = []

    def weights(self, features=None, lengths=None):
        if features is None:
            return super().weights()
        with torch.autocast(features.device.type, enabled=False):
            base = super().weights()
            if self._fixed_weights is not None:
                weights = self._fixed_weights.to(features.device).expand(features.shape[0], -1)
            elif self.mode == 'static':
                weights = base.expand(features.shape[0], -1)
            else:
                delta = self.conditioner(acoustic_summary(features, lengths))
                weights = (self.layer_logits.float()[None] + delta).softmax(-1)
            if self._collect:
                # A few small B x depth tensors only, never full encoder states.
                self._observations.append((weights.detach(), base.detach()))
            return weights

    def project_layer(self, hidden, index, weights=None):
        if weights is None or weights.ndim == 1:
            return super().project_layer(hidden, index, weights)
        if index not in self.layers or hidden.ndim != 3 or hidden.shape[-1] != self.hidden:
            raise ValueError('Unexpected layer identity or frame geometry')
        if weights.shape != (hidden.shape[0], len(self.layers)):
            raise ValueError('Per-recording layer mixture shape differs from encoder batch')
        # Exactly the parent projection/checkpoint path, with one sample-wise multiply.
        if self.checkpointing and torch.is_grad_enabled():
            # LN/Linear/SwiGLU is deterministic. Encoder/LoRA checkpoints keep
            # their own dropout RNG handling; only this reader skips the copy.
            value = checkpoint(self.transform, hidden, use_reentrant=False, preserve_rng_state=False)
        else:
            value = self.transform(hidden)
        return value.float() * weights[:, index-1, None, None]

    def forward(self, final, captured, *, condition, lengths):
        """Materialized reference for verifying the production streaming path."""
        if set(captured) != set(self.layers[:-1]):
            raise ValueError('Every intermediate layer must be present exactly once')
        weights, mixed = self.weights(condition, lengths), None
        for index in self.layers:
            hidden = final if index == self.layers[-1] else captured[index]
            if hidden.shape != final.shape:
                raise ValueError('Layer frame geometry differs')
            term = self.project_layer(hidden, index, weights)
            mixed = term if mixed is None else mixed + term
        return mixed

    def observations(self):
        if not self._observations:
            return dict(encoded_views=0)
        weights = torch.cat([w for w,_ in self._observations])
        bases = torch.cat([b.expand(w.shape[0], -1) for w,b in self._observations])
        mean = weights.mean(0)
        result = dict(encoded_views=len(weights), mean_weights=mean.cpu().tolist(),
            mean_absolute_change=float((weights-bases).abs().mean()),
            input_weight_std=float(weights.std(0, unbiased=False).mean()),
            mean_entropy=float(-(weights*weights.clamp_min(1e-30).log()).sum(-1).mean()),
            scope='all encoded views in this observation, including auxiliary references')
        return result

    @contextmanager
    def fixed(self, weights):
        value = torch.as_tensor(weights, dtype=torch.float32)
        if value.shape != (len(self.layers),) or not torch.isfinite(value).all() or (value < 0).any() or not torch.isclose(value.sum(), value.new_tensor(1.), atol=1e-5):
            raise ValueError('Fixed ablation needs a finite normalized layer mixture')
        old = self._fixed_weights
        self._fixed_weights = value
        try:
            yield
        finally:
            self._fixed_weights = old

    def describe(self):
        return dict(super().describe(), mode='input_adaptive_layer_mixture',
            fusion_mode=self.mode, conditioner_width=self.conditioner[1].out_features,
            weights_scope='shared base logits only; input mixtures are in fusion_diagnostics.jsonl',
            condition='masked mean/std of existing frozen Omni frontend features',
            zero_initialized_output=True, extra_encoder_passes=0)
