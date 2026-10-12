"""Uncapped full-depth LN/projection/SwiGLU/softmax feature reader.

Project encoder outputs as they arrive. Checkpoint the inexpensive reader
transform instead of retaining all FP32 normalization intermediates.
Original BF16 layer inputs still consume training memory.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim)
        # silu(1.2784645427610737) == 1. Identity initialization preserves
        # transferred projection coordinates without suppressing gradients.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 1.2784645427610737)
        nn.init.eye_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, x):
        return F.silu(self.gate(x)) * self.value(x)


class LayerFusion(nn.Module):
    def __init__(self, hidden, depth, projection=128, checkpointing=True):
        super().__init__()
        if min(hidden, depth, projection) < 1:
            raise ValueError('Positive encoder width, depth and projection required')
        self.layers = tuple(range(1, depth + 1))
        self.hidden, self.output_dim = hidden, projection
        self.checkpointing = checkpointing
        self.input_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.layer_logits = nn.Parameter(torch.zeros(depth))
        self.projection = nn.Linear(hidden, projection)
        self.gating = SwiGLU(projection)

    def weights(self):
        return self.layer_logits.float().softmax(0)

    def transform(self, hidden):
        # FP32 normalization, then BF16 CUDA projection under outer autocast.
        with torch.autocast(hidden.device.type, enabled=False):
            normalized = self.input_norm(hidden.float()).to(hidden.dtype)
        return self.gating(self.projection(normalized))

    def project_layer(self, hidden, index, weights=None):
        if index not in self.layers or hidden.ndim != 3 or hidden.shape[-1] != self.hidden:
            raise ValueError('Unexpected layer identity or frame geometry')
        if self.checkpointing and torch.is_grad_enabled():
            value = checkpoint(self.transform, hidden, use_reentrant=False)
        else:
            value = self.transform(hidden)
        weights = self.weights() if weights is None else weights
        return value.float() * weights[index - 1]

    def forward(self, final, captured):
        """Materialized reference; production encode streams the same terms."""
        if set(captured) != set(self.layers[:-1]):
            raise ValueError('Every intermediate layer must be present exactly once')
        weights, mixed = self.weights(), None
        for index in self.layers:
            hidden = final if index == self.layers[-1] else captured[index]
            if hidden.shape != final.shape:
                raise ValueError('Layer frame geometry differs')
            term = self.project_layer(hidden, index, weights)
            mixed = term if mixed is None else mixed + term
        return mixed

    def describe(self):
        with torch.no_grad():
            weights = self.weights().cpu()
            entropy = float(-(weights * weights.clamp_min(1e-30).log()).sum())
            order = torch.argsort(weights, descending=True)[:5].tolist()
            return dict(mode='all_layers_projected_swiglu_softmax', layers=list(self.layers),
                weights=weights.tolist(), entropy=entropy, effective_layers=math.exp(entropy),
                top_layers=[dict(layer=i+1, weight=float(weights[i])) for i in order],
                final_weight=float(weights[-1]), output_dim=self.output_dim,
                final_bypass=False, residual_cap=None)
