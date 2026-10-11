"""One encoder pass; a bounded, zero-initialized residual from intermediate layers."""
import torch
from torch import nn
from torch.nn import functional as F


class LayerFusion(nn.Module):
    def __init__(self, layers, depth, maximum=.25):
        super().__init__()
        if len(set(layers)) != len(layers) or any(i < 1 or i >= depth for i in layers):
            raise ValueError('Fusion layers must be distinct, one-based intermediate layers')
        if not 0 < maximum <= .5:
            raise ValueError('Fusion maximum must be in (0, .5]')
        self.layers, self.maximum = tuple(layers), maximum
        if layers:
            self.gate = nn.Parameter(torch.zeros(()))
            self.mix_logits = nn.Parameter(torch.zeros(len(layers)))

    def forward(self, final, captured):
        if not self.layers:
            return final
        # Inference at initialization is bitwise the parent function. During
        # training the residual MUST execute, otherwise the zero gate cannot learn.
        if not self.training and not torch.is_grad_enabled() and self.gate.item() == 0:
            return final
        with torch.autocast(final.device.type, enabled=False):
            top = final.float()
            # Read-only reference statistics avoid turning normalization into a
            # second gradient path into the final representation.
            mean = top.detach().mean(-1, keepdim=True)
            std = top.detach().var(-1, keepdim=True, unbiased=False).add(1e-5).sqrt()
            mixture = torch.zeros_like(top)
            for weight, index in zip(self.mix_logits.softmax(0), self.layers):
                mid = captured[index].float()
                if mid.shape != top.shape:
                    raise ValueError('Intermediate and final frame geometry differs')
                aligned = F.layer_norm(mid, (mid.shape[-1],)) * std + mean
                mixture = mixture + weight * aligned
            return top + self.maximum * self.gate.tanh() * (mixture - top)

    def describe(self):
        if not self.layers:
            return dict(layers=[], residual=0., weights=[])
        with torch.no_grad():
            return dict(layers=list(self.layers), residual=float(self.maximum*self.gate.tanh()),
                        maximum=self.maximum, weights=self.mix_logits.softmax(0).cpu().tolist())
