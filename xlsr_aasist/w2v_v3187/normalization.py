"""Per-recording normalization and stable population statistics."""
import torch
from torch import nn
from w2v_v318.aasist import Residual_block as OriginalResidual


def stable_std(values, dim, keepdim=False):
    """Centered population std; singleton/constant values and gradients are safe."""
    values = values.float()
    variance = (values - values.mean(dim=dim, keepdim=True)).square().mean(dim=dim, keepdim=keepdim)
    return torch.sqrt(variance.clamp_min(0) + 1e-5) - (1e-5) ** .5


def _replacement(module):
    if isinstance(module, nn.BatchNorm2d):
        channels = module.num_features
        if channels != 1 and channels % 8:
            raise ValueError('AASIST convolution channels must support eight groups')
        result = nn.GroupNorm(1 if channels == 1 else 8, channels, eps=module.eps, affine=module.affine)
    elif isinstance(module, nn.BatchNorm1d):
        # Graph layers flatten [batch,node,channel] before their normalization;
        # LN therefore normalizes channels independently for every graph node.
        result = nn.LayerNorm(module.num_features, eps=module.eps, elementwise_affine=module.affine)
    else:
        return None
    if module.affine:
        result = result.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            result.weight.copy_(module.weight)
            result.bias.copy_(module.bias)
    return result


class ResidualBlock(nn.Module):
    """Preserve executed residual equations; remove discarded bn1 preactivation."""
    def __init__(self, original):
        super().__init__()
        self.conv1, self.conv2 = original.conv1, original.conv2
        self.bn2, self.selu = _replacement(original.bn2), original.selu
        self.downsample = original.downsample
        if self.downsample:
            self.conv_downsample = original.conv_downsample

    def forward(self, x):
        identity = self.conv_downsample(x) if self.downsample else x
        return self.conv2(self.selu(self.bn2(self.conv1(x)))) + identity


def replace_batchnorm(module):
    """Transform only this new backend; older version classes stay untouched."""
    for name, child in list(module.named_children()):
        if isinstance(child, OriginalResidual):
            setattr(module, name, ResidualBlock(child))
            continue
        replacement = _replacement(child)
        if replacement is not None:
            setattr(module, name, replacement)
        else:
            replace_batchnorm(child)
    return module
