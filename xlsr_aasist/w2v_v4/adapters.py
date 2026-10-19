"""Masked temporal residual adapters; attention and checkpoint names stay intact."""
import torch
from torch import nn
from torch.nn import functional as F


class MultiConvAdapter(nn.Module):
    def __init__(self, dimension, bottleneck=64, kernels=(3, 7, 15, 23)):
        super().__init__()
        if bottleneck < len(kernels) or bottleneck % len(kernels) or any(k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError('Adapter needs equal channel groups and positive odd kernels')
        self.norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck)
        width = bottleneck // len(kernels)
        self.branches = nn.ModuleList(nn.Conv1d(width, width, k, padding=k // 2, groups=width) for k in kernels)
        self.mix = nn.Conv1d(bottleneck, bottleneck, 3, padding=1)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, lengths):
        # fairseq2 BatchLayout supplies host-side lengths; validate there, not
        # with a CUDA synchronization in every adapted attention layer.
        if len(lengths) != x.shape[0]:
            raise ValueError('Invalid adapter sequence lengths')
        if not isinstance(lengths, torch.Tensor) and any(n < 1 or n > x.shape[1] for n in lengths):
            raise ValueError('Invalid adapter sequence lengths')
        lengths = torch.as_tensor(lengths, device=x.device)
        mask = (torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]).unsqueeze(-1)
        # Mask between EVERY temporal operation: biased padded activations must
        # never travel back into valid frames through the next convolution.
        h = F.gelu(self.down(self.norm(x.float()).to(x.dtype))) * mask
        h = h.transpose(1, 2)
        time_mask = mask.transpose(1, 2)
        pieces = h.chunk(len(self.branches), dim=1)
        h = torch.cat([F.gelu(conv(part)) for conv, part in zip(self.branches, pieces)], dim=1) * time_mask
        h = (h + F.gelu(self.mix(h))) * time_mask
        delta = self.up(h.transpose(1, 2)) * mask
        return x + delta.to(x.dtype)


def attach(model):
    count = model.cfg.get('adapter_layers', 16)
    if not 0 <= count <= model.cfg['lora_layers']:
        raise ValueError('MultiConv must remain inside the inherited LoRA/checkpoint suffix')
    model.adapters = nn.ModuleDict()
    model._adapter_hooks = []
    for index in range(len(model.encoder.layers) - count, len(model.encoder.layers)):
        layer = model.encoder.layers[index].layer
        attention = getattr(layer, 'self_attn', None)
        if attention is None:
            raise ValueError('Expected pinned fairseq2 self_attn boundary; refusing a different insertion point')
        adapter = MultiConvAdapter(model.cfg['encoder_dim'], model.cfg.get('adapter_dim', 64))
        model.adapters[str(index)] = adapter

        def augment(module, args, kwargs, output, adapter=adapter):
            layout = args[1] if len(args) > 1 else kwargs.get('seqs_layout')
            if layout is None or not isinstance(output, torch.Tensor):
                raise ValueError('Pinned fairseq2 attention/layout contract changed')
            return adapter(output, layout.seq_lens)

        # Permanent attention hooks also execute during checkpoint recomputation.
        # No wrapper renames the inherited Q/K/V/O or LoRA checkpoint keys.
        model._adapter_hooks.append(attention.register_forward_hook(augment, with_kwargs=True))
