"""TFCL-inspired, two-sided alignment on discriminative fusion frame features.

Unlike the author's fixed four-second batches, classification and temporal
attention use every valid frame. Only the channel-structure branch pools to
201 bins. CKA is per source, so physical microbatching cannot change its meaning.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def fusion_frames(model):
    captured = []
    handle = model.head.blocks[0].register_forward_pre_hook(lambda _module,args:captured.append(args[0]))
    try:
        yield captured
    finally:
        handle.remove()
        captured.clear()


def channel_cka(a, b, eps=1e-8):
    """Channels are observations, pooled time bins are their descriptors."""
    a,b = a.float(),b.float()
    a = a-a.mean(0,keepdim=True); b = b-b.mean(0,keepdim=True)
    ga,gb = a@a.T,b@b.T
    denominator = torch.linalg.vector_norm(ga)*torch.linalg.vector_norm(gb)
    return 1.-((ga*gb).sum()/denominator.clamp_min(eps)).clamp(-1.,1.)


class TFCL(nn.Module):
    def __init__(self, channels=128, heads=8, bins=201):
        super().__init__()
        if channels % heads or bins < 2:
            raise ValueError('TFCL channels must divide attention heads; >=2 structure bins')
        self.attention = nn.MultiheadAttention(channels,heads,dropout=0.,batch_first=True)
        self.time_projection = nn.Linear(bins,bins)
        self.bins = bins
        # Near identity keeps the structure objective interpretable at warm start.
        with torch.no_grad():
            nn.init.eye_(self.time_projection.weight); self.time_projection.bias.zero_()

    def forward(self, reference, noisy, reference_valid, noisy_valid):
        a,b = reference[reference_valid.bool()],noisy[noisy_valid.bool()]
        if min(len(a),len(b)) < 2:
            zero = (reference.sum()+noisy.sum())*0.
            return zero,zero
        # Both reference and noisy queries/keys/values remain in the gradient graph.
        ab,_ = self.attention(a[None],b[None],b[None],need_weights=False)
        ba,_ = self.attention(b[None],a[None],a[None],need_weights=False)
        temporal = .5*((1-F.cosine_similarity(ab[0].float(),a.float(),dim=-1)).mean()
                       +(1-F.cosine_similarity(ba[0].float(),b.float(),dim=-1)).mean())
        # Pool the complete valid trajectory; never take a prefix or classify these bins.
        ap = F.adaptive_avg_pool1d(a.float().T[None],self.bins)[0]
        bp = F.adaptive_avg_pool1d(b.float().T[None],self.bins)[0]
        structure = channel_cka(self.time_projection(ap),self.time_projection(bp))
        return temporal,structure
