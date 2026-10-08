"""Two-sided TFCL on independent, complete valid feature trajectories.

The input geometry and loss definitions match V3.15: inputs are not normalized
before attention, channels are CKA observations, and only the structure branch
pools time. Batching changes execution, not the population used by an objective.
Each row returned by ``forward_batch`` belongs to exactly one supplied edge.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def fusion_frames(model):
    captured = []
    handle = model.head.blocks[0].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0]))
    try:
        yield captured
    finally:
        handle.remove()
        captured.clear()


def channel_cka(a, b, eps=1e-8):
    """Return per-edge 1-CKA for [..., channels, time] tensors in FP32.

    Center channels independently for every edge. A collapsed representation has
    zero Gram norm and receives loss 1, never a falsely perfect similarity score.
    As with cosine loss, this does not by itself guarantee avoidance of collapse:
    the supervised classification objective remains necessary.
    """
    if a.shape != b.shape or a.ndim < 2:
        raise ValueError('CKA requires matching [..., channels, time] tensors')
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, b = a.float(), b.float()
        a = a - a.mean(-2, keepdim=True)
        b = b - b.mean(-2, keepdim=True)
        ga, gb = a @ a.transpose(-1, -2), b @ b.transpose(-1, -2)
        denominator = (torch.linalg.vector_norm(ga, dim=(-2, -1))
                       * torch.linalg.vector_norm(gb, dim=(-2, -1)))
        similarity = (ga * gb).sum((-2, -1)) / denominator.clamp_min(eps)
        return 1. - similarity.clamp(-1., 1.)


class TFCL(nn.Module):
    def __init__(self, channels=128, heads=8, bins=201):
        super().__init__()
        if channels < 1 or heads < 1 or channels % heads or bins < 2:
            raise ValueError('TFCL channels must divide attention heads; >=2 structure bins')
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=0., batch_first=True)
        self.time_projection = nn.Linear(bins, bins)
        self.channels, self.bins = channels, bins
        with torch.no_grad():
            nn.init.eye_(self.time_projection.weight)
            self.time_projection.bias.zero_()

    def forward(self, reference, noisy, reference_valid, noisy_valid):
        """Compatibility with the V3.15 single-edge scalar interface."""
        temporal, structure, _ = self.forward_batch(
            [reference], [noisy], [reference_valid], [noisy_valid])
        return temporal[0], structure[0]

    def forward_batch(self, reference, noisy, reference_valid, noisy_valid):
        """Return temporal[N], structure[N], eligible[N], preserving edge order.

        Every input is a sequence; each feature tensor is [time, channels] and
        its boolean mask is [time]. Padding and known missing frames are removed
        before either comparison. Fewer than two valid frames on either side
        produces a differentiable zero and eligible=False. Invalid edges never
        enter attention, avoiding all-masked softmax NaNs. Callers retain their
        source/edge loss budget: do not implicitly average over eligible rows.

        Both sides have gradients. Auxiliary modules remain FP32 and accept BF16
        detector features, including inside an outer autocast context. No input
        prefix is cut; only the CKA branch pools the entire valid trajectory.
        """
        count = len(reference)
        if not (len(noisy) == len(reference_valid) == len(noisy_valid) == count):
            raise ValueError('TFCL feature and mask lists must have equal length')
        device = self.attention.in_proj_weight.device
        if not count:
            empty = self.attention.in_proj_weight.new_empty((0,), dtype=torch.float32)
            return empty, empty.clone(), torch.empty(0, device=device, dtype=torch.bool)

        left, right, indices, zeros = [], [], [], []
        for i, (a, b, am, bm) in enumerate(zip(
                reference, noisy, reference_valid, noisy_valid)):
            for value, mask in ((a, am), (b, bm)):
                if (value.ndim != 2 or value.shape[1] != self.channels
                        or not value.is_floating_point() or value.device != device):
                    raise ValueError('TFCL features must be floating [time, channels] on the module device')
                if mask.ndim != 1 or len(mask) != len(value):
                    raise ValueError('TFCL mask length must match its feature trajectory')
            # Empty slices connect zero gradients without reading masked NaNs.
            zeros.append(a[:0].float().sum() + b[:0].float().sum())
            a = a[am.to(device=device, dtype=torch.bool)].float()
            b = b[bm.to(device=device, dtype=torch.bool)].float()
            if min(len(a), len(b)) >= 2:
                indices.append(i)
                left.append(a)
                right.append(b)

        temporal, structure = torch.stack(zeros), torch.stack(zeros)
        eligible = torch.zeros(count, device=device, dtype=torch.bool)
        if not indices:
            return temporal, structure, eligible

        def padded(values):
            x = nn.utils.rnn.pad_sequence(values, batch_first=True)
            lengths = torch.tensor([len(v) for v in values], device=device)
            valid = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
            return x, valid

        # FP32 makes cosine/CKA and the small auxiliary projections insensitive to
        # outer BF16 autocast. The much larger detector keeps its configured AMP.
        with torch.autocast(device_type=device.type, enabled=False):
            a, am = padded(left)
            b, bm = padded(right)
            ab, _ = self.attention(a, b, b, key_padding_mask=~bm, need_weights=False)
            ba, _ = self.attention(b, a, a, key_padding_mask=~am, need_weights=False)
            ta = (1. - F.cosine_similarity(ab.float(), a, dim=-1))
            tb = (1. - F.cosine_similarity(ba.float(), b, dim=-1))
            ta = ta.masked_fill(~am, 0.).sum(1) / am.sum(1)
            tb = tb.masked_fill(~bm, 0.).sum(1) / bm.sum(1)

            def pooled(values):
                return torch.stack([F.adaptive_avg_pool1d(
                    v.T[None], self.bins)[0] for v in values])

            ap = self.time_projection(pooled(left))
            bp = self.time_projection(pooled(right))
            valid_structure = channel_cka(ap, bp)
            ix = torch.tensor(indices, device=device)
            temporal = temporal.index_copy(0, ix, .5 * (ta + tb))
            structure = structure.index_copy(0, ix, valid_structure)
            eligible[ix] = True
        return temporal, structure, eligible
