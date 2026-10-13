"""Parameter-free, source-paired consistency on pre-MultiConv frame features.

Matched clean/noisy RTC uses the known common timeline. The weak official
Online bridge uses detached, locally constrained monotonic correspondences.
This is TFCL-inspired regularization, not a reproduction of learned TFCL MHA.
Both detector endpoints receive gradients; no teacher or label-confidence gate.
"""
from contextlib import contextmanager
import math

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
    """Per-source channel structure, with channels as observations [C, T]."""
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError('CKA requires matching [channels, time] tensors')
    a, b = a.float(), b.float()
    a, b = a-a.mean(0, keepdim=True), b-b.mean(0, keepdim=True)
    ga, gb = a@a.T, b@b.T
    denominator = torch.linalg.vector_norm(ga)*torch.linalg.vector_norm(gb)
    return 1.-((ga*gb).sum()/denominator.clamp_min(eps)).clamp(-1., 1.)


class TFCL(nn.Module):
    def __init__(self, channels=128, heads=8, bins=201, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.channels, self.bins = channels, bins
        self.max_bins = int(cfg.get('alignment_max_bins', 192))
        self.min_frames = int(cfg.get('alignment_min_frames', 4))
        self.band = float(cfg.get('alignment_local_band_fraction',
                                  cfg.get('local_band_fraction', .2)))
        self.temperature = float(cfg.get('alignment_temperature', .07))
        if (channels < 1 or heads < 1 or bins < 2 or self.max_bins < self.min_frames
                or self.min_frames < 2 or not 0 < self.band <= 1
                or not math.isfinite(self.temperature) or self.temperature <= 0):
            raise ValueError('Invalid paired-consistency geometry')
        # Deliberately no trainable attention or projection: the auxiliary loss
        # cannot improve solely by updating a discarded alignment module.

    def _structure(self, a, b):
        size = min(self.bins, len(a), len(b))
        ap = F.adaptive_avg_pool1d(a.T[None], size)[0]
        bp = F.adaptive_avg_pool1d(b.T[None], size)[0]
        return channel_cka(ap, bp)

    def _pool(self, value, mask):
        positions = torch.arange(len(value), device=value.device, dtype=torch.float32)
        positions = positions / max(1, len(value)-1)
        value, positions = value[mask], positions[mask]
        size = min(self.max_bins, len(value))
        if len(value) > size:
            value = F.adaptive_avg_pool1d(value.T[None], size)[0].T
            positions = F.adaptive_avg_pool1d(positions[None, None], size)[0, 0]
        return value, positions

    def _indices(self, a, b, ap, bp):
        """GPU-only, detached correspondence; O(max_bins**2), no DP/CPU copy."""
        with torch.no_grad():
            similarities = F.normalize(a.detach(), dim=-1) @ F.normalize(b.detach(), dim=-1).T
            distance = (ap[:, None]-bp[None]).abs()
            allowed = distance <= self.band
            # A gap in valid frames can leave a temporal band empty. Include its
            # nearest valid time cell, rather than introducing all-masked NaNs.
            allowed = allowed | (distance == distance.min(dim=1, keepdim=True).values)
            scores = (similarities/self.temperature).masked_fill(~allowed, -torch.inf)
            indices = torch.arange(len(b), device=b.device, dtype=torch.float32)
            expected = scores.softmax(-1) @ indices
            matched = expected.cummax(0).values.round().long().clamp_(0, len(b)-1)
        return matched

    def forward(self, reference, noisy, reference_valid, noisy_valid):
        temporal, structure, _ = self.forward_batch(
            [reference], [noisy], [reference_valid], [noisy_valid])
        return temporal[0], structure[0]

    def forward_batch(self, reference, noisy, reference_valid, noisy_valid,
                      kinds=None, return_diagnostics=False, compute_time=True, compute_structure=True):
        count = len(reference)
        if not (len(noisy) == len(reference_valid) == len(noisy_valid) == count):
            raise ValueError('Paired feature and mask lists differ')
        kinds = ['matched']*count if kinds is None else list(kinds)
        if len(kinds) != count or any(k not in ('matched', 'bridge') for k in kinds):
            raise ValueError('Expected matched or bridge edge kinds')
        if not count:
            empty = torch.empty(0)
            result = (empty, empty.clone(), torch.empty(0, dtype=torch.bool))
            return (*result, []) if return_diagnostics else result
        temporal, structure, eligible, diagnostics = [], [], [], []
        for a, b, am, bm, kind in zip(reference, noisy, reference_valid, noisy_valid, kinds):
            for value, mask in ((a, am), (b, bm)):
                if (value.ndim != 2 or value.shape[1] != self.channels
                        or not value.is_floating_point() or value.device != a.device):
                    raise ValueError('Expected floating frame features [time, channels] on one device')
                if mask.ndim != 1 or len(mask) != len(value):
                    raise ValueError('Auxiliary mask must match the complete frame sequence')
            am, bm = am.to(device=a.device, dtype=torch.bool), bm.to(device=b.device, dtype=torch.bool)
            if kind == 'matched' and len(a) != len(b):
                raise ValueError('Matched RTC views must share the exact timeline')
            # Empty slices connect zero gradients without evaluating masked NaNs.
            zero = a[:0].float().sum()+b[:0].float().sum()
            t, s = zero, zero
            with torch.autocast(device_type=a.device.type, enabled=False):
                a, b = a.float(), b.float()
                if kind == 'matched':
                    valid = am & bm
                    av, bv = a[valid], b[valid]
                    coverage = len(av)/max(1, len(a))
                    ok = len(av) >= self.min_frames
                    if ok:
                        if compute_time:
                            t = (1.-F.cosine_similarity(av, bv, dim=-1)).mean()
                        if compute_structure:
                            s = self._structure(av, bv)
                else:
                    av, ap = self._pool(a, am)
                    bv, bp = self._pool(b, bm)
                    coverage = min(int(am.sum())/max(1, len(a)), int(bm.sum())/max(1, len(b)))
                    ok = min(len(av), len(bv)) >= self.min_frames
                    if ok:
                        ab, ba = self._indices(av, bv, ap, bp), self._indices(bv, av, bp, ap)
                        if compute_time:
                            t = .5*((1.-F.cosine_similarity(av, bv[ab], dim=-1)).mean()
                                    +(1.-F.cosine_similarity(bv, av[ba], dim=-1)).mean())
                        if compute_structure:
                            s = .5*(self._structure(av, bv[ab])+self._structure(bv, av[ba]))
                temporal.append(t if ok else zero)
                structure.append(s if ok else zero)
                eligible.append(ok)
                diagnostics.append(dict(kind=kind, eligible=ok, valid_frame_fraction=coverage,
                                        left_frames=len(a), right_frames=len(b)))
        result = (torch.stack(temporal), torch.stack(structure),
                  torch.tensor(eligible, device=reference[0].device, dtype=torch.bool))
        return (*result, diagnostics) if return_diagnostics else result
