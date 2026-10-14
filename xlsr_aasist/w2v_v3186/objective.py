"""Unchanged balanced C3 classification plus symmetric same-phone cosine loss."""
import math
import torch
from torch.nn import functional as F
from w2v_v3183r2.common import GROUPS
from w2v_v3183r2.objective import micro_losses, unit_loss  # public C3 API
from w2v_v318.objective import risk


def frame_span(start, end, frames):
    """Select Omni frame centres in a half-open interval measured at 16 kHz."""
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError('Invalid phoneme sample interval')
    # 400-sample receptive field, stride 320; first centre at sample 200.
    first = max(0, min(frames, math.ceil((start - 200.) / 320.)))
    last = max(0, min(frames, math.ceil((end - 200.) / 320.)))
    return first, last


def phoneme_loss(left, right, matches, min_frames=1):
    """Mean over matched phone instances of this source, never across sources.

    No detach/teacher, no confidence gate, no cross-recording phone negatives.
    Prefix sums avoid materializing [phones,time,features] masks. FP32 pooling
    and L2 normalization prevent gain changes alone from satisfying MSE.
    """
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1] or min_frames < 1:
        raise ValueError('Invalid phoneme features/minimum frame count')
    spans = []
    for match in matches:
        if len(match) != 4:
            raise ValueError('Each match must contain two sample intervals')
        a, b = frame_span(float(match[0]), float(match[1]), len(left))
        c, d = frame_span(float(match[2]), float(match[3]), len(right))
        if b - a >= min_frames and d - c >= min_frames:
            spans.append((a, b, c, d))
    if not spans:
        return (left.sum() + right.sum()) * 0., 0
    indices = torch.tensor(spans, device=left.device, dtype=torch.long)
    pooled = []
    for features, start, end in ((left, indices[:, 0], indices[:, 1]), (right, indices[:, 2], indices[:, 3])):
        integral = F.pad(features.float().cumsum(0), (0, 0, 1, 0))
        mean = (integral[end] - integral[start]) / (end - start).unsqueeze(1)
        pooled.append(F.normalize(mean, dim=-1, eps=1e-6))
    distance = (1. - (pooled[0] * pooled[1]).sum(-1)).clamp_min(0.)
    return distance.mean(), len(spans)


def source_weight(kind, row, cfg):
    if kind not in ('noise', 'platform'):
        raise ValueError('Unknown PCL branch')
    quota = cfg['group_quotas'][GROUPS.index((row['language'], row['label']))]
    budget = quota if kind == 'noise' else cfg.get('platform_sources_per_group', 2)
    if budget < 1:
        raise ValueError('Positive PCL source budget required')
    # Missing/invalid sources retain zero mass; the valid remainder is NOT
    # renormalized to a full group. Four complete groups each own 25%.
    return .25 / budget


def auxiliary_weights(cfg):
    if cfg.get('consistency', 'dual') == 'none':
        return dict(platform=0., noise=0.)
    update = int(cfg.get('current_update', 1))
    ramp = min(1., max(0, update) / max(1, cfg.get('pcl_ramp_updates', 300)))
    return {kind: float(cfg.get('pcl_' + kind + '_weight', .15)) * ramp
            for kind in ('platform', 'noise')}
