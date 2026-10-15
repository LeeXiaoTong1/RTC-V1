"""Actual-quota balanced C3, valid-original-phone PCL, and confidence control."""
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


def phoneme_loss(left, right, matches, min_frames=1, left_valid_frames=None, right_valid_frames=None):
    """Mean over matched phone instances of this source, never across sources.

    No detach/teacher, no confidence gate, no cross-recording phone negatives.
    Prefix sums avoid materializing [phones,time,features] masks. FP32 pooling
    and L2 normalization prevent gain changes alone from satisfying MSE.
    """
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1] or min_frames < 1:
        raise ValueError('Invalid phoneme features/minimum frame count')
    # Mirror extension is classification context only. A frame whose complete
    # 400-sample receptive field crosses the original end is not a phone view.
    for value, features in ((left_valid_frames, left), (right_valid_frames, right)):
        if value is not None and (not isinstance(value, int) or not 0 <= value <= len(features)):
            raise ValueError('Invalid original-region PCL frame count')
    left = left[:left_valid_frames] if left_valid_frames is not None else left
    right = right[:right_valid_frames] if right_valid_frames is not None else right
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
    quota = row.get('batch_group_quotas', cfg['group_quotas'])[GROUPS.index((row['language'], row['label']))]
    budget = quota if kind == 'noise' else row.get('pcl_platform_budget', min(quota, cfg.get('platform_sources_per_group', 2)))
    if budget < 1:
        raise ValueError('Positive PCL source budget required')
    # Missing/invalid sources retain zero mass; the valid remainder is NOT
    # renormalized to a full group. Four complete groups each own 25%.
    return .25 / budget


def auxiliary_weights(cfg):
    if cfg.get('consistency', 'dual') == 'none' or not cfg.get('joint_phase', True):
        return dict(platform=0., noise=0.)
    update = int(cfg.get('current_update', 1))
    ramp = min(1., max(0, update) / max(1, cfg.get('pcl_ramp_updates', 300)))
    return {kind: float(cfg.get('pcl_' + kind + '_weight', .15)) * ramp
            for kind in ('platform', 'noise')}


def batch_quotas(units, cfg):
    """All rows agree on planned quotas, including the final partial update."""
    if not units:
        raise ValueError('Empty logical batch')
    declared = {tuple(row['batch_group_quotas']) for unit in units for row in unit
                if 'batch_group_quotas' in row}
    if len(declared) > 1:
        raise ValueError('Rows disagree on actual batch group quotas')
    if declared and any('batch_group_quotas' not in row for unit in units for row in unit):
        raise ValueError('Missing actual batch group quota metadata')
    result = next(iter(declared)) if declared else tuple(cfg['group_quotas'])
    if len(result) != 4 or any(not isinstance(q, int) or q < 1 for q in result):
        raise ValueError('Each logical group requires a positive integer quota')
    return result


def confidence_regularization(logits, units, quotas, cfg):
    """Cap excessive correct EN-real confidence; never reduce its CE weight.

    This explicitly class-conditioned heuristic affects score calibration. It
    is not an assertion that large logits cause overfitting. The positive gap
    threshold makes the penalty exactly zero on incorrect/borderline Real.
    Two Noisy views receive a mean, not a second hard-view maximum. Normalizing
    by all EN-real sources keeps this independent of physical batch grouping.
    """
    beta = float(cfg.get('en_real_confidence_weight', .01))
    margin = float(cfg.get('en_real_confidence_margin', 3.5))
    if beta < 0 or not math.isfinite(beta) or margin <= 0 or not math.isfinite(margin):
        raise ValueError('Invalid EN-real confidence settings')
    index = int(cfg.get('current_update', 1))
    ramp = min(1., max(0, index) / max(1, cfg.get('pcl_ramp_updates', 300)))
    enabled = bool(cfg.get('joint_phase', True)) and beta > 0
    weighted = logits.sum() * 0.
    count, active, excess_sum, offset = 0, logits.new_zeros(()), logits.new_zeros(()), 0
    for unit in units:
        n = len(unit)
        if (unit[0]['language'], unit[0]['label']) == ('en', 1):
            gap = logits[offset:offset+n, 1].float() - logits[offset:offset+n, 0].float()
            excess = (gap - margin).clamp_min(0)
            count += n
            active = active + (excess.detach() > 0).sum()
            excess_sum = excess_sum + excess.detach().sum()
            if enabled:
                weighted = weighted + beta * ramp * (.125 / quotas[GROUPS.index(('en', 1))]) * excess.square().mean()
        offset += n
    return weighted, torch.stack((active.float(), excess_sum.float(), logits.new_tensor(float(count))))
