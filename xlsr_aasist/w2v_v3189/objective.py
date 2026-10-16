"""Balanced single-view CE and fixed-scale same-source phoneme PCL."""
import math
import torch
from torch.nn import functional as F
from w2v_v3183r2.common import GROUPS


def micro_losses(logits, units, quotas, coefficient=None, smoothing=.02):
    """Each stream owns 50%; each language/class owns 25% in that stream.

    There is no two-view max/risk term with one Noisy view. Role metadata,
    never list length, distinguishes Online from a simulated Noisy source.
    """
    if len(logits) != len(units) or any(len(unit) != 1 for unit in units):
        raise ValueError('Exactly one classification view per source is required')
    rows = [unit[0] for unit in units]
    if any(row['role'] not in ('online', 'noisy') for row in rows):
        raise ValueError('Only Online and simulated Noisy enter classification')
    labels = torch.tensor([row['label'] for row in rows], device=logits.device)
    logp = F.log_softmax(logits.float(), dim=-1)
    raw = -logp.gather(1, labels[:, None]).squeeze(1)
    ce = (1-smoothing)*raw-smoothing*logp.mean(-1)
    return [(ce[i] * (.125 / quotas[GROUPS.index((row['language'], row['label']))]), raw[i:i+1])
            for i, row in enumerate(rows)]


def frame_span(start, end, frames):
    """Select Omni frame centres in a half-open interval measured at 16 kHz."""
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError('Invalid phoneme sample interval')
    # 400-sample receptive field, stride 320; first centre at sample 200.
    first = max(0, min(frames, math.ceil((start - 200.) / 320.)))
    last = max(0, min(frames, math.ceil((end - 200.) / 320.)))
    return first, last


def phoneme_features(features, spans, min_frames=1, valid_frames=None):
    """Pool genuine original-region phone intervals without vector normalization."""
    if features.ndim != 2 or min_frames < 1:
        raise ValueError('Invalid phoneme feature geometry')
    if valid_frames is not None:
        if not isinstance(valid_frames, int) or not 0 <= valid_frames <= len(features):
            raise ValueError('Invalid original-region PCL frame count')
        features = features[:valid_frames]
    indices = [frame_span(float(a), float(b), len(features)) for a, b in spans]
    indices = [(a, b) for a, b in indices if b-a >= min_frames]
    if not indices:
        return features[:0].float()
    indices = torch.tensor(indices, device=features.device, dtype=torch.long)
    integral = F.pad(features.float().cumsum(0), (0, 0, 1, 0))
    return (integral[indices[:, 1]]-integral[indices[:, 0]]) / (indices[:, 1]-indices[:, 0]).unsqueeze(1)


def _scale(scale, features):
    if scale is None or scale.ndim != 1 or len(scale) != features.shape[1]:
        raise ValueError('PCL requires frozen per-channel Train scales')
    if scale.requires_grad:
        raise ValueError('PCL scales must be fixed, finite and positive')
    # No per-source CUDA synchronization: frozen PCLScale validates values at
    # calibration. Standalone CPU calls retain strict public input checking.
    if scale.device.type == 'cpu' and (not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any())):
        raise ValueError('PCL scales must be fixed, finite and positive')
    return scale.to(device=features.device, dtype=torch.float32)


def phoneme_loss(left, right, matches, min_frames=1, left_valid_frames=None,
                 right_valid_frames=None, *, scale=None, return_features=False):
    """Mean over matched phone instances of this source, never across sources.

    No teacher, confidence gate, cross-recording negatives or learned loss head.
    The fixed Train scale cannot grow with the current features to hide errors.
    """
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1] or min_frames < 1:
        raise ValueError('Invalid phoneme features/minimum frame count')
    scale = _scale(scale, left)
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
        result = ((left.sum() + right.sum()) * 0., 0)
        return (*result, (left[:0].float(), right[:0].float())) if return_features else result
    indices = torch.tensor(spans, device=left.device, dtype=torch.long)
    pooled = []
    for features, start, end in ((left, indices[:, 0], indices[:, 1]), (right, indices[:, 2], indices[:, 3])):
        integral = F.pad(features.float().cumsum(0), (0, 0, 1, 0))
        mean = (integral[end] - integral[start]) / (end - start).unsqueeze(1)
        pooled.append(mean)
    distance = ((pooled[0] - pooled[1]) / scale).square().mean()
    result = (distance, len(spans))
    return (*result, tuple(pooled)) if return_features else result


def phoneme_variance(records, scale, cfg):
    """Source-balanced physical-batch moments, never pretend phones are sources.

    records contain (source_id, left_phones, right_phones, planned_source_mass).
    Duplicated source occurrences contribute only once. Invalid sources retain
    zero mass; no refill, full-batch graph retention, queue or second forward.
    """
    if not records:
        raise ValueError('Variance calculation requires at least one feature record')
    scale = _scale(scale, records[0][1])
    cap = int(cfg.get('pcl_max_phones_per_source', 8))
    minimum_sources = int(cfg.get('pcl_variance_min_sources', 4))
    minimum_phones = int(cfg.get('pcl_variance_min_phones', 32))
    gamma = float(cfg.get('pcl_variance_target', .5))
    if min(cap, minimum_sources, minimum_phones) < 1 or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('Invalid PCL variance settings')
    selected, seen = [], set()
    for source, left, right, mass in records:
        if source in seen:
            continue
        if left.ndim != 2 or right.shape != left.shape or mass <= 0 or not math.isfinite(mass):
            raise ValueError('Invalid PCL variance source record')
        seen.add(source)
        if not len(left):
            continue
        indices = torch.linspace(0, len(left)-1, min(cap, len(left)), device=left.device).long()
        selected.append((left[indices], right[indices], float(mass)))
    count = sum(len(left) for left, _, _ in selected)
    stats = dict(sources=len(selected), phones=count, applied=False, mass=sum(x[2] for x in selected))
    zero = sum((left.sum()+right.sum())*0. for _, left, right, _ in records)
    if len(selected) < minimum_sources or count < minimum_phones:
        stats['reason'] = 'insufficient_sources' if len(selected) < minimum_sources else 'insufficient_phones'
        return zero, stats
    values = []
    for side in (0, 1):
        items = []
        for item in selected:
            scaled = item[side].float() / scale
            weight = item[2] / stats['mass']
            items.append((scaled, weight))
        mean = torch.stack([weight*value.mean(0) for value,weight in items]).sum(0)
        # Center first: E[x^2]-E[x]^2 loses the small discriminative residual
        # when all channels carry a large common offset.
        variance = torch.stack([weight*(value-mean).square().mean(0) for value,weight in items]).sum(0)
        deviation = variance.add(1e-6).sqrt()
        values.append((gamma-deviation).clamp_min(0).mean())
    stats.update(applied=True, reason='applied')
    return torch.stack(values).mean() * stats['mass'], stats


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
    return {kind: float(cfg.get('pcl_' + kind + '_weight', .15 if kind == 'noise' else .045)) * ramp
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
    Each source contributes exactly one view. Normalizing
    by all EN-real sources keeps this independent of physical batch grouping.
    """
    beta = float(cfg.get('en_real_confidence_weight', 0.))
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
