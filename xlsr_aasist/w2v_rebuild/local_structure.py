"""Parameter-free, selective local consistency on SSL frame features.

Supply only the SAME recording and its processed view. Classification remains
necessary: consistency alone does not establish authenticity. Feature axes are
SSL coordinates, not frequency bands or phonemes. Detached matching is mutual
nearest and order-consistent within a bounded normalized-time window. This is
not arbitrary-deletion/DTW alignment. Low variation rejects degenerate features,
but is NOT an acoustic silence detector. No cross-example statistics, learned
parameters, global CKA or D-by-D matrices are used.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LocalStructureConfig:
    scales: tuple = (16, 32)
    max_shift_fraction: float = .15
    min_cosine: float = .6
    min_margin: float = .02
    min_matches: int = 4
    temporal_radius: int = 4
    dynamic_floor: float = 1e-4
    transition_weight: float = .5

    def __post_init__(self):
        if (not self.scales or len(set(self.scales)) != len(self.scales)
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 4 for n in self.scales)):
            raise ValueError('scales must be distinct integer bin counts >= 4')
        for name in ('max_shift_fraction', 'min_cosine', 'min_margin', 'dynamic_floor', 'transition_weight'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if not 0 <= self.max_shift_fraction <= .5 or not 0 <= self.min_cosine <= 1:
            raise ValueError('Invalid timing tolerance or cosine threshold')
        if not 0 <= self.min_margin <= 1 or self.dynamic_floor <= 0:
            raise ValueError('Invalid margin or dynamic floor')
        if (isinstance(self.min_matches, bool) or not isinstance(self.min_matches, int)
                or not 3 <= self.min_matches <= min(self.scales)):
            raise ValueError('min_matches must be between 3 and the smallest scale')
        if isinstance(self.temporal_radius, bool) or not isinstance(self.temporal_radius, int) or self.temporal_radius < 1:
            raise ValueError('temporal_radius must be a positive integer')
        if not 0 <= self.transition_weight <= 1:
            raise ValueError('transition_weight must lie in [0, 1]')


def _check_frames(frames, name, config):
    if (not isinstance(frames, torch.Tensor) or frames.ndim != 3
            or not frames.is_floating_point() or frames.shape[0] < 1
            or frames.shape[1] < max(config.scales) or frames.shape[2] < 2):
        raise ValueError(f'{name} requires floating [B,T,D], B>=1, T>=max(scales), D>=2')


def _pooled(frames, bins, config):
    # FP32 under autocast; center per utterance, never across the batch.
    x = F.adaptive_avg_pool1d(frames.float().transpose(1, 2), bins).transpose(1, 2)
    x = x - x.mean(1, keepdim=True)
    rms = x.square().mean((1, 2)).sqrt()
    norms = x.square().mean(2).sqrt()
    valid = torch.isfinite(x).all(2) & (norms > config.dynamic_floor)
    x = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
    return x, F.normalize(x, dim=2, eps=config.dynamic_floor), valid, rms


def descriptors(frames, config=None):
    """Frozen per-example descriptor; default embedding [B,90].

    Concatenates the first two upper Gram diagonals at 16/32 temporal bins
    (29+61 entries), scale-normalized by sqrt(entry count). These are feature
    relationships, not literal frequencies. Class separation must be measured
    independently on source-disjoint examples. `valid` rejects degeneracy.
    """
    with torch.autocast(device_type=frames.device.type, enabled=False):
        return _descriptors(frames, config)


def _descriptors(frames, config=None):
    config = config or LocalStructureConfig()
    _check_frames(frames, 'frames', config)
    pieces, valid, rms = [], [], []
    for bins in config.scales:
        _, unit, active, variation = _pooled(frames, bins, config)
        relations = unit @ unit.transpose(1, 2)
        piece = torch.cat([relations.diagonal(offset=k, dim1=1, dim2=2) for k in (1, 2)], 1)
        pieces.append(piece / math.sqrt(piece.shape[1]))
        valid.append((active.sum(1) >= config.min_matches) & torch.isfinite(variation))
        rms.append(torch.nan_to_num(variation, nan=0., posinf=0., neginf=0.))
    return {'embedding': torch.cat(pieces, 1), 'valid': torch.stack(valid).all(0),
            'temporal_rms': torch.stack(rms).mean(0)}


@torch.no_grad()
def _match(ref, proc, ref_active, proc_active, config):
    bins = ref.shape[1]
    pos = torch.arange(bins, device=ref.device)
    tolerance = int(math.ceil(config.max_shift_fraction * (bins-1)))
    band = (pos[:, None]-pos[None, :]).abs() <= tolerance
    similarity = ref @ proc.transpose(1, 2)
    allowed = band[None] & ref_active[:, :, None] & proc_active[:, None, :]
    scores = similarity.masked_fill(~allowed, -2.)
    best, indices = scores.topk(2, dim=2)
    reverse, back_indices = scores.transpose(1, 2).topk(2, dim=2)
    match = indices[:, :, 0]
    accepted = ((best[:, :, 0] >= config.min_cosine)
                & (best[:, :, 0]-best[:, :, 1] >= config.min_margin)
                & (back_indices[:, :, 0].gather(1, match) == pos[None])
                & ((reverse[:, :, 0]-reverse[:, :, 1]).gather(1, match) >= config.min_margin)
                & ref_active & proc_active.gather(1, match))
    # Conservatively reject every crossing; invalid proposals cannot affect
    # order. Retained correspondences form a strict monotone subsequence.
    previous = torch.cat((torch.full_like(match[:, :1], -1),
                          torch.where(accepted, match, -1).cummax(1).values[:, :-1]), 1)
    following = torch.cat((torch.where(accepted, match, bins).flip(1).cummin(1).values.flip(1)[:, 1:],
                           torch.full_like(match[:, :1], bins)), 1)
    accepted = accepted & (match > previous) & (match < following)
    return match, accepted, best[:, :, 0]


def local_structure_per_pair(reference, processed, config=None):
    """Return loss [B], usable mask [B], stats; reference is stop-gradient.

    Gates are detached, label/classifier independent. Unusable pairs contribute
    a differentiable zero. Same-recording identity is the caller's responsibility.
    Stats are scalars except accept_by_pair [B] and valid_by_pair [B].
    """
    with torch.autocast(device_type=reference.device.type, enabled=False):
        return _local_structure_per_pair(reference, processed, config)


def _local_structure_per_pair(reference, processed, config=None):
    config = config or LocalStructureConfig()
    _check_frames(reference, 'reference', config)
    _check_frames(processed, 'processed', config)
    if (reference.shape[0] != processed.shape[0] or reference.shape[2] != processed.shape[2]
            or reference.device != processed.device):
        raise ValueError('Paired batches require matching B, D and device')
    losses, usable, relations, transitions = [], [], [], []
    matches, accepted_fractions, similarities, shifts = [], [], [], []
    for bins in config.scales:
        rx, ru, ra, _ = _pooled(reference.detach(), bins, config)
        px, pu, pa, _ = _pooled(processed, bins, config)
        index, accepted, similarity = _match(ru.detach(), pu.detach(), ra, pa, config)
        gathered = pu.gather(1, index[:, :, None].expand(-1, -1, pu.shape[2]))
        gathered_x = px.gather(1, index[:, :, None].expand(-1, -1, px.shape[2]))
        pos = torch.arange(bins, device=reference.device)
        distance = (pos[:, None]-pos[None, :]).abs()
        local = (distance > 0) & (distance <= config.temporal_radius)
        pair_mask = accepted[:, :, None] & accepted[:, None, :] & local[None]
        relation_error = ((ru @ ru.transpose(1, 2))-(gathered @ gathered.transpose(1, 2))).square()/4
        relation_count = pair_mask.sum((1, 2))
        relation_loss = (relation_error*pair_mask).sum((1, 2))/relation_count.clamp_min(1)
        # Adjacent-change direction complements rotation-invariant Gram terms.
        # Do not bridge a missing/unmatched interval as though it were adjacent.
        rd, pd = rx[:, 1:]-rx[:, :-1], gathered_x[:, 1:]-gathered_x[:, :-1]
        delta_valid = ((rd.detach().square().mean(2) > config.dynamic_floor**2)
                       & (pd.detach().square().mean(2) > config.dynamic_floor**2))
        transition_mask = (accepted[:, 1:] & accepted[:, :-1] & delta_valid
                           & ((index[:, 1:]-index[:, :-1]) == 1))
        transition_error = (1-(F.normalize(rd, dim=2, eps=config.dynamic_floor)
                               *F.normalize(pd, dim=2, eps=config.dynamic_floor)).sum(2).clamp(-1, 1))/2
        transition_count = transition_mask.sum(1)
        transition_loss = (transition_error*transition_mask).sum(1)/transition_count.clamp_min(1)
        okay = ((accepted.sum(1) >= config.min_matches) & (relation_count >= 2)
                & (transition_count >= 2))
        losses.append(((1-config.transition_weight)*relation_loss
                       +config.transition_weight*transition_loss)*okay)
        usable.append(okay)
        relations.append(relation_loss.detach()*okay)
        transitions.append(transition_loss.detach()*okay)
        count = accepted.sum(1)
        matches.append(count)
        accepted_fractions.append(accepted.float().mean(1))
        similarities.append((similarity*accepted).sum(1)/count.clamp_min(1))
        shifts.append(((index-pos[None]).abs().float()*accepted).sum(1)/count.clamp_min(1)/(bins-1))
    usable_scales = torch.stack(usable)
    denominator = usable_scales.sum(0).clamp_min(1)
    loss_per_pair = torch.stack(losses).sum(0)/denominator
    pair_valid = usable_scales.any(0)
    accept_by_pair = torch.stack(accepted_fractions).mean(0).detach()
    stats = {'accepted_fraction': accept_by_pair.mean(),
             'usable_fraction': pair_valid.float().mean().detach(),
             'usable_scale_fraction': usable_scales.float().mean().detach(),
             'matches': torch.stack(matches).float().mean().detach(),
             'mean_cosine': torch.stack(similarities).mean().detach(),
             'mean_shift_fraction': torch.stack(shifts).mean().detach(),
             'relation_loss': (torch.stack(relations).sum()/usable_scales.sum().clamp_min(1)).detach(),
             'transition_loss': (torch.stack(transitions).sum()/usable_scales.sum().clamp_min(1)).detach(),
             'accept_by_pair': accept_by_pair, 'valid_by_pair': pair_valid.detach()}
    return loss_per_pair, pair_valid, stats


def local_structure_loss(reference, processed, config=None):
    """Mean of usable same-recording pairs; differentiable zero if none."""
    losses, valid, stats = local_structure_per_pair(reference, processed, config)
    return losses.sum()/valid.sum().clamp_min(1), stats
