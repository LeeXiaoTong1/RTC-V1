"""Balanced CE and bidirectional, label-masked same-source InfoNCE."""
import math
import torch
from torch.nn import functional as F
from .common import require_finite


def classification(logits, rows, smoothing=.02):
    labels = torch.tensor([r['label'] for r in rows], device=logits.device)
    weights = logits.new_tensor([r['ce_weight'] for r in rows])
    raw = F.cross_entropy(logits, labels, reduction='none')
    smooth = F.cross_entropy(logits, labels, reduction='none', label_smoothing=smoothing)
    return (smooth*weights).sum(), raw, smooth


def contrastive(features, units, temperature=.1, negative_mass=4.):
    """Each source has four directed pairs: O<->view1 and O<->view2.

    Negatives use the positive's view slot, opposite labels and other sources.
    Within each language their exp similarities are averaged; each language
    contributes half of a fixed total negative mass. Thus 11/5 label counts
    do not silently become auxiliary class/negative-denominator priors.
    This is a weighted InfoNCE variant, not unmodified SupCon.
    """
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(negative_mass) or negative_mass <= 0:
        raise ValueError('Invalid contrastive temperature/negative mass')
    if features.ndim != 2 or features.shape[0] != 3*len(units):
        raise ValueError('Contrastive features must cover the whole logical source batch')
    require_finite(features, 'contrastive features')
    h = F.normalize(features.float(), dim=-1)
    sim = h @ h.T
    # Source metadata stays on the CPU. One source mask is shared by its four
    # directed pairs, avoiding four separate GPU gathers/reductions per source.
    source_weights, counts = [], []
    for unit in units:
        row = unit[0]
        weights = [0.] * len(units)
        count = 0
        for language in ('en','zh'):
            candidates = [j for j, other in enumerate(units)
                if other[0]['label'] != row['label'] and other[0]['language'] == language
                and other[0]['group_id'] != row['group_id']]
            if not candidates:
                raise ValueError('Each anchor needs opposite-label sources in both languages')
            for j in candidates:
                weights[j] = negative_mass/(2*len(candidates))
            count += len(candidates)
        source_weights.append(weights); counts.append(count)
    source = torch.arange(len(units), device=h.device)
    anchor_slots = source.new_tensor([0,1,0,2])
    positive_slots = source.new_tensor([1,0,2,0])
    anchors = (3*source[:,None] + anchor_slots).reshape(-1)
    positives = (3*source[:,None] + positive_slots).reshape(-1)
    negatives = (3*source[None,None,:] + positive_slots[None,:,None]).expand(len(units),-1,-1).reshape(-1,len(units))
    weights = h.new_tensor(source_weights).repeat_interleave(4,dim=0)
    mass = h.new_tensor([u[0]['source_mass']/4 for u in units]).repeat_interleave(4)
    positive_cosines = sim[anchors,positives]
    negative_cosines = sim[anchors[:,None],negatives]
    positive_score = positive_cosines/temperature
    # log(0) masks excluded sources with -inf, so they contribute zero gradient.
    negative_score = negative_cosines/temperature + weights.log()
    losses = torch.logsumexp(torch.cat((positive_score[:,None],negative_score),dim=1),dim=1)-positive_score
    total = (mass*losses).sum()
    diagnostics = torch.stack(((mass*positive_cosines.detach()).sum(),
        (mass*(negative_cosines.detach()*weights).sum(1)/negative_mass).sum())).cpu().tolist()
    require_finite(total, 'contrastive objective')
    return total, dict(positive_cosine=diagnostics[0], negative_cosine=diagnostics[1],
        negative_sources_min=min(counts), negative_sources_max=max(counts),
        effective_negative_mass=negative_mass, directed_positive_pairs=len(units)*4,
        candidate_sources=len(units), same_label_negatives=0)
