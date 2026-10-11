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
    total = h.sum()*0
    positive_cos = h.sum()*0
    negative_cos = h.sum()*0
    counts = []
    for i, unit in enumerate(units):
        row = unit[0]
        for anchor_slot, positive_slot in ((0,1),(1,0),(0,2),(2,0)):
            anchor, positive = 3*i+anchor_slot, 3*i+positive_slot
            indices, weights = [], []
            for language in ('en','zh'):
                candidates = [3*j+positive_slot for j, other in enumerate(units)
                    if other[0]['label'] != row['label'] and other[0]['language'] == language
                    and other[0]['group_id'] != row['group_id']]
                if not candidates:
                    raise ValueError('Each anchor needs opposite-label sources in both languages')
                indices.extend(candidates)
                weights.extend([negative_mass/(2*len(candidates))]*len(candidates))
            positive_score = sim[anchor, positive]/temperature
            negative_score = sim[anchor, indices]/temperature + h.new_tensor(weights).log()
            loss = torch.logsumexp(torch.cat((positive_score[None], negative_score)), dim=0)-positive_score
            mass = row['source_mass']/4
            total = total + mass*loss
            positive_cos = positive_cos + mass*sim[anchor, positive]
            negative_cos = negative_cos + mass*(sim[anchor, indices]*h.new_tensor(weights)).sum()/negative_mass
            counts.append(len(indices))
    require_finite(total, 'contrastive objective')
    return total, dict(positive_cosine=float(positive_cos.detach()), negative_cosine=float(negative_cos.detach()),
        negative_sources_min=min(counts), negative_sources_max=max(counts),
        effective_negative_mass=negative_mass, directed_positive_pairs=len(units)*4,
        candidate_sources=len(units), same_label_negatives=0)
