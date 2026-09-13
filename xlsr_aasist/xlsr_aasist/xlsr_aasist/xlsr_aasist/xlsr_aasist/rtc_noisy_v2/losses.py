"""Separate authenticity class weights from acoustic-condition weights."""
import torch
import torch.nn.functional as F

from .plan import group_coefficients


def grouped_classification(logits, labels, ordinary_count, real_pairs, noisy_pairs, beta=.3):
    """Order: ordinary, real_off, real_on, noisy_ref, noisy_processed.

    Ordinary CE: normalized weighted mean with [fake=.1,real=.9].
    All paired-input CE: ordinary unweighted mean (balanced source sampling).
    beta controls the mean CE of processed noisy inputs, NOT the contrastive loss.
    """
    expected = ordinary_count + 2 * real_pairs + 2 * noisy_pairs
    if logits.shape != (expected, 2) or labels.shape != (expected,):
        raise ValueError("Logits/labels do not match the four classification groups")
    if not torch.isfinite(logits).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Classification inputs must be finite with binary labels")
    sizes = (ordinary_count, 2 * real_pairs, noisy_pairs, noisy_pairs)
    weights = group_coefficients(*sizes[:3], beta)
    ce = F.cross_entropy(logits.float(), labels, reduction="none")
    ordinary_weights = logits.new_tensor([.1, .9], dtype=torch.float32)[labels[:ordinary_count]]
    parts = {"ordinary": (ce[:ordinary_count] * ordinary_weights).sum() / ordinary_weights.sum()}
    start = ordinary_count
    for name, count in zip(("real_pair", "noisy_reference", "noisy_processed"), sizes[1:]):
        parts[name] = ce[start:start + count].mean()
        start += count
    loss = sum(w * value for w, value in zip(weights, parts.values()))
    return loss, parts, dict(zip(parts, weights))
