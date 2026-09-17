"""Separate authenticity class weights from acoustic-condition weights."""
import torch
import torch.nn.functional as F

from .plan import group_coefficients


def grouped_classification(logits, labels, ordinary_count, real_pairs, noisy_pairs,
                           beta=.3, ordinary_class_weights=None):
    """Order: ordinary, real_off, real_on, noisy_ref, noisy_processed.

    Ordinary CE uses protocol-derived class weights supplied by the trainer.
    All paired-input CE remains unweighted because pair sampling is balanced.
    beta controls processed-noisy classification weight, not contrastive loss.
    """
    expected = ordinary_count + 2 * real_pairs + 2 * noisy_pairs
    if logits.shape != (expected, 2) or labels.shape != (expected,):
        raise ValueError("Logits/labels do not match the four classification groups")
    if not torch.isfinite(logits).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Classification inputs must be finite with binary labels")
    if ordinary_class_weights is None:
        raise ValueError("ordinary_class_weights must be provided by the training protocol")

    class_weights = torch.as_tensor(ordinary_class_weights, dtype=torch.float32, device=logits.device)
    if class_weights.shape != (2,) or not torch.isfinite(class_weights).all() or (class_weights <= 0).any():
        raise ValueError("ordinary_class_weights must contain two positive finite values")

    sizes = (ordinary_count, 2 * real_pairs, noisy_pairs, noisy_pairs)
    weights = group_coefficients(*sizes[:3], beta)
    ce = F.cross_entropy(logits.float(), labels, reduction="none")
    ordinary_weights = class_weights[labels[:ordinary_count]]
    parts = {"ordinary": (ce[:ordinary_count] * ordinary_weights).sum() / ordinary_weights.sum()}
    start = ordinary_count
    for name, count in zip(("real_pair", "noisy_reference", "noisy_processed"), sizes[1:]):
        parts[name] = ce[start:start + count].mean()
        start += count
    loss = sum(w * value for w, value in zip(weights, parts.values()))
    return loss, parts, dict(zip(parts, weights))
