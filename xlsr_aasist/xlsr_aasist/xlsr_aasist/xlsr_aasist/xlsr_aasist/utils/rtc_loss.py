"""Label-conditioned contrastive supervision for genuine RTC utterance pairs."""

import math

import torch
import torch.nn.functional as F


def rtc_pair_contrastive_loss(
    offline_features, online_features, pair_labels, temperature=0.1
):
    """Compare same-source Offline/Online pairs and opposite-authenticity negatives.

    Args:
        offline_features: Floating point tensor of shape ``[P, D]``.
        online_features: Floating point tensor of shape ``[P, D]``. Row ``i``
            must be the actual communication counterpart of Offline row ``i``.
        pair_labels: Tensor of shape ``[P]``, containing binary labels 0 or 1.
            Both members of each pair must share this authenticity label.
        temperature: Finite positive cosine-similarity temperature.

    With L2-normalized features ``u`` and ``v``, define

        s_ij = <u_i, v_j> / temperature
        N_i = {j : label_j != label_i}
        L_off_i = logsumexp({s_ii} union {s_ij : j in N_i}) - s_ii
        L_on_i  = logsumexp({s_ii} union {s_ji : j in N_i}) - s_ii

    The result averages both directions over anchors with at least one negative.
    The only positive is the same-row counterpart. Other same-label utterances
    are excluded, so distinct spoofing methods are not all forced together.
    Normalization, cosine similarities, and logsumexp run in FP32 even when the
    caller has automatic mixed precision enabled.

    Returns:
        ``(loss, stats)``. ``loss`` is a scalar FP32 tensor. ``stats`` contains
        integer ``pairs`` and ``valid_anchors`` (the latter counts both domains).
        Empty and single-class batches yield a differentiable zero connected to
        both feature tensors, with zero valid anchors.

    This objective requires verified source pairs; label equality alone does not
    establish that two utterances are a pair.
    """
    try:
        temperature = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("temperature must be finite and positive") from exc
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if offline_features.ndim != 2 or online_features.ndim != 2:
        raise ValueError("both feature tensors must have shape [P, D]")
    if offline_features.shape != online_features.shape:
        raise ValueError("Offline and Online feature shapes must match")
    if offline_features.shape[1] == 0:
        raise ValueError("feature dimension D must be positive")
    if not offline_features.is_floating_point() or not online_features.is_floating_point():
        raise ValueError("feature tensors must use floating point dtypes")
    if offline_features.device != online_features.device:
        raise ValueError("Offline and Online features must be on the same device")
    pairs = offline_features.shape[0]
    if pair_labels.ndim != 1 or pair_labels.shape[0] != pairs:
        raise ValueError("pair_labels must have shape [P]")
    labels = pair_labels.to(device=offline_features.device)
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("pair_labels must contain only binary labels 0 and 1")

    stats = {"pairs": pairs, "valid_anchors": 0}
    with torch.autocast(device_type=offline_features.device.type, enabled=False):
        offline = offline_features.float()
        online = online_features.float()
        opposite_label = labels[:, None] != labels[None, :]
        valid = opposite_label.any(dim=1)
        num_valid = int(valid.sum().item())
        if num_valid == 0:
            return (offline.sum() + online.sum()) * 0.0, stats

        offline = F.normalize(offline, p=2, dim=1, eps=1e-8)
        online = F.normalize(online, p=2, dim=1, eps=1e-8)
        similarities = offline @ online.transpose(0, 1)
        logits = similarities / temperature
        keep = opposite_label | torch.eye(pairs, device=labels.device, dtype=torch.bool)
        masked_logits = logits.masked_fill(~keep, float("-inf"))
        positives = logits.diagonal()
        offline_losses = torch.logsumexp(masked_logits, dim=1) - positives
        online_losses = torch.logsumexp(masked_logits, dim=0) - positives
        loss = (offline_losses[valid].sum() + online_losses[valid].sum()) / (2 * num_valid)
        stats["valid_anchors"] = 2 * num_valid
        return loss, stats
