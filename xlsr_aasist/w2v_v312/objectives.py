"""One-sided, label-checked preservation; never distill a known baseline error."""
from collections import defaultdict

import numpy as np
import torch


def reference_rows(rows, logits, cfg, features):
    """Use verified existing Train scores only as bounded correct-margin targets.

    These historical scores are NOT the new Dev baseline. Their old floating
    point policy is why we exclude low confidence rows, leave slack, and cap the
    target. Ground-truth labels, not model predictions, determine supervision.
    No teacher encoder or full new feature/audio cache is required.
    """
    z = np.asarray(logits, dtype=np.float32)
    if z.shape != (len(rows), 2) or not np.isfinite(z).all():
        raise ValueError('Reference Train scores must be finite and row-aligned')
    if any(r.get('split') != 'train' or r['label'] not in (0, 1) for r in rows):
        raise ValueError('Margin references may only use labeled official Train rows')
    labels = np.array([r['label'] for r in rows], dtype=np.int64)
    index = np.arange(len(rows))
    margins = z[index, labels] - z[index, 1-labels]
    if (not np.isfinite(margins).all() or not 0 <= cfg['retention_slack'] < cfg['retention_min_margin']
            or cfg['retention_cap'] <= 0):
        raise ValueError('Invalid reference margin or retention configuration')
    selected = margins >= cfg['retention_min_margin']
    targets = np.where(selected, np.minimum(cfg['retention_cap'],
                        margins-cfg['retention_slack']), 0.)
    if features.ndim != 2 or len(features) != len(rows) or features.shape[1] < 1:
        raise ValueError('Train reference features must align with cached score rows')
    # Stream the existing mmap; retain only a scalar norm per recording, not a
    # second copy of all vectors. The encoder is never run for these references.
    norms = np.empty(len(rows),dtype=np.float32)
    for start in range(0,len(rows),4096):
        block = np.asarray(features[start:start+4096],dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite Train reference features')
        norms[start:start+len(block)] = np.sqrt(np.mean(block*block,axis=1))
    ceilings = np.maximum(cfg['margin_soft_floor'],
        cfg['margin_reference_factor']*np.abs(margins)+cfg['margin_reference_slack'])
    counts = defaultdict(lambda: dict(rows=0, protected=0))
    output = []
    for row, target, norm, ceiling in zip(rows, targets, norms, ceilings):
        output.append(dict(row, retention_target=float(target),
            reference_rms=float(max(cfg['feature_rms_floor'],norm)), margin_ceiling=float(ceiling)))
        cell = counts[f'{row["condition"]}/{row["language"]}/{row["label"]}']
        cell['rows'] += 1
        cell['protected'] += int(target > 0)
    return output, dict(rows=len(rows), protected=int((targets > 0).sum()), groups=dict(counts),
        target_cap=cfg['retention_cap'], excluded_wrong_or_uncertain=int((~selected).sum()),
        source='checksummed historical Train logits, label checked; not a same-runtime Dev reference',
        reference_norm_quantiles=np.quantile(norms,[0,.5,.95,.99,1]).tolist(),
        derived_scalar_bytes=int(3*len(rows)*4),
        new_teacher_forward_passes=0, new_audio_bytes=0)


def retention_loss(logits, labels, weights, targets):
    """Penalize erosion below a capped target, not increased confidence or fixes.

    Logical-batch weights are preserved across microbatches. Wrong/uncertain
    original examples have target 0 and receive ordinary CE, never retention.
    """
    if not bool(torch.isfinite(targets).all()) or bool((targets < 0).any()):
        raise ValueError('Invalid retention margin target')
    margins = logits.float().gather(1, labels[:, None]).squeeze(1)
    margins = margins-logits.float().gather(1, (1-labels)[:, None]).squeeze(1)
    deficit = (targets-margins).clamp_min(0)
    protected = targets > 0
    loss = (weights * protected * deficit.square()).sum()
    breached = int((protected & (deficit > 0)).detach().sum())
    return loss, breached
