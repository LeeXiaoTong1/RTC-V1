"""One utterance label; 50/50 streams, 25% per metadata group in each stream."""
import torch
from torch.nn import functional as F


def risk(ce_a, ce_b, coefficient=.25):
    if not 0 <= coefficient <= 1:
        raise ValueError('Risk coefficient outside [0,1]')
    # torch.maximum shares gradient at ties; both views always retain mean mass.
    return (1-coefficient) * (ce_a+ce_b)/2 + coefficient * torch.maximum(ce_a, ce_b)


def unit_loss(logits, rows, stream_sources=16, coefficient=.25, smoothing=.02):
    labels = torch.tensor([r['label'] for r in rows], device=logits.device)
    raw = F.cross_entropy(logits.float(), labels, reduction='none')
    ce = F.cross_entropy(logits.float(), labels, reduction='none', label_smoothing=smoothing)
    if len(rows) == 1 and rows[0]['role'] == 'online':
        loss = ce[0]
    elif len(rows) == 2 and {r['role'] for r in rows} == {'noisy_a', 'noisy_b'}:
        if len({(r['source_id'], r['label'], r['language'], r['occurrence']) for r in rows}) != 1:
            raise ValueError('Condition risk requires the same original source and label')
        loss = risk(ce[0], ce[1], coefficient)
    else:
        raise ValueError('Expected one Online or one indivisible same-source Noisy pair')
    return loss * (.5/stream_sources), raw, ce
