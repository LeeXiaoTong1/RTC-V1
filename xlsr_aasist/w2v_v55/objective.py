"""Seo et al., Interspeech 2026, equations (1)--(7).

Distances are squared Euclidean on the actual pre-classifier readout, in FP32.
There is no detached teacher, memory bank, replacement classifier or BPL.
"""
import math
import torch
from torch.nn import functional as F


def configuration(cfg):
    result = {name: float(cfg.get('hp_' + name, default)) for name, default in
              (('guidance_weight', .01), ('triplet_weight', .01),
               ('guidance_margin', .2), ('triplet_margin', .3))}
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError('Hard-positive weights/margins must be finite and nonnegative')
    ramp = cfg.get('hp_ramp_updates', 200)
    if type(ramp) is not int or ramp < 1:
        raise ValueError('Positive integer hard-positive ramp required')
    if cfg.get('hp_mode', 'hard_positive') not in ('hard_positive', 'control'):
        raise ValueError('Unknown V5.5 objective mode')
    return result, ramp


def paper_losses(anchor, positives, negatives, *, guidance_margin=.2, triplet_margin=.3):
    """One anchor, two processed Real, three Fake; gradients reach all six."""
    a, p, n = anchor.float(), positives.float(), negatives.float()
    if a.ndim != 1 or p.shape != (2, a.numel()) or n.shape != (3, a.numel()):
        raise ValueError('Hard-positive group requires shapes D, 2xD, 3xD')
    if not all(math.isfinite(v) and v >= 0 for v in (guidance_margin, triplet_margin)):
        raise ValueError('Invalid metric margin')
    d_ap = (p - a).square().sum(-1)
    d_an = (n - a).square().sum(-1)
    hardest = d_ap.detach().argmax()
    d_hn = (n - p[hardest]).square().sum(-1)
    guidance_gap = d_ap[hardest] - d_hn + guidance_margin
    triplet_gap = d_ap[:, None] - d_an[None, :] + triplet_margin
    guidance = F.softplus(guidance_gap).mean()
    triplet = F.relu(triplet_gap).mean()
    if not bool(torch.isfinite(guidance + triplet)):
        raise FloatingPointError('Nonfinite hard-positive loss')
    stats = dict(hardest=hardest, anchor_positive=d_ap.detach(),
                 anchor_negative=d_an.detach(), hard_negative=d_hn.detach(),
                 guidance_violations=(guidance_gap.detach() > 0).sum(),
                 triplet_violations=(triplet_gap.detach() > 0).sum(),
                 guidance_raw=guidance.detach(), triplet_raw=triplet.detach(),
                 feature_norms=torch.cat((a[None], p, n)).detach().norm(dim=-1))
    return guidance, triplet, stats


def classify_with_readout(model, sequences):
    """Reuse the one AASIST forward; read its 384d aggregate BEFORE final dropout."""
    captured = []
    handle = model.head.aasist.drop.register_forward_pre_hook(
        lambda module, args: captured.append(args[0]))
    try:
        logits = model.classify_sequences(sequences)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (len(sequences), 384):
        raise ValueError('Expected V5.1 pre-classifier 384-dimensional utterance readout')
    return logits, captured[0].float()


def micro_objective(readouts, classified_rows, cfg, *, group_count, audit=False, logits=None):
    """Loss normalization is over planned groups, never the physical batch size."""
    options, ramp = configuration(cfg)
    groups = {}
    for i, row in enumerate(classified_rows):
        group = row.get('hp_group')
        role = row.get('hp_view_role')
        if group is not None and role is not None:
            groups.setdefault(group, {}).setdefault(role, []).append(i)
    zero = readouts.sum() * 0.
    guidance, triplet, records = zero, zero, []
    fraction = min(1., max(0, int(cfg.get('current_update', 1))) / ramp)
    enabled = cfg.get('hp_mode', 'hard_positive') == 'hard_positive'
    wg = options['guidance_weight'] * fraction * enabled
    wt = options['triplet_weight'] * fraction * enabled
    for name, slots in groups.items():
        if {k: len(v) for k, v in slots.items()} != {'anchor': 1, 'positive': 2, 'negative': 3}:
            raise ValueError('A hard-positive group was split or lost a view: ' + str(name))
        if group_count < 1:
            raise ValueError('Missing hard-positive group normalization budget')
        ai, pi, ni = slots['anchor'][0], slots['positive'], slots['negative']
        rows = [classified_rows[i] for i in [ai] + pi + ni]
        if any(r['label'] != y for r, y in zip(rows, [1, 1, 1, 0, 0, 0])):
            raise ValueError('Hard-positive labels changed')
        if len({r['canonical_source'] for r in rows[:3]}) != 1:
            raise ValueError('RTC positives must share their anchor source')
        if len({r['canonical_source'] for r in rows[3:]}) != 3:
            raise ValueError('Three different Fake sources are required')
        if len({r['language'] for r in rows}) != 1 or any(r['split'] != 'train' for r in rows):
            raise ValueError('Metric groups must stay within training and language')
        g, t, stat = paper_losses(readouts[ai], readouts[pi], readouts[ni],
            guidance_margin=options['guidance_margin'], triplet_margin=options['triplet_margin'])
        guidance = guidance + g * (wg / group_count)
        triplet = triplet + t * (wt / group_count)
        record = dict(group=name, source=rows[0]['id'], language=rows[0]['language'],
                      negatives=[r['id'] for r in rows[3:]],
                      guidance_weighted=g.detach() * (wg / group_count),
                      triplet_weighted=t.detach() * (wt / group_count), **stat)
        if logits is not None:
            indices = [ai] + pi + ni
            truth = torch.tensor([1, 1, 1, 0, 0, 0], device=logits.device)
            selected = logits[indices].detach().float()
            record['true_probabilities'] = selected.softmax(-1).gather(1, truth[:, None])[:, 0]
            record['correct'] = selected.argmax(-1) == truth
        if audit and readouts.requires_grad:
            # A sparse, exact gradient at the readout; no second Omni backward.
            record['readout_gradient_norms'] = {}
            for label, loss in (('guidance', g * wg / group_count), ('triplet', t * wt / group_count)):
                grad = torch.autograd.grad(loss, readouts, retain_graph=True)[0]
                record['readout_gradient_norms'][label] = grad.detach().norm()
        records.append(record)
    return guidance, triplet, records


def serialized(records):
    def convert(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value
    return convert(records)
