"""Bidirectional same-source contrast, with class-safe bounded Train negatives.

Current pairs receive gradients at both ends. Negatives are other sources of
the opposite class, SAME language and target view type. A small FIFO of detached
training embeddings supplies negatives when length packing creates singletons.
The FIFO is updated only at successful optimizer commits and saved separately
from model parameters; inference and checkpoint averaging never use it.
"""
from collections import Counter
import math
import torch
from torch.nn import functional as F

SCHEMA = 'v52_train_contrast_bank_v1'


def configuration(cfg):
    weight = float(cfg.get('contrastive_weight', .01))
    temperature = float(cfg.get('contrastive_temperature', .2))
    ramp = cfg.get('contrastive_ramp_updates', 100)
    capacity = cfg.get('contrastive_bank_per_group', 32)
    budget = cfg.get('contrastive_sources_per_group', 3)
    if not math.isfinite(weight) or weight < 0 or weight > 1:
        raise ValueError('Contrastive weight must be finite and in [0, 1]')
    if not math.isfinite(temperature) or not .01 <= temperature <= 1:
        raise ValueError('Contrastive temperature must be in [0.01, 1]')
    if any(type(v) is not int or v < 1 for v in (ramp, capacity, budget)) or capacity > 256:
        raise ValueError('Positive contrastive budgets required; bank/group <=256')
    return weight, temperature, ramp, capacity


def effective_weight(cfg):
    weight, _, ramp, _ = configuration(cfg)
    return weight * min(1., max(0, int(cfg.get('current_update', 1))) / ramp)


def empty_bank(cfg):
    return dict(schema=SCHEMA, capacity=configuration(cfg)[3], entries=[])


def validate_bank(bank, cfg):
    if bank.get('schema') != SCHEMA or bank.get('capacity') != configuration(cfg)[3]:
        raise ValueError('Contrastive bank schema/capacity changed')
    counts, seen, dimensions = Counter(), set(), set()
    for entry in bank['entries']:
        key = entry['source'], entry['view']
        group = entry['language'], entry['label'], entry['view']
        feature = entry['feature']
        if (not isinstance(entry['source'], str) or not entry['source'] or key in seen
                or group[0] not in ('en', 'zh') or group[1] not in (0, 1)
                or group[2] not in ('official', 'noisy') or entry.get('split') != 'train'
                or not isinstance(feature, torch.Tensor) or feature.ndim != 1
                or feature.device.type != 'cpu' or feature.dtype != torch.float32
                or not feature.numel() or feature.requires_grad or not bool(torch.isfinite(feature).all())):
            raise ValueError('Invalid/duplicate/non-Train contrastive bank entry')
        counts[group] += 1
        seen.add(key)
        dimensions.add(feature.numel())
    if len(dimensions) > 1 or any(n > bank['capacity'] for n in counts.values()):
        raise ValueError('Contrastive bank dimensions or group budget changed')
    return bank


def bank_for(model, cfg):
    if not hasattr(model, '_v52_contrastive_bank'):
        model._v52_contrastive_bank = empty_bank(cfg)
    return model._v52_contrastive_bank


def device_bank(bank, device):
    """One small host-to-device transfer per update, not one per negative."""
    if not bank['entries']:
        return bank
    features = torch.stack([entry['feature'] for entry in bank['entries']]).to(device)
    return dict(bank, entries=[dict(entry, feature=feature) for entry, feature in zip(bank['entries'], features)])


def next_bank(bank, pending, cfg):
    """Build before commit; never mutate the bank used by this logical update."""
    result = dict(schema=SCHEMA, capacity=configuration(cfg)[3], entries=list(bank['entries']))
    if pending:
        features = torch.stack([p['feature'].detach().float() for p in pending]).cpu()
        for pending_entry, feature in zip(pending, features):
            entry = dict(pending_entry, feature=feature.clone())
            key = entry['source'], entry['view']
            result['entries'] = [r for r in result['entries'] if (r['source'], r['view']) != key]
            result['entries'].append(entry)
        counts, kept = Counter(), []
        for entry in reversed(result['entries']):
            group = entry['language'], entry['label'], entry['view']
            if counts[group] < result['capacity']:
                kept.append(entry)
                counts[group] += 1
        result['entries'] = list(reversed(kept))
    return validate_bank(result, cfg)


def classify_with_features(model, sequences):
    """Capture the actual 384d (or parent 320d) readout before final dropout.

    This local hook observes the existing forward; it adds no model parameters,
    encoder pass, backend pass, or new inference path.
    """
    captured = []
    handle = model.head.aasist.drop.register_forward_pre_hook(
        lambda module, args: captured.append(args[0]))
    try:
        logits = model.classify_sequences(sequences)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].ndim != 2 or len(captured[0]) != len(sequences):
        raise ValueError('Contrastive learning requires one actual pre-classifier feature per view')
    features = captured[0].float()
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError('Nonfinite contrastive features')
    return logits, features


def paired_loss(pairs, bank, cfg, zero):
    """Sum four-group-balanced pair losses, without renormalizing missing pairs.

    A pair contains (Noisy feature, official Online feature, source row).
    Opposite labels only are negatives; same-label sources are neither forced
    apart nor incorrectly treated as the identical recording.
    """
    _, temperature, _, _ = configuration(cfg)
    current, pending, seen = [], [], set()
    for noisy, official, row in pairs:
        if (row.get('split') != 'train' or not row.get('canonical_source')
                or row['canonical_source'] in seen or row.get('language') not in ('en', 'zh')
                or row.get('label') not in (0, 1)):
            raise ValueError('Contrastive pairs must come from canonical Train sources')
        seen.add(row['canonical_source'])
        for view, feature in (('noisy', noisy), ('official', official)):
            record = dict(source=row['canonical_source'], language=row['language'],
                          label=row['label'], view=view, split='train',
                          feature=F.normalize(feature.float(), dim=0, eps=1e-8))
            current.append(record)
            pending.append(dict(record, feature=record['feature'].detach()))
    current_sources = {r['source'] for r in current}
    historical = [r for r in bank['entries'] if r['source'] not in current_sources]
    losses, stats = [], Counter(pairs=len(pairs), directions=0, negative_comparisons=0,
                               queue_negatives=0, skipped_no_negative=0)
    for i, (_, _, row) in enumerate(pairs):
        directions = []
        for query, positive in ((current[2*i], current[2*i+1]), (current[2*i+1], current[2*i])):
            negatives = [r for r in current + historical
                         if r['source'] != query['source'] and r['label'] != query['label']
                         and r['language'] == query['language'] and r['view'] == positive['view']]
            if not negatives:
                stats['skipped_no_negative'] += 1
                continue
            query_feature = query['feature']
            negative_features = torch.stack([r['feature'].to(query_feature.device) for r in negatives])
            similarity = torch.cat(((query_feature * positive['feature']).sum().reshape(1),
                                     negative_features @ query_feature)) / temperature
            directions.append(torch.logsumexp(similarity, dim=0) - similarity[0])
            stats['directions'] += 1
            stats['negative_comparisons'] += len(negatives)
            stats['queue_negatives'] += sum(r['source'] not in current_sources for r in negatives)
        budget = int(row.get('contrastive_group_budget', 0))
        if budget < 1:
            raise ValueError('Missing contrastive source budget')
        if directions:
            # Always divide by two: a missing direction must not upweight the other.
            losses.append(torch.stack(directions).sum() * (.25 / budget / 2))
            stats['active_pairs'] += 1
    return (torch.stack(losses).sum() if losses else zero), dict(stats), pending
