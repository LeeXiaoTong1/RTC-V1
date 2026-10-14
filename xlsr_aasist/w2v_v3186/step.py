"""One encoding per view, source-local PCL graphs, and bounded sampled audits."""
from collections import Counter, defaultdict
import math
import torch
from w2v_v3183r2.common import GROUPS, require_finite
from w2v_v3183r2.sampling import validate_units
from .objective import micro_losses, phoneme_loss, source_weight, auxiliary_weights


def schedule(optimizer, cfg, epoch, step, steps):
    if cfg.get('warm_epochs', 0) != 0:
        raise ValueError('V3.18.6 has no head warmup epochs')
    if not 0 <= epoch < cfg['epochs'] or not 0 <= step < steps:
        raise ValueError('Invalid schedule cursor')
    index, total = epoch * steps + step + 1, cfg['epochs'] * steps
    fraction = (index - 1) / max(1, total - 1)
    multiplier = cfg['lr_floor'] + (1 - cfg['lr_floor']) * .5 * (1 + math.cos(math.pi * fraction))
    ramp = min(cfg['lr_warmup_updates'], max(1, total // 10))
    for group in optimizer.param_groups:
        factor = min(1., index / ramp) if group['name'] == 'lora' else 1.
        group['lr'] = group['initial_lr'] * multiplier * factor
    # Derived cursor, not a mutation of fingerprinted experiment config.
    optimizer._v3186_update = index
    return cfg['risk_coefficient']


def _layout(unit, weights):
    """Main classification rows first; only enabled references are encoded."""
    rows, slots = list(unit), {}
    if len(unit) == 1 or not any(weights.values()):
        return rows, slots
    first = unit[0]
    refs = first.get('pcl_refs', {})
    pairs = first.get('pcl_pairs', {})
    noise = bool(weights['noise'])
    platform = bool(weights['platform']) and 'official' in refs
    if (noise or platform) and 'clean' not in refs:
        raise ValueError('PCL enabled but the paired clean RTC reference is missing')
    scheduled = []
    for kind, enabled in (('noise', noise), ('platform', platform)):
        if enabled:
            if kind not in pairs:
                raise ValueError('Missing ' + kind + ' phoneme pairs')
            scheduled.append(kind)
    slots['scheduled'] = scheduled
    # A rejected alignment remains visible in coverage, but must not spend a
    # 3B encoder pass on an auxiliary row that cannot contribute any loss.
    noise = noise and bool(len(pairs['noise']))
    platform = platform and bool(len(pairs['platform']))
    for name in ('clean', 'official'):
        if (name == 'clean' and (noise or platform)) or (name == 'official' and platform):
            row = refs[name]
            if (row['language'], row['label'], row['source_id'], row['group_id']) != (first['language'], first['label'], first['source_id'], first['group_id']):
                raise ValueError('PCL reference must match the source, language, and true class')
            if row.get('split') != 'train':
                raise ValueError('PCL references must belong to Train')
            slots[name] = len(rows)
            rows.append(row)
    if noise:
        chosen = first.get('pcl_noisy_index')
        if chosen not in (0, 1) or 'noise' not in pairs:
            raise ValueError('Missing matched Noisy index/phoneme pairs')
        slots['noise'] = (slots['clean'], chosen)
    if platform:
        if 'platform' not in pairs:
            raise ValueError('Missing official/clean RTC phoneme pairs')
        slots['platform'] = (slots['official'], slots['clean'])
    return rows, slots


def _gradient_audit(ce, branches, model):
    """One representative microbatch, reader projection only; not global gradients.

    autograd.grad stops at this shared projection instead of running three
    extra complete LoRA backwards. Invoked once only on requested audit steps.
    """
    parameter = model.fusion.projection.weight
    terms = {'classification': ce, **branches}
    values = {}
    for name, loss in terms.items():
        if not loss.requires_grad:
            continue
        gradient = torch.autograd.grad(loss, parameter, retain_graph=True, allow_unused=True)[0]
        if gradient is not None:
            values[name] = gradient.detach().float().flatten()
    result = dict(scope='first valid auxiliary microbatch; fusion.projection.weight; weighted losses; before clipping',
                  norms={}, cosines={})
    for name, value in values.items():
        result['norms'][name] = value.norm()
    if 'classification' in values:
        base = values['classification']
        for name in ('noise', 'platform'):
            if name in values:
                other = values[name]
                result['cosines']['classification/' + name] = torch.nn.functional.cosine_similarity(base, other, dim=0, eps=1e-12)
    # These small reductions transfer only on explicit audit updates.
    for key in ('norms', 'cosines'):
        result[key] = {name: float(value) for name, value in result[key].items()}
    return result


def _module_grad_norms(optimizer):
    sums = {}
    for group in optimizer.param_groups:
        terms = [p.grad.detach().float().square().sum() for p in group['params'] if p.grad is not None]
        if terms:
            sums[group['name']] = sums.get(group['name'], 0) + torch.stack(terms).sum()
    if not sums:
        return {}
    return dict(zip(sums, torch.stack(list(sums.values())).sqrt().cpu().tolist()))


def _update_snapshot(optimizer, cap=1024):
    # Bounded sample across optimizer groups; no cloning millions of parameters.
    result = []
    for group in optimizer.param_groups:
        remaining = cap
        for parameter in group['params']:
            if parameter.grad is None or remaining == 0:
                continue
            count = min(remaining, parameter.numel())
            result.append((group['name'], parameter, count, parameter.detach().flatten()[:count].clone()))
            remaining -= count
    return result


def train_step(model, optimizer, units, cfg, coefficient, audit=False):
    from .data import microbatches
    validate_units(units, cfg['group_quotas'])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    weights = auxiliary_weights(dict(cfg, current_update=getattr(optimizer, '_v3186_update', 1)))
    processing, base_views, coverage = Counter(), Counter(), defaultdict(Counter)
    metadata, telemetry, aux_telemetry = [], [], []
    encoded_views, reference_views, gradient_audit = 0, 0, None
    for unit in units:
        first = unit[0]
        base = first.get('base_augmentation')
        if base:
            kind = 'online' if len(unit) == 1 else 'offline'
            base_views[f'{kind}/{first["language"]}/{first["label"]}/{base["recipe"]["family"]}'] += 1
        for row in unit:
            recipe = row.get('augmentation', {}).get('recipe')
            if recipe:
                processing[f'{row["language"]}/{row["label"]}/{recipe["family"]}/{recipe["codec"]}/{recipe["noise_type"]}'] += 1
                if recipe.get('stress'):
                    processing['stress/' + recipe['stress']['mechanism']] += 1
    for micro in microbatches(units, cfg['microbatch'], cfg['frame_budget']):
        rows, layouts, main_indices = [], [], []
        for unit in micro:
            expanded, slots = _layout(unit, weights)
            offset = len(rows)
            layouts.append((offset, slots))
            main_indices.extend(range(offset, offset + len(unit)))
            rows.extend(expanded)
        # One encoder call for this microbatch; features are shared by CE and
        # both auxiliary branches and released immediately after backward.
        sequences = model.encode([row['wave'] for row in rows])
        encoded_views += len(rows)
        reference_views += len(rows) - len(main_indices)
        logits = model.classify_sequences([sequences[i] for i in main_indices])
        parts = micro_losses(logits, micro, cfg['group_quotas'], coefficient, cfg['label_smoothing'])
        ce = torch.stack([part[0] for part in parts]).sum()
        branches, aux_items, offset = {}, [], 0
        for unit, (objective, raw), (seq_offset, slots) in zip(micro, parts, layouts):
            first = unit[0]
            z = logits[offset:offset + len(unit)]
            offset += len(unit)
            telemetry.append(torch.stack((objective.detach(), raw.detach().sum(), raw.detach().max(),
                (z.argmax(1) == first['label']).sum(), (z.argmax(1) == 0).sum())).float())
            metadata.append((f'{"online" if len(unit) == 1 else "noisy"}/{first["language"]}/{first["label"]}', len(unit), first['label']))
            for kind in slots.get('scheduled', []):
                matches = first['pcl_pairs'][kind]
                if kind in slots:
                    a, b = slots[kind]
                    value, valid = phoneme_loss(sequences[seq_offset + a], sequences[seq_offset + b], matches,
                                               cfg.get('pcl_min_frames', 1))
                else:
                    value, valid = ce.new_zeros(()), 0
                weighted = value * source_weight(kind, first, cfg) * weights[kind]
                if valid:
                    branches[kind] = branches.get(kind, 0) + weighted
                aux_items.append((kind, value.detach(), weighted.detach()))
                cell = coverage[f'{kind}/{first["language"]}/{first["label"]}']
                cell.update(scheduled_sources=1, valid_sources=int(valid > 0), matched_phones=len(matches), valid_phones=valid)
                declared = first.get('pcl_coverage', {})
                cell['candidate_phones'] += int(declared.get(kind + '_candidates', len(matches)))
                cell['all_pairs_masked_sources'] += int(not len(matches))
                cell['no_valid_frames_sources'] += int(bool(len(matches)) and valid == 0)
                for stage in ('clean', kind):
                    timing = first.get('pcl_timing', {}).get(stage)
                    if timing is not None and not timing.get('valid', False):
                        cell['invalid_' + stage + '_' + timing.get('reason', 'unknown') + '_sources'] += 1
        loss = ce + sum(branches.values())
        require_finite(loss, 'classification and PCL loss')
        if audit and gradient_audit is None and branches:
            gradient_audit = _gradient_audit(ce, branches, model)
        loss.backward()
        aux_telemetry.extend(aux_items)
    module_norms = _module_grad_norms(optimizer) if audit else {}
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['max_grad_norm'], error_if_nonfinite=True)
    before = _update_snapshot(optimizer) if audit else []
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    updates, relative, parameter_norms, sample_counts = {}, {}, {}, Counter()
    if before:
        sums = {}
        with torch.no_grad():
            for name, parameter, count, previous in before:
                current = parameter.detach().flatten()[:count]
                values = torch.stack(((current - previous).float().square().sum(), previous.float().square().sum()))
                sums[name] = sums.get(name, 0) + values
                sample_counts[name] += count
            for name, (change, magnitude) in zip(sums, torch.stack(list(sums.values())).sqrt().cpu().tolist()):
                updates[name], parameter_norms[name] = change, magnitude
                relative[name] = change / magnitude if magnitude else (0. if change == 0 else None)
    observed = torch.stack(telemetry).cpu().tolist()
    cells = defaultdict(lambda: defaultdict(float))
    predictions = {'fake': [0, 0], 'real': [0, 0]}
    for (cell, views, label), (loss, raw, maximum, correct, fake) in zip(metadata, observed):
        for key, value in dict(objective=loss, raw_ce_sum=raw, views=views, correct=correct).items():
            cells[cell][key] += value
        pred = predictions['fake' if label == 0 else 'real']
        pred[0] += int(fake)
        pred[1] += views
    aux = {kind: dict(source_loss_sum=0., weighted_loss=0., sources=0) for kind in ('noise', 'platform')}
    if aux_telemetry:
        values = torch.stack([torch.stack((raw, weighted)) for _, raw, weighted in aux_telemetry]).cpu().tolist()
        for (kind, _, _), (raw, weighted) in zip(aux_telemetry, values):
            aux[kind]['source_loss_sum'] += raw
            aux[kind]['weighted_loss'] += weighted
            aux[kind]['sources'] += 1
    class_loss = sum(item[0] for item in observed)
    return dict(loss=class_loss + sum(item['weighted_loss'] for item in aux.values()), classification_loss=class_loss,
        raw_ce_sum=sum(item[1] for item in observed), max_example_ce=max(item[2] for item in observed),
        views=sum(item[1] for item in metadata), cells={k: dict(v) for k, v in cells.items()},
        predicted_fake_by_true_class=predictions, grad_norm=float(norm), risk_coefficient=coefficient,
        lrs={group['name']: group['lr'] for group in optimizer.param_groups}, processing_counts=dict(processing),
        base_source_view_counts=dict(base_views), unique_original_sources=len({unit[0]['group_id'] for unit in units}),
        encoded_views=encoded_views, auxiliary_reference_views=reference_views,
        pcl_weights=weights, pcl_losses=aux, pcl_coverage={k: dict(v) for k, v in coverage.items()},
        pcl_gradient_audit=gradient_audit, module_grad_norms_before_clip=module_norms,
        module_update_norms=updates, module_relative_updates=relative, module_parameter_norms=parameter_norms,
        module_update_audit_scope='first <=1024 entries per optimizer parameter group, not whole-module norms',
        module_update_sample_counts=dict(sample_counts))
