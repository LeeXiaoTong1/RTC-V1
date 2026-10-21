"""One encoding per view, source-local PCL graphs, and bounded sampled audits."""
from collections import Counter, defaultdict
import math
import torch
from w2v_v3183r2.common import GROUPS, require_finite
from w2v_v3191.sampling import validate_units
from w2v_v3191.ce import classification_weights
from w2v_v3191.objective import (micro_losses, phoneme_loss, phoneme_features, phoneme_variance,
                        source_weight, auxiliary_weights, batch_quotas, confidence_regularization)
from .objective import paired_losses


def _layout(unit, weights):
    """Encode matched Clean-RTC once for both paired CE and optional PCL."""
    rows, slots = list(unit), {}
    if unit[0]['role'] == 'online':
        return rows, slots
    first = unit[0]
    refs = first.get('pcl_refs', {})
    pairs = first.get('pcl_pairs', {})
    noise = bool(weights['noise'])
    platform = bool(weights['platform']) and 'official' in refs
    if 'clean' not in refs:
        raise ValueError('Paired classification requires a matched clean RTC reference')
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
        if name == 'clean' or (name == 'official' and platform):
            row = refs[name]
            if (row['language'], row['label'], row['source_id'], row['group_id']) != (first['language'], first['label'], first['source_id'], first['group_id']):
                raise ValueError('PCL reference must match the source, language, and true class')
            if row.get('split') != 'train':
                raise ValueError('PCL references must belong to Train')
            slots[name] = len(rows)
            rows.append(row)
    if noise:
        chosen = first.get('pcl_noisy_index')
        if chosen != 0 or 'noise' not in pairs:
            raise ValueError('Missing matched Noisy index/phoneme pairs')
        slots['noise'] = (slots['clean'], chosen)
    if platform:
        if 'platform' not in pairs:
            raise ValueError('Missing official/clean RTC phoneme pairs')
        slots['platform'] = (slots['official'], slots['clean'])
    return rows, slots


def _side_pass(model, units, cfg, scaler, *, backward, capture_logits=False, backward_factor=1.):
    coefficient=0.; audit=False
    ce_weights = classification_weights(cfg)
    side=units[0][0]['meta_side']
    if side=='B':cfg=dict(cfg,consistency='none')
    from .data import microbatches
    quotas = batch_quotas(units, cfg)
    validate_units(units, quotas)
    joint = True  # Preserve inherited CE/confidence math in both repair stages.
    if not model.training:
        model.train()
    runtime = dict(cfg, group_quotas=quotas, joint_phase=joint,
                   current_update=int(cfg.get('current_update', 1)))
    weights = auxiliary_weights(runtime)
    if not model.joint:
        weights = dict(platform=0., noise=0.)
    pcl_state = getattr(model, 'pcl_scale', None)
    if any(weights.values()) and (pcl_state is None or not pcl_state.is_frozen):
        raise RuntimeError('Joint PCL requires frozen Train phone scales from CE warmup')
    scale = pcl_state.scale if pcl_state is not None else None
    variance_weight = float(cfg.get('pcl_variance_weight', .01))
    if not math.isfinite(variance_weight) or variance_weight < 0:
        raise ValueError('Invalid PCL variance weight')
    variance_ramp = min(1., max(0, runtime['current_update']) / max(1, cfg.get('pcl_ramp_updates', 300)))
    processing, base_views, coverage = Counter(), Counter(), defaultdict(Counter)
    activity = Counter()
    metadata, telemetry, aux_telemetry, confidence_telemetry = [], [], [], []
    variance_telemetry, variance_coverage = [], defaultdict(Counter)
    calibration_phones, calibration_visits = 0, 0
    encoded_views, reference_views, gradient_audit = 0, 0, None
    paired_classification_views = 0
    captured, captured_sources = [], []
    repair_telemetry, repair_records = [], []
    ce_captured, ce_plan = [], []
    for unit in units:
        first = unit[0]
        base = first.get('base_augmentation')
        if base:
            kind = 'online' if first['role'] == 'online' else 'offline'
            base_views[f'{kind}/{first["language"]}/{first["label"]}/{base["recipe"]["family"]}'] += 1
        for row in unit:
            gate = row.get('augmentation', {}).get('rtc_activity')
            if gate:
                activity['views'] += 1
                activity['profile_' + gate['profile']] += 1
                activity[f'{row["language"]}/{row["label"]}/profile_{gate["profile"]}'] += 1
                activity['enabled_views'] += int(gate['enabled'])
                activity['affected_views'] += int(gate['suppressed_samples']>0)
                for key in ('samples','suppressed_samples','attenuated_samples','gate_closures','gate_openings'):
                    activity[key] += gate.get(key,0)
            recipe = row.get('augmentation', {}).get('recipe')
            if recipe:
                processing[f'{row["language"]}/{row["label"]}/{recipe["family"]}/{recipe["codec"]}/{recipe["noise_type"]}'] += 1
                if recipe.get('stress'):
                    processing['stress/' + recipe['stress']['mechanism']] += 1
    for micro in microbatches(units, cfg['microbatch'], cfg['frame_budget'], weights=weights):
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
        if not joint and cfg.get('consistency', 'dual') != 'none':
            if pcl_state is None:
                raise RuntimeError('PCL warmup requires the persisted scale collector')
            calibration_records = []
            for row, sequence in zip(rows, sequences):
                if 'pcl_calibration_spans' not in row:
                    continue
                values = phoneme_features(sequence.detach(), row['pcl_calibration_spans'], cfg.get('pcl_min_frames', 1),
                                          row.get('pcl_valid_frames'))
                cap = int(cfg.get('pcl_max_phones_per_source', 8))
                if len(values) > cap:
                    indices = torch.linspace(0, len(values)-1, cap, device=values.device).long()
                    values = values[indices]
                if len(values):
                    calibration_records.append((values, row['source_id'], row.get('split')))
                    calibration_phones += len(values)
                    calibration_visits += 1
            pcl_state.observe_batch(calibration_records)
        encoded_views += len(rows)
        reference_views += len(rows) - len(main_indices)
        clean_indices = [offset + slots['clean'] for offset, slots in layouts if 'clean' in slots]
        processed = model.processed_sequences([sequences[i] for i in main_indices + clean_indices])
        all_logits = model.classify_processed(processed)
        # Explicit CE capture. Output-guidance forwards below must never be
        # mistaken for the CE observations that identify difficult samples.
        from .telemetry import capture_plan
        ce_captured.append(all_logits.detach())
        ce_plan.append(capture_plan(micro, quotas, ce_weights, side, len(ce_plan)))
        logits, clean_logits = all_logits[:len(main_indices)], all_logits[len(main_indices):]
        paired_classification_views += len(clean_indices)
        if capture_logits:
            captured.append(logits.detach())
            captured_sources.extend(unit[0]['canonical_source'] for unit in micro)
        parts = micro_losses(logits, micro, quotas, coefficient, cfg['label_smoothing'])
        # micro_losses gives each original stream 50% total CE mass.
        # Rescale each view explicitly: Online 25%, Clean-RTC 25%, Noisy 50%.
        # Each view still averages EN-fake/EN-real/ZH-fake/ZH-real equally.
        clean_parts = []
        clean_position = 0
        for index, unit in enumerate(micro):
            role = unit[0]['role']
            objective, raw = parts[index]
            parts[index] = (objective * (2 * ce_weights[role]), raw)
            if role != 'noisy':
                continue
            clean_part = micro_losses(clean_logits[clean_position:clean_position+1],
                                      [unit], quotas, coefficient, cfg['label_smoothing'])[0]
            clean_parts.append((clean_part[0] * (2 * ce_weights['clean_rtc']), clean_part[1]))
            clean_position += 1
        ce = torch.stack([part[0] for part in parts + clean_parts]).sum()
        regularizer, confidence_counts = confidence_regularization(logits, micro, quotas, runtime)
        confidence_telemetry.append(torch.cat((regularizer.detach().reshape(1), confidence_counts.detach())))
        branches, aux_items, offset = {}, [], 0
        variance_records = defaultdict(list)
        clean_position = 0
        for unit, (objective, raw), (seq_offset, slots) in zip(micro, parts, layouts):
            first = unit[0]
            z = logits[offset:offset + len(unit)]
            offset += len(unit)
            observations = [(first['role'], objective, raw, z)]
            if first['role'] == 'noisy':
                clean_objective, clean_raw = clean_parts[clean_position]
                observations.append(('clean_rtc', clean_objective, clean_raw,
                                     clean_logits[clean_position:clean_position+1]))
                clean_position += 1
            for view, view_objective, view_raw, view_logits in observations:
                telemetry.append(torch.stack((view_objective.detach(), view_raw.detach().sum(),
                    view_raw.detach().max(), (view_logits.argmax(1) == first['label']).sum(),
                    (view_logits.argmax(1) == 0).sum())).float())
                metadata.append((f'{view}/{first["language"]}/{first["label"]}',
                                 len(view_logits), first['label']))
            for kind in slots.get('scheduled', []):
                matches = first['pcl_pairs'][kind]
                if kind in slots:
                    a, b = slots[kind]
                    value, valid, pooled = phoneme_loss(sequences[seq_offset + a], sequences[seq_offset + b], matches,
                        cfg.get('pcl_min_frames', 1), rows[seq_offset+a].get('pcl_valid_frames'),
                        rows[seq_offset+b].get('pcl_valid_frames'), scale=scale, return_features=True)
                else:
                    value, valid = ce.new_zeros(()), 0
                weighted = value * source_weight(kind, first, runtime) * weights[kind]
                if valid:
                    branches[kind] = branches.get(kind, 0) + weighted
                    variance_records[kind].append((first['source_id'], *pooled, source_weight(kind, first, runtime)))
                aux_items.append((kind, value.detach(), weighted.detach()))
                cell = coverage[f'{kind}/{first["language"]}/{first["label"]}']
                cell.update(scheduled_sources=1, valid_sources=int(valid > 0), matched_phones=len(matches), valid_phones=valid)
                declared = first.get('pcl_coverage', {})
                cell['candidate_phones'] += int(declared.get(kind + '_candidates', len(matches)))
                cell['all_pairs_masked_sources'] += int(not len(matches))
                cell['no_valid_frames_sources'] += int(bool(len(matches)) and valid == 0)
                for key in ('activity_affected_phones','activity_safe_segments','original_frame_rejected'):
                    cell[key] += int(declared.get(kind+'_'+key,0))
                for stage in ('clean', kind):
                    timing = first.get('pcl_timing', {}).get(stage)
                    if timing is not None and not timing.get('valid', False):
                        cell['invalid_' + stage + '_' + timing.get('reason', 'unknown') + '_sources'] += 1
        if variance_weight > 0 and any(weights.values()):
            variance_terms = []
            reference_weight = cfg['pcl_variance_reference_weight']
            for kind in ('noise', 'platform'):
                cell = variance_coverage[kind]
                cell['microbatches'] += 1
                if not variance_records[kind]:
                    cell['skipped_no_valid_pairs'] += 1
                    continue
                value, status = phoneme_variance(variance_records[kind], scale, runtime)
                cell['sources'] += status['sources']
                cell['phones'] += status['phones']
                cell['applied' if status['applied'] else 'skipped_' + status['reason']] += 1
                # Relative branch strength is explicit; reference counts alone
                # do not make the platform constraint weak.
                relative = float(cfg.get('pcl_' + kind + '_weight', .15 if kind == 'noise' else .045)) / reference_weight
                variance_terms.append(value * variance_weight * variance_ramp * relative)
            if variance_terms:
                branches['variance'] = torch.stack(variance_terms).sum()
                variance_telemetry.append(branches['variance'].detach())
        lf, ld, pair_records = paired_losses(model, micro, layouts, sequences, processed,
            main_indices, clean_indices, quotas, runtime)
        repair_telemetry.append(torch.stack((lf.detach(), ld.detach())))
        repair_records.extend(pair_records)
        loss = ce + sum(branches.values()) + regularizer + lf + ld
        require_finite(loss, 'classification and PCL loss')
        if backward:scaler.scale(loss * backward_factor).backward()
        aux_telemetry.extend(aux_items)
    norm=None; amp={}; module_norms={}; updates={}; relative={}; parameter_norms={}; sample_counts={}
    observed = torch.stack(telemetry).cpu().tolist()
    cells = defaultdict(lambda: defaultdict(float))
    predictions = {'fake': [0, 0], 'real': [0, 0]}
    for (cell, views, label), (loss, raw, maximum, correct, fake) in zip(metadata, observed):
        for key, value in dict(objective=loss, raw_ce_sum=raw, views=views, correct=correct, predicted_fake=fake).items():
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
    variance_loss = float(torch.stack(variance_telemetry).sum()) if variance_telemetry else 0.
    regularizer_loss, active, excess_sum, real_views = torch.stack(confidence_telemetry).sum(0).cpu().tolist()
    result = dict(loss=class_loss + sum(item['weighted_loss'] for item in aux.values()) + regularizer_loss + variance_loss, classification_loss=class_loss,
        raw_ce_sum=sum(item[1] for item in observed), max_example_ce=max(item[2] for item in observed),
        views=sum(item[1] for item in metadata), cells={k: dict(v) for k, v in cells.items()},
        predicted_fake_by_true_class=predictions, grad_norm=norm, amp=amp, risk_coefficient=coefficient,
        lrs={}, processing_counts=dict(processing),
        base_source_view_counts=dict(base_views), rtc_activity=dict(activity),
        unique_original_sources=len({unit[0]['group_id'] for unit in units}),
        encoded_views=encoded_views, auxiliary_reference_views=reference_views-paired_classification_views,
        paired_classification_views=paired_classification_views,
        classification_source_views=len(units),
        pcl_weights=weights, pcl_losses=aux, pcl_coverage={k: dict(v) for k, v in coverage.items()},
        pcl_variance=dict(weighted_loss=variance_loss, coverage={k:dict(v) for k,v in variance_coverage.items()},
                          configured_weight=variance_weight, gamma=cfg.get('pcl_variance_target', .5)),
        pcl_scale_calibration=dict(source_visits=calibration_visits, phones=calibration_phones),
        phase='joint' if joint else 'head_warmup', actual_group_quotas=list(quotas),
        en_real_confidence=dict(weighted_loss=regularizer_loss, active_views=int(active), views=int(real_views),
            active_fraction=active/max(1,real_views), mean_excess=excess_sum/max(1,real_views),
            configured_weight=cfg.get('en_real_confidence_weight', 0.),
            margin=cfg.get('en_real_confidence_margin',3.5), enabled=joint and cfg.get('en_real_confidence_weight',0.)>0),
        pcl_gradient_audit=gradient_audit, module_grad_norms_before_clip=module_norms,
        module_update_norms=updates, module_relative_updates=relative, module_parameter_norms=parameter_norms,
        module_update_audit_scope='<=1024 entries per optimizer group, spread across gradient-bearing tensors; not whole-module norms',
        module_update_sample_counts=dict(sample_counts))

    lf, ld = torch.stack(repair_telemetry).sum(0).cpu().tolist()
    result['loss'] += lf+ld
    result['repair_losses'] = dict(feature=lf, kd=ld, records=repair_records)
    result['_ce_captured'], result['_ce_plan'] = ce_captured, ce_plan

    if capture_logits:
        result['_logits']=torch.cat(captured).cpu()
        result['_logit_sources']=captured_sources
    return result


def side_pass(model, units, cfg, scaler, *, backward, capture_logits=False, backward_factor=1.):
    try:
        with torch.set_grad_enabled(backward):
            return _side_pass(model,units,cfg,scaler,backward=backward,capture_logits=capture_logits,
                              backward_factor=backward_factor)
    except FloatingPointError as exc:
        from w2v_v3191.meta import NonfiniteGradient
        raise NonfiniteGradient(str(exc), encoded_views=len(units)) from exc
