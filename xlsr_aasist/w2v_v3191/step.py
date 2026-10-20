"""Matched-condition classification with bounded, mechanism-balanced mining.

All arms score the same candidates with the unadapted model. Candidate scoring
cannot alter training modes, running buffers, gradients, or any RNG stream.
No teacher, virtual adaptation, KL objective, or independent detector is used.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
from functools import lru_cache
import hashlib
import math

import torch

from .meta import (Transaction, NonfiniteGradient, rng_state, restore_rng,
                   validate_units, module_norms, _update_norms, _summary)
from .precision import scaler_for
from .side import side_pass


def _key(*values):
    return hashlib.sha256(repr(values).encode('utf8')).hexdigest()


def _mechanism(candidate):
    row = candidate[0]
    return str(row.get('robust_mechanism',
                       row.get('robust_candidate', {}).get('mechanism', 'legacy')))


def _eligible(candidate):
    row = candidate[0]
    quality = row.get('robust_quality', row.get('robust_candidate', {}).get('quality', {}))
    return bool(quality.get('passed', quality.get('eligible', False)))


@contextmanager
def scoring_state(model):
    """Restore all per-module flags, buffers and random states, also on error."""
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    rng = rng_state()
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        with torch.no_grad():
            for buffer, old in buffers:
                buffer.copy_(old)
        for module, training in modes:
            module.training = training
        restore_rng(rng)


def _score_batches(rows, size, frames):
    if not isinstance(size, int) or size < 1 or not isinstance(frames, int) or frames < 1:
        raise ValueError('Positive candidate scoring budgets required')
    batch, longest = [], 0
    # Stable original order keeps candidate indexing independent of padding.
    for row in rows:
        length = max(1, (len(row['wave']) - 400) // 320 + 1)
        if length > frames:
            raise ValueError('One full candidate exceeds the scoring frame budget; audio cannot be truncated')
        if batch and (len(batch) >= size or (len(batch)+1)*max(longest, length) > frames):
            yield batch
            batch, longest = [], 0
        batch.append(row)
        longest = max(longest, length)
    if batch:
        yield batch


def _candidate_sets(units):
    records, rows = {}, []
    identity_keys = ('source_id', 'group_id', 'canonical_source', 'language', 'label', 'meta_side', 'split')
    for index, unit in enumerate(units):
        if unit[0]['role'] != 'noisy':
            continue
        original = unit[0]
        candidates = original.get('robust_candidates', [unit])
        if not candidates:
            raise ValueError('A Noisy source must retain its legacy candidate')
        for candidate in candidates:
            if len(candidate) != 1 or candidate[0].get('role') != 'noisy':
                raise ValueError('Candidates must be complete single-source Noisy units')
            row = candidate[0]
            if any(row.get(key) != original.get(key) for key in identity_keys):
                raise ValueError('Candidate changes source, class, split, or task identity')
            clean = row.get('pcl_refs', {}).get('clean')
            if clean is None:
                raise ValueError('Every candidate must have its matched Clean-RTC reference')
            if any(clean.get(key) != row.get(key) for key in identity_keys
                   if key not in ('meta_side',)):
                raise ValueError('Matched Clean-RTC must preserve Train source and label')
            rows.extend((row, clean))
        records[index] = candidates
    return records, rows


def _balanced_hard_assignment(indices, selected, candidates, margins):
    """Min-cost small assignment with exactly the uniform mechanism multiset.

    Hard mining may swap mechanisms between sources, not change their total
    exposure. This blocks a shortcut where one processing family becomes
    associated with Real and another with Fake. Quality failures fall back to
    the already selected feasible uniform assignment.
    """
    mechanisms = sorted({_mechanism(candidates[i][selected[i]]) for i in indices})
    counts = tuple(sum(_mechanism(candidates[i][selected[i]]) == m for i in indices)
                   for m in mechanisms)
    choices = {}
    for i in indices:
        for m in mechanisms:
            available = [j for j, candidate in enumerate(candidates[i])
                         if _mechanism(candidate) == m and (j == 0 or _eligible(candidate))]
            if available:
                choices[i, m] = min(available, key=lambda j: (margins[i][j][0], j))

    @lru_cache(None)
    def solve(position, remaining):
        if position == len(indices):
            return 0., ()
        i = indices[position]
        best = None
        for k, m in enumerate(mechanisms):
            if not remaining[k] or (i, m) not in choices:
                continue
            next_counts = list(remaining)
            next_counts[k] -= 1
            rest = solve(position+1, tuple(next_counts))
            if rest is None:
                continue
            j = choices[i, m]
            item = (margins[i][j][0] + rest[0], (j,) + rest[1])
            if best is None or item < best:
                best = item
        return best

    result = solve(0, counts)
    return dict(zip(indices, result[1])) if result is not None else {}


def select_candidates(model, units, cfg):
    """Return one unit per source plus auditable, JSON-safe selection telemetry."""
    arm = cfg.get('robust_arm', 'baseline')
    if arm not in ('baseline', 'expanded', 'hard'):
        raise ValueError('Unknown robust arm: ' + str(arm))
    fraction = float(cfg.get('robust_hard_fraction', .5))
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Hard-source fraction must be finite and within [0, 1]')
    candidates, rows = _candidate_sets(units)
    scores, batches = [], 0
    with scoring_state(model):
        for batch in _score_batches(rows, int(cfg.get('robust_score_microbatch', cfg['microbatch'])),
                                   int(cfg.get('robust_score_frame_budget', cfg['frame_budget']))):
            sequences = model.encode([row['wave'] for row in batch])
            logits = model.classify_sequences(sequences).detach().float()
            if logits.shape != (len(batch), 2) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Candidate scorer produced invalid binary logits')
            labels = torch.tensor([row['label'] for row in batch], device=logits.device)
            true_margin = logits.gather(1, labels[:, None]) - logits.gather(1, (1-labels)[:, None])
            scores.extend(true_margin[:, 0].cpu().tolist())
            batches += 1
    margins, offset = {}, 0
    for i, values in candidates.items():
        margins[i] = [(scores[offset+2*j], scores[offset+2*j+1]) for j in range(len(values))]
        offset += 2*len(values)
    seed, update = int(cfg.get('sampling_seed', cfg.get('seed', 0))), int(cfg.get('current_update', 0))
    mechanisms = sorted({_mechanism(candidate) for values in candidates.values()
                         for candidate in values[1:]})
    selected, uniform, desired, hard_sources = {}, {}, {}, set()
    cells = defaultdict(list)
    for i in candidates:
        cells[units[i][0]['language'], units[i][0]['label']].append(i)
    for (language, label), members in sorted(cells.items()):
        # Source ordering differs between labels but the desired processing
        # schedule does not. Equal-size cells therefore get equal budgets.
        members.sort(key=lambda i: _key(seed, update, units[i][0]['canonical_source']))
        shift = int(_key(seed, update, language), 16) % max(1, len(mechanisms))
        for position, i in enumerate(members):
            target = mechanisms[(position+shift) % len(mechanisms)] if mechanisms else 'legacy'
            desired[i] = target
            available = [j for j, value in enumerate(candidates[i]) if j and _eligible(value)]
            matching = [j for j in available if _mechanism(candidates[i][j]) == target]
            choices = matching or available
            uniform[i] = min(choices, key=lambda j: _key(seed, update,
                             units[i][0]['canonical_source'], j)) if choices else 0
            selected[i] = 0 if arm == 'baseline' else uniform[i]
        if arm == 'hard':
            count = min(len(members), int(math.floor(len(members)*fraction)))
            indices = members[:count]
            hard_sources.update(indices)
            selected.update(_balanced_hard_assignment(indices, uniform, candidates, margins))
    chosen = list(units)
    counts, rejected, by_class, by_side, transitions = Counter(), Counter(), Counter(), Counter(), Counter()
    shifted, clean_shift, legacy_margins, selected_margins = [], [], [], []
    details = []
    for i, values in candidates.items():
        j = selected[i]
        # Remove the candidate tree from the selected training unit. This does
        # not mutate worker outputs and releases unselected waves after step.
        row = dict(values[j][0])
        row.pop('robust_candidates', None)
        chosen[i] = [row]
        label, language = row['label'], row['language']
        mechanism = _mechanism(values[j])
        base, actual, clean = margins[i][0][0], margins[i][j][0], margins[i][j][1]
        counts['noisy_sources'] += 1
        counts['candidate_views'] += len(values)
        counts['quality_passed_expanded'] += sum(_eligible(v) for v in values[1:])
        counts['quality_rejected_expanded'] += sum(not _eligible(v) for v in values[1:])
        counts['all_expanded_rejected_fallback'] += int(j == 0 and not any(_eligible(v) for v in values[1:]))
        counts['desired_mechanism_unavailable'] += int(arm != 'baseline' and _mechanism(values[uniform[i]]) != desired[i])
        counts['hard_sources'] += int(i in hard_sources)
        counts['hard_changed_from_uniform'] += int(i in hard_sources and j != uniform[i])
        by_class[f'{language}/{label}/{mechanism}'] += 1
        by_side[f'{row["meta_side"]}/{language}/{label}/{mechanism}'] += 1
        correct = lambda margin: margin >= 0 if label == 0 else margin > 0
        transitions[f'{language}/{label}/legacy_correct_selected_wrong'] += int(correct(base) and not correct(actual))
        transitions[f'{language}/{label}/clean_correct_noisy_wrong'] += int(correct(clean) and not correct(actual))
        shifted.append(actual-base)
        clean_shift.append(actual-clean)
        legacy_margins.append(base)
        selected_margins.append(actual)
        for candidate in values[1:]:
            quality = candidate[0].get('robust_quality', {})
            if not _eligible(candidate):
                for reason in quality.get('reasons', [quality.get('reason', 'quality_rejected')]):
                    rejected[_mechanism(candidate) + '/' + str(reason)] += 1
        if cfg.get('robust_record_details', False):
            details.append(dict(source=row['canonical_source'], selected=j, uniform=uniform[i],
                                hard=i in hard_sources, mechanism=mechanism,
                                legacy_margin=base, selected_margin=actual, matched_clean_margin=clean))
    imbalance = {}
    for language in sorted({unit[0]['language'] for unit in units}):
        for m in sorted({_mechanism(v) for values in candidates.values() for v in values}):
            difference = by_class[f'{language}/0/{m}'] - by_class[f'{language}/1/{m}']
            if difference:
                imbalance[f'{language}/{m}'] = difference
    average = lambda x: sum(x)/len(x) if x else None
    return chosen, dict(arm=arm, counts=dict(counts), quality_rejections=dict(rejected),
                       selected_mechanism_by_class=dict(by_class), class_mechanism_imbalance=imbalance,
                       selected_mechanism_by_side_class=dict(by_side),
                       transitions=dict(transitions), mean_selected_minus_legacy_margin=average(shifted),
                       mean_noisy_minus_clean_margin=average(clean_shift),
                       mean_legacy_margin=average(legacy_margins), mean_selected_margin=average(selected_margins),
                       scoring_encoded_views=len(rows), scoring_batches=batches,
                       hard_fraction=fraction, hard_budget_scope='language/class across both task sides',
                       policy='equal desired Real/Fake mechanism budgets; hard assignment preserves uniform multiset',
                       details=details)


def train_step(model, optimizer, units, cfg, alpha=0., beta=1., joint=True, audit=False):
    """One source-balanced paired-classification AdamW update, no adaptation."""
    if float(alpha) != 0. or not joint:
        raise ValueError('V3.19.1 uses joint classification without virtual adaptation or head warmup')
    beta = float(beta)
    if not math.isfinite(beta) or beta != 1.:
        raise ValueError('All arms require equal A/B classification weights (beta=1)')
    validate_units(units)
    if hasattr(model, 'set_phase') and getattr(model, 'joint', None) is not True:
        model.set_phase(True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    runtime = dict(cfg, current_update=int(getattr(optimizer, '_v3189_joint_update',
                                                 cfg.get('current_update', 1))))
    scaler = scaler_for(optimizer, runtime)
    scaler_initial = copy.deepcopy(scaler.state_dict())
    scale = float(scaler.get_scale())
    transaction = Transaction(model)
    scaler.scale(torch.zeros((), device=transaction.parameters[0][1].device))
    encoded = 0
    try:
        selected, selection = select_candidates(model, units, runtime)
        encoded += selection['scoring_encoded_views']
        sides = validate_units(selected)
        a = side_pass(model, sides['A'], runtime, scaler, backward=True, backward_factor=1/(1+beta))
        encoded += a['encoded_views']
        b = side_pass(model, sides['B'], runtime, scaler, backward=True, backward_factor=beta/(1+beta))
        encoded += b['encoded_views']
        scaler.unscale_(optimizer)
        norms = module_norms(optimizer, transaction, [p.grad for _, p in transaction.parameters]) if audit else {}
        parameters = [p for _, p in transaction.parameters if p.grad is not None]
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(cfg.get('max_grad_norm', 1.)),
                                                  error_if_nonfinite=False))
        if not math.isfinite(norm):
            raise NonfiniteGradient('Nonfinite paired-classification gradient')
    except (NonfiniteGradient, FloatingPointError) as exc:
        transaction.restore()
        optimizer.zero_grad(set_to_none=True)
        if not scaler.is_enabled():
            scaler.load_state_dict(scaler_initial)
            raise
        scaler.update(new_scale=scale*float(scaler.get_backoff_factor()))
        return dict(loss=None, classification_loss=None, encoded_views=encoded, views=len(units),
                    classification_source_views=len(units), unique_original_sources=len(units),
                    auxiliary_reference_views=0, cells={}, raw_ce_sum=None, max_example_ce=None,
                    predicted_fake_by_true_class={'fake': [0, 0], 'real': [0, 0]}, grad_norm=None,
                    phase='joint', lrs={g.get('name', str(i)): g['lr'] for i, g in enumerate(optimizer.param_groups)},
                    amp=dict(enabled=True, previous_scale=scale, scale=scaler.get_scale(),
                             skipped_update=True, reason=str(exc)),
                    meta=dict(alpha=0., beta=beta, objective='paired_CE_no_virtual_update'))
    except BaseException:
        transaction.restore()
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.update(new_scale=scale)
        scaler.load_state_dict(scaler_initial)
        raise
    # Single commit point. Earlier failures cannot modify Adam moments.
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    result = _summary(selected, a, b, beta)
    result.update(views=a['views']+b['views'], classification_source_views=len(units),
                  paired_classification_views=a['paired_classification_views']+b['paired_classification_views'],
                  auxiliary_reference_views=a['auxiliary_reference_views']+b['auxiliary_reference_views'],
                  encoded_views=encoded, training_encoded_views=a['encoded_views']+b['encoded_views'],
                  grad_norm=norm, selection=selection,
                  amp=dict(enabled=scaler.is_enabled(), previous_scale=scale, scale=scaler.get_scale(),
                           skipped_update=scaler.get_scale() < scale),
                  lrs={g.get('name', str(i)): g['lr'] for i, g in enumerate(optimizer.param_groups)},
                  meta=dict(alpha=0., beta=beta, A_loss=a['loss'], B_loss=b['loss'],
                            objective='(A_paired_CE + A_parent_PCL + beta*B_paired_CE)/(1+beta)',
                            virtual_update_norm=0., module_outer_gradient_norms=norms,
                            module_outer_update_norms=_update_norms(optimizer, transaction) if audit else {}))
    return result
