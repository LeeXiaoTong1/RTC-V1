"""Label-supervised, source-paired objectives for the V3.13 detector.

There is no global real centre and no requirement that English and Chinese
features coincide.  Each positive is another observed condition of the *same*
source.  Negative mining uses only labelled Train recordings of the same
language and condition.  Features are the actual final classifier input.

``classification`` preserves the supplied logical-batch CE weight mass.
``pair`` and ``ranking`` are local means: the caller must multiply them by the
microgroup's fraction of the logical source batch before gradient accumulation.
Do not split a source pair across separate calls.
"""
from collections import defaultdict
import math

import torch
from torch.nn import functional as F


DEFAULTS = {
    'hard_weight_strength': 1.0,
    'hard_weight_max': 2.0,
    'pair_min_confidence': .75,
    'pair_margin_cap': 4.0,
    'pair_margin_slack': .25,
    'pair_margin_weight': 1.0,
    'pair_feature_weight': .1,
    'ranking_feature_margin': .15,
    'ranking_logit_margin': .2,
    'ranking_temperature': .2,
    'ranking_score_temperature': 2.0,
    'ranking_feature_weight': 1.0,
    'ranking_logit_weight': .25,
}


def _options(cfg):
    values = {key: float(cfg.get(key, default)) for key, default in DEFAULTS.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('Objective coefficients must be finite')
    if not .5 < values['pair_min_confidence'] < 1.:
        raise ValueError('Pair teachers must be confidently correct')
    if values['pair_margin_cap'] <= 0 or values['pair_margin_slack'] < 0:
        raise ValueError('Invalid capped pair margin')
    if values['hard_weight_max'] < 1 or values['hard_weight_strength'] < 0:
        raise ValueError('Invalid bounded hard-example weights')
    if values['ranking_temperature'] <= 0 or values['ranking_score_temperature'] <= 0:
        raise ValueError('Ranking temperatures must be positive')
    nonnegative = ('pair_margin_weight', 'pair_feature_weight', 'ranking_feature_margin',
                   'ranking_logit_margin', 'ranking_feature_weight', 'ranking_logit_weight')
    if any(values[key] < 0 for key in nonnegative):
        raise ValueError('Loss weights and ranking margins cannot be negative')
    if values['ranking_feature_margin'] > 2 or values['ranking_logit_margin'] > 2:
        raise ValueError('Margins must fit the bounded cosine/evidence scale')
    return values


def _validate(logits, features, rows):
    if logits.ndim != 2 or logits.shape != (len(rows), 2) or not len(rows):
        raise ValueError('Expected a nonempty [N,2] labelled logit batch')
    if features.ndim != 2 or len(features) != len(rows) or features.shape[1] < 1:
        raise ValueError('Task features must align with the logit batch')
    if features.device != logits.device:
        raise ValueError('Task features and logits must be on the same device')
    occurrences = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get('split', 'train') != 'train':
            raise ValueError('Training objectives may only use labelled Train recordings')
        if row.get('label') not in (0, 1) or row.get('language') not in ('en', 'zh'):
            raise ValueError('Expected known binary labels and EN/ZH language metadata')
        if not row.get('source_id') or not row.get('pair_occurrence') or not row.get('condition'):
            raise ValueError('Every row requires source_id, pair_occurrence and condition')
        occurrences[str(row['pair_occurrence'])].append(index)
    for indices in occurrences.values():
        if len(indices) != 2:
            raise ValueError('Each source occurrence requires exactly two complete views')
        identity = {(rows[i]['source_id'], rows[i]['language'], rows[i]['label']) for i in indices}
        if len(identity) != 1:
            raise ValueError('A pair occurrence cannot cross sources, languages or labels')
        if rows[indices[0]]['condition'] == rows[indices[1]]['condition']:
            raise ValueError('Paired views must represent different processing conditions')
    return list(occurrences.values())


def _ce(logits, labels, rows, occurrences, options, ce_weights):
    per_row = F.cross_entropy(logits, labels, reduction='none')
    if ce_weights is not None:
        base = torch.as_tensor(ce_weights, device=logits.device, dtype=logits.dtype).detach()
    elif any('ce_weight' in row for row in rows):
        if not all('ce_weight' in row for row in rows):
            raise ValueError('CE logical-batch weights must be supplied for every row')
        base = logits.new_tensor([float(row['ce_weight']) for row in rows])
    else:
        # Every draw of a source has the same total supervised mass.  A balanced
        # sampler supplies one EN-real, EN-fake, ZH-real and ZH-fake source.
        base = torch.zeros(len(rows), device=logits.device, dtype=logits.dtype)
        for indices in occurrences:
            base[indices] = 1. / (len(occurrences) * len(indices))
    if base.shape != (len(rows),) or not bool(torch.isfinite(base).all()) or bool((base < 0).any()):
        raise ValueError('CE weights must be finite, nonnegative and row-aligned')
    factors = torch.ones_like(base)
    for indices in occurrences:
        # One bounded detached difficulty factor per source, shared by both
        # conditions.  A difficult processed view cannot silently replace its
        # original recording's supervised budget or backpropagate via weights.
        difficulty = per_row[indices].detach().mean()
        factor = 1. + options['hard_weight_strength'] * difficulty / (1. + difficulty)
        factors[indices] = factor.clamp(max=options['hard_weight_max'])
    mass = base.sum()
    weighted = base * factors
    weights = weighted * mass / weighted.sum().clamp_min(torch.finfo(logits.dtype).eps)
    return (per_row * weights).sum(), weights, factors


def loss_terms(logits, features, rows, cfg, *, ce_weights=None):
    """Return CE, paired preservation and matched hard-negative discrimination.

    Labels are 0=fake, 1=real. ``pair_occurrence`` identifies one sampled source
    draw, whereas ``source_id`` identifies its original recording across draws.
    ``ce_weight`` / ``ce_weights`` are *base* weights, before bounded hard-example
    emphasis.  Their sum (including a fractional microbatch sum) is preserved.

    Pair evidence is one-sided and capped: only a correct confident view may
    teach its partner, and neither a wrong teacher nor a demand for ever-larger
    logits is allowed.  Low-weight cosine preservation is applied only when
    both views are correct. No temporal or waveform alignment is assumed.

    The returned prototype loss is an explicit differentiable zero: this
    version preserves multiple genuine modes with instance positives instead
    of introducing an unsupported learned single/global centre.
    """
    options = _options(cfg)
    occurrences = _validate(logits, features, rows)
    z = logits.float()
    h = F.normalize(features.float(), p=2, dim=-1, eps=1e-6)
    zero = (z.sum() + h.sum()) * 0.
    labels = torch.tensor([row['label'] for row in rows], device=z.device, dtype=torch.long)
    score = z[:, 1] - z[:, 0]
    signed = score * (labels * 2 - 1)
    classification, weights, hard_factors = _ce(z, labels, rows, occurrences, options, ce_weights)

    pair_margins, pair_features = [], []
    eligible, active, feature_active = 0, 0, 0
    partner = {}
    minimum_margin = math.log(options['pair_min_confidence'] / (1. - options['pair_min_confidence']))
    for indices in occurrences:
        left, right = indices
        partner[left], partner[right] = right, left
        # Selection and teacher values are stop-gradient.  Labels determine
        # which side is allowed to teach, not the model's predicted class.
        teacher, student = (left, right) if bool(signed[left].detach() >= signed[right].detach()) else (right, left)
        teacher_margin = signed[teacher].detach()
        if bool(teacher_margin < minimum_margin):
            pair_margins.append(zero)
            pair_features.append(zero)
            continue
        eligible += 1
        target = (teacher_margin - options['pair_margin_slack']).clamp(min=0., max=options['pair_margin_cap'])
        deficit = (target - signed[student]).clamp_min(0.)
        # Huber's linear tail limits the gradient of severely corrupted views;
        # no hard clipping suppresses their corrective supervised direction.
        pair_margins.append(F.smooth_l1_loss(deficit, torch.zeros_like(deficit), reduction='sum'))
        active += int(bool(deficit.detach() > 0.))
        if bool(signed[student].detach() > 0.):
            pair_features.append((1. - (h[student] * h[teacher].detach()).sum()).clamp_min(0.))
            feature_active += 1
        else:
            pair_features.append(zero)
    pair_margin = torch.stack(pair_margins).mean()
    pair_feature = torch.stack(pair_features).mean()
    pair = options['pair_margin_weight'] * pair_margin + options['pair_feature_weight'] * pair_feature

    # Smoothly bounded evidence avoids rewarding arbitrary logit rescaling.
    # Softsign has polynomial tails: unlike a clamp, there is no dead zone.
    evidence = score / (options['ranking_score_temperature'] + score.abs())
    ranking_features, ranking_logits = [], []
    ranking_anchors, ranking_edges, ranking_feature_active = 0, 0, 0
    ranking_candidate_views = 0
    def original_identity(row):
        return row.get('group_id') or row['source_id']
    for anchor, row in enumerate(rows):
        if row['label'] != 1:
            continue
        candidate_groups = defaultdict(list)
        for index, other in enumerate(rows):
            if (other['label'] == 0 and other['language'] == row['language']
                    and other['condition'] == row['condition']
                    and other['source_id'] != row['source_id']
                    and original_identity(other) != original_identity(row)):
                candidate_groups[original_identity(other)].append(index)
        if not candidate_groups:
            # Include an explicit zero in the mean over all real views so that
            # effective strength does not grow when matching coverage drops.
            ranking_features.append(zero)
            ranking_logits.append(zero)
            continue
        ranking_anchors += 1
        ranking_edges += len(candidate_groups)
        ranking_candidate_views += sum(len(indices) for indices in candidate_groups.values())
        positive = partner[anchor]
        # Aliases/repeated draws are one original-source candidate. Keep the
        # hardest observed feature/score independently within each source (their
        # dropout realizations may differ), then mine across unique sources.
        feature_candidates, score_candidates = [], []
        for indices in candidate_groups.values():
            similarities = h[indices] @ h[anchor]
            feature_candidates.append(indices[int(similarities.detach().argmax())])
            score_candidates.append(indices[int(evidence[indices].detach().argmax())])
        similarities = h[feature_candidates] @ h[anchor]
        feature_negative = feature_candidates[int(similarities.detach().argmax())]
        score_negative = score_candidates[int(evidence[score_candidates].detach().argmax())]
        gap = (h[anchor] * h[feature_negative]).sum() - (h[anchor] * h[positive]).sum()
        feature_loss = (options['ranking_feature_margin'] + gap).clamp_min(0.)
        ranking_feature_active += int(bool(feature_loss.detach() > 0.))
        ranking_features.append(feature_loss)
        logit_gap = options['ranking_logit_margin'] - evidence[anchor] + evidence[score_negative]
        # Temperature-scaled softplus is smooth, has bounded derivative, and
        # sees a bounded input because evidence lies in [-1, 1].
        ranking_logits.append(options['ranking_temperature'] * F.softplus(logit_gap / options['ranking_temperature']))
    ranking_feature = torch.stack(ranking_features).mean() if ranking_features else zero
    ranking_logit = torch.stack(ranking_logits).mean() if ranking_logits else zero
    ranking = options['ranking_feature_weight'] * ranking_feature + options['ranking_logit_weight'] * ranking_logit

    return {
        'classification': classification,
        'pair': pair,
        'ranking': ranking,
        'prototype': zero,
        'pair_margin': pair_margin,
        'pair_feature': pair_feature,
        'ranking_feature': ranking_feature,
        'ranking_logit': ranking_logit,
        'source_count': z.new_tensor(len(occurrences)),
        'pair_eligible_sources': z.new_tensor(eligible),
        'pair_active_sources': z.new_tensor(active),
        'pair_feature_sources': z.new_tensor(feature_active),
        'ranking_anchors': z.new_tensor(ranking_anchors),
        'ranking_candidate_edges': z.new_tensor(ranking_edges),
        'ranking_candidate_views': z.new_tensor(ranking_candidate_views),
        'ranking_feature_active': z.new_tensor(ranking_feature_active),
        'ce_weight_mass': weights.sum().detach(),
        'hard_weight_mean': hard_factors.mean().detach(),
        'hard_weight_max_observed': hard_factors.max().detach(),
    }
