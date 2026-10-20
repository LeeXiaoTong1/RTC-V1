"""Fixed competition objective and source-complete V5 screening helpers.

Only the caller-provided selection rows enter these functions. No test labels,
Audit scores, language-dependent decision thresholds, or non-regression gates
are involved in candidate choice.
"""
from collections import defaultdict
import hashlib
import json
import math
from numbers import Integral

import numpy as np

GROUPS = (('en', 0), ('en', 1), ('zh', 0), ('zh', 1))


def _clean(row):
    return row['condition'] in ('online', 'official_online')


def _validate_rows(rows):
    if not rows:
        raise ValueError('V5 selection needs nonempty labeled rows')
    for row in rows:
        if row.get('language') not in ('en', 'zh') or row.get('label') not in (0, 1):
            raise ValueError('V5 selection requires EN/ZH and fake=0, real=1 labels')
        if not isinstance(row.get('condition'), str) or not row['condition']:
            raise ValueError('V5 selection requires an explicit evaluation condition')


def _metrics(labels, prediction):
    if not len(labels):
        return dict(count=0, class_counts=[0, 0], confusion=[[0, 0], [0, 0]],
            recall=[None, None], macro_f1=None)
    confusion = np.bincount(labels * 2 + prediction, minlength=4).reshape(2, 2)
    support = confusion.sum(axis=1)
    denominator = confusion.sum(axis=0) + support
    f1 = np.divide(2 * confusion.diagonal(), denominator,
        out=np.zeros(2, dtype=np.float64), where=denominator > 0)
    recall = [float(confusion[k, k] / support[k]) if support[k] else None for k in range(2)]
    return dict(count=int(len(labels)), class_counts=support.tolist(),
        confusion=confusion.tolist(), recall=recall, macro_f1=float(f1.mean()))


def score(rows, logits):
    """P(fake)>=0.5; 30% pooled Online + 70% pooled all Noisy macro-F1."""
    rows = list(rows)
    _validate_rows(rows)
    z = np.asarray(logits, dtype=np.float64)
    if z.shape != (len(rows), 2) or not np.isfinite(z).all():
        raise ValueError('V5 selection logits must be finite N x 2 with fake in column 0')
    labels = np.asarray([r['label'] for r in rows], np.int64)
    # Comparing the two logits is exactly the fixed 0.5 softmax threshold,
    # without overflow or sigmoid/softmax rounding around that threshold.
    prediction = np.where(z[:, 0] >= z[:, 1], 0, 1)
    panels = {}
    masks = {}
    for panel, want_clean in (('clean', True), ('noisy', False)):
        mask = np.asarray([_clean(row) == want_clean for row in rows], bool)
        if set(labels[mask].tolist()) != {0, 1}:
            raise ValueError('V5 selection needs both Real and Fake in each Clean/Noisy panel: ' + panel)
        masks[panel] = mask
        panels[panel] = _metrics(labels[mask], prediction[mask])
    groups = {}
    for condition, language in sorted({(r['condition'], r['language']) for r in rows}):
        mask = np.asarray([r['condition'] == condition and r['language'] == language for r in rows])
        groups[condition + '/' + language] = _metrics(labels[mask], prediction[mask])
    four = {}
    for language, label in GROUPS:
        key = language + '-' + ('fake' if label == 0 else 'real')
        value = {}
        group_mask = np.asarray([r['language'] == language and r['label'] == label for r in rows])
        for panel in ('clean', 'noisy'):
            mask = masks[panel] & group_mask
            count = int(mask.sum())
            correct = int(np.sum(prediction[mask] == label))
            value[panel] = dict(count=count, correct=correct, errors=count - correct,
                recall=correct / count if count else None)
        four[key] = value
    return dict(clean_f1=panels['clean']['macro_f1'], noisy_f1=panels['noisy']['macro_f1'],
        weighted_f1=.3 * panels['clean']['macro_f1'] + .7 * panels['noisy']['macro_f1'],
        threshold=.5, score_column='fake=0', panels=panels, groups=groups,
        four_groups=four, count=len(rows),
        objective='0.3 pooled Clean macro-F1 + 0.7 pooled Noisy macro-F1',
        selection_rule='weighted_f1 only; group recalls are diagnostics, not vetoes')


def _rank(seed, group, source):
    return hashlib.sha256(json.dumps([int(seed), group, source],
        sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def balanced_source_subset(dev, allowed_indices, per_group, seed, *, source_keys):
    """Select equal source budgets and keep every panel of each chosen source.

    `source_keys` must come from the final official-pair/identity partition, not
    independent filename guesses. Incomplete allowed sources are rejected.
    Missing Online partners are permitted, with aggregate panel coverage
    protected by a deterministic minimal coverage anchor in each group.
    """
    dev = list(dev)
    _validate_rows(dev)
    if len(source_keys) != len(dev):
        raise ValueError('V5 source keys must cover the exact Dev row manifest')
    if not isinstance(per_group, Integral) or isinstance(per_group, bool) or per_group <= 0:
        raise ValueError('V5 source budget must be a positive integer per group')
    raw = list(allowed_indices)
    if not raw or any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= len(dev) for i in raw):
        raise ValueError('V5 allowed selection indices are invalid or empty')
    if len(set(raw)) != len(raw):
        raise ValueError('V5 allowed selection indices contain duplicates')
    allowed = set(map(int, raw))
    all_members = defaultdict(list)
    source_group = {}
    for i, (row, source) in enumerate(zip(dev, source_keys)):
        if not isinstance(source, str) or not source:
            raise ValueError('V5 canonical source keys must be nonempty strings')
        group = (row['language'], row['label'])
        if source in source_group and source_group[source] != group:
            raise ValueError('A canonical Dev source has conflicting language or class labels')
        source_group[source] = group
        all_members[source].append(i)
    sources = {source_keys[i] for i in allowed}
    for source in sources:
        if not set(all_members[source]).issubset(allowed):
            raise ValueError('Allowed selection rows split a canonical source across panels or partitions')
    by_group = {group: [] for group in GROUPS}
    for source in sources:
        by_group[source_group[source]].append(source)
    if any(not by_group[group] for group in GROUPS):
        raise ValueError('V5 screening needs all four EN/ZH x Fake/Real source groups')
    budget = min(int(per_group), *(len(by_group[group]) for group in GROUPS))
    selected = set()
    for group in GROUPS:
        ordered = sorted(by_group[group], key=lambda source: (_rank(seed, group, source), source))
        coverage = {source: {_clean(dev[i]) for i in all_members[source]} for source in ordered}
        if set().union(*(coverage[source] for source in ordered)) != {False, True}:
            raise ValueError('Selection group lacks Clean or Noisy source coverage: ' + str(group))
        # One paired anchor is enough when present; if there are only unpaired
        # records, two anchors may be required. Remaining picks follow the
        # deterministic uniform hash order, including unpaired recordings.
        paired = [source for source in ordered if coverage[source] == {False, True}]
        chosen = paired[:1]
        if not chosen:
            chosen = [next(source for source in ordered if panel in coverage[source]) for panel in (False, True)]
        if len(chosen) > budget:
            raise ValueError('V5 per-group budget cannot retain both Clean and Noisy source coverage')
        chosen.extend(source for source in ordered if source not in chosen)
        selected.update(chosen[:budget])
    return sorted(i for source in selected for i in all_members[source])


def plan_tail_updates(total, snapshots=4, start=.75):
    """Unique committed optimizer updates, strictly in the final tail fraction."""
    for name, value in (('total', total), ('snapshots', snapshots)):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(name + ' must be a positive integer')
    if not math.isfinite(float(start)) or not 0 <= float(start) < 1:
        raise ValueError('Tail start must be in [0, 1)')
    first = min(int(total), int(math.floor(int(total) * float(start))) + 1)
    count = min(int(snapshots), int(total) - first + 1)
    if count == 1:
        return [int(total)]
    values = np.rint(np.linspace(first, int(total), count)).astype(np.int64)
    return sorted(set(map(int, values)) | {int(total)})


def choose_candidate(candidates):
    """Return the actual-weight candidate with best selection weighted F1.

    Exact ties prefer identity, then lower explicit conservative_rank, then
    input order. Neither Clean nor any individual class recall vetoes a gain.
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError('No V5 candidates were evaluated')
    def key(item):
        index, candidate = item
        value = candidate.get('metrics', {}).get('weighted_f1')
        if not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError('Candidate needs a finite selection weighted_f1 in [0, 1]')
        rank = candidate.get('conservative_rank', 0)
        if not isinstance(rank, (int, float)) or not math.isfinite(float(rank)):
            raise ValueError('Candidate conservative_rank must be finite')
        identity = candidate.get('name') == 'identity' or candidate.get('is_identity') is True
        return (-float(value), not identity, float(rank), index)
    return min(enumerate(candidates), key=key)[1]
