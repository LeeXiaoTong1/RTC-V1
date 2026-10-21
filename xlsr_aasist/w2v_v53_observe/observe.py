"""Attribute existing CE forwards to their source; never alter training tensors."""
from contextlib import contextmanager
import math
import torch
from w2v_v3183r2.common import GROUPS
from w2v_v3191.ce import classification_weights
from w2v_v3191.data import microbatches
from w2v_v3191.meta import validate_units
from w2v_v3191.objective import auxiliary_weights, batch_quotas

SAMPLE_EXCESS_THRESHOLD = .05
MAX_EXAMPLES_PER_UPDATE = 8


def entropy_floor(smoothing):
    q = (1. - smoothing/2., smoothing/2.)
    return -sum(p*math.log(p) for p in q if p)


def classification_plan(units, cfg, current_update):
    """Mirror only deterministic packing and classifier order in side_pass.

    Main Online/Noisy logits precede matched Clean-RTC logits in EACH microbatch.
    Official PCL-only references are encoded but never included in this plan.
    """
    sides = validate_units(units)
    ce_weights = classification_weights(cfg)
    batches = []
    pair_id = 0
    for name in ('A', 'B'):
        side_units = sides[name]
        quotas = batch_quotas(side_units, cfg)
        runtime = dict(cfg, group_quotas=quotas, current_update=current_update,
                       joint_phase=True)
        if name == 'B':
            runtime['consistency'] = 'none'
        weights = auxiliary_weights(runtime)
        for micro in microbatches(side_units, cfg['microbatch'], cfg['frame_budget'], weights=weights):
            main, clean = [], []
            for unit in micro:
                row = unit[0]
                quota = quotas[GROUPS.index((row['language'], row['label']))]
                # Parent's 1/(8*quota), view rescaling 2*w, A/B factor 1/2.
                def entry(view, actual):
                    return dict(row=actual, source=row, view=view, side=name,
                                pair_id=pair_id, mass=ce_weights[view]/(8.*quota))
                main.append(entry(row['role'], row))
                if row['role'] == 'noisy':
                    clean.append(entry('clean_rtc', row['pcl_refs']['clean']))
                pair_id += 1
            batches.append(main + clean)
    return batches


@contextmanager
def capture_classifier(model, batches):
    """Temporary local observer of the exact existing logits, including dropout.

    Detached logits are tiny; one CPU transfer after the whole committed update.
    Restore an existing instance override (or the original class descriptor) on
    every exit, including AMP retries and unexpected exceptions.
    """
    original = model.classify_sequences
    had_override = 'classify_sequences' in model.__dict__
    previous = model.__dict__.get('classify_sequences')
    captured = []

    def observed(sequences):
        index = len(captured)
        if index >= len(batches) or len(sequences) != len(batches[index]):
            raise RuntimeError('Hard-example observer classifier packing differs from the pinned pipeline')
        logits = original(sequences)
        if logits.shape != (len(batches[index]), 2):
            raise RuntimeError('Hard-example observer requires binary classification logits')
        captured.append(logits.detach())
        return logits

    model.classify_sequences = observed
    try:
        yield captured
    finally:
        if had_override:
            model.classify_sequences = previous
        else:
            delattr(model, 'classify_sequences')


def _small(value, depth=0):
    """Bounded JSON metadata only: no waveforms, tensors or phoneme arrays."""
    if depth > 4:
        return None
    if value is None or isinstance(value, (bool, int, str)):
        return value[:1500] if isinstance(value, str) else value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _small(v, depth+1) for k,v in value.items()
                if k not in ('wave', 'suppressed_spans', 'pcl_pairs', 'pcl_refs')
                and isinstance(v, (type(None), bool, int, float, str, dict))}
    return None


def source_metadata(item):
    row, source = item['row'], item['source']
    result = {key: _small(source[key]) for key in
        ('id', 'source_id', 'group_id', 'canonical_source', 'view_id', 'path',
         'language', 'label', 'v5_origin_split', 'split', 'occurrence', 'source_cycle',
         'condition_block', 'noise_recipe_view', 'row_index') if key in source}
    if 'path' in row:
        result['path'] = str(row['path'])
    result.update(view=item['view'], meta_side=item['side'],
                  true_class='fake' if source['label'] == 0 else 'real',
                  samples=len(row['wave']), duration_seconds=len(row['wave'])/16000.,
                  original_samples=row.get('original_samples', len(row['wave'])))
    for key in ('base_augmentation', 'augmentation', 'input_processing', 'short_padding'):
        if key in row:
            result[key] = _small(row[key])
    # Clean shares the paired recipe, but environmental noise applies to Noisy only.
    result['environment_noise_applied'] = item['view'] == 'noisy'
    return result


def analyze(captured, batches, cfg, stats):
    if len(captured) != len(batches):
        raise RuntimeError('Incomplete hard-example observation')
    items = [item for batch in batches for item in batch]
    # Do numerical diagnostics on CPU, outside autograd, without RNG consumption.
    logits = torch.cat(captured).float().cpu()
    logp = torch.log_softmax(logits.double(), dim=-1).tolist()
    smoothing = float(cfg['label_smoothing'])
    floor = entropy_floor(smoothing)
    values, paired, cells = [], {}, {}
    for item, lp in zip(items, logp):
        label = item['source']['label']
        raw = -lp[label]
        ce = (1-smoothing)*raw - smoothing*(lp[0]+lp[1])/2
        p_true = math.exp(lp[label])
        predicted = 0 if lp[0] >= lp[1] else 1
        value = dict(raw_ce=raw, smoothed_ce=ce, ce_excess=max(0., ce-floor),
                     ce_mass=item['mass'], weighted_ce=ce*item['mass'],
                     weighted_excess=max(0., ce-floor)*item['mass'],
                     p_fake=math.exp(lp[0]), p_true=p_true, correct=predicted == label,
                     predicted_class='fake' if predicted == 0 else 'real')
        value['reason'] = ('misclassified' if not value['correct'] else
                           'overconfident_correct' if p_true > 1-smoothing/2 else
                           'low_confidence_correct')
        values.append(value)
        paired[(item['pair_id'], item['view'])] = value
        key = f'{item["view"]}/{item["source"]["language"]}/{label}'
        cell = cells.setdefault(key, dict(views=0, correct=0, objective=0., raw_ce_sum=0.))
        cell['views'] += 1
        cell['correct'] += int(value['correct'])
        cell['objective'] += value['weighted_ce']
        cell['raw_ce_sum'] += raw
    observed_ce = sum(v['weighted_ce'] for v in values)
    expected_ce = float(stats['classification_loss'])
    # Validate group attribution as well as the global total: wrong ordering can
    # preserve a binary batch mean while assigning the error to the wrong source.
    matches = math.isclose(observed_ce, expected_ce, rel_tol=2e-5, abs_tol=2e-6)
    expected_cells = stats.get('cells', {})
    matches = matches and set(cells) == set(expected_cells)
    for key, cell in cells.items():
        expected = expected_cells.get(key, {})
        matches = matches and all(cell[k] == expected.get(k) for k in ('views', 'correct'))
        matches = matches and all(math.isclose(cell[k], expected.get(k, float('inf')),
                    rel_tol=2e-5, abs_tol=2e-6) for k in ('objective', 'raw_ce_sum'))
    result = dict(validated=matches, views=len(values), ce_floor=floor,
                  classification_loss=expected_ce, reconstructed_ce=observed_ce,
                  errors=sum(not v['correct'] for v in values),
                  raw_ce_mean=sum(v['raw_ce'] for v in values)/len(values),
                  max_raw_ce=max(v['raw_ce'] for v in values),
                  weighted_excess=sum(v['weighted_excess'] for v in values),
                  sample_excess_threshold=SAMPLE_EXCESS_THRESHOLD, cells=cells)
    if not matches:
        # The optimizer has already committed. Report the observer failure rather
        # than raising and replaying that update or publishing unreliable IDs.
        return dict(result, error='CE/group attribution mismatch; sample IDs suppressed', examples=[])
    candidates = [i for i,v in enumerate(values)
                  if not v['correct'] or v['ce_excess'] >= SAMPLE_EXCESS_THRESHOLD]
    candidates.sort(key=lambda i: values[i]['weighted_excess'], reverse=True)
    examples = []
    for index in candidates[:MAX_EXAMPLES_PER_UPDATE]:
        item, value = items[index], values[index]
        example = dict(source_metadata(item), **value)
        other_view = {'noisy': 'clean_rtc', 'clean_rtc': 'noisy'}.get(item['view'])
        other = paired.get((item['pair_id'], other_view))
        if other is not None:
            example['paired'] = dict(view=other_view, **{k: other[k] for k in
                ('p_fake', 'p_true', 'correct', 'raw_ce', 'smoothed_ce')})
            clean = value if item['view'] == 'clean_rtc' else other
            noisy = value if item['view'] == 'noisy' else other
            example['pair_outcome'] = ('both_wrong' if not clean['correct'] and not noisy['correct'] else
                'noisy_only_wrong' if clean['correct'] and not noisy['correct'] else
                'clean_only_wrong' if not clean['correct'] and noisy['correct'] else 'both_correct')
        examples.append(example)
    selected_excess = sum(e['weighted_excess'] for e in examples)
    return dict(result, candidate_count=len(candidates), recorded_count=len(examples),
                omitted_candidates=max(0, len(candidates)-len(examples)),
                recorded_excess_fraction=selected_excess/max(result['weighted_excess'], 1e-30),
                examples=examples)


def observed_step(step_fn, model, optimizer, units, cfg, **kwargs):
    current = int(getattr(optimizer, '_v3189_joint_update', cfg.get('current_update', 1)))
    batches = classification_plan(units, cfg, current)
    with capture_classifier(model, batches) as captured:
        stats = step_fn(model, optimizer, units, cfg, **kwargs)
    if not stats['amp']['skipped_update']:
        stats['hard_examples'] = analyze(captured, batches, cfg, stats)
    return stats
