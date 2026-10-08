"""Bounded, source-disjoint selection/audit panels; never add them to Weighted."""
from collections import defaultdict
import hashlib
import json
import math

import numpy as np
import torch

from w2v_v315.augment import recipe, seed_for
from w2v_v314.diagnostics import offline_rows
from w2v_v315.diagnostics import treatment_errors
from w2v_aasist.progress import progress
from w2v_v3.model import microbatches as exact_microbatches
from w2v_v313.replay import fp32_inference
from w2v_v315.step import clear_features
from .data import panel_loader, tensors
from .metrics import binary_metrics


FAMILIES = ('ffmpeg', 'webrtc', 'light')


def infer_panel(model, rows, recipes, cfg, run, label):
    """One exact-length FP32 forward per processed source; zero waveform disk cache."""
    model.eval()
    output, offset = [], 0
    try:
        with fp32_inference(cfg['device']):
            for batch in progress(panel_loader(rows, recipes, cfg, run),
                    total=math.ceil(len(rows)/cfg['feature_batch']), label=label, every=50):
                examples = tensors(batch)
                scores = np.empty((len(examples), 2), np.float32)
                for indices, x, mask in exact_microbatches(examples, cfg['microbatch'], cfg['frame_budget']):
                    logits, _ = model(x.to(cfg['device']), mask.to(cfg['device']))
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError('Nonfinite robustness panel logits')
                    scores[indices] = logits.float().cpu().numpy()
                    clear_features(model)
                expected = rows[offset:offset+len(examples)]
                if [(r['source_id'], r['group_id']) for r in examples] != [(r['source_id'], r['group_id']) for r in expected]:
                    raise ValueError('Frozen robustness panel source order changed')
                output.append(scores)
                offset += len(examples)
    finally:
        clear_features(model)
    if offset != len(rows) or not output:
        raise ValueError('Incomplete robustness panel inference')
    return np.concatenate(output)


def build_panel(rows, cfg):
    """One recipe per unique original source, bounded before any audio decoding.

    Tune and audit original-waveform hashes are disjoint from one another.
    Both come from existing official Dev: neither is an unseen competition test.
    Audit source recordings still occur in the historical fixed Dev; only these
    additional processed views are final-only. This is not an independent-source
    generalization estimate. Never rank candidates on the audit arm.
    """
    limit = int(cfg.get('panel_sources', 4*cfg.get('panel_sources_per_group', 128)))
    if limit < 48 or limit % 8:
        raise ValueError('panel_sources must be a multiple of eight and at least 48')
    unique = {}
    for row in rows:
        if row.get('split') != 'dev' or row.get('condition') != 'offline':
            raise ValueError('Panels require verified original official Dev rows')
        key = row['group_id']
        if key != row['audio_sha256'] or key != row['source_sha256']:
            raise ValueError('Panel source identity must be the verified original waveform hash')
        if key in unique and (row['label'], row['language']) != (unique[key]['label'], unique[key]['language']):
            raise ValueError('Conflicting labels/languages for the same panel source')
        if key not in unique or str(row['source_id']) < str(unique[key]['source_id']):
            unique[key] = row
    strata = defaultdict(list)
    for row in unique.values():
        strata[(row['language'], row['label'])].append(row)
    if set(strata) != {('en', 0), ('en', 1), ('zh', 0), ('zh', 1)}:
        raise ValueError('Panel needs EN/ZH and fake/real source strata')
    per_stratum = min(limit//4, *(len(value) for value in strata.values()))
    per_stratum -= per_stratum % 2
    if per_stratum < 12:
        raise ValueError('Panel needs at least 12 unique sources in every language/class stratum')
    selected, recipes = [], []
    seed = int(cfg.get('panel_seed', 31510901))
    for group, values in sorted(strata.items()):
        ordered = sorted(values, key=lambda row: (seed_for(seed, 'source', row['group_id']), row['source_id']))
        for rank, row in enumerate(ordered[:per_stratum]):
            partition = 'tune' if rank % 2 == 0 else 'audit'
            within = rank//2
            # Equal family cycles; held-out settings come from recipe(split=dev).
            # Offset the two arms so their settings/noise draws are independent.
            phase = (0, 2, 4)[within % 3] + 5*(within//3)
            r = recipe(seed, 'v3151:'+partition+':'+row['group_id'], phase, row['group_id'], split='dev')
            selected.append(dict(row, panel_partition=partition))
            recipes.append(dict(r, panel_partition=partition))
    signature = hashlib.sha256(json.dumps([(r['group_id'], r['language'], r['label'], p)
                    for r, p in zip(selected, recipes)], sort_keys=True).encode()).hexdigest()
    manifest = dict(schema='v3151_bounded_source_panel_v1', total_sources=len(selected),
        source_cap=limit, sources_per_partition=len(selected)//2, source_disjoint=True,
        selection_partition='tune', post_selection_only_partition='audit', sha256=signature,
        included_in_weighted=False, ranking_unit='equal mechanism macro; balanced source language/class strata',
        scope='Existing official Dev sources; new source-disjoint panels, not unseen official test data',
        audit_limitation='Only additional audit processed views are final-only; their source recordings '
            'also occur in historical fixed Dev, so this is not independent-source generalization evidence',
        settings_scope='Complete settings and noise recordings held out from this Train recipe; '
            'engines are shared and may have been seen by parent checkpoints',
        all_inputs_frozen_before_training=True,
        disk_policy='no generated waveform/feature disk cache; bounded decoded-wave/noise RAM only')
    return selected, recipes, manifest


def panel_partition(rows, recipes, partition):
    if len(rows) != len(recipes) or partition not in ('tune', 'audit'):
        raise ValueError('Invalid panel inventory/partition')
    indices = [i for i, row in enumerate(rows) if row.get('panel_partition') == partition]
    if not indices or any(recipes[i].get('panel_partition') != partition for i in indices):
        raise ValueError('Panel recipe/source partition mismatch')
    return [rows[i] for i in indices], [recipes[i] for i in indices]


def panel_metrics(rows, recipes, logits, target=.99):
    z = np.asarray(logits, dtype=np.float64)
    if not rows or len(rows) != len(recipes) or z.shape != (len(rows), 2) or not np.isfinite(z).all():
        raise ValueError('Invalid panel scores/inventory')
    partitions = {row.get('panel_partition') for row in rows}
    if len(partitions) != 1 or partitions - {'tune', 'audit'}:
        raise ValueError('Never pool selection and final-only audit sources')
    if any(r.get('panel_partition') != row['panel_partition'] for row, r in zip(rows, recipes)):
        raise ValueError('Panel recipe/source partition mismatch')
    if len({row['group_id'] for row in rows}) != len(rows):
        raise ValueError('A panel source may appear only once')
    margins = z[:, 0]-z[:, 1]

    def score(indices):
        return binary_metrics([rows[i]['label'] for i in indices], margins[indices], target)

    indices = list(range(len(rows)))
    pooled = score(indices)
    families = {name: score([i for i in indices if recipes[i]['family'] == name])
                for name in sorted({r['family'] for r in recipes})}
    complete = (set(families) == set(FAMILIES) and
                all(min(group['class_counts']) > 0 for group in families.values()))
    result = dict(f1=pooled['macro_f1'], pooled=pooled, families=families, complete=complete,
        macro_f1=float(np.mean([g['macro_f1'] for g in families.values()])),
        macro_auc=float(np.mean([g['auc'] for g in families.values()])) if complete else None,
        macro_matched_real=float(np.mean([g['matched']['real_recall'] for g in families.values()])) if complete else None,
        groups={language: score([i for i in indices if rows[i]['language'] == language]) for language in ('en', 'zh')},
        noise_types={name: score([i for i in indices if recipes[i]['noise_type'] == name])
                     for name in sorted({r['noise_type'] for r in recipes})},
        severities={name: score([i for i in indices if recipes[i]['severity'] == name])
                    for name in sorted({r['severity'] for r in recipes})},
        mechanism_language={name+'/'+language: score([i for i in indices
            if recipes[i]['family'] == name and rows[i]['language'] == language])
            for name in families for language in ('en', 'zh')},
        views=len(rows), partition=next(iter(partitions)), included_in_weighted=False,
        scope='Bounded official Dev sources, same engines with held-out settings/noise; not an official score')
    result['alerts'] = ([name+': near-chance macro F1; inspect class errors and audio recipe before interpreting'
        for name, g in families.items() if g['macro_f1'] < .60] +
        [name+': fewer than 100 fake sources; matched99 recall is discrete'
         for name, g in families.items() if g['class_counts'][0] < 100])
    return result


def paired_prediction_changes(rows, before, after):
    """Exactly paired source counts, with no inference from aggregate F1 alone."""
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape or a.shape != (len(rows), 2) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected paired finite logits in the same row order')
    output = defaultdict(lambda: dict(count=0, rescued=0, newly_wrong=0, unchanged_correct=0, unchanged_wrong=0))
    for row, old, new in zip(rows, a.argmax(1), b.argmax(1)):
        label = row['label']
        group = output[row['language']+'/'+('fake' if label == 0 else 'real')]
        group['count'] += 1
        key = ('unchanged_correct' if new == label else 'newly_wrong') if old == label else (
            'rescued' if new == label else 'unchanged_wrong')
        group[key] += 1
    return dict(groups=dict(output), decision_changes=int(np.sum(a.argmax(1) != b.argmax(1))))
