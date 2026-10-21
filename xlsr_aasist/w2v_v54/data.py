"""Reuse V5 receiver synthesis; add CTC-independent pair validity in workers."""
import numpy as np
import hashlib
import json
import time
from w2v_v5.data import Waves as ParentWaves, TicketSampler, loader, close
from w2v_v3191.data import microbatches
from .alignment import align_pair


def valid_frames(row, *, activity=True):
    count = max(0, (len(row['wave'])-400)//320+1)
    starts = np.arange(count)*320
    original = int(row.get('original_samples', len(row['wave'])))
    valid = starts+400 <= original
    excluded = list(row.get('input_processing', {}).get('pcl_excluded_spans', []))
    aug = row.get('augmentation', {})
    # V5 shifts these spans back from the discarded processor context.
    clean = row.get('role') == 'clean_ref'
    if activity:
        excluded += aug.get('clean_activity' if clean else 'rtc_activity', {}).get('suppressed_spans', [])
    if activity and not clean:
        excluded += aug.get('erased_spans', [])
    for a, b in excluded:
        valid &= ~((starts < b) & (starts+400 > a))
    return valid


def pair_info(clean, noisy):
    started = time.perf_counter()
    keys = ('source_id', 'group_id', 'language', 'label')
    if any(clean[k] != noisy[k] for k in keys) or clean.get('split') != 'train' or noisy.get('split') != 'train':
        raise ValueError('Repair target must have the same allowed source and true class')
    a = clean['wave'][:clean['original_samples']]
    b = noisy['wave'][:noisy['original_samples']]
    spans = [(s, min(len(a), s+2560)) for s in range(0, len(a), 2560)]
    delay, reliable = align_pair(a, b, spans)
    cm, nm = valid_frames(clean), valid_frames(noisy)
    if delay['valid']:
        # Local signal survival, independent of CTC coverage and labels.
        c_ok, n_ok = np.zeros_like(cm), np.zeros_like(nm)
        cs, ns = np.arange(len(cm))*320, np.arange(len(nm))*320
        for (start, end), keep in zip(spans, reliable):
            if keep:
                c_ok |= (cs >= start) & (cs+400 <= end)
                n_ok |= (ns >= start+delay['offset']) & (ns+400 <= end+delay['offset'])
        cm &= c_ok
        nm &= n_ok
    else:
        cm[:] = False
        nm[:] = False
    def checksum(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for key in ('base_augmentation',):
        if clean.get(key) != noisy.get(key):
            raise ValueError('Receiver pair has different base augmentation')
    ca,na = clean.get('augmentation',{}),noisy.get('augmentation',{})
    if any(na.get(k) is False for k in ('same_settings','shared_base','fresh_state_per_utterance')):
        raise ValueError('Receiver pair must share settings/base and reset processor states')
    for key in ('recipe','common_gain','receiver_context'):
        if ca.get(key) != na.get(key):
            raise ValueError('Receiver pair has different RTC settings: '+key)
    recipe = na.get('recipe',{})
    snr = recipe.get('snr',recipe.get('snr_db',na.get('snr_db')))
    return dict(schema='receiver_frame_pair_v2', shift_frames=delay['offset']/320.,
        clean_valid=cm, noisy_valid=nm, delay=delay,
        source_id=noisy['source_id'], source_group=noisy['group_id'], label=noisy['label'],
        language=noisy['language'], source_visit_id=noisy.get('view_id',noisy.get('id')),
        noisy_slot='unit.0',clean_slot='pcl_refs.clean',
        planned_group_budget=noisy.get('batch_group_quotas'),
        recipe_hash=checksum(recipe), common_base_hash=checksum(noisy.get('base_augmentation',{})),
        original_samples=dict(clean=len(a),noisy=len(b)),
        condition='/'.join(str(recipe.get(k,'unknown')) for k in ('family','codec','noise_type')),
        snr_db=snr, seconds=len(b)/16000., alignment_seconds=time.perf_counter()-started,
        original_origin=noisy.get('v5_origin_split', 'train'))


class Waves(ParentWaves):
    def __getitem__(self, ticket):
        units = super().__getitem__(ticket)
        if self.cfg.get('repair_arm') != 'paired':
            return units
        for row in units:
            if row.get('role') == 'noisy' and row.get('meta_side') == 'A':
                # Parent returns early without PCL for Dev. Pair generation is
                # deliberately AFTER that return: allowed Dev gets LF as well.
                row['repair_pair'] = pair_info(row['pcl_refs']['clean'], row)
        return units
