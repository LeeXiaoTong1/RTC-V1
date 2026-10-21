"""Reuse V5 receiver synthesis; add CTC-independent pair validity in workers."""
import numpy as np
from w2v_v5.data import Waves as ParentWaves, TicketSampler, loader, close
from w2v_v3191.data import microbatches
from w2v_v3186.augment import estimate_delay, reliable_spans


def valid_frames(row):
    count = max(0, (len(row['wave'])-400)//320+1)
    starts = np.arange(count)*320
    original = int(row.get('original_samples', len(row['wave'])))
    valid = starts+400 <= original
    excluded = list(row.get('input_processing', {}).get('pcl_excluded_spans', []))
    aug = row.get('augmentation', {})
    # V5 shifts these spans back from the discarded processor context.
    clean = row.get('role') == 'clean_ref'
    excluded += aug.get('clean_activity' if clean else 'rtc_activity', {}).get('suppressed_spans', [])
    if not clean:
        excluded += aug.get('erased_spans', [])
    for a, b in excluded:
        valid &= ~((starts < b) & (starts+400 > a))
    return valid


def pair_info(clean, noisy):
    keys = ('source_id', 'group_id', 'language', 'label')
    if any(clean[k] != noisy[k] for k in keys) or clean.get('split') != 'train' or noisy.get('split') != 'train':
        raise ValueError('Repair target must have the same allowed source and true class')
    a = clean['wave'][:clean['original_samples']]
    b = noisy['wave'][:noisy['original_samples']]
    delay = estimate_delay(a, b)
    cm, nm = valid_frames(clean), valid_frames(noisy)
    if delay['valid']:
        # Local signal survival, independent of CTC coverage and labels.
        spans = [(s, min(len(a), s+2560)) for s in range(0, len(a), 2560)]
        reliable = reliable_spans(a, b, spans, 0., delay['offset'])
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
    return dict(schema='receiver_frame_pair_v1', shift_frames=delay['offset']/320.,
        clean_valid=cm, noisy_valid=nm, delay=delay,
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
