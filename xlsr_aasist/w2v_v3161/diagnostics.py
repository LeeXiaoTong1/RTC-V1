"""Treatment diagnosis must distinguish real official pairs from simulated rows."""
import numpy as np
from w2v_v316_tfcl.diagnostics import treatment_errors


def paired_treatment(originals, original_logits, dev, dev_logits, verified_pairs=None):
    hashes = {r['audio_sha256'] for r in originals}
    simulated = [i for i,r in enumerate(dev)
                 if r['condition'] in ('seen','heldout') and r['source_sha256'] in hashes]
    result = dict(official=dict(status='unavailable', reason='No verified Dev Offline-Online mapping supplied'),
        simulated=treatment_errors(originals, original_logits, [dev[i] for i in simulated],
                                   np.asarray(dev_logits)[simulated]), used_for_selection=False)
    if verified_pairs is not None:
        # Records must explicitly bind BOTH file hashes; filenames, labels or
        # Online's own source_sha256 alone are not evidence of an Offline pair.
        mapping = {p['online_sha256']:p['offline_sha256'] for p in verified_pairs}
        if len(mapping) != len(verified_pairs): raise ValueError('Ambiguous verified Dev pair mapping')
        rows, indices = [], []
        for i,row in enumerate(dev):
            if row['condition']!='online' or row['audio_sha256'] not in mapping: continue
            key = mapping[row['audio_sha256']]
            if key in hashes:
                rows.append(dict(row, source_sha256=key)); indices.append(i)
        result['official'] = dict(status='measured', **treatment_errors(
            originals, original_logits, rows, np.asarray(dev_logits)[indices]))
    return result
