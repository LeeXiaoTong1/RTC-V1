"""Fit only reserved sources; select identity vs affine on the selection subset."""
import numpy as np
from w2v_v318.calibration import fit, apply
from .metrics import measure


def choose(rows, logits, partition):
    candidate = fit(rows, logits, partition['calibration'])
    ix = partition['select']
    selection_rows = [rows[i] for i in ix]
    z = np.asarray(logits)
    raw = measure(selection_rows, z[ix])
    calibrated = measure(selection_rows, apply(z[ix], candidate))
    enabled = calibrated['weighted_f1'] > raw['weighted_f1']+1e-12
    return dict(scale=candidate['scale'] if enabled else 1., bias=candidate['bias'] if enabled else 0.,
        enabled=enabled, selected_candidate='affine' if enabled else 'identity', candidate=candidate,
        raw_selection=raw, affine_selection=calibrated, classification_threshold=.5,
        note='Affine fitted only on calibration sources. Identity/affine selected on Dev selection, not an independent test.')
