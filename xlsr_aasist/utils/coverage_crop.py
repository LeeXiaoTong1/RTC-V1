"""Select one Train window without consuming the augmentation RNG stream.

The energy screen rejects mostly silent proposals; it is not a speech VAD.
Labels must apply to the full utterance, as in the official RTCFake protocol.
"""
import math
import random

import numpy as np


def select_window(audio, cut=64600, seed=0, prefix_probability=.5):
    x = np.asarray(audio)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Coverage cropping requires nonempty finite mono audio')
    if type(cut) is not int or cut < 1:
        raise ValueError('cut must be a positive integer')
    if not math.isfinite(prefix_probability) or not 0 <= prefix_probability <= 1:
        raise ValueError('prefix_probability must lie in [0, 1]')
    info = {'start': 0, 'source_samples': len(x), 'window_samples': min(cut, len(x)),
            'kind': 'short', 'active_fraction': None}
    if len(x) <= cut:
        return info
    rng = random.Random(int(seed))
    if rng.random() < prefix_probability:
        info['kind'] = 'prefix'
        return info
    frame = 320  # 20 ms at 16 kHz, measured before RawBoost/noise.
    frames = x[:len(x)//frame*frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64)**2, axis=1))
    threshold = max(1e-5, float(rms.max()) * .01)
    active = rms > threshold
    cumulative = np.concatenate(([0], np.cumsum(active)))
    for _ in range(8):
        start = rng.randint(0, len(x)-cut)
        first, last = (start+frame-1)//frame, (start+cut)//frame
        fraction = float((cumulative[last]-cumulative[first]) / max(1, last-first))
        if fraction >= .2:
            info.update(start=start, kind='random', active_fraction=fraction)
            return info
    # Keep the established prefix if the bounded search finds no usable window.
    info['kind'] = 'fallback_prefix'
    return info
