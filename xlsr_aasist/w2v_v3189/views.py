"""Label-blind, full-content variants with exact affine phoneme timing."""
import math
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt
from w2v_v3185.common import seed_for

SR = 16000
FAMILIES = ('original', 'speed_slow', 'speed_fast', 'frequency_response', 'amplitude_envelope')


def recipe(seed, source, kind, variant, *, family=None, generation=0, dynamic=False):
    if kind not in ('online', 'offline') or not 0 <= variant < 5:
        raise ValueError('Expected an Online/Offline variant slot')
    family = FAMILIES[variant] if family is None else family
    if family not in FAMILIES or type(generation) is not int or generation < 0:
        raise ValueError('Unknown family or invalid generation')
    if type(dynamic) is not bool or family == 'original' and (dynamic or generation):
        raise ValueError('Original audio cannot be refreshed or dynamic')
    key = seed_for(seed, 'v3189-full-content', source, kind, variant, family, generation, dynamic)
    rng = np.random.default_rng(key)
    return dict(version=3, seed=key, variant=variant, family=family,
                generation=generation, dynamic=dynamic,
                rate=int(rng.integers(970, 991)) if family == 'speed_slow' else int(rng.integers(1010, 1031)),
                low_hz=float(rng.uniform(180, 600)), high_hz=float(rng.uniform(2500, 5000)),
                low_db=float(rng.choice((-1, 1)) * rng.uniform(1, 3)),
                high_db=float(rng.choice((-1, 1)) * rng.uniform(1, 3)),
                envelope_db=float(rng.uniform(1, 3)), phase=float(rng.uniform(0, 2 * np.pi)))


def augment(wave, rec):
    x = np.asarray(wave, np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid augmentation source')
    family = rec['family']; scale = 1.
    if family == 'original':
        y = x.copy()
    elif family in ('speed_slow', 'speed_fast'):
        rate = int(rec['rate']); divisor = math.gcd(1000, rate)
        y = resample_poly(x, 1000 // divisor, rate // divisor).astype(np.float32)
        scale = 1000. / rate
    elif family == 'frequency_response':
        low = sosfilt(butter(2, rec['low_hz'], fs=SR, output='sos'), x)
        high = sosfilt(butter(2, rec['high_hz'], btype='highpass', fs=SR, output='sos'), x)
        y = (x + (10 ** (rec['low_db'] / 20) - 1) * low +
             (10 ** (rec['high_db'] / 20) - 1) * high).astype(np.float32)
    elif family == 'amplitude_envelope':
        phase = np.linspace(0, 2 * np.pi, len(x), dtype=np.float64) + rec['phase']
        y = (x * np.power(10., rec['envelope_db'] * np.sin(phase) / 20)).astype(np.float32)
    else:
        raise ValueError('Unknown full-content augmentation family')
    if family != 'original':
        peak = float(np.abs(y).max())
        if peak > .98: y = y * (.98 / peak)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError('Augmentation produced invalid audio')
    return y, dict(recipe=rec, input_samples=len(x), output_samples=len(y), fallback=None,
                   source_identity_preserved=True, independent_new_recording=False,
                   content_cropped=False,
                   time_map=dict(scale=scale, offset=0., input_samples=len(x), output_samples=len(y)))
