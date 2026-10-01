"""Gentle, label-independent processing variation of existing full noisy Train.

These operators broaden signal conditions. They are not a codec, a packet-loss
simulator, or a reconstruction of any evaluation platform. No new audio is read.
"""
import math
import numpy as np
from w2v_aasist.data import stable_seed
from .silence import apply_local_silence

SAMPLE_RATE = 16000
OPERATORS = ('frequency_response', 'smooth_gain', 'local_attenuation')


def processing_settings(cfg):
    values = {
        'processing_enabled': cfg.get('processing_enabled', True),
        'processing_identity_probability': float(cfg.get('processing_identity_probability', .5)),
        'processing_single_probability': float(cfg.get('processing_single_probability', .4)),
        # Old configurations remain non-silencing unless explicitly opted in.
        'processing_silence_probability': float(cfg.get('processing_silence_probability', 0.)),
    }
    if type(values['processing_enabled']) is not bool:
        raise ValueError('processing_enabled must be boolean')
    identity, single, silence = (values[k] for k in ('processing_identity_probability',
        'processing_single_probability','processing_silence_probability'))
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (identity, single, silence)) or identity + single + silence > 1 + 1e-12:
        raise ValueError('Processing probabilities must be finite, nonnegative, and sum to at most one')
    return values


def processing_plan(source_id, version, *, seed, epoch, **settings):
    """Only recording identity/version and epoch determine the random recipe.

    There is deliberately no label, language, source-domain group or worker ID
    argument. The same seed remains valid with a different number of workers.
    """
    settings = processing_settings(settings)
    rng = np.random.default_rng(stable_seed(seed, epoch, ('v32-processing', source_id, version)))
    draw = float(rng.random())
    identity, single = settings['processing_identity_probability'], settings['processing_single_probability']
    if not settings['processing_enabled'] or draw < identity:
        return []
    if identity + single <= draw < identity + single + settings['processing_silence_probability']:
        return [dict(operator='local_silence',duration_seconds=float(rng.uniform(.04,.16)),
                     position=float(rng.random()),maximum_fraction=.05,fade_seconds=.005)]
    count = 1 if draw < identity + single else 2
    choices = rng.choice(len(OPERATORS), count, replace=False)
    plan = []
    for index in choices:
        name = OPERATORS[int(index)]
        if name == 'frequency_response':
            parameters = dict(highpass_hz=float(rng.uniform(60., 160.)),
                              lowpass_hz=float(rng.uniform(3400., 7200.)), order=2)
        elif name == 'smooth_gain':
            parameters = dict(knot_seconds=float(rng.uniform(.5, 1.5)), maximum_db=3.,
                              envelope_seed=int(rng.integers(0, 2**32)))
        else:
            parameters = dict(duration_seconds=float(rng.uniform(.08, .24)),
                              depth_db=float(rng.uniform(3., 9.)), position=float(rng.random()),
                              maximum_fraction=.1)
        plan.append({'operator': name, **parameters})
    return plan


def apply_plan(wave, plan):
    """Apply at most two gentle operations, preserving every sample position.

    Work on an independent buffer: cached waveforms and the caller's arrays are
    never modified. Filters use constant-signal initial conditions to limit a
    spurious onset transient. No clipping, tiling, resampling or length change.
    """
    if np.ndim(wave) != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Processing requires finite, nonempty mono audio')
    if len(plan) > 2 or len({p['operator'] for p in plan}) != len(plan):
        raise ValueError('At most two distinct processing operators are supported')
    if any(p['operator']=='local_silence' for p in plan):
        if len(plan)!=1:
            raise ValueError('Local silence must not be combined with other processing')
        return apply_local_silence(wave,plan[0])[0]
    result = np.array(wave, dtype=np.float32, copy=True)
    for parameters in plan:
        name = parameters['operator']
        if name == 'frequency_response':
            from scipy.signal import butter, sosfilt, sosfilt_zi
            sos = butter(parameters['order'], [parameters['highpass_hz'], parameters['lowpass_hz']],
                         btype='bandpass', fs=SAMPLE_RATE, output='sos')
            filtered, _ = sosfilt(sos, result, zi=sosfilt_zi(sos) * float(result[0]))
            result = filtered.astype(np.float32)
        elif name == 'smooth_gain':
            interval = max(1, round(parameters['knot_seconds'] * SAMPLE_RATE))
            rng = np.random.default_rng(parameters['envelope_seed'])
            knot_count = len(result) // interval + 2
            db = rng.uniform(-parameters['maximum_db'], parameters['maximum_db'], knot_count)
            positions = np.arange(len(result), dtype=np.int64)
            segment, fraction = positions // interval, (positions % interval) / interval
            blend = .5 - .5 * np.cos(np.pi * fraction)
            envelope = db[segment] * (1-blend) + db[segment+1] * blend
            result *= np.power(10., envelope / 20.).astype(np.float32)
        elif name == 'local_attenuation':
            duration = min(round(parameters['duration_seconds'] * SAMPLE_RATE),
                           max(1, math.floor(parameters['maximum_fraction'] * len(result))))
            start = min(len(result)-duration, int(parameters['position'] * (len(result)-duration+1)))
            # Raised cosine reaches a nonzero attenuation floor and returns to
            # unity at both ends, avoiding hard-edged artificial dropouts.
            pulse = np.sin(np.linspace(0., np.pi, duration)) ** 2 if duration > 1 else np.ones(1)
            envelope = 1 - (1 - 10. ** (-parameters['depth_db']/20.)) * pulse
            result[start:start+duration] *= envelope.astype(np.float32)
        else:
            raise ValueError('Unknown processing operator: ' + str(name))
    if not np.isfinite(result).all():
        raise FloatingPointError('Processing produced non-finite audio')
    return np.ascontiguousarray(result)


def process_wave(wave, source_id, version, *, seed, epoch, protected_views=(), **settings):
    plan = processing_plan(source_id, version, seed=seed, epoch=epoch, **settings)
    details={}
    if plan and plan[0]['operator']=='local_silence':
        result,details=apply_local_silence(wave,plan[0],protected_views)
    else:
        result = apply_plan(wave, plan)
    applied=bool(plan) and details.get('silence_applied',True)
    return result, {
        'processing_condition': ('+'.join(sorted(p['operator'] for p in plan)) if applied else 'unchanged'),
        'processing_parameters': plan,
        'processing_applied': applied,
        **details,
    }
