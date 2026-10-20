"""V5 deterministic receiver context and identical train/inference input policy.

The context is a same-recording proxy for processor history, not extra speech
data or an assertion that a particular RTC service has been reproduced.
"""
import copy
import hashlib
import json
import math

import numpy as np
from scipy import signal

SR = 16000
MIN_SAMPLES = 3920
SCHEMA = 'v5_input_processing_v1'
STANDARDS = ('identity', 'loudness', 'lowpass', 'loudness_lowpass')


def _wave(wave):
    x = np.asarray(wave, np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('V5 processing needs finite, nonempty, mono PCM')
    return x


def policy(standardization='identity', short_seconds=0.):
    """Return a complete, hashable policy; inference must use this saved policy."""
    if standardization not in STANDARDS:
        raise ValueError('Unknown V5 input standardization: ' + str(standardization))
    if not math.isfinite(float(short_seconds)) or not 0 <= float(short_seconds) <= 6:
        raise ValueError('V5 short duration must be between 0 and 6 seconds')
    return dict(schema=SCHEMA, standardization=standardization,
        short_seconds=float(short_seconds), sample_rate=SR,
        minimum_samples=MIN_SAMPLES, rms_dbfs=-23., gain_limit_db=6.,
        peak_limit=.98, lowpass_hz=6800., lowpass_taps=65,
        repeat_policy='append_exact_cycles_preserve_original_prefix',
        pcl_policy='original_prefix_only_exclude_filter_boundary_support')


def validate_policy(value=None):
    if value is None:
        return policy()
    if isinstance(value, str):
        return policy(value)
    if not isinstance(value, dict):
        raise ValueError('V5 input policy must be a saved policy dictionary')
    expected = policy(value.get('standardization', 'identity'), value.get('short_seconds', 0.))
    if value != expected:
        raise ValueError('V5 input policy differs from the supported versioned recipe')
    return copy.deepcopy(expected)


def policy_hash(value):
    value = validate_policy(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def candidate_recipes():
    """Identity is compulsory; Omni's waveform LN cancels simple gain changes."""
    return [
        dict(name='identity', policy=policy()),
        dict(name='short_3s', policy=policy(short_seconds=3.)),
        dict(name='lowpass_6800', policy=policy('lowpass')),
        dict(name='short_3s_lowpass_6800', policy=policy('lowpass', 3.)),
    ]


def standardize_wave(wave, recipe=None):
    """Deterministic sample-preserving processing, with no labels/IDs as input."""
    cfg = validate_policy(recipe)
    x = _wave(wave)
    y = x.copy()
    standard = cfg['standardization']
    guard = 0
    if standard in ('lowpass', 'loudness_lowpass'):
        taps = signal.firwin(cfg['lowpass_taps'], cfg['lowpass_hz'], fs=SR)
        guard = len(taps) // 2
        # Center the FIR explicitly, retaining exact source time coordinates.
        # Reflection avoids edge zero-padding changing the level of short clips.
        padded = np.pad(y, (guard, guard), mode='reflect' if len(y) > 1 else 'edge')
        y = signal.fftconvolve(padded, taps, mode='valid').astype(np.float32)
    gain = 1.
    if standard in ('loudness', 'loudness_lowpass'):
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        if rms > 1e-8:
            gain = float(np.clip(10 ** (cfg['rms_dbfs'] / 20) / rms,
                10 ** (-cfg['gain_limit_db'] / 20), 10 ** (cfg['gain_limit_db'] / 20)))
            # Attenuation may exceed the nominal +/-6 dB when needed to prevent
            # clipping. Never hard-clip a diagnostic signal.
            gain = min(gain, cfg['peak_limit'] / max(float(np.max(np.abs(y))), 1e-12))
            y = (y * gain).astype(np.float32)
    if y.shape != x.shape or not np.isfinite(y).all():
        raise RuntimeError('V5 standardization changed source geometry')
    excluded = [] if not guard else [[0, min(guard, len(x))],
        [max(0, len(x) - guard), len(x)]]
    return y, dict(standardization=standard, gain=gain, filter_guard_samples=guard,
        pcl_excluded_spans=excluded,
        time_map=dict(scale=1., offset=0., input_samples=len(x), output_samples=len(y)))


def prepare_wave(wave, recipe=None, *, short_seconds=None):
    """Standardize, optionally repeat short clips, and retain original PCL scope.

    Artificial cycles are classifier evidence only. Phone timestamps refer to
    the original prefix, and CNN fields crossing its end are excluded from PCL.
    """
    if short_seconds is not None:
        if isinstance(recipe, dict):
            raise ValueError('Do not override a frozen V5 policy')
        recipe = policy(recipe or 'identity', short_seconds)
    cfg = validate_policy(recipe)
    x, meta = standardize_wave(wave, cfg)
    original = len(x)
    repeat_target = int(round(cfg['short_seconds'] * SR))
    added_repeat = max(0, repeat_target - original)
    if added_repeat:
        x = np.tile(x, int(math.ceil(repeat_target / original)))[:repeat_target].copy()
    # Match V3.19.1's existing minimum-length fallback when repetition is off.
    if len(x) < cfg['minimum_samples']:
        x = np.pad(x, (0, cfg['minimum_samples'] - len(x)),
            mode='reflect' if len(x) > 1 else 'edge')
    excluded = list(meta['pcl_excluded_spans'])
    if len(x) > original:
        excluded.append([original, len(x)])
    meta.update(schema=SCHEMA, policy=cfg, policy_sha256=policy_hash(cfg),
        original_samples=original, output_samples=len(x), repeated_samples=added_repeat,
        minimum_padding_samples=len(x) - original - added_repeat,
        pcl_valid_samples=original, pcl_valid_frames=max(0, (original - 400) // 320 + 1),
        pcl_excluded_spans=excluded,
        time_map=dict(scale=1., offset=0., input_samples=original,
            output_samples=len(x), valid_output_samples=original))
    return x.astype(np.float32, copy=False), meta


def preceding_context(wave, context_seconds=2.):
    """Use only this recording, so context cannot cross any source partition."""
    x = _wave(wave)
    seconds = float(context_seconds)
    if not math.isfinite(seconds) or not 0 <= seconds <= 10:
        raise ValueError('V5 RTC context must be between 0 and 10 seconds')
    count = int(round(seconds * SR))
    if not count:
        return np.empty(0, np.float32)
    # End immediately before original sample zero in a cyclic same-source view.
    # No label, speaker table, other source, or test-data inventory is consulted.
    indices = np.arange(-count, 0, dtype=np.int64) % len(x)
    return x[indices].copy()


def _crop_spans(spans, start, length):
    result = []
    for a, b in spans:
        a, b = max(0., float(a) - start), min(float(length), float(b) - start)
        if b > a:
            result.append([a, b])
    return result


def _target_metadata(meta, context_samples, target_samples):
    """Rebase exclusion geometry; retain honest full-stream activity statistics."""
    value = copy.deepcopy(meta)
    for key in ('erased_spans', 'affected_spans', 'suppressed_spans'):
        if key in value:
            value[key] = _crop_spans(value[key], context_samples, target_samples)
    for key in ('clean_activity', 'rtc_activity'):
        if isinstance(value.get(key), dict):
            activity = value[key]
            activity['suppressed_spans'] = _crop_spans(
                activity.get('suppressed_spans', []), context_samples, target_samples)
            # Aggregate gain statistics cannot be reconstructed from guarded
            # spans. Keep their actual scope instead of inventing target counts.
            activity['statistics_scope'] = 'context_plus_target'
            activity['span_coordinates'] = 'target_only'
            activity['target_samples'] = target_samples
    value['input_snr_statistics_scope'] = 'context_plus_target'
    return value


def generate_matched_context(wave, rec, bank, engines, *, context_seconds=2.):
    """Run inherited matched RTC once on an extended stream, then keep target.

    The inherited generator resets Clean/Noisy processor state independently,
    sharing the source, gain and recipe. Noise is generated over the entire
    extended stream. Context warms NS/AGC/activity/codec state within each view;
    this function never post-encodes an already encoded target.
    """
    from w2v_v3189.augment import generate_matched

    x = _wave(wave)
    context = preceding_context(x, context_seconds)
    stream = np.concatenate((context, x)) if len(context) else x
    clean, noisy, meta = generate_matched(stream, rec, bank, engines)
    clean, noisy = _wave(clean), _wave(noisy)
    if len(clean) != len(stream) or len(noisy) != len(stream):
        raise ValueError('RTC context processing changed aligned stream length')
    offset = len(context)
    clean, noisy = clean[offset:].copy(), noisy[offset:].copy()
    meta = _target_metadata(meta, offset, len(x)) if offset else copy.deepcopy(meta)
    meta['receiver_context'] = dict(schema='v5_same_source_rtc_context_v1',
        enabled=bool(offset), context_samples=offset, target_samples=len(x),
        source='cyclic_suffix_of_this_recording_only',
        same_source_context=True, fresh_state_per_view=True,
        processing_scope='one_extended_stream_per_view_then_crop',
        clean_noisy_share_source_and_settings=True,
        noise_scope='entire_noisy_stream_including_context',
        time_map=dict(scale=1., offset=0., input_samples=len(x), output_samples=len(x)))
    return clean, noisy, meta
