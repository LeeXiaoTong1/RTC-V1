"""Existing scene/RTC chain plus activity attenuation BEFORE its single codec pass."""
import copy
from w2v_v3184.augment import NoiseBank, Engines, recipe as base_recipe, generate as base_generate
from w2v_v3184.common import seed_for
from w2v_v3186.augment import estimate_delay, reliable_spans
from .activity import process, settings


def recipe(*args, activity_policy='mixed', activity_profile=None, **kwargs):
    if activity_policy not in ('mixed', 'off'):
        raise ValueError('Unknown RTC activity policy')
    rec = base_recipe(*args, **kwargs)
    # Historical fixed probes stay identical. Scene panels get deterministic
    # activity draws. Explicit profiles are for Train diagnostics only.
    fixed = rec['rtc_policy'] == 'legacy-fixed-probe'
    profile = 'off' if activity_policy == 'off' or fixed else activity_profile
    rec['activity'] = settings(seed_for(rec['seed'], 'v3189-activity'), profile)
    return rec


def endpoint(wave, rec, engines):
    cfg = rec.get('activity')
    if cfg is None:
        # Plain legacy recipes are permitted only for inherited fixed probes.
        cfg = settings(0, 'off')
    if cfg['profile'] == 'off':
        y = engines(wave, rec)
        _, meta = process(y, cfg)
        return y, meta
    if rec['codec'] not in ('none', 'opus') or rec['family'] not in ('webrtc', 'ffmpeg', 'light', 'bypass'):
        raise ValueError('Activity profiles require a supported RTC path')
    # No extra codec work: run NS/AGC once, the envelope once, then Opus once.
    y = engines(wave, dict(rec, codec='none'))
    y, meta = process(y, cfg)
    if rec['codec'] == 'opus':
        y = engines(y, dict(rec, family='bypass', stress=None))
    if len(y) != len(wave):
        raise ValueError('RTC activity must not insert or remove samples')
    return y, meta


def generate(wave, rec, bank, engines, *, input_observer=None):
    activity = {}
    def receive(mixed, selected):
        if input_observer is not None:
            input_observer(mixed)
        y, meta = endpoint(mixed, selected, engines)
        activity.update(meta)
        return y
    y, meta = base_generate(wave, rec, bank, receive)
    return y, dict(meta, rtc_activity=activity, erased_spans=activity['suppressed_spans'])


def generate_matched(wave, rec, bank, engines):
    noisy, meta = generate(wave, rec, bank, engines)
    clean, clean_activity = endpoint(wave*meta['common_gain'], rec, engines)
    if clean.shape != noisy.shape:
        raise ValueError('Matched RTC changed audio geometry')
    return clean, noisy, dict(meta, clean_activity=clean_activity,
        same_settings=True, shared_base=True, fresh_state_per_utterance=True,
        activity_state='Independent signal-conditioned state, identical parameters',
        clean_has_scene_echo=False, recipe=copy.deepcopy(rec))
