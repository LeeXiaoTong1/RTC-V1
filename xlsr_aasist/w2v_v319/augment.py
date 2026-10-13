"""Matched clean/noisy communication after one shared augmented Offline base."""
import copy
import numpy as np
from w2v_v315.augment import common_inputs, seed_for
from w2v_v318.augment import noise_wave
from w2v_v3184.augment import NoiseBank, Engines, recipe as scene_recipe
from w2v_v3184.noise import CATEGORIES, scene_wave

RECIPE = 'rtc_v319_matched_scene_full_wave_v1'


def make_recipe(cfg, occurrence, source, epoch=0, split='train', scene=None):
    return recipe(cfg['seed'], occurrence, 0, source, split=split, cfg=cfg,
                  epoch=epoch, scene=scene)


def recipe(seed, occurrence, phase, source, split='train', warm=1., *, cfg=None, epoch=0, scene=None):
    cfg = cfg or {}
    if split not in ('train', 'dev'):
        raise ValueError('Only Train/Dev may generate noise recipes')
    probability = float(cfg.get('scene_probability', .8))
    if not 0 <= probability <= 1:
        raise ValueError('Scene probability must lie in [0, 1]')
    policy = cfg.get('rtc_policy', 'three-profile')
    r = scene_recipe(seed, occurrence, source, 0, epoch, 0, split,
        stress_probability=0., stress_ramp_epochs=1.,
        scene_probability=probability, rtc_policy=policy, scene=scene)
    # V3.18.4's full scheme is 5--25 dB BEFORE processing, not residual SNR.
    # No head warmup/noise curriculum is imposed on a trained V3.15 parent.
    if split == 'dev' and scene is None:
        rng = np.random.default_rng(seed_for(seed, occurrence, source, 'v319-dev-scenes'))
        selected = str(rng.choice(CATEGORIES)) if rng.random() < probability else None
        if selected:
            r = scene_recipe(seed, occurrence, source, 0, epoch, 0, split,
                stress_probability=0., stress_ramp_epochs=1., scene_probability=probability,
                rtc_policy=policy, scene=selected)
    r.update(format=RECIPE, phase=phase, split=split, severity='full_5_25',
             matched_rtc=True, warm=1., dropout=None)
    return r


def _checked(wave, length):
    wave = np.asarray(wave, np.float32)
    if wave.ndim != 1 or len(wave) != length or not np.isfinite(wave).all():
        raise RuntimeError('Matched RTC changed full duration or produced nonfinite audio')
    return wave


def clean(wave, r, engines):
    """Independent simulated clean RTC fallback; never return raw Offline."""
    x = np.asarray(wave, np.float32)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Invalid Offline base for simulated clean fallback')
    gain = min(10**(r['input_gain_db']/20.), .97/max(float(np.abs(x).max()), 1e-12))
    return _checked(engines((x*gain).astype(np.float32), r), len(x))


def generate(wave, r, bank, engines, *, include_reference=True):
    """R(g*x) and R(g*(x+n)): same RTC recipe, independently reset state.

    A scene echo perturbs only the noisy input. It is a delayed-speech room
    simulation, not a claim to simulate a complete echo-cancellation system.
    """
    x = np.asarray(wave, np.float32)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Matched inputs require a finite full mono recording')
    if r.get('format') != RECIPE:
        raise ValueError('V3.19 requires its explicit matched scene recipe')
    category = r.get('category')
    hashes, sources, attempt = [], [], 0
    if category == 'echo':
        delay = min(r['echo']['delay_samples'], len(x)-1)
        mixed = x.copy()
        mixed[delay:] += r['echo']['gain']*x[:-delay]
        gain = min(10**(r['input_gain_db']/20.),
                   .97/max(float(np.abs(mixed).max()), float(np.abs(x).max()), 1e-12))
        a, b = (x*gain).astype(np.float32), (mixed*gain).astype(np.float32)
        meta = dict(common_gain=gain, noise_scale=0., pair_eligible=bool(np.max(np.abs(x)) > 1e-6),
            measured_input_snr_db=None, snr_scope='delayed-speech gain, not additive-noise SNR',
            echo_delay_samples=delay, echo_gain=r['echo']['gain'])
    else:
        rng = np.random.default_rng(r['seed'])
        for attempt in range(8):
            if category:
                if category not in CATEGORIES or bank.scenes is None:
                    raise ValueError('Scene bank is missing or target category is invalid')
                noise, hashes, sources = scene_wave(bank.scenes, category, len(x), rng)
            else:
                noise, hashes = noise_wave(bank, len(x), r['noise_type'], rng)
            a, b, meta = common_inputs(x, noise, r['snr_db'], r['input_gain_db'])
            if meta['pair_eligible']:
                break
        if not meta['pair_eligible']:
            # Silent speech may still participate in CE, but never gets an
            # invented positive auxiliary pair or an undisclosed noise retry.
            if np.max(np.abs(x)) > 1e-6:
                raise RuntimeError('No active speech/noise overlap after eight attempts')
    # Engines create fresh APM state per call. The same stress recipe, if any,
    # is applied to both inputs. No arbitrary second noise draw enters the pair.
    # Scene evaluation needs only the noisy endpoint. Avoid a second complete
    # RTC invocation when its clean output would be immediately discarded.
    ref = _checked(engines(a, r), len(x)) if include_reference else None
    noisy = _checked(engines(b, r), len(x))
    erased = []
    if r.get('dropout'):
        d = r['dropout']
        count = min(max(1, round(d['seconds']*16000)), max(1, len(noisy)//50))
        start = min(len(noisy)-count, max(0, round(d['position']*len(noisy))))
        noisy[start:start+count] *= d['gain']
        erased.append([start, start+count])
    meta.update(recipe=copy.deepcopy(r), noise_hashes=hashes, noise_original_sources=sources,
                noise_attempts=attempt+1, erased_spans=erased, output_samples=len(x),
                reference_samples=len(x), same_settings=True, shared_base=True,
                fresh_state_per_utterance=True)
    return ref, noisy, meta
