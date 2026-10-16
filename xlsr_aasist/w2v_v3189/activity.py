"""Deterministic pause/weak-boundary attenuation, not a platform or DTX emulator.

The detector sees only the processed mixture, never a clean source, phonemes,
class labels or evaluation recordings. Utterance quantiles initialize robust
level thresholds; this offline augmentation is not a causal speech VAD.
The envelope then has state, hysteresis, hangover and finite attack/release.
"""
import math
import numpy as np

SR, HOP, WINDOW = 16000, 160, 320
POLICY = 'energy_hysteresis_v1'
PROFILES = ('off', 'mild', 'strong')
PROBABILITIES = (.25, .45, .30)


def settings(seed, profile=None):
    rng = np.random.default_rng(seed)
    profile = str(rng.choice(PROFILES, p=PROBABILITIES)) if profile is None else profile
    if profile not in PROFILES:
        raise ValueError('Unknown RTC activity profile')
    strong = profile == 'strong'
    return dict(policy=POLICY, profile=profile,
        floor_db=float(rng.uniform(-45, -30) if strong else rng.uniform(-26, -16)),
        open_margin_db=float(rng.uniform(12, 16) if strong else rng.uniform(9, 12)),
        close_margin_db=float(rng.uniform(6, 9) if strong else rng.uniform(4, 7)),
        hold_ms=float(rng.uniform(10, 40) if strong else rng.uniform(30, 70)),
        attack_ms=float(rng.uniform(8, 25) if strong else rng.uniform(4, 12)),
        release_ms=float(rng.uniform(8, 25) if strong else rng.uniform(20, 45)),
        mask_db=12., codec_guard_samples=320, min_dynamic_db=8.)


def merge_spans(spans, length):
    result = []
    for a, b in sorted((max(0., float(a)), min(float(length), float(b))) for a, b in spans):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def _runs(mask, guard):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [[int(a), int(b)] for a, b in merge_spans(
        [(a-guard, b+guard) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))], len(mask))]


def process(wave, cfg):
    x = np.asarray(wave, np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Activity input must be a finite nonempty mono waveform')
    if cfg.get('policy') != POLICY or cfg.get('profile') not in PROFILES:
        raise ValueError('Invalid RTC activity recipe')
    for key in ('floor_db', 'open_margin_db', 'close_margin_db', 'hold_ms', 'attack_ms',
                'release_ms', 'mask_db', 'codec_guard_samples', 'min_dynamic_db'):
        if not math.isfinite(cfg[key]):
            raise ValueError('Nonfinite RTC activity parameter: ' + key)
    if not (-60 <= cfg['floor_db'] < 0 and 0 < cfg['close_margin_db'] < cfg['open_margin_db']
            and 0 <= cfg['hold_ms'] <= 250 and 0 < cfg['attack_ms'] <= 250
            and 0 < cfg['release_ms'] <= 250 and cfg['mask_db'] > 0
            and 0 <= cfg['codec_guard_samples'] <= 800 and cfg['min_dynamic_db'] > 0):
        raise ValueError('Out-of-range RTC activity parameters')
    meta = dict(policy=POLICY, profile=cfg['profile'], samples=len(x), enabled=False,
        suppressed_samples=0, attenuated_samples=0, min_gain_db=0., mean_gain_db=0.,
        suppressed_spans=[], gate_closures=0, gate_openings=0,
        mask_definition='Gain <= -12 dB, expanded by 20 ms for codec smearing; not proof of erasure')
    if cfg['profile'] == 'off' or len(x) < 4*HOP:
        return x, dict(meta, reason='off' if cfg['profile'] == 'off' else 'too_short')
    ends = np.minimum(np.arange(HOP, len(x)+HOP, HOP), len(x))
    starts = np.maximum(0, ends-WINDOW)
    power = np.r_[0., np.cumsum(x.astype(np.float64)**2)]
    levels = 10*np.log10(np.maximum((power[ends]-power[starts])/(ends-starts), 1e-14))
    low, high = np.percentile(levels, [10, 95])
    spread = high-low
    if high < -70 or spread < cfg['min_dynamic_db']:
        return x, dict(meta, reason='insufficient_activity_contrast', level_range_db=float(spread))
    # Never put the opening threshold above clearly voiced frames. A flat
    # recording passes unchanged instead of being indiscriminately muted.
    opening = low+min(cfg['open_margin_db'], .60*spread)
    closing = low+min(cfg['close_margin_db'], .25*spread)
    gains, current, opened, hangover = [], 0., True, 0
    hold = int(math.ceil(cfg['hold_ms']/10))
    for level in levels:
        if level >= opening:
            if not opened:
                meta['gate_openings'] += 1
            opened, hangover = True, hold
        elif level < closing:
            if hangover > 0:
                hangover -= 1
            elif opened:
                opened = False
                meta['gate_closures'] += 1
        # In the hysteresis band, retain the previous state and hangover.
        target = 0. if opened else cfg['floor_db']
        tau = cfg['attack_ms'] if target > current else cfg['release_ms']
        current += (target-current)*(1-math.exp(-10/tau))
        gains.append(current)
    envelope_db = np.interp(np.arange(len(x)), np.r_[0, ends], np.r_[0., gains]).astype(np.float32)
    y = (x*np.power(10., envelope_db/20)).astype(np.float32)
    suppressed = envelope_db <= -cfg['mask_db']
    meta.update(enabled=True, reason='applied', suppressed_samples=int(suppressed.sum()),
        attenuated_samples=int((envelope_db < -3).sum()), min_gain_db=float(envelope_db.min()),
        mean_gain_db=float(envelope_db.mean()), suppressed_spans=_runs(suppressed, int(cfg['codec_guard_samples'])),
        open_threshold_db=float(opening), close_threshold_db=float(closing), level_range_db=float(spread))
    return y, meta


def mask_pairs(pairs, left_spans, right_spans, guard=200):
    """Remove suppressed RF support from BOTH aligned ends, retaining subspans.

    Pairs use potentially different phone durations. Union exclusions in
    normalized phone time before projecting back, preserving correspondence.
    The 200-sample guard protects a 400-sample frontend receptive field.
    """
    result, affected = [], 0
    for a, b, c, d in pairs:
        if not (0 <= a < b and 0 <= c < d):
            raise ValueError('Invalid PCL pair geometry')
        blocked = []
        for start, end, spans in ((a, b, left_spans), (c, d, right_spans)):
            for x, y in spans:
                if y+guard > start and x-guard < end:
                    blocked.append(((x-guard-start)/(end-start), (y+guard-start)/(end-start)))
        merged = merge_spans(blocked, 1.)
        if not merged:
            result.append([a, b, c, d])
            continue
        affected += 1
        cursor = 0.
        for start, end in merged+[[1., 1.]]:
            if start > cursor:
                result.append([a+cursor*(b-a), a+start*(b-a), c+cursor*(d-c), c+start*(d-c)])
            cursor = max(cursor, end)
    return result, affected
