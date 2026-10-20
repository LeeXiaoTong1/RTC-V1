"""Bounded, local receiver postprocessors; never claimed to emulate a platform.

Every call creates fresh signal-conditioned state. Parameters and packet masks
are shared by the two views, while their processing state is independent.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy import signal

TRAIN_MECHANISMS = ('spectral_color', 'packet_repeat', 'rnnoise')
HELDOUT_MECHANISMS = ('spectral_wiener', 'color_packet')
IMPLEMENTATIONS = {
    'baseline': 'v3189_legacy_receiver_v1',
    'spectral_color': 'fir65_zero_phase_equalizer_v1',
    'packet_repeat': 'bounded_20ms_repeat_fade_v1',
    'rnnoise': 'ffmpeg_arnndn_pinned_model_v1',
    'spectral_wiener': 'scipy_stft_minimum_quantile_wiener_v1',
    'color_packet': 'fir65_then_packet_repeat_v1',
}
SR = 16000


def _wave(wave):
    value = np.asarray(wave, np.float32)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError('Receiver processing requires finite nonempty mono PCM')
    return value


def mechanism_catalog(cfg=None):
    cfg = cfg or {}
    train = tuple(cfg.get('robust_train_mechanisms', TRAIN_MECHANISMS))
    heldout = tuple(cfg.get('robust_eval_mechanisms', HELDOUT_MECHANISMS))
    if len(set(train)) != len(train) or any(name not in TRAIN_MECHANISMS for name in train):
        raise ValueError('Training receiver mechanisms must be unique declared training implementations')
    if len(set(heldout)) != len(heldout) or any(name not in HELDOUT_MECHANISMS for name in heldout):
        raise ValueError('Heldout receiver mechanisms must use the declared heldout implementations')
    if set(train) & set(heldout):
        raise ValueError('A heldout implementation cannot occur in training')
    return dict(train=list(train), heldout=list(heldout),
        implementations={name: IMPLEMENTATIONS[name] for name in ('baseline', *train, *heldout)},
        scope='Postprocessors of inherited RTC outputs; not real platform/network reproduction',
        heldout_contract={
            'spectral_wiener': 'Distinct non-neural STFT gain estimator, never used in Train',
            'color_packet': 'Composition of two Train implementations; composition never used in Train'},
        neural_model_sha256=cfg.get('rnnoise_sha256'))


@lru_cache(maxsize=8)
def _verified_model(path_text, expected, size, mtime_ns):
    path = Path(path_text)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', path.name):
        raise ValueError('RNNoise model filename must contain only letters, digits, dot, dash, underscore')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not re.fullmatch(r'[0-9a-f]{64}', str(expected).lower()) or digest != str(expected).lower():
        raise ValueError('RNNoise model does not match its pinned SHA256')
    return dict(path=str(path), sha256=digest, bytes=size, mtime_ns=mtime_ns)


def model_provenance(cfg):
    if not cfg.get('rnnoise_model') or not cfg.get('rnnoise_sha256'):
        raise ValueError('RNNoise requires an explicit pinned model path and SHA256; no fallback is allowed')
    # Reading a pinned file needs an absolute path, not permission to resolve
    # every ancestor handle (restricted Windows runtimes may deny the latter).
    path = Path(cfg['rnnoise_model']).absolute()
    stat = path.stat()
    return _verified_model(str(path), cfg['rnnoise_sha256'], stat.st_size, stat.st_mtime_ns)


def runtime_preflight(cfg):
    """Validate neural asset/filter by actually processing PCM, before training."""
    catalog = mechanism_catalog(cfg)
    provenance = dict(catalog=catalog)
    if 'rnnoise' in catalog['train']:
        provenance['rnnoise'] = model_provenance(cfg)
        result = subprocess.run([cfg['ffmpeg'], '-hide_banner', '-version'],
            check=True, capture_output=True, text=True, timeout=20)
        provenance['ffmpeg_version'] = result.stdout.splitlines()[0]
        x = (.02*np.random.default_rng(3191).standard_normal(SR)).astype(np.float32)
        y, meta = apply_view(x, 'rnnoise', 3191, cfg)
        if y.shape != x.shape or not np.isfinite(y).all():
            raise RuntimeError('RNNoise preflight produced invalid PCM')
        if not meta['alignment']['valid']:
            raise RuntimeError('RNNoise startup delay could not be verified; refusing unknown receiver timing')
        provenance['rnnoise_smoke'] = dict(samples=len(y), alignment=meta['alignment'])
    return provenance


def _parameters(name, seed):
    rng = np.random.default_rng(int(seed))
    if name == 'spectral_color':
        return dict(gains_db=rng.uniform(-5., 5., 7).tolist(), taps=65,
                    frequencies=[0., 80., 300., 1000., 3000., 6000., 8000.])
    if name == 'packet_repeat':
        return dict(loss_fraction=float(rng.uniform(.01, .05)),
                    max_burst_packets=int(rng.integers(1, 4)),
                    packet_samples=320, fade_samples=80, mask_seed=int(rng.integers(0, 2**31)))
    if name == 'spectral_wiener':
        return dict(noise_quantile=.15, strength=float(rng.uniform(.8, 1.4)),
                    gain_floor=float(rng.uniform(.15, .3)), fft_samples=512, hop_samples=128)
    if name == 'rnnoise':
        return dict(mix=1., model_rate=48000)
    if name == 'color_packet':
        return dict(color=_parameters('spectral_color', int(rng.integers(0, 2**31))),
                    packet=_parameters('packet_repeat', int(rng.integers(0, 2**31))))
    if name == 'baseline':
        return {}
    raise ValueError('Unknown receiver mechanism: '+str(name))


def _color(x, parameters):
    taps = signal.firwin2(parameters['taps'], parameters['frequencies'],
        np.power(10., np.asarray(parameters['gains_db'])/20.), fs=SR)
    output = signal.fftconvolve(x, taps, mode='same').astype(np.float32)
    guard = parameters['taps']//2
    return output, [[0, min(guard, len(x))], [max(0, len(x)-guard), len(x)]]


def _packet(x, parameters):
    size = parameters['packet_samples']
    count = int(np.ceil(len(x)/size))
    output = x.copy()
    if count < 4:
        return output, []
    rng = np.random.default_rng(parameters['mask_seed'])
    # Keep the first and last packets intact; never exceed a 5% budget rounded
    # down. Very short utterances consequently have no synthetic packet loss.
    budget = int(count*parameters['loss_fraction'])
    lost = set()
    for start in rng.permutation(np.arange(1, count-1)):
        if len(lost) >= budget:
            break
        burst = int(rng.integers(1, parameters['max_burst_packets']+1))
        for index in range(int(start), min(int(start)+burst, count-1)):
            if len(lost) < budget:
                lost.add(index)
    spans = []
    previous = x[:size].copy()
    run = 0
    for index in range(1, count):
        start, end = index*size, min((index+1)*size, len(x))
        if index not in lost:
            previous = x[start:end].copy()
            run = 0
            continue
        run += 1
        replaced = previous[:end-start]*(.85**run)
        fade = min(parameters['fade_samples'], len(replaced)//2)
        if fade:
            # Local crossfades suppress artificial clicks, without moving time.
            weight = np.ones(len(replaced), np.float32)
            weight[:fade] = np.linspace(0., 1., fade)
            weight[-fade:] = np.linspace(1., 0., fade)
            replaced = replaced*weight+x[start:end]*(1.-weight)
        output[start:end] = replaced
        spans.append([max(0, start-320), min(len(x), end+320)])
    return output, spans


def _wiener(x, parameters):
    # A heldout offline, signal-conditioned gain estimator; it is not claimed
    # to match a causal receiver. Padding supports even very short recordings.
    nfft, hop = parameters['fft_samples'], parameters['hop_samples']
    padded = np.pad(x, (0, max(0, nfft-len(x))))
    _, _, z = signal.stft(padded, fs=SR, window='hann', nperseg=nfft,
        noverlap=nfft-hop, boundary='zeros', padded=True)
    power = np.abs(z)**2
    floor = np.quantile(power, parameters['noise_quantile'], axis=1, keepdims=True)
    gain = np.maximum(1.-parameters['strength']*floor/np.maximum(power, 1e-14),
                      parameters['gain_floor'])
    _, output = signal.istft(z*gain, fs=SR, window='hann', nperseg=nfft,
        noverlap=nfft-hop, boundary=True)
    return output[:len(x)].astype(np.float32), []


def _neural(x, parameters, cfg):
    provenance = model_provenance(cfg)
    path = Path(provenance['path'])
    # arnndn operates at 48 kHz. Independent ffmpeg processes reset all filter
    # state. Extra tail is for filter flush; measured delay is handled below.
    tail = np.zeros(1600, np.float32)
    filters = f"aresample=48000,arnndn=m={path.name}:mix=1,aresample=16000"
    command = [cfg['ffmpeg'], '-hide_banner', '-loglevel', 'error', '-nostdin',
        '-threads', '1', '-filter_threads', '1', '-f', 'f32le', '-ar', str(SR),
        '-ac', '1', '-i', 'pipe:0', '-af', filters, '-f', 'f32le', '-ar', str(SR),
        '-ac', '1', 'pipe:1']
    result = subprocess.run(command, input=np.r_[x, tail].astype('<f4').tobytes(),
        capture_output=True, cwd=str(path.parent), timeout=cfg.get('robust_processor_timeout', 120))
    if result.returncode:
        raise RuntimeError('RNNoise processing failed (no fallback): '+result.stderr.decode(errors='replace')[-1500:])
    output = np.frombuffer(result.stdout, '<f4').copy()
    if len(output) < len(x):
        raise RuntimeError('RNNoise unexpectedly truncated its input')
    return output, provenance


def _suppressed(before, after):
    """Conservative energy-loss intervals, not a claim of perceptual validity."""
    hop = 160
    starts = np.arange(0, len(before), hop)
    ends = np.minimum(starts+320, len(before))
    power_before = np.r_[0., np.cumsum(before.astype(np.float64)**2)]
    power_after = np.r_[0., np.cumsum(after.astype(np.float64)**2)]
    a = (power_before[ends]-power_before[starts])/(ends-starts)
    b = (power_after[ends]-power_after[starts])/(ends-starts)
    active = a > max(float(np.max(a))*.01, 1e-9)
    return [[max(0, int(start)-200), min(len(before), int(end)+200)]
        for start, end, drop in zip(starts, ends, active & (b < a*.0631)) if drop]


def _apply(x, name, parameters, cfg):
    provenance = {}
    alignment = dict(valid=True, offset=0., compensated=False, policy='sample_preserving')
    if name == 'baseline':
        y, spans = x.copy(), []
    elif name == 'spectral_color':
        y, spans = _color(x, parameters)
    elif name == 'packet_repeat':
        y, spans = _packet(x, parameters)
    elif name == 'spectral_wiener':
        y, spans = _wiener(x, parameters)
    elif name == 'color_packet':
        y, first = _color(x, parameters['color'])
        y, second = _packet(y, parameters['packet'])
        spans = first+second
    elif name == 'rnnoise':
        y, provenance = _neural(x, parameters, cfg)
        from w2v_v3186.augment import estimate_delay
        alignment = estimate_delay(x, y[:len(x)], max_shift=800)
        alignment.update(compensated=False, policy='measured_neural_delay')
        shift = int(round(alignment['offset'])) if alignment['valid'] else 0
        if alignment['valid'] and shift >= 0:
            y = y[shift:shift+len(x)]
            if len(y) < len(x):
                y = np.pad(y, (0, len(x)-len(y)))
            spans = [[max(0, len(x)-abs(shift)-400), len(x)]]
            alignment['compensated'] = True
        elif alignment['valid']:
            y = np.r_[np.zeros(-shift, np.float32), y][:len(x)]
            spans = [[0, min(len(x), -shift+400)]]
            alignment['compensated'] = True
        else:
            # A weak utterance correlation must not cause the known filter
            # startup delay to clip the final speech. Use the same asset/runtime
            # preflight measurement for PCM only; PCL remains entirely masked.
            calibrated = cfg.get('robust_runtime', {}).get('rnnoise_smoke', {}).get('alignment', {})
            fallback = int(round(calibrated.get('offset', 0.))) if calibrated.get('valid') else 0
            if fallback > 0:
                y = y[fallback:fallback+len(x)]
                alignment.update(compensated=True, startup_delay_fallback=fallback)
            elif fallback < 0:
                y = np.r_[np.zeros(-fallback, np.float32), y][:len(x)]
                alignment.update(compensated=True, startup_delay_fallback=fallback)
            else:
                y = y[:len(x)]
            spans = [[0, len(x)]]  # Never reuse an unverified PCL alignment.
    else:
        raise ValueError('Unknown receiver mechanism')
    y = np.asarray(y, np.float32)
    if y.shape != x.shape or not np.isfinite(y).all():
        raise RuntimeError('Receiver processor changed geometry or produced nonfinite PCM: '+name)
    spans += _suppressed(x, y)
    return y, dict(affected_spans=spans, alignment=alignment, provenance=provenance)


def apply_view(wave, mechanism, seed, cfg=None):
    cfg = cfg or {}
    x = _wave(wave)
    parameters = _parameters(mechanism, seed)
    y, details = _apply(x, mechanism, parameters, cfg)
    return y, dict(mechanism=mechanism, implementation=IMPLEMENTATIONS[mechanism],
        seed=int(seed), parameters=parameters, fresh_state_per_view=True,
        composition='after inherited RTC', **details)


def quality_check(reference, clean, noisy, cfg=None):
    """Necessary bounded-signal checks; does not certify preservation of spoof cues."""
    cfg = cfg or {}
    reference = np.asarray(reference)
    reasons = []
    views = (np.asarray(clean), np.asarray(noisy))
    if any(v.shape != reference.shape for v in views):
        return dict(passed=False, eligible=False, reason='length', reasons=['length'])
    if not np.isfinite(reference).all() or any(not np.isfinite(v).all() for v in views):
        return dict(passed=False, eligible=False, reason='nonfinite', reasons=['nonfinite'])
    hop = 320
    def energy(x):
        return np.array([np.mean(x[a:a+hop].astype(np.float64)**2)
                         for a in range(0, len(x), hop)])
    ref_energy = energy(reference)
    active = ref_energy > max(float(ref_energy.max())*.01, 1e-9)
    if not active.any():
        reasons.append('no_reference_activity')
    retention, clipping, gain = [], [], []
    for wave in views:
        power = energy(wave)
        fraction = float(np.mean(power[active] >= ref_energy[active]*.01)) if active.any() else 0.
        clip = float(np.mean(np.abs(wave) >= .999))
        ratio = float(10.*np.log10(max(float(np.mean(wave.astype(np.float64)**2)), 1e-14)/
            max(float(np.mean(reference.astype(np.float64)**2)), 1e-14)))
        retention.append(fraction); clipping.append(clip); gain.append(ratio)
    if min(retention) < cfg.get('robust_min_active_retention', .8):
        reasons.append('active_energy_lost')
    if max(clipping) > cfg.get('robust_max_clipped_fraction', .005):
        reasons.append('clipping')
    if max(gain) > cfg.get('robust_max_gain_db', 12.):
        reasons.append('excess_gain')
    return dict(passed=not reasons, eligible=not reasons,
        reason='ok' if not reasons else ','.join(reasons), reasons=reasons,
        active_retention=retention, clipped_fraction=clipping, rms_gain_db=gain,
        scope='Signal validity only; cannot prove preservation of spoof evidence')


def apply_pair(clean, noisy, mechanism, seed, cfg=None):
    cfg = cfg or {}
    clean, noisy = _wave(clean), _wave(noisy)
    if clean.shape != noisy.shape:
        raise ValueError('Matched receiver views must have the same length')
    parameters = _parameters(mechanism, seed)
    clean_out, clean_meta = _apply(clean, mechanism, parameters, cfg)
    noisy_out, noisy_meta = _apply(noisy, mechanism, parameters, cfg)
    return clean_out, noisy_out, dict(mechanism=mechanism,
        implementation=IMPLEMENTATIONS[mechanism], seed=int(seed), parameters=parameters,
        shared_parameters=True, fresh_state_per_view=True,
        composition='after inherited RTC; independent state for matched clean/noisy',
        clean=clean_meta, noisy=noisy_meta,
        quality=quality_check(clean, clean_out, noisy_out, cfg))
