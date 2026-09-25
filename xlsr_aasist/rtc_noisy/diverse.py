"""Offline algorithm diversity. WebRTC is real local DSP, not a platform upload.

Binding API: github.com/xiongyihui/python-webrtc-audio-processing.
No external speech, resynthesis, or label-conditioned augmentation is used.
"""
import importlib.metadata
import numpy as np
from scipy.signal import fftconvolve
from .simulator import LocalRTC

SCHEMA = 'rtc_diverse_processing_v1'
PROFILES = {'legacy': ('ffmpeg',), 'diverse': ('ffmpeg', 'webrtc'), 'unseen': ('anlmdn',)}


def profile_definition(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown processing profile')
    return {'schema': SCHEMA, 'profile': profile, 'families': list(PROFILES[profile]),
            'room_probability': .5, 'intermittent_probability': .5,
            'room': 'causal sparse reflections plus decaying synthetic impulse response; same length',
            'snr_reference': 'active-frame SNR before room and RTC processing'}


def room_response(waveform, rng):
    """A mild synthetic room impulse response, not a claim to reproduce real RTC."""
    x = np.asarray(waveform, dtype=np.float32)
    if rng.random_sample() >= .5:
        return x, {'applied': False}
    rt60 = float(rng.uniform(.08, .4))
    length = max(2, int(rt60 * 16000))
    t = np.arange(length) / 16000.
    impulse = rng.normal(size=length) * np.exp(-6.91 * t / rt60)
    impulse *= .025 / max(np.linalg.norm(impulse), 1e-12)
    impulse[0] = 1.
    reflections = []
    for _ in range(int(rng.randint(2, 6))):
        delay = int(rng.randint(160, min(length, 2400)))
        gain = float(rng.uniform(.04, .18) * np.exp(-3 * delay / length))
        impulse[delay] += gain
        reflections.append([delay, gain])
    y = fftconvolve(x, impulse, mode='full')[:len(x)]
    gain = min(1., .99 / max(float(np.abs(y).max()), 1e-12))
    return np.asarray(y * gain, dtype=np.float32), {'applied': True, 'rt60': rt60, 'reflections': reflections, 'gain': gain}


def webrtc_process(waveform, ns_level, target_dbfs):
    try:
        from webrtc_audio_processing import AudioProcessingModule
    except ImportError as exc:
        raise RuntimeError('Diverse cache requires webrtc-audio-processing==0.1.3; see README_W2V_IMPROVEMENTS.md') from exc
    # Fresh state per utterance; never share a mutable processor between threads.
    ap = AudioProcessingModule(enable_ns=True, agc_type=1, enable_vad=False)
    ap.set_stream_format(16000, 1)
    ap.set_ns_level(int(ns_level))
    ap.set_agc_target(int(target_dbfs))
    x = np.asarray(waveform, dtype=np.float32)
    gain = min(1., .99 / max(float(np.abs(x).max()), 1e-12))
    pcm = np.round(x * gain * 32767.).astype('<i2')
    output = []
    for start in range(0, len(pcm), 160):
        frame = pcm[start:start+160]
        padded = np.pad(frame, (0, 160-len(frame)))
        result = np.frombuffer(ap.process_stream(padded.tobytes()), dtype='<i2')
        if len(result) != 160:
            raise RuntimeError('WebRTC must return exactly one 10 ms mono PCM16 frame')
        output.append(result[:len(frame)].astype(np.float32) / 32768.)
    return np.concatenate(output)


class DiverseRTC(LocalRTC):
    def __init__(self, ffmpeg='ffmpeg', profile='diverse'):
        super().__init__(ffmpeg)
        self.profile = profile
        self.definition = profile_definition(profile)
        self.webrtc_version = None
        if 'webrtc' in PROFILES[profile]:
            try:
                self.webrtc_version = importlib.metadata.version('webrtc-audio-processing')
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError('Install requirements_rtc_diverse.txt before generating diverse caches') from exc
            if self.webrtc_version != '0.1.3':
                raise RuntimeError('Expected webrtc-audio-processing==0.1.3; do not silently change DSP versions')

    def __call__(self, waveform, settings, rng, family=None):
        family = family or str(rng.choice(PROFILES[self.profile]))
        if family not in PROFILES[self.profile]:
            raise ValueError('Processing family is not allowed in this profile')
        x, room = room_response(waveform, rng)
        ns = int(rng.randint(4))
        target = int(rng.choice([6, 12, 18]))
        if family == 'ffmpeg':
            result = super().__call__(x, settings)
        else:
            if family == 'webrtc':
                x = webrtc_process(x, ns, target)
                filters = None
            else:
                filters = 'anlmdn=s=0.0001:p=0.002:r=0.006,dynaudnorm=f=150:g=7:p=0.9:m=4'
            arguments = ['-f', 'f32le', '-ar', '16000', '-ac', '1', '-i', 'pipe:0']
            if filters:
                arguments += ['-af', filters]
            encoded = self._run(arguments + ['-c:a', 'libopus', '-b:a', str(settings.bitrate),
                                '-application', 'voip', '-frame_duration', '20', '-vbr', 'on', '-f', 'ogg', 'pipe:1'],
                                np.asarray(x, dtype='<f4').tobytes())
            decoded = self._run(['-f', 'ogg', '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-f', 'f32le', 'pipe:1'], encoded)
            result = np.frombuffer(decoded, dtype='<f4').copy()
        if len(result) != len(waveform) or not np.isfinite(result).all():
            raise RuntimeError('Diverse RTC output length/nonfinite error')
        return result, {'family': family, 'room': room,
                        'ns_level': ns if family == 'webrtc' else None,
                        'agc_target_dbfs': target if family == 'webrtc' else None}
