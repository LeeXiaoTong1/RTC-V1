"""Additive environmental noise for RTC-SDD experiment B (NumPy + SciPy).

Training only: original RawBoost -> extra environment noise -> original padding.
Noise probability and SNR are independent of the real/spoof label.
"""
import hashlib
import json
import os
from collections import OrderedDict
import threading
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd


def active_mask(audio, sample_rate):
    """20-ms energy mask, within 30 dB of the loudest frame; not an oracle VAD."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite mono audio")
    frame = max(1, int(sample_rate * 0.02))
    mask = np.zeros(len(x), dtype=bool)
    powers = [np.mean(x[i:i + frame] ** 2) for i in range(0, len(x), frame)]
    threshold = max(max(powers) * 1e-3, 1e-12)
    for j, power in enumerate(powers):
        if power >= threshold:
            mask[j * frame:(j + 1) * frame] = True
    return mask


def mix_at_snr(audio, noise, sample_rate, snr_db, speech_mask=None):
    """Measure powers on jointly active frames; do not over-amplify sparse events."""
    x = np.asarray(audio, dtype=np.float64)
    n = np.asarray(noise, dtype=np.float64)
    if x.shape != n.shape or not np.isfinite(n).all():
        raise ValueError("Speech and noise must have the same finite 1-D shape")
    mask = active_mask(x, sample_rate) if speech_mask is None else speech_mask.copy()
    if not mask.any():
        return x.astype(np.float32), {"applied": False, "reason": "silent_speech"}
    mask &= active_mask(n, sample_rate)
    if not mask.any():
        raise ValueError("Noise is silent on the active speech region")
    ps, pn = np.mean(x[mask] ** 2), np.mean(n[mask] ** 2)
    if pn < 1e-12:
        raise ValueError("Noise is silent on the active speech region")
    alpha = np.sqrt(ps / (pn * 10.0 ** (float(snr_db) / 10.0)))
    mixture = x + alpha * n
    peak = float(np.max(np.abs(mixture)))
    gain = min(1.0, 0.99 / max(peak, 1e-12))
    # Do not hard-clip: a common gain preserves the intended SNR.
    return (mixture * gain).astype(np.float32), {
        "applied": True, "snr_db": float(snr_db),
        "alpha": float(alpha), "gain": gain,
    }


def read_noise(path, target_sr):
    try:
        sr, data = wavfile.read(path, mmap=True)
    except ValueError:
        sr, data = wavfile.read(path, mmap=False)
    if np.issubdtype(data.dtype, np.unsignedinteger):
        midpoint = (np.iinfo(data.dtype).max + 1) / 2
        data = (data.astype(np.float32) - midpoint) / midpoint
    elif np.issubdtype(data.dtype, np.signedinteger):
        data = data.astype(np.float32) / float(-np.iinfo(data.dtype).min)
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all():
        raise ValueError("Invalid noise audio: " + str(path))
    if sr != target_sr:
        divisor = gcd(int(sr), int(target_sr))
        data = resample_poly(data, target_sr // divisor, sr // divisor)
    return data


class NoiseAugment:
    def __init__(self, manifest, probability=0.5, snr_min=10.0, snr_max=30.0):
        if not 0 <= probability <= 1 or not np.isfinite([snr_min, snr_max]).all() or snr_min > snr_max:
            raise ValueError("Invalid noise probability or SNR range")
        manifest = Path(manifest).expanduser().resolve()
        raw = manifest.read_bytes()
        self.records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if not self.records:
            raise ValueError("Empty noise manifest: " + str(manifest))
        for row in self.records:
            path = Path(row["path"]).expanduser()
            row["path"] = str((manifest.parent / path).resolve() if not path.is_absolute() else path)
            if not Path(row["path"]).is_file():
                raise FileNotFoundError(row["path"])
        self.probability, self.snr_min, self.snr_max = probability, snr_min, snr_max
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.cache_bytes = max(0, int(os.environ.get('RTC_NOISE_CACHE_MB', '128'))) * 1024**2
        self._cache, self._bytes, self._lock = OrderedDict(), 0, threading.Lock()
        self.intermittent_probability = 0.

    def __getstate__(self):
        state = vars(self).copy()
        state.pop('_lock')
        state['_cache'], state['_bytes'] = OrderedDict(), 0
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _read_noise(self, path, sample_rate):
        if not self.cache_bytes:
            return read_noise(path, sample_rate)
        stat = Path(path).stat()
        key = (path, sample_rate, stat.st_size, stat.st_mtime_ns)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        value = read_noise(path, sample_rate)
        if value.nbytes <= self.cache_bytes:
            value.setflags(write=False)
            with self._lock:
                if key not in self._cache:
                    while self._cache and self._bytes + value.nbytes > self.cache_bytes:
                        _, old = self._cache.popitem(last=False)
                        self._bytes -= old.nbytes
                    self._cache[key] = value
                    self._bytes += value.nbytes
        return value

    @classmethod
    def from_env(cls):
        path = os.environ.get("RTC_B_NOISE_MANIFEST", "")
        if not path:
            return None
        probability = float(os.environ.get("RTC_B_NOISE_PROB", "0.5"))
        if probability == 0:
            return None
        obj = cls(path, probability,
                  float(os.environ.get("RTC_B_SNR_MIN", "10")),
                  float(os.environ.get("RTC_B_SNR_MAX", "30")))
        splits = {r.get("split") for r in obj.records}
        if splits != {"train"}:
            raise ValueError("Training requires the train noise manifest, not dev")
        print("[RTC-B] " + json.dumps({
            "manifest": str(Path(path).resolve()), "sha256": obj.manifest_sha256,
            "files": len(obj.records), "probability": probability,
            "snr_db": [obj.snr_min, obj.snr_max], "order": "RawBoost -> environment",
            "snr_reference": "joint speech/noise active 20ms frames, -30dB thresholds",
        }), flush=True)
        return obj

    def __call__(self, audio, sample_rate, rng=None, snr_db=None, return_info=False):
        # np.random is seeded separately by PyTorch DataLoader in each worker.
        rng = np.random if rng is None else rng
        if self.probability == 0 or (self.probability < 1 and rng.random_sample() >= self.probability):
            result = (audio, {"applied": False, "reason": "probability"})
            return result if return_info else result[0]
        x = np.asarray(audio)
        speech_mask = active_mask(x, sample_rate)
        if not speech_mask.any():
            result = (audio, {"applied": False, "reason": "silent_speech"})
            return result if return_info else result[0]
        for _ in range(20):
            row = self.records[int(rng.randint(len(self.records)))]
            n = self._read_noise(row["path"], sample_rate)
            if len(n) >= len(x):
                start = int(rng.randint(len(n) - len(x) + 1))
                n = n[start:start + len(x)].copy()
            else:
                # Preserve a short event once instead of looping it periodically.
                start = int(rng.randint(len(x) - len(n) + 1))
                event = np.zeros(len(x), dtype=np.float32)
                event[start:start + len(n)] = n
                n = event
            event = None
            if self.intermittent_probability and rng.random_sample() < self.intermittent_probability:
                # A smooth amplitude envelope produces intermittent environmental
                # events from the same permitted non-speech recording.
                envelope = np.full(len(n), .05, dtype=np.float32)
                events = []
                for _ in range(int(rng.randint(1, 5))):
                    width = min(len(n), int(rng.uniform(.08, .7) * sample_rate))
                    offset = int(rng.randint(len(n) - width + 1))
                    envelope[offset:offset+width] = np.maximum(envelope[offset:offset+width],
                                                              np.hanning(width).astype(np.float32))
                    events.append([offset, width])
                n = n * envelope
                event = events
            level = float(rng.uniform(self.snr_min, self.snr_max)) if snr_db is None else float(snr_db)
            try:
                mixed, info = mix_at_snr(x, n, sample_rate, level, speech_mask=speech_mask)
            except ValueError as exc:
                if "Noise is silent" in str(exc):
                    continue
                raise
            info.update({"noise_path": row["path"], "offset": start})
            if event is not None:
                info['intermittent_events'] = event
            return (mixed, info) if return_info else mixed
        raise RuntimeError("20 silent noise draws; inspect or remove silent noise files")
