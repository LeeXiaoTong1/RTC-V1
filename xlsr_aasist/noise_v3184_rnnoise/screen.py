"""CPU-only noise screening and recording/session-disjoint split construction."""
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
FRAME = 512
POLICY = dict(vad_threshold=.35, speech_run_frames=4, speech_pad_samples=4000,
              min_noise_samples=48000, raw_clip_fraction=.01, min_noise_rms=1e-5,
              rate_duration_tolerance_seconds=1.5, session_gap_ms=30*60*1000,
              vad_batch_size=16)


def infer_rate(byte_count):
    """The archive has no sample-rate metadata. Infer ONLY near-minute uploads.

    The collection recorded up to a minute at 44.1 or 48 kHz. Short/ambiguous
    uploads are excluded; the chosen rate remains explicitly marked inferred.
    """
    if byte_count <= 0 or byte_count % 2:
        raise ValueError('Invalid PCM16 byte count')
    candidates = [(abs(byte_count / (2 * rate) - 60), rate) for rate in (44100, 48000)]
    close = [(err, rate) for err, rate in candidates if err <= POLICY['rate_duration_tolerance_seconds']]
    if len(close) != 1:
        raise ValueError('No unambiguous near-minute sample-rate inference')
    return close[0][1]


def decode_raw(data, rate):
    x = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.
    if len(x) < rate or not np.isfinite(x).all():
        raise ValueError('Invalid raw audio')
    clip_fraction = float(np.mean(np.abs(x) >= 32760/32768))
    if clip_fraction > POLICY['raw_clip_fraction']:
        raise ValueError('Excessive source clipping')
    divisor = math.gcd(rate, SAMPLE_RATE)
    y = resample_poly(x, SAMPLE_RATE//divisor, rate//divisor).astype(np.float32)
    # A small resampling overshoot must not wrap when writing PCM16.
    peak = float(np.max(np.abs(y)))
    if peak > .999:
        y *= .999 / peak
    return y, clip_fraction


class SpeechScreen:
    def __init__(self, model_path):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        self.model = torch.jit.load(str(model_path), map_location='cpu').eval()

    def probabilities(self, wave):
        return self.probabilities_batch([wave])[0]

    def probabilities_batch(self, waves):
        """Batch independent recordings, keeping each recurrent stream intact.

        Never batch consecutive chunks of one recording as independent items:
        that would discard the state used for speech detection.
        """
        if not 1 <= len(waves) <= POLICY['vad_batch_size']:
            raise ValueError('Invalid VAD preparation batch size')
        torch = self.torch
        self.model.reset_states()
        # Use a separate normalized copy for VAD, never for the training bank.
        frames = [math.ceil(len(wave)/FRAME) for wave in waves]
        signal = np.zeros((len(waves), max(frames)*FRAME), dtype=np.float32)
        for i, wave in enumerate(waves):
            peak = max(float(np.max(np.abs(wave))), 1e-12)
            signal[i,:len(wave)] = np.asarray(wave, np.float32) * (.9 / peak)
        values = np.zeros((len(waves), max(frames)), dtype=np.float32)
        with torch.inference_mode():
            for frame in range(max(frames)):
                chunk = signal[:,frame*FRAME:(frame+1)*FRAME].copy()
                values[:,frame] = self.model(torch.from_numpy(chunk), SAMPLE_RATE).reshape(-1).numpy()
        return [values[i,:n].copy() for i,n in enumerate(frames)]


def runs(mask):
    """Half-open intervals for True runs, including runs at either endpoint."""
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def safe_intervals(probabilities, samples):
    probs = np.asarray(probabilities)
    if probs.ndim != 1 or len(probs) != math.ceil(samples/FRAME) or not np.isfinite(probs).all():
        raise ValueError('VAD output does not cover the recording')
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError('Invalid VAD probability')
    blocked = []
    for first, stop in runs(probs >= POLICY['vad_threshold']):
        if stop-first < POLICY['speech_run_frames']:
            continue
        lo = max(0, first*FRAME-POLICY['speech_pad_samples'])
        hi = min(samples, stop*FRAME+POLICY['speech_pad_samples'])
        if blocked and lo <= blocked[-1][1]:
            blocked[-1][1] = max(hi, blocked[-1][1])
        else:
            blocked.append([lo, hi])
    safe = []
    cursor = 0
    for lo, hi in blocked + [[samples, samples]]:
        if lo-cursor >= POLICY['min_noise_samples']:
            safe.append([cursor, lo])
        cursor = hi
    return safe, sum(hi-lo for lo, hi in blocked) / samples


def assign_splits(rows):
    """Union adjacent same-browser uploads and exact duplicates before splitting.

    Browser/session grouping is conservative and heuristic: the archive does
    not identify recording rooms or donors. It is not donor-disjoint evidence.
    """
    names = sorted({r['recording'] for r in rows})
    parent = {name: name for name in names}
    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a,b)] = min(a,b)
    info = {r['recording']: r for r in rows}
    browsers = defaultdict(list)
    for name in names:
        row = info[name]
        browsers[row['user_agent_sha256']].append((row['timestamp_ms'], name))
    for group in browsers.values():
        group.sort()
        for (ta, a), (tb, b) in zip(group, group[1:]):
            if tb-ta <= POLICY['session_gap_ms']:
                union(a, b)
    for field in ('source_sha256', 'pcm_sha256'):
        seen = {}
        for row in rows:
            digest = row[field]
            if digest in seen:
                union(row['recording'], seen[digest])
            else:
                seen[digest] = row['recording']
    groups = defaultdict(set)
    for row in rows:
        groups[find(row['recording'])].add(row['category'])
    rank = lambda name: hashlib.sha256(('rtc-rnnoise-split-v1:'+name).encode()).hexdigest()
    dev = set()
    for category in ('office', 'coffee'):
        available = sorted((g for g, cats in groups.items() if category in cats), key=rank)
        if len(available) < 5:
            raise ValueError('Too few independent usable groups for '+category)
        dev.update(available[:max(1, math.ceil(len(available)*.2))])
    for row in rows:
        group = find(row['recording'])
        row['source_group'] = 'rnnoise-session:'+group
        row['split'] = 'dev' if group in dev else 'train'
        row['original_sources'] = ['rnnoise-recording:'+row['recording'], row['source_group']]
    for category in ('office', 'coffee'):
        for split in ('train', 'dev'):
            if not any(r['category'] == category and r['split'] == split for r in rows):
                raise ValueError('Session isolation leaves an empty '+category+'/'+split)
    return rows
