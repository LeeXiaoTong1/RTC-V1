"""Noise BEFORE matched RTC. Fresh APM state per full utterance, never per frame."""
from collections import OrderedDict
import hashlib
from importlib.metadata import version
import itertools
import json
import os
from pathlib import Path

import numpy as np

from rtc_noisy.simulator import LocalRTC, RTCSettings
from utils.env_noise import active_mask, read_noise
from w2v_v39.common import digest

SR = 16000
RECIPE = 'rtc_v315_matched_full_wave_v1'
FAMILIES = ('ffmpeg', 'ffmpeg', 'webrtc', 'webrtc', 'light')
NOISE_TYPES = ('continuous', 'events', 'mixed')


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:8], 'little')


def runtime(ffmpeg=None):
    rtc = LocalRTC(ffmpeg or os.environ.get('FFMPEG_BIN') or 'ffmpeg')
    installed = version('webrtc-audio-processing')
    if installed != '0.1.3':
        raise RuntimeError('V3.15 requires the existing webrtc-audio-processing==0.1.3')
    return dict(ffmpeg=rtc.version, ffmpeg_path=rtc.ffmpeg, ffmpeg_sha256=digest(rtc.ffmpeg),
                webrtc_audio_processing=installed, recipe=RECIPE)


def _find(cfg, key):
    for value in (cfg, cfg.get('source_config', {}), cfg.get('dev_config', {})):
        if value.get(key):
            return str(Path(value[key]).expanduser().resolve())
    raise ValueError('Recorded source configuration lacks '+key)


def bind_augmentation(cfg):
    from rtc_noisy.common import noise_catalog, assert_noise_disjoint
    catalogs, records, files = {}, {}, {}
    for split, key in (('train', 'train_noise_manifest'), ('dev', 'dev_noise_manifest')):
        path = _find(cfg, key)
        augmenter, catalogs[split] = noise_catalog(path, split)
        records[split] = []
        files[path] = digest(path)
        for row in augmenter.records:
            p = Path(row['path']); stat = p.stat()
            sha = digest(p)
            records[split].append(dict(path=str(p), sha256=sha, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, original_recording=row['original_recording'], split=split))
            files[str(p)] = sha
    assert_noise_disjoint({'noise':catalogs['train']}, {'noise':catalogs['dev']})
    return dict(augmentation_runtime=runtime(cfg.get('ffmpeg')), augmentation_files=files,
                noise_records=records, noise_catalogs=catalogs)


def settings_grid(family, split):
    if family == 'ffmpeg':
        values = [dict(noise_reduction=nr, max_gain=g, bitrate=b)
                  for nr, g, b in itertools.product((6,12,18), (2,4,8), (16000,24000,32000))]
    elif family == 'webrtc':
        values = [dict(ns_level=n, agc_target_dbfs=g, bitrate=b)
                  for n, g, b in itertools.product(range(4), (6,12,18), (16000,24000,32000))]
    elif family == 'light':
        values = [dict(cutoff=h, bitrate=b) for h,b in itertools.product((4800,6000,7200),(16000,24000,32000))]
    else:
        raise ValueError('Unknown processing family')
    if split not in ('train', 'dev'):
        raise ValueError('Only official Train/Dev recipes are allowed')
    # Disjoint complete settings, not a claim that an engine was unseen by the base.
    return [v for i,v in enumerate(values) if (i % 4 == 0) == (split == 'dev')]


def recipe(seed, occurrence, phase, source_sha, split='train', warm=1.):
    rng = np.random.default_rng(seed_for(seed, occurrence, source_sha, split, RECIPE))
    family, kind = FAMILIES[phase % 5], NOISE_TYPES[(phase // 5) % 3]
    severity = 'moderate' if phase % 2 == 0 else 'strong'
    grid = settings_grid(family, split)
    lo, hi = (15.,30.) if severity == 'moderate' else (5.,15.)
    if split == 'train':
        lo += (1.-warm)*5.; hi += (1.-warm)*5.
    # Same room-like early reflection and input gain on both sides.
    echo = dict(delay_samples=int(rng.integers(320,1601)), gain=float(rng.uniform(.05,.18))) if rng.random()<.15 else None
    dropout = None
    if rng.random() < .08:
        dropout = dict(position=float(rng.uniform(.1,.9)), seconds=float(rng.uniform(.02,.10)),
                       gain=0. if rng.random()<.35 else float(rng.uniform(.03,.25)))
    return dict(format=RECIPE, split=split, seed=seed_for(seed,occurrence,source_sha,'noise'),
        family=family, noise_type=kind, severity=severity, snr_db=float(rng.uniform(lo,hi)),
        settings=grid[int(rng.integers(len(grid)))], input_gain_db=float(rng.uniform(-3.,3.)),
        echo=echo, dropout=dropout)


class NoiseBank:
    def __init__(self, records, cap_bytes=96*1024**2):
        if not records:
            raise ValueError('No non-speech noise recordings')
        self.records, self.cap = records, cap_bytes
        self.cache, self.used = OrderedDict(), 0

    def sample(self, rng):
        row = self.records[int(rng.integers(len(self.records)))]
        p = Path(row['path']); stat = p.stat()
        if stat.st_size != row['size'] or stat.st_mtime_ns != row['mtime_ns']:
            raise ValueError('Noise recording changed during training')
        key = row['sha256']
        if key not in self.cache:
            x = read_noise(p, SR)
            if x.nbytes <= self.cap:
                while self.cache and self.used+x.nbytes > self.cap:
                    _, old = self.cache.popitem(last=False); self.used -= old.nbytes
                self.cache[key] = x; self.used += x.nbytes
        else:
            x = self.cache[key]; self.cache.move_to_end(key)
        return x, key


def noise_wave(bank, length, kind, rng):
    used = set()

    def continuous():
        result = np.zeros(length, np.float32)
        cursor = 0
        # Random excerpts with small crossfades, not periodic tiling of a short file.
        while cursor < length:
            x, key = bank.sample(rng); used.add(key)
            count = min(len(x), length-cursor+min(cursor,320), int(rng.integers(SR,4*SR+1)))
            start = int(rng.integers(len(x)-count+1))
            piece = x[start:start+count].copy()
            fade = min(320, cursor, count//4)
            if fade:
                w = np.linspace(0,1,fade,dtype=np.float32)
                result[cursor-fade:cursor] = result[cursor-fade:cursor]*(1-w)+piece[:fade]*w
            remaining = min(count-fade, length-cursor)
            result[cursor:cursor+remaining] = piece[fade:fade+remaining]
            cursor += remaining
        return result

    def events():
        result = np.zeros(length, np.float32)
        for _ in range(int(rng.integers(1,5))):
            x,key = bank.sample(rng); used.add(key)
            count = min(len(x), length, int(rng.integers(800,24001)))
            a = int(rng.integers(len(x)-count+1)); b = int(rng.integers(length-count+1))
            piece = x[a:a+count].copy()
            fade = min(160,count//4)
            if fade:
                piece[:fade] *= np.linspace(0,1,fade); piece[-fade:] *= np.linspace(1,0,fade)
            result[b:b+count] += piece
        return result

    if kind == 'continuous':
        result = continuous()
    elif kind == 'events':
        result = events()
    elif kind == 'mixed':
        a,b = continuous(),events()
        # Equal component RMS on their own support; preserves transient structure.
        a /= max(float(np.sqrt(np.mean(a*a))), 1e-6)
        support = np.abs(b)>1e-7
        b /= max(float(np.sqrt(np.mean(b[support]**2))) if support.any() else 0.,1e-6)
        result = .5*a+.5*b
    else:
        raise ValueError('Unknown noise type')
    return result, sorted(used)


def common_inputs(wave, noise, snr_db, input_gain_db=0., echo=None):
    x,n = np.asarray(wave,dtype=np.float32),np.asarray(noise,dtype=np.float32)
    if x.shape != n.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(n).all():
        raise ValueError('Matched inputs require equally long finite waveforms')
    if echo:
        delay, gain = echo['delay_samples'], echo['gain']
        # Preserve the full original plus reflection tail, identically in both arms.
        def reflect(y):
            out = np.pad(y,(0,delay)); out[delay:delay+len(y)] += gain*y
            return out
        x,n = reflect(x),reflect(n)
    active = active_mask(x,SR) & active_mask(n,SR)
    ps = float(np.mean(x[active].astype(np.float64)**2)) if active.any() else 0.
    pn = float(np.mean(n[active].astype(np.float64)**2)) if active.any() else 0.
    valid = ps > 1e-12 and pn > 1e-12
    scale = float(np.sqrt(ps/(pn*10**(snr_db/10)))) if valid else 0.
    y = x+scale*n
    gain = min(10**(input_gain_db/20), .97/max(float(np.abs(x).max()),float(np.abs(y).max()),1e-12))
    return (x*gain).astype(np.float32),(y*gain).astype(np.float32),dict(
        common_gain=gain, noise_scale=scale, pair_eligible=valid,
        measured_input_snr_db=10*np.log10(ps/(scale*scale*pn)) if valid else None,
        snr_scope='joint active speech/noise samples; event-local for sparse noise')


class Engines:
    def __init__(self, ffmpeg=None):
        self.local = LocalRTC(ffmpeg or os.environ.get('FFMPEG_BIN') or 'ffmpeg')
        self.webrtc = None

    def __call__(self, x, r):
        family, settings = r['family'], r['settings']
        if family == 'ffmpeg':
            return self.local(x, RTCSettings(**settings))
        if family == 'webrtc':
            if self.webrtc is None:
                from w2v_v33.cache import PairedRTC
                self.webrtc = PairedRTC(self.local.ffmpeg)
            return self.webrtc.process(x, dict(family='webrtc', ns_level=settings['ns_level'],
                agc_target_dbfs=settings['agc_target_dbfs'], rtc={'bitrate':settings['bitrate']}))
        if family != 'light':
            raise ValueError('Unknown RTC engine')
        encoded = self.local._run(['-f','f32le','-ar','16000','-ac','1','-i','pipe:0',
            '-af',f'lowpass=f={settings["cutoff"]}', '-c:a','libopus','-b:a',str(settings['bitrate']),
            '-application','voip','-frame_duration','20','-vbr','on','-f','ogg','pipe:1'],x.astype('<f4').tobytes())
        raw = self.local._run(['-f','ogg','-i','pipe:0','-ar','16000','-ac','1','-f','f32le','pipe:1'],encoded)
        return np.frombuffer(raw,dtype='<f4').copy()


def generate(wave, r, bank, engines):
    rng = np.random.default_rng(r['seed'])
    # Retry only silent/no-overlap noise, independent of labels and model scores.
    for attempt in range(8):
        noise, used = noise_wave(bank,len(wave),r['noise_type'],rng)
        a,b,meta = common_inputs(wave,noise,r['snr_db'],r['input_gain_db'],r['echo'])
        if meta['pair_eligible'] or not active_mask(wave,SR).any():
            break
    ref, noisy = engines(a,r), engines(b,r)
    if len(ref)!=len(a) or len(noisy)!=len(a) or not np.isfinite(ref).all() or not np.isfinite(noisy).all():
        raise RuntimeError('RTC changed full duration or produced nonfinite audio')
    erased = []
    if r['dropout']:
        d = r['dropout']; count = min(round(d['seconds']*SR), max(1,len(noisy)//50))
        start = min(len(noisy)-count,round(d['position']*len(noisy)))
        # Short communication dropout only in the noisy arm; excludes aux alignment.
        noisy[start:start+count] *= d['gain']
        erased.append([start,start+count])
    meta.update(recipe=r, noise_hashes=used, noise_attempts=attempt+1, erased_spans=erased,
        output_samples=len(ref), reference_samples=len(wave), same_settings=True,
        fresh_state_per_utterance=True, pair_eligible=bool(meta['pair_eligible']))
    return ref,noisy,meta


def cache_key(source_sha, r, catalog, engine_runtime):
    return hashlib.sha256(json.dumps([source_sha,r,catalog,engine_runtime],sort_keys=True).encode()).hexdigest()
