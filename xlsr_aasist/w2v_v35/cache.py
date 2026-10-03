"""Epoch-addressed full-wave views, generated lazily by bounded CPU workers.

Only this package's marked generation directories may be retired.  Existing
V3 caches and all official waveforms remain read-only.  Seeds and recipes, not
worker scheduling, determine each view, so a partial epoch is reproducible.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import soundfile as sf

from rtc_noisy.common import noise_catalog, assert_noise_disjoint
from rtc_noisy.simulator import RTCSettings
from rtc_noisy_v2.plan import BANDS, settings_for, stable_seed, digest_json
from utils.env_noise import NoiseAugment, mix_at_snr
from w2v_aasist.data import read_wave
from w2v_aasist.launch import run_lock
from w2v_aasist.progress import progress
from w2v_aasist.runtime import atomic_json, sha256
from w2v_v33.cache import PairedRTC

FORMAT = 'rtc_v35_epoch_full_views_v1'
TRAIN_MODES = ('identity', 'codec', 'denoise', 'gain', 'full', 'full')


class DiverseRTC(PairedRTC):
    """Fresh utterance state; matched module choices for both real and fake."""
    def process(self, waveform, assignment):
        x = np.asarray(waveform, np.float32)
        family, mode = assignment['family'], assignment['mode']
        if mode not in TRAIN_MODES or family not in ('ffmpeg', 'webrtc', 'anlmdn'):
            raise ValueError('Unknown V3.5 processing condition')
        if mode == 'identity':
            return x.copy()  # Mixed waveform; bypass communication processing only.
        cfg = assignment['rtc']
        filters = []
        if family == 'webrtc' and mode in ('denoise', 'gain', 'full'):
            ap = self.apm_class(enable_ns=mode in ('denoise', 'full'),
                                agc_type=1 if mode in ('gain', 'full') else 0, enable_vad=False)
            ap.set_stream_format(16000, 1)
            if mode in ('denoise', 'full'): ap.set_ns_level(assignment['ns_level'])
            if mode in ('gain', 'full'): ap.set_agc_target(assignment['agc_target_dbfs'])
            gain = min(1., .99 / max(float(np.abs(x).max()), 1e-12))
            pcm = np.round(x * gain * 32767.).astype('<i2')
            frames = []
            for start in range(0, len(pcm), 160):
                frame = pcm[start:start+160]
                result = np.frombuffer(ap.process_stream(np.pad(frame, (0, 160-len(frame))).tobytes()), dtype='<i2')
                if len(result) != 160: raise RuntimeError('WebRTC changed frame size')
                frames.append(result[:len(frame)].astype(np.float32)/32768.)
            x = np.concatenate(frames)
        elif family in ('ffmpeg', 'anlmdn'):
            if mode in ('denoise', 'full'):
                filters.append('anlmdn=s=0.0001:p=0.002:r=0.006' if family == 'anlmdn'
                               else f'afftdn=nr={cfg["noise_reduction"]}:nf=-35:tn=1')
            if mode in ('gain', 'full'):
                filters.append(f'dynaudnorm=f=150:g=7:p=0.9:m={cfg["max_gain"]}')
        arguments = ['-f', 'f32le', '-ar', '16000', '-ac', '1', '-i', 'pipe:0']
        if filters: arguments += ['-af', ','.join(filters)]
        if mode in ('codec', 'full'):
            encoded = self._run(arguments + ['-c:a', 'libopus', '-b:a', str(cfg['bitrate']),
                '-application', 'voip', '-frame_duration', '20', '-vbr', 'on', '-f', 'ogg', 'pipe:1'], x.astype('<f4').tobytes())
            output = self._run(['-f', 'ogg', '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-f', 'f32le', 'pipe:1'], encoded)
            y = np.frombuffer(output, dtype='<f4').copy()
        elif filters:
            y = np.frombuffer(self._run(arguments + ['-f', 'f32le', 'pipe:1'], x.astype('<f4').tobytes()), dtype='<f4').copy()
        else:
            y = x
        if y.shape != waveform.shape or not np.isfinite(y).all():
            raise RuntimeError('Communication processing changed full length or produced nonfinite audio')
        return np.asarray(y, np.float32)


def assignments(records, seed, epoch, role='train'):
    """Independent group shuffles give identical condition distributions by class.

    Each train source gets both families and opposite SNR halves.  Across epochs
    its identity, module choices, noise excerpt and SNR all change.  Fixed Dev
    uses one shared mixture with seen processing and truly excluded anlmdn.
    """
    if role not in ('train', 'dev'): raise ValueError('Unknown V3.5 cache role')
    groups = defaultdict(list)
    for row in records: groups[(row['language'], row['label'])].append(row['id'])
    result = {}
    for group, ids in sorted(groups.items()):
        ids = sorted(ids)
        random.Random(stable_seed(seed, 'v35-order', role, epoch, *group)).shuffle(ids)
        for rank, source in enumerate(ids):
            views = []
            for version in range(2):
                family = ('ffmpeg', 'webrtc')[(rank//4+version+epoch) % 2] if role == 'train' else ('ffmpeg' if (rank//4) % 2 == 0 else 'webrtc') if version == 0 else 'anlmdn'
                setting_role = 'dev_heldout' if family == 'anlmdn' else 'train'
                allowed = settings_for(setting_role)
                rng = random.Random(stable_seed(seed, 'v35-setting', role, epoch, source, version))
                setting = allowed[rng.randrange(len(allowed))]
                # Dev uses full processing so heldout always exercises anlmdn.
                mode = TRAIN_MODES[(rank // 8 + version + epoch) % len(TRAIN_MODES)] if role == 'train' else 'full'
                band = (rank + 2*version + epoch) % 4 if role == 'train' else rank % 4
                views.append(dict(condition=('noisy_a', 'noisy_b')[version] if role == 'train' else ('seen', 'heldout')[version],
                    version=version, family=family, mode=mode, band=band,
                    rtc=RTCSettings(*setting).as_dict(), ns_level=rng.randrange(4),
                    agc_target_dbfs=(6, 12, 18)[rng.randrange(3)]))
            result[source] = views
    return result


def extend_noise(noise, length, rng, fade_samples=800):
    """Length-preserving contiguous crop or crossfaded repetition, never a zero tail."""
    noise = np.asarray(noise, np.float32)
    if not len(noise) or not np.isfinite(noise).all(): raise ValueError('Invalid noise')
    if len(noise) >= length:
        offset = int(rng.randint(len(noise)-length+1))
        return noise[offset:offset+length].copy(), {'offset': offset, 'crossfade_repeated': False}
    # Randomize the initial phase; only noise (never speech) is repeated.
    offset = int(rng.randint(len(noise)))
    noise = np.concatenate((noise[offset:], noise[:offset]))
    fade = min(fade_samples, len(noise)//4)
    result = noise.copy()
    while len(result) < length:
        if fade:
            alpha = np.linspace(0., 1., fade, endpoint=False, dtype=np.float32)
            cross = (1-alpha)*result[-fade:] + alpha*noise[:fade]
            result = np.concatenate((result[:-fade], cross, noise[fade:]))
        else:
            result = np.concatenate((result, noise))
    return result[:length], {'offset': offset, 'crossfade_repeated': True, 'crossfade_samples': fade}


def full_mix(augment, durations, wave, rng, snr):
    # Per-recording selection: long training outliers never shrink everyone's pool.
    eligible = [i for i, duration in enumerate(durations) if duration*16000 >= len(wave)]
    pool = eligible or list(range(len(augment.records)))
    if not pool: raise ValueError('Empty noise catalog')
    for _ in range(40):
        record = augment.records[pool[int(rng.randint(len(pool)))]]
        noise, excerpt = extend_noise(augment._read_noise(record['path'], 16000), len(wave), rng)
        try:
            mixed, stats = mix_at_snr(wave, noise, 16000, snr)
        except ValueError as exc:
            if 'Noise is silent' in str(exc): continue
            raise
        if not stats.get('applied'): raise ValueError('Official source has no active speech')
        return mixed, dict(stats, **excerpt, noise_path=record['path'],
                           original_recording=record['original_recording'], eligible_noise_count=len(pool))
    raise ValueError('No active noise excerpt after 40 deterministic attempts')


def _root(cfg, role):
    key = 'train_cache_root' if role == 'train' else 'full_dev_cache_root'
    output = Path(cfg[key]).expanduser().resolve()
    protected = ('train_data_path', 'dev_data_path', 'dev_noisy_cache', 'dev_heldout_cache',
                 'train_noisy_cache_v33', 'legacy_full_cache', 'ssl_path')
    for name in protected:
        if not cfg.get(name): continue
        path = Path(cfg[name]).expanduser().resolve()
        if output == path or output in path.parents or path in output.parents:
            raise ValueError('V3.5 cache overlaps protected data: '+name)
    other = cfg.get('full_dev_cache_root' if role == 'train' else 'train_cache_root')
    if other:
        path = Path(other).expanduser().resolve()
        if output == path or output in path.parents or path in output.parents:
            raise ValueError('Train and Dev V3.5 cache roots overlap')
    return output


def _write_exact(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('V3.5 cache recipe changed; use a separate owned cache root: '+str(path))
    else:
        atomic_json(path, value)


def prepare_base(cfg, train, dev):
    """Once per run: validate split provenance and record immutable input identities."""
    train_aug, train_noise = noise_catalog(cfg['train_noise_manifest'], 'train')
    dev_aug, dev_noise = noise_catalog(cfg['dev_noise_manifest'], 'dev')
    assert_noise_disjoint({'noise': train_noise}, {'noise': dev_noise})
    engine = DiverseRTC(cfg.get('ffmpeg') or 'ffmpeg')
    all_hashes = {}
    for role, records, augment, noise in (('train', train, train_aug, train_noise), ('dev', dev, dev_aug, dev_noise)):
        output = _root(cfg, role)
        output.mkdir(parents=True, exist_ok=True)
        source_rows = {}
        marker = output/'owner.json'
        with run_lock(output/'.prepare.lock'):
            if not marker.exists() and any(p.name != '.prepare.lock' for p in output.iterdir()):
                raise ValueError('Refusing to adopt unrecognized nonempty V3.5 cache directory')
            previous = json.loads((output/'sources.json').read_text()) if (output/'sources.json').is_file() else {}
            for row in records:
                path = Path(row['audio']); stat = path.stat(); prior = previous.get(row['id'], {})
                info = sf.info(path)
                if info.samplerate != 16000 or info.channels != 1 or info.frames < 1:
                    raise ValueError('Official audio must be nonempty 16kHz mono: '+str(path))
                unchanged = prior.get('audio') == str(path.resolve()) and prior.get('size') == stat.st_size and prior.get('mtime_ns') == stat.st_mtime_ns
                fingerprint = prior['sha256'] if unchanged and not cfg.get('verify_source_hashes', False) else sha256(path)
                source_rows[row['id']] = dict(audio=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    samples=info.frames, sha256=fingerprint, language=row['language'], label=row['label'], domain=row['domain'])
            recipe = dict(format=FORMAT, role=role, seed=cfg['seed'] if role == 'train' else cfg.get('dev_seed', 935017),
                protocol_sha256=sha256(cfg[role+'_protocol']), source_digest=digest_json(source_rows), noise=noise,
                noise_manifest=str(Path(cfg[role+'_noise_manifest']).resolve()),
                noise_durations=[sf.info(r['path']).duration for r in augment.records],
                ffmpeg_version=engine.version, webrtc_version=engine.webrtc_version,
                generator_sha256=sha256(__file__), modes=list(TRAIN_MODES), bands=[list(v) for v in BANDS],
                full_length=True, views_per_source=2, subtype='FLOAT',
                dev_policy='one seen and one anlmdn heldout per source; identical mixture and SNR',
                source_read_policy='startup file identity; reuse SHA when size+mtime are unchanged')
            _write_exact(marker, {'format': FORMAT, 'role': role, 'root': str(output)})
            _write_exact(output/'sources.json', source_rows)
            _write_exact(output/'recipe.json', recipe)
        all_hashes[role] = {row['sha256'] for row in source_rows.values()}
    if all_hashes['train'] & all_hashes['dev']:
        raise ValueError('Identical official waveform file content occurs in Train and Dev')


def prepare_epoch(cfg, epoch, role='train'):
    if type(epoch) is not int or epoch < 0: raise ValueError('Epoch must be nonnegative integer')
    if role == 'dev' and epoch != 0: raise ValueError('Dev generation is permanently fixed at zero')
    root = _root(cfg, role)
    recipe = json.loads((root/'recipe.json').read_text(encoding='utf-8'))
    if recipe['format'] != FORMAT or recipe['role'] != role: raise ValueError('Unrecognized cache owner')
    folder = root/f'epoch_{epoch:03d}'
    marker = dict(format=FORMAT, role=role, epoch=epoch, recipe_sha256=sha256(root/'recipe.json'), root=str(root))
    with run_lock(root/'.prepare.lock'):
        if not folder.exists():
            existing = [p for p in root.glob('epoch_*') if p.is_dir()]
            if role == 'train' and len(existing) >= 2:
                raise RuntimeError('Two Train generations already exist; retire a completed generation after saving last.pt')
            folder.mkdir()
        if folder.is_symlink(): raise ValueError('Cache generation must not be a symlink')
        if not (folder/'generation.json').exists() and any(folder.iterdir()):
            raise ValueError('Refusing to adopt unmarked nonempty cache generation')
        _write_exact(folder/'generation.json', marker)
    return folder


def retire_generations(cfg, keep_epochs):
    """Caller must have committed last.pt and closed all DataLoader workers."""
    root = _root(cfg, 'train')
    owner = json.loads((root/'owner.json').read_text(encoding='utf-8'))
    if owner != {'format': FORMAT, 'role': 'train', 'root': str(root)}:
        raise ValueError('Refusing to retire an unowned cache')
    keep = {f'epoch_{int(epoch):03d}' for epoch in keep_epochs}
    if not keep: raise ValueError('Must preserve the resumable current epoch generation')
    removed = []
    with run_lock(root/'.prepare.lock'):
        for folder in root.glob('epoch_*'):
            if folder.name in keep: continue
            resolved = folder.resolve()
            if folder.is_symlink() or resolved.parent != root or not folder.is_dir():
                raise ValueError('Unsafe generation retirement target')
            marker = json.loads((folder/'generation.json').read_text(encoding='utf-8'))
            if (marker.get('format') != FORMAT or marker.get('role') != 'train'
                    or marker.get('root') != str(root) or folder.name != f'epoch_{marker["epoch"]:03d}'):
                raise ValueError('Refusing to delete unrecognized generation')
            for directory, subdirs, files in os.walk(resolved, followlinks=False):
                for name in subdirs+files:
                    child = Path(directory)/name
                    if child.is_symlink() or resolved not in child.resolve().parents:
                        raise ValueError('Refusing retirement through a symlink or junction')
            # The per-generation lifetime lock is shared on Linux by workers.
            with run_lock(folder/'.active.lock'):
                pass
            shutil.rmtree(resolved)
            removed.append(str(resolved))
    return removed


class EpochCache:
    """Small worker-local metadata; no model or full speech dataset in RAM."""
    def __init__(self, cfg, epoch, role='train'):
        self.cfg, self.epoch, self.role = cfg, epoch, role
        self.root = _root(cfg, role)
        self.folder = self.root/f'epoch_{epoch:03d}'
        self.recipe = json.loads((self.root/'recipe.json').read_text(encoding='utf-8'))
        self.inventory = json.loads((self.root/'sources.json').read_text(encoding='utf-8'))
        self.recipe_hash = sha256(self.root/'recipe.json')
        records = [dict(id=k, **v) for k, v in self.inventory.items() if v['domain']=='offline']
        self.assignments = assignments(records, self.recipe['seed'], epoch, role)
        self.augment = NoiseAugment(self.recipe['noise_manifest'], probability=1.)
        self.durations = self.recipe['noise_durations']
        self.engine = None
        self._lifetime = None
        if role == 'train' and os.name == 'posix':
            import fcntl
            self._lifetime = (self.folder/'.active.lock').open('a+b')
            fcntl.flock(self._lifetime.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)

    def close(self):
        handle = getattr(self, '_lifetime', None)
        if handle is not None:
            handle.close()
            self._lifetime = None

    def __del__(self):
        self.close()

    def get(self, row, wave=None):
        """Read the source once outside or here; reuse same mixture in Dev pair."""
        source = row['id']; entry = self.inventory[source]
        stat = Path(row['audio']).stat()
        if stat.st_size != entry['size'] or stat.st_mtime_ns != entry['mtime_ns']:
            raise ValueError('Official source changed during V3.5 run')
        if wave is None: wave = read_wave(row['audio'])
        if len(wave) != entry['samples'] or not np.isfinite(wave).all(): raise ValueError('Source waveform differs')
        key = hashlib.sha256(source.encode()).hexdigest()
        folder = self.folder/'audio'/key[:2]
        folder.mkdir(parents=True, exist_ok=True)
        outputs = []
        mixture = None
        # Same source is never duplicated by the epoch sampler. Per-source lock
        # also protects interrupted/resumed preparation from concurrent writers.
        with run_lock(folder/(key+'.lock')):
            for assignment in self.assignments[source]:
                condition = assignment['condition']
                path = folder/f'{key}_{condition}.wav'
                sidecar = path.with_suffix('.json')
                identity = dict(format=FORMAT, epoch=self.epoch, role=self.role, source=source,
                    source_sha256=entry['sha256'], recipe_sha256=self.recipe_hash, assignment=assignment)
                if path.is_file() and sidecar.is_file():
                    saved = json.loads(sidecar.read_text(encoding='utf-8'))
                    if all(saved.get(k) == v for k, v in identity.items()) and sha256(path) == saved.get('audio_sha256'):
                        out = read_wave(path)
                        if len(out) != len(wave): raise ValueError('Cached full waveform changed duration')
                        outputs.append((out, saved, str(path)))
                        continue
                # The same Dev mixture is regenerated identically even if one
                # half was already cached before interruption.
                condition_seed = condition if self.role == 'train' else 'shared-dev-mixture'
                rng = np.random.RandomState(stable_seed(self.recipe['seed'], self.role, self.epoch,
                                                        source, condition_seed) % 2**32)
                snr = float(rng.uniform(*BANDS[assignment['band']]))
                if self.role == 'train' or mixture is None:
                    mixed, noise = full_mix(self.augment, self.durations, wave, rng, snr)
                    mixture = (mixed, noise)
                else:
                    mixed, noise = mixture
                if self.engine is None: self.engine = DiverseRTC(self.cfg.get('ffmpeg') or 'ffmpeg')
                processed = self.engine.process(mixed, assignment)
                if len(processed) != len(wave) or not np.isfinite(processed).all():
                    raise ValueError('Processed view is not finite/full length')
                if shutil.disk_usage(self.root).free < processed.nbytes + int(self.cfg.get('cache_free_floor_bytes', 1024**3)):
                    raise OSError('V3.5 cache free-space floor reached; old caches were not deleted')
                tmp = path.with_suffix('.tmp.wav')
                try:
                    sf.write(tmp, processed, 16000, subtype='FLOAT')
                    os.replace(tmp, path)
                finally:
                    if tmp.exists(): tmp.unlink()
                saved = dict(identity, snr_db=snr, noise=noise, samples=len(wave),
                    mixture_sha256=hashlib.sha256(mixed.astype('<f4').tobytes()).hexdigest(),
                    audio_sha256=sha256(path), audio=str(path.relative_to(self.folder)))
                atomic_json(sidecar, saved)
                outputs.append((processed, saved, str(path)))
        return outputs


def prepare_dev(cfg, records):
    """One-time full Dev generation; never reads Progress or Eval."""
    prepare_epoch(cfg, 0, 'dev')
    # Each thread owns its processor/noise LRU. One get handles both views and
    # reads each source only once. Modest parallelism bounds audio RAM.
    import threading
    local = threading.local()
    def make(row):
        if not hasattr(local, 'cache'): local.cache = EpochCache(cfg, 0, 'dev')
        result = []
        for _, saved, path in local.cache.get(row):
            a = saved['assignment']
            result.append(dict(row, audio=path, source_id=row['id'], condition=a['condition'],
                noisy=True, band=a['band'], full_length=True, output_samples=saved['samples'],
                processing_family=a['family'], source_audio=row['audio'], view='full'))
        return result
    views = {'seen': [], 'heldout': []}
    with ThreadPoolExecutor(max_workers=max(1, int(cfg.get('cache_workers', 4)))) as pool:
        for result in progress(pool.map(make, records), total=len(records), label='V3.5 fixed full Dev sources', every=200):
            for row in result: views[row['condition']].append(row)
    root = _root(cfg, 'dev')/'epoch_000'
    manifest = {name: sorted(rows, key=lambda r:r['id']) for name, rows in views.items()}
    _write_exact(root/'manifest.json', manifest)
    _write_exact(root/'complete.json', {'manifest_sha256': sha256(root/'manifest.json'),
        'sources': len(records), 'views': 2*len(records), 'full_length': True})
    return manifest
