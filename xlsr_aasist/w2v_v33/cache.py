"""Owned V3.3 full-duration FFmpeg/WebRTC cache; no legacy data is deleted here."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import soundfile as sf

from rtc_noisy.common import noise_catalog, assert_noise_disjoint
from rtc_noisy.simulator import LocalRTC, RTCSettings
from rtc_noisy_v2.plan import BANDS, settings_for, stable_seed, digest_json
from w2v_aasist.data import read_protocol, read_wave, safe_audio_path
from w2v_aasist.full_cache import continuous_mix, read_index as read_legacy_index
from w2v_aasist.launch import run_lock
from w2v_aasist.progress import progress
from w2v_aasist.runtime import atomic_json, sha256

FORMAT = 'rtc_v33_full_condition_cache_v1'
CONDITIONS = ('noisy_a', 'noisy_b')


class PairedRTC(LocalRTC):
    """Fresh real WebRTC APM per utterance, before Opus; no platform claim."""
    def __init__(self, ffmpeg='ffmpeg'):
        super().__init__(ffmpeg or 'ffmpeg')
        try:
            self.webrtc_version = importlib.metadata.version('webrtc-audio-processing')
            from webrtc_audio_processing import AudioProcessingModule
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise RuntimeError('V3.3 cache requires webrtc-audio-processing==0.1.3') from exc
        if self.webrtc_version != '0.1.3': raise RuntimeError('Expected webrtc-audio-processing==0.1.3')
        self.apm_class = AudioProcessingModule

    def process(self, waveform, assignment):
        if assignment['family'] == 'ffmpeg':
            return super().__call__(waveform, RTCSettings(**assignment['rtc']))
        if assignment['family'] != 'webrtc': raise ValueError('Unknown processing family')
        x = np.asarray(waveform, np.float32)
        ap = self.apm_class(enable_ns=True, agc_type=1, enable_vad=False)
        ap.set_stream_format(16000, 1)
        ap.set_ns_level(assignment['ns_level'])
        ap.set_agc_target(assignment['agc_target_dbfs'])
        gain = min(1., .99 / max(float(np.abs(x).max()), 1e-12))
        pcm = np.round(x * gain * 32767.).astype('<i2')
        blocks = []
        for start in range(0, len(pcm), 160):
            frame = pcm[start:start + 160]
            padded = np.pad(frame, (0, 160 - len(frame)))
            processed = np.frombuffer(ap.process_stream(padded.tobytes()), dtype='<i2')
            if len(processed) != 160: raise RuntimeError('WebRTC returned a non-10ms frame')
            blocks.append(processed[:len(frame)].astype(np.float32) / 32768.)
        y = np.concatenate(blocks)
        encoded = self._run(['-f', 'f32le', '-ar', '16000', '-ac', '1', '-i', 'pipe:0',
                    '-c:a', 'libopus', '-b:a', str(assignment['rtc']['bitrate']),
                    '-application', 'voip', '-frame_duration', '20', '-vbr', 'on', '-f', 'ogg', 'pipe:1'],
                    y.astype('<f4').tobytes())
        decoded = self._run(['-f', 'ogg', '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-f', 'f32le', 'pipe:1'], encoded)
        result = np.frombuffer(decoded, dtype='<f4').copy()
        if len(result) != len(x) or not np.isfinite(result).all():
            raise RuntimeError('WebRTC/Opus changed full duration or produced nonfinite audio')
        return result


def assignments(records, seed, legacy=None):
    groups = defaultdict(list)
    for r in records:
        if r['domain'] == 'offline': groups[(r['language'], r['label'])].append(r['id'])
    allowed, result = settings_for('train'), {}
    for group, ids in sorted(groups.items()):
        ids = sorted(ids)
        random.Random(stable_seed(seed, 'v33-source-order', *group)).shuffle(ids)
        for rank, source in enumerate(ids):
            result[source] = [
                dict(condition='noisy_a', version=0, family='ffmpeg', band=rank % 4,
                     rtc=RTCSettings(*allowed[rank % len(allowed)]).as_dict()),
                dict(condition='noisy_b', version=1, family='webrtc', band=(rank + 2) % 4,
                     rtc=RTCSettings(*allowed[(rank + 7) % len(allowed)]).as_dict(),
                     ns_level=(rank // 4) % 4, agc_target_dbfs=(6, 12, 18)[(rank // 16) % 3])]
        if legacy is not None:
            # Old v0/v1 cover strong/weak halves. Split each old parity stratum
            # between versions to balance A itself across all four SNR bands.
            parity = defaultdict(list)
            for source in ids:
                if (source, 0) not in legacy or (source, 1) not in legacy:
                    raise ValueError('Legacy cache has missing source/version')
                parity[legacy[(source, 0)]['band']].append(source)
            for band, selected in sorted(parity.items()):
                random.Random(stable_seed(seed, 'v33-legacy-selection', *group, band)).shuffle(selected)
                for rank, source in enumerate(selected):
                    version = rank % 2
                    old = legacy[(source, version)]
                    result[source][0].update(band=old['band'], rtc=old['rtc'], legacy_version=version,
                                             legacy_audio_sha256=old['audio_sha256'])
    return result


def _header(path, frames=None):
    info = sf.info(path)
    if info.samplerate != 16000 or info.channels != 1 or info.frames < 1 or (frames is not None and info.frames != frames):
        raise ValueError('Expected complete 16 kHz mono waveform: ' + str(path))
    return info


def _finite(path):
    with sf.SoundFile(path) as stream:
        for block in stream.blocks(blocksize=16000 * 30, dtype='float32'):
            if not np.isfinite(block).all(): raise ValueError('Nonfinite cache audio: ' + str(path))


def _protected_output(cfg, output):
    keys = ('train_data_path', 'dev_data_path', 'dev_noisy_cache', 'dev_heldout_cache', 'legacy_full_cache')
    for key in keys:
        if not cfg.get(key): continue
        p = Path(cfg[key]).resolve()
        if output == p or output in p.parents or p in output.parents:
            raise ValueError('V3.3 output overlaps protected data: ' + key)


def _legacy(cfg, records):
    if not cfg.get('legacy_full_cache'): return None, None
    folder = Path(cfg['legacy_full_cache']).resolve()
    _, (raw, configuration) = read_legacy_index(folder, 'train', cfg['train_protocol'], records)
    indexed = {}
    for r in raw:
        indexed[(r['source'], r['version'])] = dict(r, resolved_audio=str(safe_audio_path(folder, r['audio'])))
    identity = {name: sha256(folder / name) for name in ('config.json', 'manifest.jsonl', 'complete.json')}
    return indexed, dict(path=str(folder), hashes=identity, ffmpeg_version=configuration['ffmpeg_version'],
                         noise=configuration['noise'])


def _source_inventory(records):
    result = {}
    for row in progress(records, total=len(records), label='V33 source identity', every=2000):
        path = Path(row['audio']); header = _header(path); stat = path.stat()
        result[row['id']] = dict(sha256=sha256(path), samples=header.frames,
                                file_size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


def read_index(folder, cfg, official_records=None, *, verify_audio_hash=True, verify_finite=False, _completion=None):
    folder = Path(folder).resolve()
    configuration = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    complete = _completion if _completion is not None else json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
    recipe_hash = sha256(folder / 'config.json')
    if (configuration.get('format') != FORMAT or configuration.get('role') != 'train'
            or configuration.get('families') != ['ffmpeg', 'webrtc'] or configuration.get('cut') is not None
            or configuration.get('protocol_sha256') != sha256(cfg['train_protocol'])
            or configuration.get('generator_sha256') != sha256(__file__)
            or configuration.get('seed') != cfg['seed']
            or configuration.get('webrtc_version') != '0.1.3' or configuration.get('subtype') != 'FLOAT'
            or configuration.get('processing') != {'profile': 'v33-paired', 'families': ['ffmpeg', 'webrtc']}
            or configuration.get('sr') != 16000 or configuration.get('views_per_source') != 2
            or complete.get('config_sha256') != recipe_hash
            or complete.get('manifest_sha256') != sha256(folder / 'manifest.jsonl')
            or not complete.get('all_audio_finite_verified')):
        raise ValueError('V3.3 cache is incomplete or has a different recipe/protocol')
    if cfg.get('train_pair_manifest') and configuration['pair_manifest_sha256'] != sha256(cfg['train_pair_manifest']):
        raise ValueError('Official pair manifest changed since cache preparation')
    official_records = official_records or read_protocol(cfg['train_protocol'], cfg['train_data_path'])
    official = {r['id']: r for r in official_records if r['domain'] == 'offline'}
    assignments_map = json.loads((folder / 'assignments.json').read_text(encoding='utf-8'))
    inventory = json.loads((folder / 'sources.json').read_text(encoding='utf-8'))
    if (digest_json(assignments_map) != configuration['assignments_digest']
            or digest_json(inventory) != configuration['source_inventory_digest'] or set(inventory) != set(official)):
        raise ValueError('Cache assignment/source inventory differs')
    source_hashes = {}
    rows, raw, seen = [], [], set()
    for line in (folder / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(line); source, condition = r['source'], r['condition']
        if source not in official or condition not in CONDITIONS or (source, condition) in seen:
            raise ValueError('Duplicate/unknown paired cache row')
        a = assignments_map[source][CONDITIONS.index(condition)]
        expected = official[source]
        if (any(r.get(k) != v for k, v in a.items()) or r['label'] != expected['label']
                or r['language'] != expected['language'] or r['source_sha256'] != inventory[source]['sha256']
                or r['output_samples'] != inventory[source]['samples'] or r['source_samples'] != r['output_samples']
                or not BANDS[r['band']][0] <= r['snr_db'] <= BANDS[r['band']][1]
                or not r.get('noise', {}).get('contiguous_full_length')
                or r.get('recipe_sha256') != recipe_hash):
            raise ValueError('Paired cache condition, label, source or duration differs')
        if source not in source_hashes:
            _header(expected['audio'], inventory[source]['samples'])
            source_hashes[source] = sha256(expected['audio'])
        if source_hashes[source] != inventory[source]['sha256']:
            raise ValueError('Official source changed since V3.3 cache generation')
        path = safe_audio_path(folder, r['audio'])
        if _header(path, r['output_samples']).subtype != 'FLOAT': raise ValueError('Expected FLOAT cache WAV')
        if verify_audio_hash and sha256(path) != r['audio_sha256']: raise ValueError('Paired cache audio hash differs')
        if verify_finite: _finite(path)
        rows.append(dict(expected, audio=str(path), source_audio=expected['audio'],
                         source_id=source, source_sha256=r['source_sha256'], condition=condition,
                         noisy=True, full_length=True, version=r['version'], output_samples=r['output_samples'],
                         band=r['band'], bank=str(folder), processing_family=r['family']))
        raw.append(r); seen.add((source, condition))
    if len(rows) != 2 * len(official) or complete.get('rows') != len(rows) or configuration['source_count'] != len(official):
        raise ValueError('Every Offline source must have exactly two full condition views')
    return rows, (raw, configuration)


def validate_cache(cfg, folder):
    return read_index(folder, cfg, verify_audio_hash=True, verify_finite=True)


def prepare_cache(cfg, output, workers=4):
    """Resume atomically inside a new owned root; return its manifest Path."""
    if type(workers) is not int or workers < 1: raise ValueError('workers must be positive')
    from .data import pair_manifest, canonical_sources
    output = Path(output).resolve(); _protected_output(cfg, output)
    records = read_protocol(cfg['train_protocol'], cfg['train_data_path'])
    pair_path = pair_manifest(cfg); cfg['train_pair_manifest'] = str(pair_path)
    canonical_sources(records, pair_path)
    offline = [r for r in records if r['domain'] == 'offline']
    # A verified owned cache no longer depends on the legacy hard-link names.
    if (output / 'complete.json').is_file():
        existing = json.loads((output / 'config.json').read_text(encoding='utf-8'))
        if (existing.get('seed') != cfg['seed'] or existing.get('pair_manifest_sha256') != sha256(pair_path)
                or existing.get('noise', {}).get('manifest_sha256') != sha256(cfg['train_noise_manifest'])):
            raise ValueError('Completed cache has a different seed/pair recipe')
        with run_lock(output / '.prepare.lock'):
            validate_cache(cfg, output)
        print('V33_FULL_CACHE_COMPLETE=True (reused complete owned cache)', flush=True)
        return output / 'manifest.jsonl'
    legacy, parent = _legacy(cfg, records)
    choices = assignments(records, cfg['seed'], legacy)
    inventory = _source_inventory(offline)
    noise_path = cfg['train_noise_manifest']
    augment, catalog = noise_catalog(noise_path, 'train')
    if parent and parent['noise'] != catalog:
        raise ValueError('Legacy/new Train noise catalogs differ; reuse cannot establish common noise provenance')
    rtc = PairedRTC(cfg.get('ffmpeg') or 'ffmpeg')
    longest = max(x['samples'] for x in inventory.values()) / 16000
    pool = [(r, sf.info(r['path']).duration) for r in augment.records]
    pool = [(r, seconds) for r, seconds in pool if seconds >= longest]
    if not pool: raise ValueError('No contiguous Train noise recording covers the longest source')
    augment.records, durations = [r for r, _ in pool], [seconds for _, seconds in pool]
    for key in ('dev_noisy_cache', 'dev_heldout_cache'):
        dev = json.loads((Path(cfg[key]) / 'config.json').read_text(encoding='utf-8'))
        assert_noise_disjoint({'noise': catalog}, dev)
        if dev['ffmpeg_version'] != rtc.version: raise ValueError('Train/Dev FFmpeg versions differ')
    if parent and parent['ffmpeg_version'] != rtc.version: raise ValueError('Legacy/new FFmpeg version differs')
    recipe = dict(format=FORMAT, role='train', split='train', generation=0, seed=cfg['seed'],
        families=['ffmpeg', 'webrtc'], processing={'profile': 'v33-paired', 'families': ['ffmpeg', 'webrtc']},
        sr=16000, subtype='FLOAT', cut=None, views_per_source=2, source_count=len(offline),
        protocol_sha256=sha256(cfg['train_protocol']), pair_manifest_sha256=sha256(pair_path),
        assignments_digest=digest_json(choices), source_inventory_digest=digest_json(inventory),
        noise=catalog, ffmpeg_version=rtc.version, webrtc_version=rtc.webrtc_version,
        legacy=parent, reuse_policy='verified legacy A; hardlink or copy' if parent else 'generate both full families',
        order='contiguous Train background noise -> family DSP -> Opus -> full FLOAT WAV',
        snr_bands=[list(b) for b in BANDS], continuous_noise_recordings=len(pool),
        generator_sha256=sha256(__file__), simulator_sha256=sha256(Path(__file__).parents[1] / 'rtc_noisy/simulator.py'),
        mix_code_sha256=sha256(Path(__file__).parents[1] / 'utils/env_noise.py'))
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output / '.prepare.lock'):
        config_file = output / 'config.json'
        if config_file.exists():
            if json.loads(config_file.read_text(encoding='utf-8')) != recipe:
                raise ValueError('Partial cache recipe changed; use a new cache directory')
        elif any(p.name != '.prepare.lock' for p in output.iterdir()):
            raise ValueError('Unrecognized nonempty cache root')
        else:
            atomic_json(config_file, recipe)
            atomic_json(output / 'assignments.json', choices)
            atomic_json(output / 'sources.json', inventory)
        recipe_hash = sha256(config_file)
        hardlink = False
        if legacy:
            first = next(iter(legacy.values()))['resolved_audio']; probe = output / '.hardlink-probe'
            try:
                os.link(first, probe); hardlink = True
            except OSError:
                pass
            finally:
                if probe.exists(): probe.unlink()
        total = sum(r['samples'] for r in inventory.values()) * 4
        existing = sum(p.stat().st_size for p in (output / 'audio').rglob('*.wav')
                       if not hardlink or p.name.endswith('_noisy_b.wav')) if (output / 'audio').exists() else 0
        required = max(0, total * (1 if legacy and hardlink else 2) - existing) + 2 * 1024 ** 3
        if shutil.disk_usage(output).free < required:
            raise OSError(f'Need {required / 1024**3:.2f} GiB free; no old cache was removed')
        print(f'V33_CACHE_SOURCES={len(offline)} NEW_REQUIRED_GiB={required / 1024**3:.2f} LEGACY_A_HARDLINK={hardlink}', flush=True)

        def generate(row):
            source = row['id']; original = inventory[source]; path = Path(row['audio'])
            before = path.stat()
            if sha256(path) != original['sha256']: raise ValueError('Source changed during cache preparation')
            wave = read_wave(path)
            if len(wave) != original['samples']: raise ValueError('Source duration changed')
            key = hashlib.sha256(source.encode()).hexdigest()[:24]
            folder = output / 'audio' / key[:2]; folder.mkdir(parents=True, exist_ok=True)
            result = []
            for a in choices[source]:
                condition = a['condition']; audio = folder / f'{key}_{condition}.wav'; sidecar = audio.with_suffix('.json')
                if audio.is_file() and sidecar.is_file():
                    saved = json.loads(sidecar.read_text(encoding='utf-8'))
                    if (saved.get('recipe_sha256') == recipe_hash and saved.get('source_sha256') == original['sha256']
                            and saved.get('source') == source and saved.get('condition') == condition
                            and all(saved.get(k) == v for k, v in a.items())
                            and saved.get('audio_sha256') == sha256(audio)):
                        _header(audio, original['samples']); _finite(audio)
                        result.append(saved); continue
                tmp = audio.with_suffix('.tmp.wav')
                try:
                    if condition == 'noisy_a' and legacy:
                        old = legacy[(source, a['legacy_version'])]
                        if old['source_sha256'] != original['sha256'] or sha256(old['resolved_audio']) != old['audio_sha256']:
                            raise ValueError('Legacy cache/source integrity mismatch')
                        _header(old['resolved_audio'], original['samples']); _finite(old['resolved_audio'])
                        if tmp.exists(): tmp.unlink()
                        if hardlink:
                            os.link(old['resolved_audio'], tmp); reuse = 'hardlink'
                        else:
                            shutil.copyfile(old['resolved_audio'], tmp); reuse = 'copy'
                        snr, noise, mix_id = old['snr_db'], old['noise'], old['mix_id']
                    else:
                        rng = np.random.RandomState(stable_seed(cfg['seed'], 'v33-full', source, condition) % 2**32)
                        snr = float(rng.uniform(*BANDS[a['band']]))
                        mixed, noise = continuous_mix(augment, durations, wave, rng, snr)
                        processed = rtc.process(mixed, a)
                        if len(processed) != len(wave) or not np.isfinite(processed).all():
                            raise ValueError('DSP changed duration or produced nonfinite audio')
                        if shutil.disk_usage(output).free < processed.nbytes + 1024 ** 3:
                            raise OSError('Cache free-space floor reached; old cache was not removed')
                        sf.write(tmp, processed, 16000, subtype='FLOAT')
                        mix_id = hashlib.sha256(mixed.astype('<f4').tobytes()).hexdigest(); reuse = 'generated'
                    os.replace(tmp, audio)
                finally:
                    if tmp.exists(): tmp.unlink()
                saved = dict(a, source=source, label=row['label'], language=row['language'],
                    source_sha256=original['sha256'], source_samples=len(wave), output_samples=len(wave),
                    snr_db=snr, noise=noise, mix_id=mix_id, reuse=reuse, role='train', generation=0,
                    recipe_sha256=recipe_hash, audio=audio.relative_to(output).as_posix(), audio_sha256=sha256(audio))
                atomic_json(sidecar, saved); result.append(saved)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Official source changed during cache generation')
            return result

        temporary = output / 'manifest.jsonl.tmp'
        with ThreadPoolExecutor(max_workers=workers) as pool_executor, temporary.open('w', encoding='utf-8', newline='\n') as stream:
            for items in progress(pool_executor.map(generate, offline), total=len(offline), label='V33 full noisy sources', every=100):
                for item in items: stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        os.replace(temporary, output / 'manifest.jsonl')
        complete = dict(rows=2 * len(offline), config_sha256=recipe_hash,
                        manifest_sha256=sha256(output / 'manifest.jsonl'), all_audio_finite_verified=True)
        # Publish completion only after every owned audio hash, header and sample
        # has been validated; readers can never observe a provisional completion.
        read_index(output, cfg, records, verify_audio_hash=True, verify_finite=True, _completion=complete)
        atomic_json(output / 'complete.json', complete)
        print('V33_FULL_CACHE_COMPLETE=True; old cache remains untouched', flush=True)
    return output / 'manifest.jsonl'
