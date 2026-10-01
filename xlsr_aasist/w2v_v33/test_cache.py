"""Resumable full-wave cache integrity and real-DSP dispatch, using tiny fixtures."""
from collections import Counter, defaultdict
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from rtc_noisy_v2.plan import digest_json
from w2v_aasist.runtime import atomic_json, sha256
from . import cache
from .test_data import fixture


class FakeRTC:
    version = 'fixture-ffmpeg-v1'
    webrtc_version = '0.1.3'
    calls = []
    def __init__(self, *args): pass
    def process(self, wave, assignment):
        self.calls.append((np.array(wave), dict(assignment)))
        return np.asarray(wave * (.9 if assignment['family'] == 'ffmpeg' else .8), np.float32)


def cache_fixture(root):
    root = Path(root)
    cfg, records, sources, rows = fixture(root, per_group=1)
    noise = root / 'train_noise.wav'
    sf.write(noise, np.random.default_rng(8).normal(0, .1, 24000).astype(np.float32), 16000, subtype='FLOAT')
    manifest = root / 'train_noise.jsonl'
    manifest.write_text(json.dumps(dict(path=str(noise), split='train', original_recording='train-noise-1', sha256=sha256(noise))) + '\n')
    cfg.update(train_noise_manifest=str(manifest), train_noisy_cache_v33=str(root / 'new_cache'),
               ffmpeg=None, dev_data_path=str(root / 'dev_audio'))
    for key in ('dev_noisy_cache', 'dev_heldout_cache'):
        folder = root / key; folder.mkdir()
        atomic_json(folder / 'config.json', dict(ffmpeg_version=FakeRTC.version,
                    noise=dict(recording_ids=['dev-noise-1'], file_sha256=['dev-sha'], manifest_sha256='dev-manifest')))
        cfg[key] = str(folder)
    return cfg, records, sources


class CacheTests(unittest.TestCase):
    def setUp(self): FakeRTC.calls = []

    def test_each_family_balances_all_snr_bands_within_label_language(self):
        records = [dict(id=f'offline/{language}/{label}_{i}', domain='offline', label=label, language=language)
                   for language in ('en', 'zh') for label in (0, 1) for i in range(65)]
        plan = cache.assignments(records, 99)
        self.assertEqual(plan, cache.assignments(list(reversed(records)), 99))
        groups = defaultdict(Counter)
        for row in records:
            for a in plan[row['id']]:
                groups[(row['language'], row['label'], a['family'])][a['band']] += 1
        self.assertEqual(len(groups), 8)
        for counter in groups.values():
            self.assertEqual(set(counter), set(range(4)))
            self.assertLessEqual(max(counter.values()) - min(counter.values()), 1)
        web = [a[1] for a in plan.values()]
        self.assertEqual(len({(a['band'], a['ns_level']) for a in web}), 16)

    def test_generation_is_full_length_finite_atomic_and_complete_reuse_is_independent(self):
        with tempfile.TemporaryDirectory() as d, patch.object(cache, 'PairedRTC', FakeRTC):
            cfg, records, _ = cache_fixture(d); out = Path(cfg['train_noisy_cache_v33'])
            result = cache.prepare_cache(cfg, out, workers=1)
            self.assertEqual(result, (out / 'manifest.jsonl').resolve())
            rows, (raw, configuration) = cache.validate_cache(cfg, out)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(FakeRTC.calls), 8)
            self.assertEqual({r['family'] for r in raw}, {'ffmpeg', 'webrtc'})
            for r in raw:
                self.assertEqual(r['source_samples'], r['output_samples'])
                self.assertTrue(r['noise']['contiguous_full_length'])
                self.assertEqual(sf.info(out / r['audio']).subtype, 'FLOAT')
            self.assertFalse(list(out.rglob('*.tmp*')))
            before = sha256(result)
            with patch.object(cache, '_legacy', side_effect=AssertionError('No dependency on retired parent')):
                cache.prepare_cache(cfg, out, workers=1)
            self.assertEqual(len(FakeRTC.calls), 8)
            self.assertEqual(sha256(result), before)

    def test_failed_generation_resumes_valid_sidecars_without_repeating_dsp(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, _, _ = cache_fixture(d); out = Path(cfg['train_noisy_cache_v33'])
            class Interrupted(FakeRTC):
                failed = False
                def process(self, wave, a):
                    if a['family'] == 'webrtc' and not self.failed:
                        self.failed = True
                        raise RuntimeError('fixture interruption')
                    return super().process(wave, a)
            with patch.object(cache, 'PairedRTC', Interrupted):
                with self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
                    cache.prepare_cache(cfg, out, workers=1)
            self.assertFalse((out / 'complete.json').exists())
            existing = len(list((out / 'audio').rglob('*.json')))
            before = len(FakeRTC.calls)
            with patch.object(cache, 'PairedRTC', FakeRTC): cache.prepare_cache(cfg, out, workers=1)
            self.assertEqual(len(FakeRTC.calls) - before, 8 - existing)
            cache.validate_cache(cfg, out)

    def test_complete_cache_rejects_audio_and_source_changes(self):
        with tempfile.TemporaryDirectory() as d, patch.object(cache, 'PairedRTC', FakeRTC):
            cfg, _, _ = cache_fixture(d); out = Path(cfg['train_noisy_cache_v33'])
            cache.prepare_cache(cfg, out, workers=1)
            raw = [json.loads(line) for line in (out / 'manifest.jsonl').read_text().splitlines()]
            audio = out / raw[0]['audio']; payload = audio.read_bytes()
            changed = bytearray(payload); changed[-1] ^= 1; audio.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, 'audio hash'): cache.validate_cache(cfg, out)
            audio.write_bytes(payload)
            source = Path(cfg['train_data_path']) / raw[0]['source']
            changed = bytearray(source.read_bytes()); changed[-1] ^= 1; source.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, 'source changed'): cache.validate_cache(cfg, out)

    def test_finite_validation_catches_nan_even_if_manifest_hashes_are_rewritten(self):
        with tempfile.TemporaryDirectory() as d, patch.object(cache, 'PairedRTC', FakeRTC):
            cfg, _, _ = cache_fixture(d); out = Path(cfg['train_noisy_cache_v33'])
            cache.prepare_cache(cfg, out, workers=1)
            manifest = out / 'manifest.jsonl'; raw = [json.loads(line) for line in manifest.read_text().splitlines()]
            audio = out / raw[0]['audio']; wave, sr = sf.read(audio, dtype='float32'); wave[42] = np.nan
            sf.write(audio, wave, sr, subtype='FLOAT'); raw[0]['audio_sha256'] = sha256(audio)
            manifest.write_text(''.join(json.dumps(r) + '\n' for r in raw))
            complete = json.loads((out / 'complete.json').read_text()); complete['manifest_sha256'] = sha256(manifest)
            atomic_json(out / 'complete.json', complete)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'): cache.validate_cache(cfg, out)

    def test_reused_a_hardlink_survives_retired_legacy_filename(self):
        with tempfile.TemporaryDirectory() as d, patch.object(cache, 'PairedRTC', FakeRTC):
            cfg, records, _ = cache_fixture(d); out = Path(cfg['train_noisy_cache_v33'])
            legacy_root = Path(d) / 'legacy'; legacy_root.mkdir(); cfg['legacy_full_cache'] = str(legacy_root)
            legacy = {}
            native = cache.assignments(records, cfg['seed'])
            for i, row in enumerate(r for r in records if r['domain'] == 'offline'):
                wave, sr = sf.read(row['audio'], dtype='float32')
                for version in (0, 1):
                    path = legacy_root / f'{i}_{version}.wav'; sf.write(path, wave * (.7 + .1 * version), sr, subtype='FLOAT')
                    band = 2 * version
                    legacy[(row['id'], version)] = dict(source=row['id'], version=version, band=band,
                        rtc=native[row['id']][0]['rtc'], source_sha256=sha256(row['audio']), resolved_audio=str(path),
                        audio_sha256=sha256(path), snr_db=sum(cache.BANDS[band]) / 2,
                        noise=dict(contiguous_full_length=True), mix_id='test-mix')
            _, catalog = cache.noise_catalog(cfg['train_noise_manifest'], 'train')
            parent = dict(path=str(legacy_root), hashes={}, ffmpeg_version=FakeRTC.version, noise=catalog)
            with patch.object(cache, '_legacy', return_value=(legacy, dict(parent, noise={}))):
                with self.assertRaisesRegex(ValueError, 'noise catalogs differ'):
                    cache.prepare_cache(cfg, out, workers=1)
            self.assertFalse(out.exists())
            with patch.object(cache, '_legacy', return_value=(legacy, parent)):
                cache.prepare_cache(cfg, out, workers=1)
            rows, (raw, _) = cache.validate_cache(cfg, out)
            self.assertEqual(len(FakeRTC.calls), 4)
            for row in raw:
                if row['condition'] != 'noisy_a': continue
                old = Path(legacy[(row['source'], row['legacy_version'])]['resolved_audio'])
                if row['reuse'] == 'hardlink': self.assertEqual(os.stat(old).st_ino, os.stat(out / row['audio']).st_ino)
                old.unlink()  # Simulate root's explicit retirement, never done by prepare_cache.
            with patch.object(cache, '_legacy', side_effect=AssertionError('Retired cache must not be opened')):
                cache.prepare_cache(cfg, out, workers=1)
            self.assertEqual(len(cache.validate_cache(cfg, out)[0]), 8)

    def test_wrong_recipe_noise_leak_and_protected_destination_fail(self):
        with tempfile.TemporaryDirectory() as d, patch.object(cache, 'PairedRTC', FakeRTC):
            cfg, _, _ = cache_fixture(d); out = Path(cfg['train_noisy_cache_v33'])
            with self.assertRaisesRegex(ValueError, 'protected data'):
                cache.prepare_cache(cfg, Path(cfg['train_data_path']) / 'nested')
            cache.prepare_cache(cfg, out, workers=1)
            with self.assertRaisesRegex(ValueError, 'different seed'):
                cache.prepare_cache({**cfg, 'seed': cfg['seed'] + 1}, out, workers=1)
            dev_file = Path(cfg['dev_noisy_cache']) / 'config.json'
            dev = json.loads(dev_file.read_text()); dev['noise']['recording_ids'] = ['train-noise-1']; atomic_json(dev_file, dev)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                cache.prepare_cache(cfg, Path(d) / 'another_cache', workers=1)

    def test_webrtc_dsp_precedes_opus_and_preserves_partial_final_frame(self):
        events = []
        class APM:
            def __init__(self, **kwargs): events.append(('init', kwargs))
            def set_stream_format(self, *args): events.append(('format', args))
            def set_ns_level(self, value): events.append(('ns', value))
            def set_agc_target(self, value): events.append(('agc', value))
            def process_stream(self, blob):
                events.append(('dsp', len(blob)))
                return np.zeros(160, dtype='<i2').tobytes()
        rtc = object.__new__(cache.PairedRTC); rtc.apm_class = APM
        wave = np.ones(703, np.float32) * .1
        def command(args, blob):
            if '-c:a' in args:
                events.append(('opus', len(blob)))
                self.assertTrue(np.all(np.frombuffer(blob, dtype='<f4') == 0))
                return b'encoded'
            return np.zeros(len(wave), dtype='<f4').tobytes()
        rtc._run = command
        assignment = dict(family='webrtc', ns_level=2, agc_target_dbfs=12, rtc=dict(bitrate=24000))
        got = rtc.process(wave, assignment)
        self.assertEqual(len(got), len(wave))
        self.assertEqual(sum(k == 'dsp' for k, _ in events), 5)
        self.assertGreater(next(i for i, e in enumerate(events) if e[0] == 'opus'),
                           max(i for i, e in enumerate(events) if e[0] == 'dsp'))

    @unittest.skipUnless(importlib.util.find_spec('webrtc_audio_processing') and shutil.which('ffmpeg'),
                         'Actual WebRTC/Opus smoke requires server DSP dependencies')
    def test_actual_webrtc_and_ffmpeg_full_utterance_smoke(self):
        rtc = cache.PairedRTC('ffmpeg')
        wave = np.random.default_rng(127).normal(0, .02, 32073).astype(np.float32)
        assignment = cache.assignments([dict(id='offline/en/t', domain='offline', language='en', label=1)], 5)['offline/en/t']
        for a in assignment:
            result = rtc.process(wave, a)
            self.assertEqual(result.shape, wave.shape)
            self.assertTrue(np.isfinite(result).all())
            self.assertFalse(np.array_equal(result, wave))


if __name__ == '__main__': unittest.main()
