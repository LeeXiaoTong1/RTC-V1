"""Teacher provenance, all-sample pooling, real frontend and cache recovery tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from w2v_aasist.runtime import sha256
from . import language
from .language_cache import extract_language_cache, load_language_cache


class FakeEncoder:
    identity = {'teacher': 'fixed-fixture', 'policy': language.encoding_policy({})}

    def __init__(self, stop=None):
        self.calls = 0
        self.stop = stop

    def encode_waveforms(self, waves):
        self.calls += 1
        if self.calls == self.stop:
            raise RuntimeError('fixture interruption')
        result = np.zeros((len(waves), 256), dtype=np.float32)
        for i, wave in enumerate(waves):
            result[i, :3] = [1., np.mean(wave), np.std(wave)]
            result[i] /= np.linalg.norm(result[i])
        return result


def fixture(folder):
    rows = []
    for i, (length, sr) in enumerate(((3001, 16000), (8000, 8000), (2101, 16000))):
        path = (Path(folder)/f'{i}.wav').resolve()
        wave = np.random.default_rng(i).normal(0, .1, length).astype(np.float32)
        sf.write(path, wave, sr, subtype='FLOAT')
        stat = path.stat()
        rows.append(dict(id=f'offline/en/{i}.wav', source_id=str(i), group_id=str(i),
                         condition='offline', label=i % 2, language='en', split='train', view='full',
                         audio=str(path), audio_sha256=sha256(path), audio_size=stat.st_size,
                         audio_mtime_ns=stat.st_mtime_ns, output_samples=length*16000//sr, noisy=False))
    cfg = dict(language_commit_rows=1, language_free_margin_bytes=0)
    return rows, cfg


def close(bundle):
    bundle['lid']._mmap.close()


class LanguageCacheTests(unittest.TestCase):
    def test_recovery_exact_row_binding_resampling_and_completed_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            rows, cfg = fixture(folder); out = Path(folder)/'cache'; identity = dict(split='train', base='x')
            interrupted = FakeEncoder(stop=2)
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                extract_language_cache(rows, cfg, out, identity, interrupted)
            self.assertEqual(json.loads((out/'cursor.json').read_text())['cursor'], 1)
            encoder = FakeEncoder()
            resumed = extract_language_cache(rows, cfg, out, identity, encoder)
            self.assertEqual(encoder.calls, 2)
            clean = extract_language_cache(rows, cfg, Path(folder)/'clean', identity, FakeEncoder())
            np.testing.assert_array_equal(clean['lid'], resumed['lid'])
            np.testing.assert_allclose(np.linalg.norm(resumed['lid'], axis=1), 1., atol=1e-6)
            self.assertEqual(resumed['rows'], rows)
            reuse = FakeEncoder(stop=1)
            completed = extract_language_cache(rows, cfg, out, identity, reuse)
            self.assertEqual(reuse.calls, 0)
            self.assertEqual(resumed['lid'].nbytes, len(rows)*1024)
            for bundle in (resumed, clean, completed):
                close(bundle)
            with self.assertRaisesRegex(ValueError, 'identity/order'):
                load_language_cache(out, identity, rows[::-1])

    def test_partial_corruption_and_completed_corruption_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            rows, cfg = fixture(folder); out = Path(folder)/'cache'
            with self.assertRaises(RuntimeError):
                extract_language_cache(rows, cfg, out, {}, FakeEncoder(stop=2))
            arr = np.load(out/'lid.npy', mmap_mode='r+'); arr[0, 0] += .1; arr.flush(); arr._mmap.close()
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                extract_language_cache(rows, cfg, out, {}, FakeEncoder(stop=1))
            out = Path(folder)/'other'
            close(extract_language_cache(rows, cfg, out, {}, FakeEncoder()))
            arr = np.load(out/'lid.npy', mmap_mode='r+'); arr[-1, 0] += .1; arr.flush(); arr._mmap.close()
            with self.assertRaisesRegex(ValueError, 'file changed'):
                load_language_cache(out)

    def test_audio_hash_identity_dev_and_disk_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            rows, cfg = fixture(folder); out = Path(folder)/'cache'
            with self.assertRaisesRegex(ValueError, 'official Train'):
                extract_language_cache([dict(rows[0], split='dev')], cfg, out, {}, FakeEncoder())
            self.assertFalse(out.exists())
            with patch('w2v_v37.language_cache.shutil.disk_usage', return_value=type('Disk', (), {'free': 0})()):
                with self.assertRaisesRegex(OSError, 'no existing data deleted'):
                    extract_language_cache(rows, cfg, out, {}, FakeEncoder())
            self.assertFalse((out/'owner.json').exists())
            altered = [dict(rows[0], audio_sha256='f'*64)] + rows[1:]
            with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
                extract_language_cache(altered, cfg, out, {}, FakeEncoder(stop=1))
            with self.assertRaisesRegex(ValueError, 'changed frozen LID'):
                extract_language_cache(rows, cfg, out, {}, FakeEncoder())

    def test_uncommitted_tail_replays_and_changed_teacher_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            rows, cfg = fixture(folder); cfg['language_commit_rows'] = 2; out = Path(folder)/'cache'
            with self.assertRaises(RuntimeError):
                extract_language_cache(rows, cfg, out, {}, FakeEncoder(stop=3))
            self.assertEqual(json.loads((out/'cursor.json').read_text())['cursor'], 2)
            different = FakeEncoder(); different.identity = {'teacher': 'changed'}
            with self.assertRaisesRegex(ValueError, 'changed frozen LID'):
                extract_language_cache(rows, cfg, out, {}, different)
            encoder = FakeEncoder(); close(extract_language_cache(rows, cfg, out, {}, encoder))
            self.assertEqual(encoder.calls, 1)


class PoolingTests(unittest.TestCase):
    def encoder(self):
        encoder = object.__new__(language.LanguageEncoder)
        encoder.device = torch.device('cpu')
        encoder.policy = language.encoding_policy({'language_segment_seconds': .1})
        return encoder

    def test_all_samples_nonoverlap_balanced_duration_pooling_and_no_metadata(self):
        encoder = self.encoder(); seen = []
        def segment(wave):
            seen.append(wave.copy())
            result = np.zeros(256, dtype=np.float32)
            result[len(seen)-1] = 1.
            return result
        wave = np.arange(3201, dtype=np.float32)
        with patch.object(encoder, '_segment', side_effect=segment):
            result = encoder.encode_waveforms([wave])
        np.testing.assert_array_equal(np.concatenate(seen), wave)
        self.assertEqual([len(x) for x in seen], [1067, 1067, 1067])
        np.testing.assert_allclose(result[0, :3], 1/np.sqrt(3), atol=1e-7)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(encoder.encode_waveforms([]).shape, (0, 256))

    def test_invalid_input_and_nonfinite_pool_fail(self):
        encoder = self.encoder()
        for invalid in (np.array([]), np.array([np.nan]), np.ones((2, 3))):
            with self.assertRaises(ValueError):
                encoder.encode_waveforms([invalid])
        with patch.object(encoder, '_segment', return_value=np.zeros(256, dtype=np.float32)):
            with self.assertRaises(FloatingPointError):
                encoder.encode_waveforms([np.ones(10)])

    def test_real_speechbrain_frontend_tiny_ecapa_repeat_and_frozen_determinism(self):
        try:
            from speechbrain.lobes.features import Fbank
            from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
            from speechbrain.processing.features import InputNormalization
        except ImportError as exc:
            self.skipTest('Optional SpeechBrain frontend dependencies unavailable: ' + str(exc))
        torch.set_num_threads(1)
        encoder = self.encoder()
        encoder.features = Fbank(n_mels=60, left_frames=0, right_frames=0, deltas=False, sample_rate=16000)
        encoder.normalizer = InputNormalization(norm_type='sentence', std_norm=False)
        encoder.model = ECAPA_TDNN(input_size=60, channels=[32, 32, 32, 32, 96],
                                  attention_channels=8, se_channels=8, lin_neurons=256)
        for module in (encoder.features, encoder.normalizer, encoder.model):
            module.eval().requires_grad_(False)
        waves = [np.random.default_rng(3).normal(0, .1, 123).astype(np.float32),
                 np.random.default_rng(4).normal(0, .1, 3501).astype(np.float32)]
        first, second = encoder.encode_waveforms(waves), encoder.encode_waveforms(waves)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 256))
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1., atol=1e-6)
        self.assertFalse(any(p.requires_grad for p in encoder.model.parameters()))


class AssetTests(unittest.TestCase):
    def test_tensor_only_strict_load_with_real_normalizer_freezes_and_preserves_policy(self):
        from speechbrain.processing.features import InputNormalization
        with tempfile.TemporaryDirectory() as folder:
            files = {}
            expected = torch.nn.Linear(60, 256)
            for name in language.ASSET_FILES:
                path = Path(folder)/name
                if name == 'embedding_model.ckpt':
                    torch.save(expected.state_dict(), path)
                else:
                    # Deliberately invalid YAML is never interpreted or executed.
                    path.write_text('!unknown_remote_object: no-execution', encoding='utf-8')
                files[name] = dict(path=str(path), sha256=sha256(path), size=path.stat().st_size)
            assets = dict(format=language.ASSET_FORMAT, repo_id=language.REPO_ID,
                revision=language.REVISION, license='apache-2.0', speechbrain_version=language.SPEECHBRAIN_VERSION,
                embedding_dim=256, sample_rate=16000, files=files,
                encoding_policy=language.encoding_policy({'language_segment_seconds': 2.}))
            modules = (torch.nn.Identity(), InputNormalization(norm_type='sentence', std_norm=False),
                       torch.nn.Linear(60, 256))
            before = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
            def components():
                torch.backends.cuda.matmul.allow_tf32 = not before[0]
                torch.backends.cudnn.allow_tf32 = not before[1]
                return modules
            with patch.object(language, 'WEIGHT_SHA256', files['embedding_model.ckpt']['sha256']), \
                    patch.object(language, 'WEIGHT_BYTES', files['embedding_model.ckpt']['size']), \
                    patch.object(language, '_components', side_effect=components), \
                    patch('torch.load', wraps=torch.load) as load:
                encoder = language.LanguageEncoder(assets, device='cpu')
                self.assertTrue(load.call_args.kwargs['weights_only'])
            self.assertEqual(encoder.policy['segment_samples'], 32000)
            self.assertEqual(before, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
            self.assertTrue(all(not m.training for m in modules))
            self.assertFalse(any(p.requires_grad for m in modules for p in m.parameters()))
            torch.testing.assert_close(expected.weight, encoder.model.weight, rtol=0, atol=0)

    def test_only_pinned_required_files_cached_no_remote_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            files = {}
            for name in language.ASSET_FILES:
                path = Path(folder)/name; path.write_bytes(('fixture ' + name).encode()); files[name] = str(path)
            weights = Path(files['embedding_model.ckpt'])
            manifest = Path(folder)/'manifest.json'
            def cached(**kwargs):
                self.assertEqual(kwargs['repo_id'], language.REPO_ID)
                self.assertEqual(kwargs['revision'], language.REVISION)
                self.assertTrue(kwargs['local_files_only'])
                return files[kwargs['filename']]
            with patch('huggingface_hub.hf_hub_download', side_effect=cached) as download, \
                    patch.object(language, 'WEIGHT_SHA256', sha256(weights)), \
                    patch.object(language, 'WEIGHT_BYTES', weights.stat().st_size):
                assets = language.ensure_language_assets(dict(language_cache_dir=folder,
                    language_assets_manifest=str(manifest), language_offline=True, language_segment_seconds=2.))
                self.assertEqual(download.call_count, 3)
                self.assertEqual(assets['encoding_policy']['segment_samples'], 32000)
                self.assertEqual(json.loads(manifest.read_text()), assets)
                self.assertNotIn('path', language.assets_identity(assets)['files']['embedding_model.ckpt'])
                weights.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'asset changed'):
                    language.verify_language_assets(assets)

    def test_unpinned_revision_rejected_before_download(self):
        with patch('huggingface_hub.hf_hub_download') as download:
            with self.assertRaisesRegex(ValueError, 'pinned official'):
                language.ensure_language_assets({'language_revision': 'main'})
            download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
