"""Failure-path checks: storage exhaustion must not corrupt checkpoints/features."""
import errno
from contextlib import nullcontext
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .core import atomic_save
from .data import FeatureCollator
from .feature_cache import FeatureCache, FEATURE_FORMAT_HASH
from .storage import (GIB, MIB, checkpoint_headroom, checkpoint_sizes,
                      prune_feature_files, require_space, storage_bytes)


class StorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_unique_tensor_backing_storage_and_adam_size(self):
        t = torch.arange(12, dtype=torch.float32)
        self.assertEqual(storage_bytes({'a': t, 'b': [t[2:4], t]}), 48)
        model = torch.nn.Linear(4, 2)
        weights, full = checkpoint_sizes(model)
        self.assertEqual(weights, 40+64*MIB)
        self.assertEqual(full, 120+128*MIB)

    def test_budget_includes_future_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(checkpoint_headroom(root, 2*GIB, 6*GIB), 15*GIB)
            (root/'best_model.pt').touch()
            self.assertEqual(checkpoint_headroom(root, 2*GIB, 6*GIB), 13*GIB)
            (root/'last.pt').touch()
            self.assertEqual(checkpoint_headroom(root, 2*GIB, 6*GIB), 7*GIB)

    def test_low_disk_rejected_before_checkpoint_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'last.pt'
            original = b'previous complete checkpoint'
            path.write_bytes(original)
            with patch('w2v_rebuild.storage.available_bytes', return_value=0), patch('torch.save') as save:
                with self.assertRaises(OSError) as error:
                    atomic_save({'model': torch.zeros(2)}, path)
                self.assertEqual(error.exception.errno, errno.ENOSPC)
                save.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(path.with_name('last.pt.tmp').exists())

    def test_partial_write_keeps_previous_checkpoint_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'last.pt'
            path.write_bytes(b'complete')
            def fail(value, tmp):
                Path(tmp).write_bytes(b'partial')
                raise RuntimeError('PytorchStreamWriter failed writing file')
            with patch('w2v_rebuild.storage.available_bytes', return_value=10*GIB), patch('torch.save', side_effect=fail):
                with self.assertRaises(RuntimeError):
                    atomic_save({'model': torch.zeros(2)}, path)
            self.assertEqual(path.read_bytes(), b'complete')
            self.assertFalse(path.with_name('last.pt.tmp').exists())

    def test_feature_floor_and_enospc_recompute_identical_inputs(self):
        from transformers import SeamlessM4TFeatureExtractor
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            extractor = SeamlessM4TFeatureExtractor()
            extractor.save_pretrained(root/'model')
            plain = FeatureCollator(root/'model')
            plain.extractor = extractor
            wave = torch.randn(64600, generator=torch.Generator().manual_seed(83))*.02
            rows = [(wave, 0, 'a'), (wave, 0, 'a')]
            expected = plain(rows)
            for label, space, failure in [('floor', 0, None), ('full', 10*GIB, errno.ENOSPC),
                                          ('quota', 10*GIB, getattr(errno, 'EDQUOT', 122))]:
                cached = FeatureCollator(root/'model', cache_root=root/label)
                cached.extractor = extractor
                def save(file, value, **kwargs):
                    file.write(b'partial')
                    raise OSError(failure, 'simulated full storage')
                with patch('w2v_rebuild.feature_cache.available_bytes', return_value=space):
                    if failure:
                        with patch('w2v_rebuild.feature_cache.np.save', side_effect=save):
                            actual = cached(rows)
                    else:
                        actual = cached(rows)
                torch.testing.assert_close(actual['features'], expected['features'], rtol=0, atol=0)
                torch.testing.assert_close(actual['mask'], expected['mask'], rtol=0, atol=0)
                self.assertEqual(list((root/label).rglob('*.npy')), [])
                self.assertEqual(list((root/label).rglob('*.tmp')), [])

    def test_existing_cache_hit_at_zero_free_space_and_legacy_identity(self):
        from transformers import SeamlessM4TFeatureExtractor
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            extractor = SeamlessM4TFeatureExtractor()
            extractor.save_pretrained(root/'model')
            cache = FeatureCache(root/'cache', root/'model', extractor)
            # The pre-fix Linux identity must still read existing disk entries.
            identity = {'schema': 'w2v_official_fixed_pcm_v1', 'cut': 64600, 'sr': 16000,
                        'config': hashlib.sha256((root/'model'/'preprocessor_config.json').read_bytes()).hexdigest(),
                        'extractor': hashlib.sha256(inspect.getsource(type(extractor)).encode()).hexdigest(),
                        'implementation': FEATURE_FORMAT_HASH,
                        'collator': hashlib.sha256(Path(__file__).with_name('data.py').read_bytes()).hexdigest(),
                        'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'torch', 'transformers')}}
            self.assertEqual(cache.root.name, hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())
            wave = torch.zeros(64600)
            path = cache.path(wave)
            path.parent.mkdir(parents=True)
            value = np.ones((200, 160), dtype=np.float32)
            np.save(path, value)
            with patch('w2v_rebuild.feature_cache.available_bytes', return_value=0):
                actual, _ = cache.transform([wave], lambda _: self.fail('existing cache must be read'))
            np.testing.assert_array_equal(actual[0].numpy(), value)

    def test_cleanup_only_generated_feature_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shard = root/('a'*64)/'bb'
            shard.mkdir(parents=True)
            feature = shard/('b'*64+'.npy')
            feature.write_bytes(b'generated')
            keep = [root/'best_model.pt', shard/'audio.wav', shard/'notes.npy', shard/('c'*64+'.npy')]
            for path in keep:
                path.write_bytes(b'keep')
            count, size = prune_feature_files(root)
            self.assertEqual((count, size), (1, 9))
            self.assertFalse(feature.exists())
            self.assertTrue(all(p.read_bytes() == b'keep' for p in keep))

    def test_cleanup_does_not_follow_namespace_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            namespace = root/('a'*64)
            shard = namespace/'bb'
            shard.mkdir(parents=True)
            feature = shard/('b'*64+'.npy')
            feature.write_bytes(b'outside')
            original = Path.is_symlink
            with patch.object(Path, 'is_symlink', lambda p: p == namespace or original(p)):
                self.assertEqual(prune_feature_files(root), (0, 0))
            self.assertEqual(feature.read_bytes(), b'outside')

    def test_recovery_preview_then_launch_preserves_baseline_and_audio(self):
        import recover_w2v_storage as recovery
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            failed = root/'exp'/'failed'
            (failed/'stage3').mkdir(parents=True)
            baseline = root/'original.pt'
            baseline.write_bytes(b'original model')
            torch.save({'epoch': 0, 'kind': 'weights', 'model': {'w': torch.zeros(2)}},
                       failed/'stage3'/'best_model.pt')
            cache = root/'audio_cache'
            cache.mkdir()
            for name in ('config.json', 'manifest.jsonl'):
                (cache/name).write_text('{}')
            audio = cache/'audio.wav'
            audio.write_bytes(b'keep audio')
            feature_root = root/'data'/'w2v_feature_cache'
            feature = feature_root/('a'*64)/'bb'/('b'*64+'.npy')
            feature.parent.mkdir(parents=True)
            feature.write_bytes(b'generated')
            config = dict(stage=3, adaptation=True, baseline_path=str(baseline), feature_cache=str(feature_root),
                          train_noisy_cache=str(cache), dev_noisy_cache=str(cache), dev_heldout_cache=str(cache),
                          noise_environment={'RTC_B_NOISE_PROB': '0.5'}, epochs=8)
            config_file = failed/'stage3'/'config.json'
            config_file.write_text(json.dumps(config))
            argv = ['recover', '--failed-run', str(failed), '--release-feature-cache']
            with patch.object(recovery, 'ROOT', root), patch('sys.argv', argv), \
                    patch.object(recovery.subprocess, 'run') as launch:
                recovery.main()
                launch.assert_not_called()
            self.assertTrue(feature.exists())
            with patch.object(recovery, 'ROOT', root), patch('sys.argv', argv+['--run']), \
                    patch.object(recovery, 'recovery_lock', return_value=nullcontext()), \
                    patch.object(recovery.subprocess, 'run') as launch:
                recovery.main()
            self.assertEqual(launch.call_count, 2)
            self.assertIn('--preflight', launch.call_args_list[0].args[0])
            self.assertNotIn('--preflight', launch.call_args_list[1].args[0])
            self.assertEqual(launch.call_args.kwargs['env']['RTC_B_NOISE_PROB'], '0.5')
            self.assertFalse(feature.exists())
            self.assertEqual(baseline.read_bytes(), b'original model')
            self.assertEqual(audio.read_bytes(), b'keep audio')
            self.assertEqual(json.loads(config_file.read_text()), config)
            self.assertTrue((failed/'stage3'/'best_model.pt').is_file())

    def test_recovery_keeps_learning_and_data_settings(self):
        from recover_w2v_storage import training_command
        from .train import parser
        config = dict(stage=3, out='old', ssl_path='ssl', train_data_path='train', dev_data_path='dev',
                      train_protocol='tp', dev_protocol='dp', rtc_pairs='pairs', train_noise_manifest='noise',
                      train_noisy_cache='original', extra_train_noisy_cache=['extra'], dev_noisy_cache='seen',
                      dev_heldout_cache='held', ordinary_sampling='balanced', noisy_bank_policy='mixed',
                      consistency_weight=.02, consistency_confidence=.8, epochs=8, encoder_lr=2e-7,
                      head_lr=5e-6, amp='bf16', microbatch=4, eval_microbatch=4, feature_cache='cache',
                      algo=5, resume='wrong', preflight=False)
        for preflight in (True, False):
            command = training_command(config, 'new/stage3', 'baseline.pt', preflight)
            args = parser().parse_args(command[4:])
            for key, value in config.items():
                if key not in ('out', 'feature_cache', 'resume', 'preflight'):
                    self.assertEqual(getattr(args, key), value)
            self.assertEqual(args.out, 'new/stage3')
            self.assertEqual(args.finetune_from, 'baseline.pt')
            self.assertIsNone(args.resume)
            self.assertIsNone(args.feature_cache)
            self.assertTrue(args.no_feature_cache)
            self.assertEqual(args.preflight, preflight)


if __name__ == '__main__':
    unittest.main(verbosity=2)
