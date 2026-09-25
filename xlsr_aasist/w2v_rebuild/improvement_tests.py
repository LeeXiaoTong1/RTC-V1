"""Regression checks for the new recipe and semantics-preserving optimizations.

python -m unittest w2v_rebuild.improvement_tests -v
Actual FFmpeg checks run when RTC_TEST_FFMPEG points to an executable.
"""
import copy
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import wavfile
import torch
from torch.nn import functional as F
from .sampling import BalancedOrdinarySampler
from .data import SeededDataset, FeatureCollator, combine_batches
from .core import objective, pair_loss, prediction_consistency, materialize_stats
from .tests import RandomDataset
from rtc_noisy.diverse import room_response, webrtc_process, DiverseRTC, PROFILES, profile_definition
from rtc_noisy.simulator import RTCSettings
from rtc_noisy_v2.cache import check_metadata, check_suite
from rtc_noisy_v2.plan import SCHEMA, PLAN_ID, plan_definition, settings_for
from rtc_noisy_v2.sampling import RotatingViewBatchSampler
from utils.env_noise import NoiseAugment, active_mask, mix_at_snr


class SourceSampler:
    def __init__(self, count=32): self.count = count
    def __len__(self): return 1
    def set_epoch(self, epoch): pass
    def __iter__(self): yield list(range(self.count))


def cache_fixture(role, profile=None):
    config = {'optimization_schema': SCHEMA, 'role': role, 'plan_id': PLAN_ID,
              'plan': plan_definition(), 'allowed_settings': [list(x) for x in settings_for(role)],
              'generation': 0, 'seed': 1234, 'ffmpeg_version': 'fixture',
              'noise': {'recording_ids': [role == 'train'], 'file_sha256': [role == 'train']}}
    if profile:
        config.update(processing=profile_definition(profile), webrtc_version='0.1.3')
    row = {'source': 'offline/a.wav', 'band': 0, 'source_sha256': 'a'*64, 'label': 0,
           'mix_id': 'b'*64, 'role': role, 'generation': 0,
           'rtc': dict(zip(('noise_reduction', 'max_gain', 'bitrate'), settings_for(role)[0]))}
    if profile:
        row['processing'] = {'family': PROFILES[profile][0]}
    return [row], config


class Improvements(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_balanced_budget_replay_coverage_and_new_draws(self):
        labels = [0]*17 + [1]*3
        sampler = BalancedOrdinarySampler(labels, 6, 4, 73)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        seen, draws = set(), set()
        for epoch in (1, 2, 3):
            sampler.set_epoch(epoch)
            for batch in sampler:
                self.assertEqual([labels[i] for i, _ in batch].count(1), 3)
                for index, draw in batch:
                    self.assertNotIn(draw, draws)
                    draws.add(draw); seen.add(index)
        self.assertEqual(seen, set(range(20)))
        resumed = BalancedOrdinarySampler(labels, 6, 4, 73)
        resumed.set_epoch(3)
        self.assertEqual(list(resumed), list(sampler))
        ds = SeededDataset(RandomDataset(), 7)
        torch.testing.assert_close(ds[(1, 3)], ds[(1, 3)], rtol=0, atol=0)
        self.assertFalse(torch.equal(ds[(1, 3)], ds[(1, 4)]))

    def test_mixed_banks_start_immediately_cover_all_views_and_resume(self):
        sources = [{'offline': f'offline/{i}.wav'} for i in range(32)]
        s = RotatingViewBatchSampler(SourceSampler(), sources, 73, 2, 'mixed')
        history = [[] for _ in sources]
        for epoch in range(1, 9):
            s.set_epoch(epoch)
            batches = list(s)
            if epoch == 1:
                self.assertEqual({bank for _, bank, _ in batches[0]}, {0, 1})
            for i, bank, band in batches[0]: history[i].append((bank, band))
            s.commit_epoch(1)
        self.assertTrue(all(len(set(views)) == 8 for views in history))
        restored = RotatingViewBatchSampler(SourceSampler(), sources, 73, 2, 'mixed')
        restored.load_state_dict(s.state_dict())
        s.set_epoch(9); restored.set_epoch(9)
        self.assertEqual(list(s), list(restored))
        legacy = RotatingViewBatchSampler(SourceSampler(), sources, 73, 2)
        with self.assertRaises(ValueError): legacy.load_state_dict(restored.state_dict())

    def test_consistency_only_correct_confident_teacher_and_stop_gradient(self):
        reference = torch.tensor([[5., 0.], [5., 0.], [0., 0.], [0., 5.]], requires_grad=True)
        processed = torch.zeros(4, 2, requires_grad=True)
        loss, accepted = prediction_consistency(reference, processed, torch.tensor([0, 1, 1, 1]), .8)
        self.assertEqual(float(accepted), .5)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertTrue(torch.equal(processed.grad[1:3], torch.zeros(2, 2)))
        self.assertLess(float(processed.grad[0, 0]), 0)
        self.assertLess(float(processed.grad[3, 1]), 0)

    def test_no_accepted_reference_is_finite_zero(self):
        z = torch.zeros(4, 2, requires_grad=True)
        loss, accepted = prediction_consistency(z, z, torch.arange(4)%2)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(accepted), 0.)
        self.assertTrue(torch.equal(z.grad, torch.zeros_like(z)))

    def test_disabled_consistency_and_batched_stats_preserve_loss_and_gradients(self):
        torch.manual_seed(31)
        z, h = torch.randn(40, 2, requires_grad=True), torch.randn(40, 16, requires_grad=True)
        y, weights = torch.arange(40)%2, torch.tensor([.617, 2.640])
        a, stats = objective(z, h, y, 24, 4, 4, weights, .1, .1, tensor_stats=True, check_labels=False)
        ce = F.cross_entropy(z, y, reduction='none')
        w = weights[y[:24]]
        reference = (.7*24/36*((ce[:24]*w).sum()/w.sum()) + .7*8/36*ce[24:32].mean()
                     + .7*4/36*ce[32:36].mean() + .3*ce[36:].mean()
                     + .1*pair_loss(h[24:28], h[28:32], y[24:28])
                     + .1*pair_loss(h[32:36], h[36:], y[32:36]))
        torch.testing.assert_close(a, reference, rtol=0, atol=0)
        ga = torch.autograd.grad(a, (z, h), retain_graph=True)
        gb = torch.autograd.grad(reference, (z, h))
        for left, right in zip(ga, gb): torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertIsInstance(materialize_stats(stats)['ce'], float)

    def test_official_feature_cache_bitwise_equivalence_hits_and_invalidations(self):
        from transformers import SeamlessM4TFeatureExtractor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extractor = SeamlessM4TFeatureExtractor()
            extractor.save_pretrained(root/'model')
            plain = FeatureCollator(root/'model')
            cached = FeatureCollator(root/'model', cache_root=root/'cache')
            plain.extractor = cached.extractor = extractor
            rng = torch.Generator().manual_seed(8)
            rows = [(torch.randn(64600, generator=rng)*.02, i%2, str(i)) for i in range(3)]
            rows.append(rows[0])
            expected, actual = plain(rows), cached(rows)
            torch.testing.assert_close(expected['features'], actual['features'], rtol=0, atol=0)
            torch.testing.assert_close(expected['mask'], actual['mask'], rtol=0, atol=0)
            with patch.object(cached, 'extract', side_effect=AssertionError('Cache hit re-extracted features')):
                torch.testing.assert_close(cached(list(reversed(rows)))['features'], actual['features'].flip(0), rtol=0, atol=0)
            count = len(list((root/'cache').rglob('*.npy')))
            cached([(rows[0][0]*.9, 0, 'changed')])
            self.assertEqual(len(list((root/'cache').rglob('*.npy'))), count+1)
            damaged = cached.cache.path(rows[0][0])
            np.save(damaged, np.full((20, 160), np.nan, dtype=np.float32))
            with self.assertRaises(ValueError): cached([rows[0]])

    def test_noise_cache_and_mask_reuse_are_bitwise_equal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            noise = (np.random.RandomState(9).normal(size=96000)*1000).astype(np.int16)
            wavfile.write(path/'noise.wav', 24000, noise)
            manifest = path/'train.jsonl'
            manifest.write_text(json.dumps({'path': str(path/'noise.wav'), 'split': 'train'}), encoding='utf-8')
            a, b = NoiseAugment(manifest, 1.), NoiseAugment(manifest, 1.)
            a.cache_bytes = 0
            x = np.random.RandomState(4).normal(size=64600).astype(np.float32)*.02
            for seed in range(3):
                left, lm = a(x, 16000, np.random.RandomState(seed), return_info=True)
                right, rm = b(x, 16000, np.random.RandomState(seed), return_info=True)
                np.testing.assert_array_equal(left, right)
                self.assertEqual(lm, rm)
            self.assertLessEqual(b._bytes, b.cache_bytes)
            clone = pickle.loads(pickle.dumps(b))
            np.testing.assert_array_equal(clone(x, 16000, np.random.RandomState(7)), b(x, 16000, np.random.RandomState(7)))
            n = np.random.RandomState(42).normal(size=len(x))
            np.testing.assert_array_equal(mix_at_snr(x, n, 16000, 10)[0],
                                          mix_at_snr(x, n, 16000, 10, active_mask(x, 16000))[0])

    def test_processing_roles_and_algorithm_holdout(self):
        train, seen, held = cache_fixture('train', 'diverse'), cache_fixture('dev_seen', 'diverse'), cache_fixture('dev_heldout', 'unseen')
        for rows, cfg in (train, seen, held): check_metadata(cfg, rows, cfg['role'])
        check_suite([train], seen, held)
        bad = copy.deepcopy(train)
        bad[1]['processing'] = profile_definition('unseen')
        with self.assertRaises(ValueError): check_metadata(bad[1], bad[0], 'train')
        bad = copy.deepcopy(held)
        bad[1]['processing']['families'] = ['webrtc']
        with self.assertRaises(ValueError): check_suite([train], seen, bad)

    def test_room_is_deterministic_finite_same_length(self):
        x = np.random.RandomState(3).normal(size=70321).astype(np.float32)*.02
        a, metadata = room_response(x, np.random.RandomState(1))
        b, _ = room_response(x, np.random.RandomState(1))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), len(x))
        self.assertTrue(metadata['applied'])
        self.assertTrue(np.isfinite(a).all())

    def test_webrtc_adapter_frames_fresh_state_and_tail_with_test_double(self):
        # Tests binding integration contract; the server smoke command tests native DSP.
        from types import SimpleNamespace
        instances = []
        class Processor:
            def __init__(self, **kwargs): instances.append(self); self.lengths = []; self.kwargs = kwargs
            def set_stream_format(self, rate, channels): self.format = (rate, channels)
            def set_ns_level(self, level): self.ns = level
            def set_agc_target(self, target): self.target = target
            def process_stream(self, data): self.lengths.append(len(data)); return data
        x = np.linspace(-.4, .4, 321).astype(np.float32)
        with patch.dict('sys.modules', {'webrtc_audio_processing': SimpleNamespace(AudioProcessingModule=Processor)}):
            a = webrtc_process(x, 2, 12); b = webrtc_process(x, 2, 12)
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[0].lengths, [320, 320, 320])
        self.assertEqual(instances[0].kwargs['agc_type'], 1)
        self.assertEqual(len(a), len(x)); np.testing.assert_array_equal(a, b)

    @unittest.skipUnless(os.environ.get('RTC_TEST_FFMPEG'), 'Set RTC_TEST_FFMPEG for native FFmpeg regression')
    def test_native_ffmpeg_heldout_length_and_determinism(self):
        rtc = DiverseRTC(os.environ['RTC_TEST_FFMPEG'], 'unseen')
        x = np.random.RandomState(7).normal(size=70321).astype(np.float32)*.02
        a, _ = rtc(x, RTCSettings(), np.random.RandomState(3))
        b, _ = rtc(x, RTCSettings(), np.random.RandomState(3))
        self.assertEqual(len(a), len(x)); np.testing.assert_array_equal(a, b)


if __name__ == '__main__': unittest.main(verbosity=2)
