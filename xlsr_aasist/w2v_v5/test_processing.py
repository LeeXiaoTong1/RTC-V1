import copy
import unittest
from unittest.mock import patch

import numpy as np

from .processing import (MIN_SAMPLES, SR, candidate_recipes,
    generate_matched_context, policy, policy_hash, preceding_context,
    prepare_wave, standardize_wave, validate_policy)


class InputProcessingTests(unittest.TestCase):
    def test_identity_preserves_original_and_legacy_minimum_padding(self):
        wave = np.arange(1500, dtype=np.float32) / 2000
        out, meta = prepare_wave(wave)
        np.testing.assert_array_equal(out[:len(wave)], wave)
        np.testing.assert_array_equal(out, np.pad(wave, (0, MIN_SAMPLES - len(wave)), mode='reflect'))
        self.assertEqual(meta['pcl_valid_frames'], (len(wave) - 400) // 320 + 1)
        self.assertEqual(meta['repeated_samples'], 0)

    def test_repeat_retains_all_original_content_and_only_original_phone_scope(self):
        wave = np.sin(np.arange(8000) / 100).astype(np.float32) * .1
        out, meta = prepare_wave(wave, policy(short_seconds=3.))
        self.assertEqual(len(out), 3 * SR)
        np.testing.assert_array_equal(out[:8000], wave)
        np.testing.assert_array_equal(out[8000:16000], wave)
        self.assertEqual(meta['pcl_valid_samples'], 8000)
        self.assertEqual(meta['pcl_valid_frames'], (8000 - 400) // 320 + 1)
        self.assertIn([8000, 48000], meta['pcl_excluded_spans'])
        self.assertEqual(meta['time_map']['offset'], 0.)

    def test_repeat_never_crops_a_long_recording(self):
        wave = np.linspace(-.1, .1, 5 * SR, dtype=np.float32)
        out, meta = prepare_wave(wave, policy(short_seconds=3.))
        np.testing.assert_array_equal(out, wave)
        self.assertEqual(meta['repeated_samples'], 0)

    def test_lowpass_is_aligned_and_attenuates_high_frequency_without_resampling(self):
        samples = np.arange(2 * SR)
        low = np.sin(2 * np.pi * 500 * samples / SR)
        high = np.sin(2 * np.pi * 7600 * samples / SR)
        wave = (.1 * low + .1 * high).astype(np.float32)
        out, meta = standardize_wave(wave, policy('lowpass'))
        self.assertEqual(out.shape, wave.shape)
        span = slice(200, -200)
        low_amp = abs(np.dot(out[span], low[span])) / np.dot(low[span], low[span])
        high_amp = abs(np.dot(out[span], high[span])) / np.dot(high[span], high[span])
        self.assertGreater(low_amp, .098)
        self.assertLess(high_amp, .002)
        impulse = np.zeros(8000, np.float32)
        impulse[4000] = .5
        aligned, _ = standardize_wave(impulse, policy('lowpass'))
        self.assertEqual(int(np.argmax(aligned)), 4000)
        self.assertEqual(meta['filter_guard_samples'], 32)

    def test_loudness_is_deterministic_bounded_and_does_not_hard_clip(self):
        wave = np.array([.01, -.02, .005] * 1500, np.float32)
        first, meta = standardize_wave(wave, policy('loudness'))
        second, _ = standardize_wave(wave, policy('loudness'))
        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(meta['gain'], 10 ** (6 / 20) + 1e-6)
        loud = np.array([2., -2., .5] * 2000, np.float32)
        bounded, _ = standardize_wave(loud, policy('loudness'))
        self.assertLessEqual(float(np.abs(bounded).max()), .980001)
        np.testing.assert_allclose(bounded / loud, np.full(len(loud), bounded[0] / loud[0]))
        silent, details = standardize_wave(np.zeros(8000, np.float32), policy('loudness'))
        self.assertEqual(details['gain'], 1.)
        self.assertTrue(np.isfinite(silent).all())

    def test_frozen_recipe_rejects_silent_train_inference_parameter_drift(self):
        cfg = policy('lowpass', 3.)
        self.assertEqual(validate_policy(cfg), cfg)
        self.assertEqual(policy_hash(cfg), policy_hash(copy.deepcopy(cfg)))
        changed = dict(cfg, lowpass_hz=6000.)
        with self.assertRaisesRegex(ValueError, 'versioned'):
            validate_policy(changed)
        self.assertNotEqual(policy_hash(policy()), policy_hash(cfg))
        self.assertEqual(candidate_recipes()[0]['policy'], policy())

    def test_invalid_audio_and_durations_fail_early(self):
        for invalid in ([], [float('nan')], [[1., 2.]], [float('inf')]):
            with self.assertRaises(ValueError):
                prepare_wave(invalid)
        for seconds in (-1, 7, float('nan')):
            with self.assertRaises(ValueError):
                policy(short_seconds=seconds)
        out, meta = prepare_wave(np.array([.2], np.float32))
        self.assertEqual(len(out), MIN_SAMPLES)
        self.assertEqual(meta['pcl_valid_frames'], 0)


class ReceiverContextTests(unittest.TestCase):
    def test_context_only_uses_same_source_and_handles_short_clip(self):
        wave = np.array([.1, .2, .3], np.float32)
        context = preceding_context(wave, 10 / SR)
        np.testing.assert_array_equal(context, wave[np.arange(-10, 0) % 3])
        self.assertEqual(len(preceding_context(wave, 0)), 0)

    def test_single_extended_processing_pass_and_correct_target_exclusion_geometry(self):
        wave = np.linspace(-.1, .1, 8000, dtype=np.float32)
        prefix = 1600
        rec = {'seed': 123, 'family': 'webrtc'}
        calls = []
        def generator(stream, given, bank, engines):
            calls.append((stream.copy(), given, bank, engines))
            meta = dict(recipe=given, pair_eligible=True, common_gain=.9,
                erased_spans=[[0, prefix - 5], [prefix - 20, prefix + 100], [9000, 9600]],
                rtc_activity=dict(samples=len(stream), suppressed_samples=140,
                    suppressed_spans=[[prefix - 20, prefix + 100]]),
                clean_activity=dict(samples=len(stream), suppressed_samples=0, suppressed_spans=[]))
            return stream * .9, stream * .8, meta
        with patch('w2v_v3189.augment.generate_matched', side_effect=generator):
            clean, noisy, meta = generate_matched_context(wave, rec, 'bank', 'engines', context_seconds=.1)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0][prefix:], wave)
        np.testing.assert_array_equal(clean, wave * .9)
        np.testing.assert_array_equal(noisy, wave * .8)
        self.assertEqual(meta['erased_spans'], [[0., 100.], [7400., 8000.]])
        self.assertEqual(meta['rtc_activity']['suppressed_spans'], [[0., 100.]])
        self.assertEqual(meta['receiver_context']['time_map']['offset'], 0.)
        self.assertEqual(meta['rtc_activity']['statistics_scope'], 'context_plus_target')
        self.assertEqual(rec, {'seed': 123, 'family': 'webrtc'})

    def test_context_actually_changes_state_seen_by_target_without_cross_view_state(self):
        wave = np.ones(8000, np.float32) * .05
        def generator(stream, *args):
            # A simple causal state estimator, reset independently per view.
            def run(x):
                state = 0.
                values = []
                for sample in x:
                    state = .999 * state + .001 * float(sample)
                    values.append(sample / (1 + state))
                return np.array(values, np.float32)
            return run(stream), run(stream), dict(pair_eligible=True)
        with patch('w2v_v3189.augment.generate_matched', side_effect=generator):
            cold, _, _ = generate_matched_context(wave, {}, None, None, context_seconds=0)
            warm, paired, _ = generate_matched_context(wave, {}, None, None, context_seconds=.2)
            again, _, _ = generate_matched_context(wave, {}, None, None, context_seconds=.2)
        self.assertNotEqual(float(cold[0]), float(warm[0]))
        np.testing.assert_array_equal(warm, paired)
        np.testing.assert_array_equal(warm, again)

    def test_context_disabled_reproduces_inherited_output(self):
        wave = np.linspace(-.1, .1, 8000, dtype=np.float32)
        metadata = dict(pair_eligible=True, erased_spans=[[5, 10]])
        def generator(stream, *_):
            return stream * .8, stream * .7, metadata
        with patch('w2v_v3189.augment.generate_matched', side_effect=generator):
            clean, noisy, meta = generate_matched_context(wave, {}, None, None, context_seconds=0)
        np.testing.assert_array_equal(clean, wave * .8)
        np.testing.assert_array_equal(noisy, wave * .7)
        self.assertEqual(meta['erased_spans'], [[5, 10]])
        self.assertFalse(meta['receiver_context']['enabled'])
        self.assertNotIn('receiver_context', metadata)

    def test_unverified_processor_geometry_is_rejected(self):
        def bad(stream, *_):
            return stream[:-1], stream, {}
        with patch('w2v_v3189.augment.generate_matched', side_effect=bad):
            with self.assertRaisesRegex(ValueError, 'length'):
                generate_matched_context(np.ones(8000), {}, None, None)


if __name__ == '__main__':
    unittest.main()
