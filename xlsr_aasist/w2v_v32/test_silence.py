"""Local zeroing stays rare, conservative and part of the valid waveform."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from w2v_v31.test_data_step import write_audio
from .augment import apply_plan, process_wave, processing_plan, processing_settings
from .data import AudioDataset, ViewCollator
from .silence import apply_local_silence

torch.set_num_threads(1)
RATE = 16000


def parameters(**changes):
    return dict(dict(operator='local_silence', duration_seconds=.16,
                     position=.5, maximum_fraction=.05, fade_seconds=.005), **changes)


def forced_silence():
    return dict(processing_identity_probability=0., processing_single_probability=0.,
                processing_silence_probability=1.)


class SilenceWaveformTests(unittest.TestCase):
    def test_zero_interior_smooth_edges_and_no_input_mutation(self):
        wave = np.ones(8*RATE, dtype=np.float32)
        before = wave.copy()
        result, meta = apply_local_silence(wave, parameters())
        self.assertTrue(meta['silence_applied'])
        start, count = meta['silence_start_sample'], meta['silence_samples']
        self.assertEqual(count, 2560)
        np.testing.assert_array_equal(result[:start], wave[:start])
        np.testing.assert_array_equal(result[start+count:], wave[start+count:])
        self.assertTrue(np.all(result[start+80:start+count-80] == 0))
        self.assertEqual(meta['silence_zero_samples'], int((result == 0).sum()))
        self.assertEqual(result[start], 1.)
        self.assertEqual(result[start+count-1], 1.)
        self.assertLess(float(np.max(np.abs(np.diff(result)))), .021)
        self.assertEqual(result.shape, wave.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertFalse(np.shares_memory(result, wave))
        np.testing.assert_array_equal(wave, before)

    def test_duration_limit_includes_ramps_and_handles_endpoint_positions(self):
        wave = np.ones(RATE, dtype=np.float32)
        for position, expected_start in ((0., 0), (1., RATE-800)):
            with self.subTest(position=position):
                result, meta = apply_local_silence(wave, parameters(position=position))
                self.assertTrue(meta['silence_applied'])
                self.assertEqual(meta['silence_samples'], 800)
                self.assertEqual(meta['silence_start_sample'], expected_start)
                self.assertLessEqual(int((result != wave).sum()), .05*len(wave))

    def test_minimum_supported_length_and_shorter_recordings(self):
        for samples, expected in ((6720, False), (12799, False), (12800, True)):
            with self.subTest(samples=samples):
                wave = np.ones(samples, dtype=np.float32)
                result, meta = apply_local_silence(wave, parameters())
                self.assertEqual(meta['silence_applied'], expected)
                if expected:
                    self.assertEqual(meta['silence_samples'], 640)
                else:
                    self.assertIn('too_short', meta['silence_skip_reason'])
                    np.testing.assert_array_equal(result, wave)

    def test_short_view_caps_entire_shared_gap_not_only_the_overlap(self):
        wave = np.ones(12*RATE, dtype=np.float32)
        for crop_start in (0, 4*RATE, 9*RATE):
            with self.subTest(crop_start=crop_start):
                result, meta = apply_local_silence(wave, parameters(), [(crop_start, 3*RATE)])
                self.assertTrue(meta['silence_applied'])
                self.assertEqual(meta['silence_samples'], 2400)
                self.assertLessEqual(max(meta['silence_view_fractions']), .05)
                self.assertLessEqual(int((result[crop_start:crop_start+3*RATE] != 1).sum()), 2400)

    def test_very_short_protected_view_skips_even_with_long_source(self):
        wave = np.ones(10*RATE, dtype=np.float32)
        result, meta = apply_local_silence(wave, parameters(), [(RATE, 6720)])
        self.assertFalse(meta['silence_applied'])
        np.testing.assert_array_equal(result, wave)

    def test_energy_guard_does_not_remove_only_audible_sound(self):
        wave = np.zeros(4*RATE, dtype=np.float32)
        wave[31200:32800] = .5
        result, meta = apply_local_silence(wave, parameters(duration_seconds=.1))
        self.assertFalse(meta['silence_applied'])
        self.assertEqual(meta['silence_skip_reason'], 'insufficient_remaining_view_energy')
        np.testing.assert_array_equal(result, wave)

    def test_energy_guard_is_checked_on_short_view_not_only_full_recording(self):
        wave = np.ones(4*RATE, dtype=np.float32)
        wave[24000:40000] = 0
        wave[31200:32800] = 1
        result, meta = apply_local_silence(wave, parameters(duration_seconds=.05), [(24000, RATE)])
        self.assertFalse(meta['silence_applied'])
        self.assertGreaterEqual(meta['silence_remaining_energy_fractions'][0], .9)
        self.assertLess(meta['silence_remaining_energy_fractions'][1], .9)
        np.testing.assert_array_equal(result, wave)

    def test_existing_quiet_gap_is_not_reported_as_applied(self):
        wave = np.ones(4*RATE, dtype=np.float32)
        wave[30000:34000] = 0
        result, meta = apply_local_silence(wave, parameters(duration_seconds=.1))
        self.assertFalse(meta['silence_applied'])
        self.assertEqual(meta['silence_skip_reason'], 'selected_region_already_silent')
        np.testing.assert_array_equal(result, wave)

    def test_invalid_parameters_and_protected_views_fail_clearly(self):
        wave = np.ones(4*RATE, dtype=np.float32)
        for change in ({'duration_seconds':.5}, {'duration_seconds':.039},
                       {'position':-1.}, {'position':float('nan')},
                       {'maximum_fraction':.1}, {'fade_seconds':0.}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                apply_local_silence(wave, parameters(**change))
        for crop in ((-1, 1000), (1, 0), (1000, 64000), (True, 1000), (0, 1000.5)):
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                apply_local_silence(wave, parameters(), [crop])


class SilenceIntegrationTests(unittest.TestCase):
    def test_rare_exclusive_draw_keeps_half_unchanged_and_legacy_plan_unchanged(self):
        settings = dict(processing_identity_probability=.5, processing_single_probability=.35,
                        processing_silence_probability=.05)
        counts = Counter()
        for i in range(8192):
            plan = processing_plan(str(i), i%2, seed=1234, epoch=1, **settings)
            silence = any(p['operator'] == 'local_silence' for p in plan)
            if silence:
                self.assertEqual(len(plan), 1)
            counts['silence' if silence else len(plan)] += 1
        for key, probability in ((0,.5), (1,.35), ('silence',.05), (2,.1)):
            self.assertAlmostEqual(counts[key]/8192, probability, delta=.02)
        self.assertEqual(processing_settings({})['processing_silence_probability'], 0.)
        for i in range(128):
            self.assertFalse(any(p['operator'] == 'local_silence' for p in
                processing_plan(str(i), 0, seed=1, epoch=1)))

    def test_silence_rejects_combination_and_records_rejected_short_audio_as_unchanged(self):
        with self.assertRaises(ValueError):
            apply_plan(np.ones(RATE, dtype=np.float32), [parameters(),
                dict(operator='local_attenuation', duration_seconds=.1, position=.5,
                     maximum_fraction=.1, depth_db=3.)])
        wave = np.ones(6720, dtype=np.float32)
        actual, meta = process_wave(wave, 'same.wav', 0, seed=1, epoch=1, **forced_silence())
        self.assertFalse(meta['processing_applied'])
        self.assertEqual(meta['processing_condition'], 'unchanged')
        self.assertEqual(meta['processing_parameters'][0]['operator'], 'local_silence')
        np.testing.assert_array_equal(actual, wave)

    def test_same_condition_for_classes_and_languages_and_shared_full_short_waveform(self):
        with tempfile.TemporaryDirectory() as td:
            row, wave = write_audio(Path(td), 'same.wav', samples=8*RATE)
            rows = [dict(row, label=label, language=language, noisy=True, full_length=True,
                         output_samples=len(wave), source_sha256='same-source', version=0)
                    for label in (0,1) for language in ('en','zh')]
            ds = AudioDataset(rows, training=True, epoch=1, seed=3,
                              short_min_seconds=3., short_max_seconds=3., **forced_silence())
            actual = [ds[(i,i,'full',i)] for i in range(4)]
            first = actual[0]
            self.assertTrue(first[0]['silence_applied'])
            for full, short in actual:
                np.testing.assert_array_equal(full['wave'], first[0]['wave'])
                self.assertEqual(full['silence_start_sample'], first[0]['silence_start_sample'])
                start, count = short['crop_start_sample'], short['crop_samples']
                np.testing.assert_array_equal(short['wave'], full['wave'][start:start+count])
                self.assertLessEqual(full['silence_samples'], .05*count)
                self.assertEqual(short['processing_parameters'], full['processing_parameters'])
                self.assertAlmostEqual(full['view_weight']+short['view_weight'], 1.)

    def test_internal_silence_does_not_become_padding_in_feature_mask(self):
        from transformers import SeamlessM4TFeatureExtractor
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ssl = root/'ssl'
            SeamlessM4TFeatureExtractor().save_pretrained(ssl)
            row, wave = write_audio(root, 'mask.wav', samples=8*RATE)
            row.update(noisy=True, full_length=True, output_samples=len(wave),
                       source_sha256='source', version=0)
            ds = AudioDataset([row], training=True, epoch=1, seed=3,
                              short_min_seconds=3., short_max_seconds=3., **forced_silence())
            views = ds[(0,0,'full',0)]
            self.assertGreater(int((views[0]['wave']==0).sum()), 0)
            for example in ViewCollator(ssl)([views]):
                self.assertTrue(torch.all(example['mask'] == 1))
                self.assertEqual(example['mask'].shape[1], example['features'].shape[1])
                self.assertTrue(torch.isfinite(example['features']).all())

    def test_force_silence_never_changes_ordinary_or_validation(self):
        with tempfile.TemporaryDirectory() as td:
            row, wave = write_audio(Path(td), 'ordinary.wav')
            with patch('w2v_v32.data.process_wave', side_effect=AssertionError('Only noisy Train')):
                ordinary = AudioDataset([row], training=True, **forced_silence())[0]
                validated = AudioDataset([row], training=False, **forced_silence())[0]
            np.testing.assert_array_equal(ordinary[0]['wave'], wave)
            np.testing.assert_array_equal(validated['wave'], wave)


if __name__ == '__main__':
    unittest.main()
