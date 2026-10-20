import unittest
import numpy as np
import torch
from transformers import SeamlessM4TFeatureExtractor
from .features import OfficialFeatures, frame_count, frame_intervals, prepare_wave


class FeatureTests(unittest.TestCase):
    def test_official_features_independent_of_batch_padding(self):
        extractor = OfficialFeatures(SeamlessM4TFeatureExtractor(padding_value=1))
        rng = np.random.default_rng(31)
        short = rng.normal(0, .1, 8000).astype(np.float32)
        long = rng.normal(0, .1, 15000).astype(np.float32)
        one, _, a = extractor([short])
        two, mask, b = extractor([short, long])
        self.assertEqual(a[0], b[0])
        torch.testing.assert_close(one[0], two[0, :a[0]], rtol=0, atol=0)
        self.assertEqual(int(mask[0].sum()), frame_count(len(short)))
        self.assertEqual(two.shape[-1], 160)

    def test_short_reflection_and_new_grid(self):
        wave = np.arange(8, dtype=np.float32)
        prepared, original = prepare_wave(wave)
        self.assertEqual(original, 8)
        self.assertEqual(len(prepared), 3920)
        np.testing.assert_array_equal(prepared[:8], wave)
        self.assertEqual(frame_count(3920), 11)
        self.assertEqual(frame_count(560, extend_short=False), 1)
        self.assertEqual(frame_count(559, extend_short=False), 0)
        np.testing.assert_allclose(frame_intervals(2), [[0., .035], [.02, .055]])

    def test_finite_nonempty_inputs_required(self):
        for wave in ([], [float('nan')], np.ones((2, 5))):
            with self.assertRaises(ValueError):
                prepare_wave(wave)

    def test_even_and_odd_fbank_tail_masks(self):
        extractor = OfficialFeatures(SeamlessM4TFeatureExtractor())
        for length in (3920, 4080, 4240, 16000, 16160):
            _, mask, lengths = extractor([np.random.default_rng(length).normal(size=length).astype(np.float32)])
            self.assertEqual(lengths, [frame_count(length)])
            self.assertEqual(int(mask.sum()), frame_count(length))


if __name__ == '__main__':
    unittest.main()
