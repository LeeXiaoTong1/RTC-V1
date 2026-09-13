"""Run with: python -m unittest discover -s tests -p 'test_rtc_loss.py'."""

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.rtc_loss import rtc_pair_contrastive_loss


class RTCPairContrastiveLossTests(unittest.TestCase):
    def test_both_feature_branches_receive_gradients(self):
        generator = torch.Generator().manual_seed(7)
        offline = torch.randn(4, 5, generator=generator, requires_grad=True)
        online = torch.randn(4, 5, generator=generator, requires_grad=True)
        loss, stats = rtc_pair_contrastive_loss(offline, online, torch.tensor([0, 0, 1, 1]))
        loss.backward()
        self.assertEqual(stats, {"pairs": 4, "valid_anchors": 8})
        for features in (offline, online):
            self.assertTrue(torch.isfinite(features.grad).all())
            self.assertTrue((features.grad.abs().sum(dim=1) > 0).all())

    def test_same_label_nonpair_utterances_are_excluded(self):
        features = torch.tensor([[1.0, 0], [1.0, 0], [-1.0, 0], [-1.0, 0]])
        labels = torch.tensor([0, 0, 1, 1])
        loss, _ = rtc_pair_contrastive_loss(features, features, labels, temperature=0.5)
        # Each anchor has one positive at +1 and two opposite-label negatives
        # at -1. The identical same-label nonpair is absent from the denominator.
        expected = math.log1p(2 * math.exp(-4))
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_same_label_pair_swap_breaks_actual_source_correspondence(self):
        offline = torch.eye(4)
        labels = torch.tensor([0, 0, 1, 1])
        correct, _ = rtc_pair_contrastive_loss(offline, offline, labels)
        swapped, _ = rtc_pair_contrastive_loss(offline, offline[[1, 0, 2, 3]], labels)
        self.assertGreater(swapped.item(), correct.item() + 0.1)

    def test_well_separated_pairs_have_lower_loss_than_confused_pairs(self):
        offline = torch.tensor([[1.0, 0], [-1.0, 0]])
        labels = torch.tensor([0, 1])
        separated, _ = rtc_pair_contrastive_loss(offline, offline, labels, temperature=0.5)
        confused, _ = rtc_pair_contrastive_loss(offline, offline.flip(0), labels, temperature=0.5)
        self.assertLess(separated.item(), 0.02)
        self.assertGreater(confused.item(), 4.0)

    def test_bidirectional_result_is_symmetric(self):
        generator = torch.Generator().manual_seed(11)
        offline = torch.randn(5, 4, generator=generator)
        online = torch.randn(5, 4, generator=generator)
        labels = torch.tensor([0, 1, 0, 1, 1])
        first, _ = rtc_pair_contrastive_loss(offline, online, labels)
        second, _ = rtc_pair_contrastive_loss(online, offline, labels)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_single_class_and_empty_pairs_return_differentiable_zero(self):
        for count, label in ((3, 0), (3, 1), (1, 0), (0, 0)):
            with self.subTest(count=count, label=label):
                offline = torch.randn(count, 4, requires_grad=True)
                online = torch.randn(count, 4, requires_grad=True)
                labels = torch.full((count,), label, dtype=torch.long)
                loss, stats = rtc_pair_contrastive_loss(offline, online, labels)
                self.assertEqual(loss.item(), 0.0)
                self.assertEqual(stats, {"pairs": count, "valid_anchors": 0})
                loss.backward()
                self.assertIsNotNone(offline.grad)
                self.assertIsNotNone(online.grad)
                self.assertEqual(offline.grad.abs().sum().item(), 0.0)
                self.assertEqual(online.grad.abs().sum().item(), 0.0)

    def test_bfloat16_autocast_uses_finite_fp32_loss_and_gradients(self):
        offline = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 1, 1], [-1, 1, 0]], dtype=torch.bfloat16, requires_grad=True)
        online = torch.tensor([[1, 0.5, 0], [0, 0, 0], [0.5, 1, 1], [-1, 0, 0]], dtype=torch.bfloat16, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        reference, _ = rtc_pair_contrastive_loss(offline.float(), online.float(), labels, temperature=0.01)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual, _ = rtc_pair_contrastive_loss(offline, online, labels, temperature=0.01)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual))
        self.assertTrue(torch.allclose(actual, reference, atol=1e-6))
        actual.backward()
        self.assertTrue(torch.isfinite(offline.grad).all())
        self.assertTrue(torch.isfinite(online.grad).all())

    def test_invalid_temperature_and_shapes_raise(self):
        features = torch.eye(2)
        labels = torch.tensor([0, 1])
        for temperature in (0, -1, float("nan"), float("inf"), "invalid"):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                rtc_pair_contrastive_loss(features, features, labels, temperature)
        for offline, online, target in (
            (features[0], features, labels),
            (features, features[:1], labels),
            (features, features, labels[:, None]),
            (features, features, torch.tensor([0, 2])),
        ):
            with self.assertRaises(ValueError):
                rtc_pair_contrastive_loss(offline, online, target)


if __name__ == "__main__":
    unittest.main()
