"""Objective invariants: balanced CE and source-local, symmetric phone gradients."""
import math
import unittest
import torch
from .objective import phoneme_loss, frame_span, micro_losses, source_weight, auxiliary_weights
from w2v_v3183r2.common import GROUPS


class TestObjective(unittest.TestCase):
    def test_existing_ce_mass_survives_unequal_quotas(self):
        quotas = [4, 3, 5, 4]
        units = []
        for language, label in GROUPS:
            for _ in range(quotas[GROUPS.index((language, label))]):
                row = dict(language=language, label=label)
                units.extend([[row], [row, row]])
        logits = torch.zeros(48, 2, requires_grad=True)
        terms = micro_losses(logits, units, quotas, .25, .02)
        total = sum(value for value, _ in terms)
        self.assertAlmostEqual(float(total), math.log(2), places=6)
        for group in GROUPS:
            mass = sum(float(value) for unit, (value, _) in zip(units, terms)
                       if (unit[0]['language'], unit[0]['label']) == group)
            self.assertAlmostEqual(mass, math.log(2) * .25, places=6)
        total.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_sample_boundaries_map_frame_centres(self):
        self.assertEqual(frame_span(0, 200, 10), (0, 0))
        self.assertEqual(frame_span(200, 520, 10), (0, 1))
        self.assertEqual(frame_span(520, 840, 10), (1, 2))
        self.assertEqual(frame_span(0, 20000, 10), (0, 10))
        with self.assertRaises(ValueError):
            frame_span(-1, 200, 10)

    def test_same_phone_instances_have_symmetric_gradients(self):
        left = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
        right = torch.tensor([[.6, .8], [.6, .8], [.8, .6], [.8, .6]], requires_grad=True)
        matches = [[0, 840, 0, 840], [840, 1480, 840, 1480]]
        loss, count = phoneme_loss(left, right, matches)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(float(loss), .4, places=6)
        loss.backward()
        self.assertGreater(float(left.grad.norm()), 0.)
        self.assertGreater(float(right.grad.norm()), 0.)
        same, _ = phoneme_loss(left.detach(), left.detach() * 3, matches)
        self.assertAlmostEqual(float(same), 0., places=6)

    def test_empty_or_short_phones_do_not_inflate_other_sources(self):
        a = torch.randn(3, 4, requires_grad=True)
        b = torch.randn(3, 4, requires_grad=True)
        value, count = phoneme_loss(a, b, [[0, 100, 0, 100]])
        self.assertEqual(count, 0)
        self.assertEqual(float(value), 0.)
        value.backward()
        self.assertEqual(float(a.grad.abs().sum()), 0.)
        cfg = dict(group_quotas=[4, 3, 5, 4], platform_sources_per_group=2)
        row = dict(language='en', label=1)
        self.assertAlmostEqual(source_weight('noise', row, cfg), .25 / 3)
        self.assertAlmostEqual(source_weight('platform', row, cfg), .125)

    def test_auxiliary_ramp_and_classification_only(self):
        cfg = dict(pcl_ramp_updates=300, pcl_noise_weight=.15, pcl_platform_weight=.15)
        self.assertAlmostEqual(auxiliary_weights(dict(cfg, current_update=150))['noise'], .075)
        self.assertAlmostEqual(auxiliary_weights(dict(cfg, current_update=900))['platform'], .15)
        self.assertEqual(auxiliary_weights(dict(cfg, consistency='none')), dict(noise=0., platform=0.))


if __name__ == '__main__':
    unittest.main()
