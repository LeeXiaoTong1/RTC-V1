"""Geometry and gradient contracts for the V3.19 consistency redesign."""
import unittest

import torch

from .objectives import TFCL, channel_cka


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(319)
        torch.set_num_threads(1)
        self.aux = TFCL(channels=8, heads=2, bins=13,
                        cfg=dict(alignment_max_bins=12, alignment_min_frames=4))

    def test_identical_matched_zero_and_no_trainable_shortcut(self):
        a = torch.randn(21, 8, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        mask = torch.ones(21, dtype=torch.bool)
        t, s = self.aux(a, b, mask, mask)
        self.assertAlmostEqual(float(t.detach()), 0., places=6)
        self.assertAlmostEqual(float(s.detach()), 0., places=6)
        self.assertEqual(sum(p.numel() for p in self.aux.parameters()), 0)

    def test_matched_joint_timeline_masks_exclude_nan_and_gradients(self):
        a, b = torch.randn(20, 8), torch.randn(20, 8)
        am, bm = torch.ones(20, dtype=torch.bool), torch.ones(20, dtype=torch.bool)
        am[3:5], bm[7:9] = False, False
        a[~am], b[~bm] = float('nan'), float('nan')
        a.requires_grad_(); b.requires_grad_()
        t, s = self.aux(a, b, am, bm)
        expected = (1.-torch.nn.functional.cosine_similarity(a[am & bm], b[am & bm], dim=-1)).mean()
        torch.testing.assert_close(t, expected)
        (t+s).backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())
        self.assertEqual(a.grad[~(am & bm)].count_nonzero().item(), 0)
        self.assertEqual(b.grad[~(am & bm)].count_nonzero().item(), 0)
        self.assertGreater(a.grad[am & bm].norm().item(), 0.)
        self.assertGreater(b.grad[am & bm].norm().item(), 0.)

    def test_unalignable_matched_length_rejected_not_interpolated(self):
        with self.assertRaisesRegex(ValueError, 'exact timeline'):
            self.aux(torch.randn(20, 8), torch.randn(21, 8), torch.ones(20), torch.ones(21))

    def test_bridge_different_lengths_bidirectional_gradient_and_mask(self):
        a, b = torch.randn(27, 8), torch.randn(36, 8)
        am, bm = torch.ones(27, dtype=torch.bool), torch.ones(36, dtype=torch.bool)
        am[2:4] = False; bm[8:11] = False
        a[~am] = float('nan'); b[~bm] = float('nan')
        a.requires_grad_(); b.requires_grad_()
        t, s, eligible, details = self.aux.forward_batch([a], [b], [am], [bm],
            kinds=['bridge'], return_diagnostics=True)
        self.assertTrue(eligible[0])
        self.assertLess(details[0]['valid_frame_fraction'], 1.)
        (t+s).sum().backward()
        for value, mask in ((a, am), (b, bm)):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertEqual(value.grad[~mask].count_nonzero().item(), 0)
            self.assertGreater(value.grad[mask].norm().item(), 0.)

    def test_correspondence_is_monotonic_detached_and_full_sequence(self):
        a, b = torch.randn(12, 8), torch.randn(19, 8)
        ap, bp = torch.linspace(0, 1, 12), torch.linspace(0, 1, 19)
        result = self.aux._indices(a.requires_grad_(), b.requires_grad_(), ap, bp)
        self.assertTrue((result[1:] >= result[:-1]).all())
        self.assertFalse(result.requires_grad)
        self.assertLessEqual(result[0].item(), 5)
        self.assertGreaterEqual(result[-1].item(), 13)

    def test_invalid_pair_zero_but_collapsed_representation_not_perfect(self):
        a, b = torch.randn(20, 8, requires_grad=True), torch.randn(20, 8, requires_grad=True)
        mask = torch.zeros(20, dtype=torch.bool)
        t, s, valid = self.aux.forward_batch([a], [b], [mask], [mask], kinds=['bridge'])
        self.assertFalse(valid[0]); self.assertEqual(float((t+s).detach()), 0.)
        (t+s).sum().backward()
        self.assertEqual(a.grad.count_nonzero().item(), 0)
        self.assertEqual(b.grad.count_nonzero().item(), 0)
        self.assertEqual(float(channel_cka(torch.zeros(8, 20), torch.zeros(8, 20))), 1.)


if __name__ == '__main__':
    unittest.main()
