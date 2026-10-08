"""CPU contracts for full-trajectory, independent-edge TFCL execution."""
import copy
import unittest

import torch

from w2v_v315.objectives import TFCL as OriginalTFCL
from w2v_v3151.objectives import TFCL, channel_cka


class TFCLTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3115)
        torch.set_num_threads(1)
        self.model = TFCL(channels=8, heads=2, bins=11)

    @staticmethod
    def examples():
        left = [torch.randn(t, 8, requires_grad=True) for t in (5, 9, 6)]
        right = [torch.randn(t, 8, requires_grad=True) for t in (8, 4, 10)]
        am = [torch.ones(len(v), dtype=torch.bool) for v in left]
        bm = [torch.ones(len(v), dtype=torch.bool) for v in right]
        am[0][1] = False
        bm[2][[1, 7]] = False
        return left, right, am, bm

    def test_batched_matches_single_losses_and_all_gradients(self):
        left, right, am, bm = self.examples()
        serial = copy.deepcopy(self.model)
        sl = [v.detach().clone().requires_grad_() for v in left]
        sr = [v.detach().clone().requires_grad_() for v in right]
        temporal, structure, valid = self.model.forward_batch(left, right, am, bm)
        single = [serial(a, b, m, n) for a, b, m, n in zip(sl, sr, am, bm)]
        expected_t, expected_s = map(torch.stack, zip(*single))
        torch.testing.assert_close(temporal, expected_t, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(structure, expected_s, atol=2e-6, rtol=1e-5)
        self.assertTrue(valid.all())
        # Different weights catch accidental mixing or cross-edge reductions.
        weights = torch.tensor([.1, .3, .6])
        ((temporal + .7 * structure) * weights).sum().backward()
        ((expected_t + .7 * expected_s) * weights).sum().backward()
        for a, b in zip(left + right, sl + sr):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
            self.assertGreater(a.grad.norm().item(), 0.)
        for a, b in zip(self.model.parameters(), serial.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=2e-5)

    def test_preserves_original_raw_feature_objective(self):
        old = OriginalTFCL(channels=8, heads=2, bins=11)
        old.load_state_dict(self.model.state_dict(), strict=True)
        left, right, am, bm = self.examples()
        for a, b, m, n in zip(left, right, am, bm):
            # Nonuniform amplitudes reveal unintended feature normalization.
            a = a * torch.linspace(.2, 4., len(a))[:, None]
            for actual, expected in zip(self.model(a, b, m, n), old(a, b, m, n)):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)

    def test_padding_and_known_missing_frames_have_zero_influence(self):
        a, b = torch.randn(7, 8), torch.randn(9, 8)
        am = torch.tensor([1, 1, 0, 1, 1, 0, 0], dtype=torch.bool)
        bm = torch.tensor([1, 0, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool)
        expected = self.model(a, b, am, bm)
        a[~am] = float('nan')
        b[~bm] = float('inf')
        a.requires_grad_(); b.requires_grad_()
        actual = self.model(a, b, am, bm)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y)
        sum(actual).backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(b.grad).all())
        self.assertEqual(a.grad[~am].count_nonzero().item(), 0)
        self.assertEqual(b.grad[~bm].count_nonzero().item(), 0)

    def test_ineligible_edges_are_zero_connected_without_all_masked_attention(self):
        a = torch.full((4, 8), float('nan'), requires_grad=True)
        b = torch.randn(5, 8, requires_grad=True)
        am, bm = torch.zeros(4, dtype=torch.bool), torch.ones(5, dtype=torch.bool)
        t, s, valid = self.model.forward_batch([a], [b], [am], [bm])
        self.assertFalse(valid.item())
        self.assertEqual((t + s).item(), 0.)
        (t + s).sum().backward()
        self.assertEqual(a.grad.count_nonzero().item(), 0)
        self.assertEqual(b.grad.count_nonzero().item(), 0)

    def test_edge_order_and_independence_with_ineligible_middle(self):
        left, right, am, bm = self.examples()
        am[1].fill_(False)
        t, s, valid = self.model.forward_batch(left, right, am, bm)
        self.assertEqual(valid.tolist(), [True, False, True])
        for i in (0, 2):
            a, b = self.model(left[i], right[i], am[i], bm[i])
            torch.testing.assert_close(t[i], a, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(s[i], b, atol=1e-6, rtol=1e-5)
        grads = torch.autograd.grad(t[0] + s[0], left + right, allow_unused=True)
        for i in (1, 2, 4, 5):
            self.assertEqual(grads[i].count_nonzero().item(), 0)

    def test_zero_features_and_collapsed_cka_remain_finite_and_not_perfect(self):
        a = torch.zeros(5, 8, requires_grad=True)
        b = torch.zeros(7, 8, requires_grad=True)
        t, s = self.model(a, b, torch.ones(5, dtype=torch.bool),
                          torch.ones(7, dtype=torch.bool))
        self.assertEqual(t.item(), 1.)
        self.assertEqual(s.item(), 1.)
        (t + s).backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(b.grad).all())
        self.assertEqual(channel_cka(torch.ones(8, 11), torch.ones(8, 11)).item(), 1.)

    def test_cka_identity_and_per_edge_scale_invariance(self):
        x = torch.randn(3, 8, 11)
        scales = torch.tensor([.1, 1., 10.])[:, None, None]
        torch.testing.assert_close(channel_cka(x, x * scales), torch.zeros(3),
                                   atol=3e-7, rtol=0.)

    def test_bfloat16_features_under_autocast_use_finite_fp32_auxiliary(self):
        a = torch.randn(5, 8).bfloat16().requires_grad_()
        b = torch.randn(9, 8).bfloat16().requires_grad_()
        am, bm = torch.ones(5, dtype=torch.bool), torch.ones(9, dtype=torch.bool)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            t, s = self.model(a, b, am, bm)
        expected = self.model(a.float(), b.float(), am, bm)
        for actual, target in zip((t, s), expected):
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, target)
        (t + s).backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(b.grad).all())

    def test_complete_trajectory_including_tail_receives_gradient(self):
        a = torch.randn(43, 8, requires_grad=True)
        b = torch.randn(57, 8, requires_grad=True)
        t, s = self.model(a, b, torch.ones(43, dtype=torch.bool),
                          torch.ones(57, dtype=torch.bool))
        (t + s).backward()
        self.assertGreater(a.grad[-1].norm().item(), 0.)
        self.assertGreater(b.grad[-1].norm().item(), 0.)

    def test_empty_and_invalid_input_contracts(self):
        t, s, valid = self.model.forward_batch([], [], [], [])
        self.assertEqual((t.numel(), s.numel(), valid.numel()), (0, 0, 0))
        a, b = torch.randn(3, 8), torch.randn(4, 8)
        with self.assertRaises(ValueError):
            self.model.forward_batch([a], [], [], [])
        with self.assertRaises(ValueError):
            self.model(a, b, torch.ones(2), torch.ones(4))
        with self.assertRaises(ValueError):
            TFCL(channels=8, heads=3)


if __name__ == '__main__':
    unittest.main()
