"""Synthetic mechanism checks, not evidence of real speech accuracy gains."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from .local_structure import (LocalStructureConfig, descriptors, local_structure_loss,
                              local_structure_per_pair, _match)
from .model import Detector, forward_chunks


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.projection = nn.Linear(160, 16)

    def forward(self, input_features, attention_mask, **unused):
        return SimpleNamespace(last_hidden_state=self.projection(input_features))


class LocalStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7623)
        self.config = LocalStructureConfig()
        self.frames = torch.randn(3, 128, 64)

    def test_identity_and_static_affine_change(self):
        transformed = self.frames*2.3 + torch.randn(3, 1, 64)*5
        loss, stats = local_structure_loss(self.frames, transformed)
        self.assertLess(float(loss), 1e-6)
        self.assertEqual(float(stats['usable_fraction']), 1.)
        self.assertEqual(float(stats['accepted_fraction']), 1.)
        before, after = descriptors(self.frames), descriptors(transformed)
        self.assertEqual(before['embedding'].shape, (3, 90))
        self.assertTrue(bool(before['valid'].all()))
        torch.testing.assert_close(before['embedding'], after['embedding'], atol=2e-7, rtol=2e-5)

    def test_processed_has_finite_useful_gradient_and_reference_is_detached(self):
        reference = self.frames.clone().requires_grad_()
        processed = (self.frames+.2*torch.randn_like(self.frames)).requires_grad_()
        loss, stats = local_structure_loss(reference, processed)
        self.assertGreater(float(loss), 0.)
        self.assertEqual(float(stats['usable_fraction']), 1.)
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertTrue(bool(torch.isfinite(processed.grad).all()))
        self.assertGreater(float(processed.grad.abs().sum()), 0.)

    def test_bounded_delay_recovers_ordered_correspondences(self):
        # Eight frames correspond to one coarse/two fine bins.
        shifted = torch.cat((self.frames[:, :8], self.frames[:, :-8]), 1)
        loss, stats = local_structure_loss(self.frames, shifted)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(stats['accepted_fraction']), .8)
        self.assertGreater(float(stats['mean_shift_fraction']), .04)
        self.assertEqual(float(stats['usable_fraction']), 1.)
        _, exact_stats = local_structure_loss(self.frames, shifted, replace(self.config, max_shift_fraction=0.))
        self.assertLess(float(exact_stats['accepted_fraction']), .2)

    def test_unrelated_and_degenerate_pairs_skip_with_zero_gradient(self):
        for reference, raw in [(self.frames, torch.randn_like(self.frames)),
                               (self.frames, torch.zeros_like(self.frames)),
                               (torch.ones_like(self.frames), torch.ones_like(self.frames)*8)]:
            with self.subTest(kind=float(raw.std())):
                processed = raw.clone().requires_grad_()
                loss, stats = local_structure_loss(reference, processed)
                self.assertEqual(float(stats['usable_fraction']), 0.)
                self.assertEqual(float(loss), 0.)
                loss.backward()
                self.assertTrue(bool(torch.isfinite(processed.grad).all()))
                self.assertEqual(float(processed.grad.abs().sum()), 0.)
        for x in (torch.zeros_like(self.frames), torch.ones_like(self.frames)):
            self.assertFalse(bool(descriptors(x)['valid'].any()))

    def test_matches_are_mutual_monotone_and_bounded(self):
        ref = torch.eye(16)[None]
        # Swap two distinctive neighboring tokens. Both crossing matches must
        # be rejected rather than violating the temporal order constraint.
        order = torch.arange(16)
        order[5], order[6] = 6, 5
        proc = ref[:, order]
        active = torch.ones(1, 16, dtype=torch.bool)
        match, valid, _ = _match(ref, proc, active, active, self.config)
        self.assertFalse(bool(valid[0, 5]))
        self.assertFalse(bool(valid[0, 6]))
        indices = match[0, valid[0]]
        self.assertTrue(bool((indices[1:] > indices[:-1]).all()))
        self.assertEqual(len(indices), 14)
        self.assertLessEqual(int((match[valid]-torch.arange(16)[None][valid]).abs().max()), 3)
        # Exact duplicate local features are ambiguous; min_margin rejects.
        ambiguous = ref[:, :1].expand(1, 16, 16)
        self.assertFalse(bool(_match(ambiguous, ambiguous, active, active, self.config)[1].any()))

    def test_per_example_batch_invariance_and_permutation(self):
        processed = self.frames+.2*torch.randn_like(self.frames)
        batch, valid, stats = local_structure_per_pair(self.frames, processed)
        for i in range(3):
            single, okay, single_stats = local_structure_per_pair(self.frames[i:i+1], processed[i:i+1])
            torch.testing.assert_close(batch[i:i+1], single)
            torch.testing.assert_close(valid[i:i+1], okay)
            torch.testing.assert_close(stats['accept_by_pair'][i:i+1], single_stats['accept_by_pair'])
        order = torch.tensor([2, 0, 1])
        shuffled, shuffled_valid, _ = local_structure_per_pair(self.frames[order], processed[order])
        torch.testing.assert_close(shuffled, batch[order])
        torch.testing.assert_close(shuffled_valid, valid[order])
        augmented, augmented_valid, _ = local_structure_per_pair(
            torch.cat((self.frames, self.frames[:1])),
            torch.cat((processed, torch.randn_like(processed[:1]))))
        torch.testing.assert_close(augmented[:3], batch)
        self.assertFalse(bool(augmented_valid[-1]))

    def test_fp32_thresholds_are_not_changed_by_outer_autocast(self):
        processed = self.frames+.2*torch.randn_like(self.frames)
        expected, _ = local_structure_loss(self.frames, processed)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual, _ = local_structure_loss(self.frames, processed)
            encoded = descriptors(self.frames)['embedding']
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(encoded.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_bad_shapes_and_configs_rejected_nonfinite_descriptors_invalid(self):
        for kw in ({'scales': (2, 32)}, {'min_cosine': float('nan')},
                   {'max_shift_fraction': 1.}, {'min_matches': 2}, {'transition_weight': -1}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                LocalStructureConfig(**kw)
        for proc in (self.frames[:, :16], self.frames[:, :, :5], self.frames[:2]):
            with self.assertRaises(ValueError):
                local_structure_loss(self.frames, proc)
        damaged = self.frames.clone()
        damaged[0, 0, 0] = float('nan')
        encoded = descriptors(damaged)
        self.assertFalse(bool(encoded['valid'][0]))
        self.assertTrue(bool(torch.isfinite(encoded['embedding']).all()))

    def test_frame_export_preserves_eval_state_dict_and_chunk_padding(self):
        model = Detector(TinyEncoder()).eval()
        inputs, mask = torch.randn(3, 96, 160), torch.ones(3, 96, dtype=torch.long)
        before = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            normal = model(inputs, mask)
            exported = model(inputs, mask, return_frames=True)
            chunked = forward_chunks(model, inputs, mask, 2, pad_last=True, return_frames=True)
            legacy = forward_chunks(model, inputs, mask, 2, pad_last=True)
        self.assertEqual(len(normal), 2)
        self.assertEqual(len(exported), 3)
        self.assertEqual(exported[2].shape, (3, 96, 16))
        for i in range(2):
            torch.testing.assert_close(normal[i], exported[i], atol=0, rtol=0)
            torch.testing.assert_close(chunked[i], legacy[i], atol=0, rtol=0)
            torch.testing.assert_close(exported[i], chunked[i], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(exported[2], chunked[2])
        self.assertEqual(set(before), set(model.state_dict()))
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
