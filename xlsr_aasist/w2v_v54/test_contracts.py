"""CPU contract tests; no claim about GPU integration or RTC accuracy."""
import copy
import io
import json
from pathlib import Path
import unittest
import torch
from torch import nn
from .repair import (FeatureRepair, length_mask, aligned_reference, source_feature_loss,
                       planned_group_mean, source_kd_loss, deterministic_head, migrated_anchor)


class RepairContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(5421)

    def test_zero_init_is_exact_identity_and_gate_nonzero(self):
        model = FeatureRepair()
        x = torch.randn(2, 25, 128)
        out = model(x, [12, 25])
        torch.testing.assert_close(out['fused'][0, :12], x[0, :12], rtol=0, atol=0)
        torch.testing.assert_close(out['fused'][1], x[1], rtol=0, atol=0)
        torch.testing.assert_close(out['gate'][1], torch.full_like(x[1], .1))
        self.assertEqual(float(out['fused'][0, 12:].abs().sum()), 0.)

    def test_init_does_not_advance_rng(self):
        from unittest.mock import patch
        before = torch.get_rng_state()
        with patch('torch.cuda.manual_seed_all') as cuda_seed:
            a, b = FeatureRepair(), FeatureRepair()
            cuda_seed.assert_not_called()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)

    def test_nonzero_processor_is_independent_of_padding_and_batch_companions(self):
        model = FeatureRepair()
        nn.init.normal_(model.out.weight, std=.05)
        nn.init.normal_(model.gate[-1].weight, std=.05)
        x = torch.randn(2, 39, 128)
        x[0, 13:] = float('nan')
        batched = model(x, [13, 39])
        solo = model(x[:1, :13], [13])
        for key in ('fused', 'repaired', 'gate'):
            torch.testing.assert_close(batched[key][:1, :13], solo[key], rtol=1e-5, atol=2e-6)
            self.assertTrue(torch.isfinite(batched[key]).all())
            self.assertEqual(float(batched[key][0, 13:].abs().sum().detach()), 0.)

    def test_feature_gradient_updates_repair_only_and_opens_hidden_layers(self):
        model = FeatureRepair()
        x = torch.randn(2, 20, 128, requires_grad=True)
        target = torch.randn_like(x, requires_grad=True)
        mask = length_mask(x, [20, 16])
        opt = torch.optim.AdamW(model.parameters(), lr=.005)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            pred = model.auxiliary_prediction(x, [20, 16])
            loss, _ = source_feature_loss(pred, target, mask, model.frame_scale)
            loss.mean().backward()
            self.assertIsNone(x.grad)
            self.assertIsNone(target.grad)
            self.assertIsNone(model.frame_scale.grad)
            self.assertTrue(all(p.grad is None for p in model.gate.parameters()))
            self.assertGreater(float(model.out.weight.grad.norm()), 0.)
            if step == 1:
                self.assertGreater(float(model.blocks[0].expand.weight.grad.norm()), 0.)
            opt.step()

    def test_classification_route_retains_encoder_gradient(self):
        model = FeatureRepair()
        x = torch.randn(2, 20, 128, requires_grad=True)
        model(x, [20, 20])['fused'].square().mean().backward()
        self.assertGreater(float(x.grad.norm()), 0.)
        self.assertGreater(float(model.out.weight.grad.norm()), 0.)

    def test_auxiliary_loss_does_not_change_input_gradient_of_ce(self):
        model = FeatureRepair()
        x = torch.randn(2, 20, 128, requires_grad=True)
        ce = model(x, [20, 20])['fused'].square().mean()
        ce_grad = torch.autograd.grad(ce, x, retain_graph=True)[0]
        values, _ = source_feature_loss(model.auxiliary_prediction(x, [20, 20]),
                                         torch.zeros_like(x), length_mask(x, [20, 20]), model.frame_scale)
        total_grad = torch.autograd.grad(ce + values.mean(), x)[0]
        torch.testing.assert_close(total_grad, ce_grad, rtol=0, atol=0)

    def test_fractional_alignment_and_valid_interpolation_neighbours(self):
        clean = torch.arange(6.).reshape(6, 1).requires_grad_()
        target, mask = aligned_reference(clean, torch.ones(6, dtype=torch.bool), 6, .5)
        torch.testing.assert_close(target[1:, 0], torch.tensor([.5, 1.5, 2.5, 3.5, 4.5]))
        self.assertFalse(bool(mask[0])); self.assertTrue(mask[1:].all())
        self.assertFalse(target.requires_grad)
        reliable = torch.ones(6, dtype=torch.bool); reliable[2] = False
        _, mask = aligned_reference(clean, reliable, 6, .5)
        self.assertFalse(bool(mask[2])); self.assertFalse(bool(mask[3]))

    def test_integer_alignment_last_frame_not_discarded(self):
        clean = torch.arange(6.).reshape(6, 1)
        target, mask = aligned_reference(clean, torch.ones(6, dtype=torch.bool), 6, 0.)
        torch.testing.assert_close(target, clean)
        self.assertTrue(mask.all())

    def test_negative_delay_advances_reference_without_wrapping(self):
        clean = torch.arange(6.).reshape(6, 1)
        target, mask = aligned_reference(clean, torch.ones(6, dtype=torch.bool), 6, -1.5)
        torch.testing.assert_close(target[:4, 0], torch.tensor([1.5, 2.5, 3.5, 4.5]))
        self.assertEqual(mask.tolist(), [True, True, True, True, False, False])
        self.assertTrue(torch.equal(target[4:], torch.zeros_like(target[4:])))

    def test_per_source_mean_and_missing_pairs_do_not_shift_group_mass(self):
        x = torch.ones(4, 20, 128, requires_grad=True)
        mask = length_mask(x, [5, 20, 2, 20])
        target = torch.zeros_like(x); target[2] = float('nan')
        values, active = source_feature_loss(x, target, mask, torch.ones(128))
        torch.testing.assert_close(values, torch.tensor([.5, .5, 0., .5]))
        self.assertEqual(active.tolist(), [True, True, False, True])
        balanced = planned_group_mean(values, [0, 1, 2, 3], [1, 1, 1, 1])
        self.assertAlmostEqual(float(balanced.detach()), .375)
        half_budget = planned_group_mean(values, [0, 1, 2, 3], [2, 2, 2, 2])
        self.assertAlmostEqual(float(half_budget.detach()), .1875)

    def test_no_valid_pairs_returns_differentiable_zero(self):
        x = torch.randn(2, 10, 128, requires_grad=True)
        values, active = source_feature_loss(x, torch.full_like(x, float('nan')),
                                             torch.zeros(2, 10, dtype=torch.bool), torch.ones(128))
        self.assertFalse(active.any())
        values.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))

    def test_kd_wrong_teacher_and_better_student_are_not_forced_to_match(self):
        teacher = torch.tensor([[4., 0.], [0., 4.], [4., 0.], [4., 0.]], requires_grad=True)
        student = torch.tensor([[1., 0.], [1., 0.], [6., 0.], [1., 0.]], requires_grad=True)
        values, active = source_kd_loss(student, teacher, torch.zeros(4, dtype=torch.long),
                                       torch.tensor([True, True, True, False]))
        self.assertEqual(active.tolist(), [True, False, False, False])
        self.assertGreater(float(values[0].detach()), 0.)
        values.sum().backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad[0].abs().sum()), 0.)
        self.assertEqual(float(student.grad[1:].abs().sum()), 0.)

    def test_kd_all_inactive_is_finite_zero(self):
        logits = torch.randn(3, 2, requires_grad=True)
        values, active = source_kd_loss(logits, logits.detach(), torch.zeros(3, dtype=torch.long),
                                       torch.ones(3, dtype=torch.bool))
        self.assertFalse(active.any()); self.assertTrue(torch.isfinite(values).all())
        values.sum().backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_actual_aasist_logits_unchanged_at_init_and_no_batchnorm(self):
        from w2v_v53.test_fusion import configuration, detector
        parent = detector(configuration()).eval()
        repair = FeatureRepair()
        self.assertEqual(sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in parent.head.modules()), 0)
        self.assertGreater(sum(isinstance(m, nn.Dropout) for m in parent.head.modules()), 0)
        sequences = [torch.randn(n, 128) for n in (12, 55, 150)]
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        output = repair(padded, [len(s) for s in sequences])['fused']
        processed = [output[i, :len(s)] for i, s in enumerate(sequences)]
        with torch.no_grad():
            torch.testing.assert_close(parent.classify_sequences(sequences),
                                       parent.classify_sequences(processed), rtol=0, atol=0)

    def test_deterministic_head_restores_modes_without_consuming_rng(self):
        head = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.5), nn.Linear(4, 2)).train()
        head[0].eval()
        modes = [m.training for m in head.modules()]
        x = torch.randn(3, 4, requires_grad=True)
        state = torch.get_rng_state()
        with deterministic_head(head):
            a, b = head(x), head(x)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            a.sum().backward()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual([m.training for m in head.modules()], modes)
        self.assertGreater(float(x.grad.abs().sum()), 0.)

    def test_frozen_actual_backend_still_teaches_repair_during_warmup(self):
        from w2v_v53.test_fusion import configuration, detector
        parent = detector(configuration()).eval().requires_grad_(False)
        repair = FeatureRepair()
        raw = torch.randn(2, 60, 128)
        fused = repair(raw, [30, 60])['fused']
        logits = parent.classify_sequences([fused[0, :30], fused[1]])
        nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertTrue(all(p.grad is None for p in parent.parameters()))
        self.assertGreater(float(repair.out.weight.grad.norm()), 0.)

    def test_new_state_strict_save_reload_and_parent_migration(self):
        model = FeatureRepair()
        nn.init.normal_(model.out.weight, std=.1)
        model.frame_scale.fill_(2.)
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        loaded = FeatureRepair()
        loaded.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        x = torch.randn(1, 15, 128)
        torch.testing.assert_close(model(x, [15])['fused'], loaded(x, [15])['fused'], rtol=0, atol=0)
        old = {'fusion.weight': torch.ones(2)}
        initialized = dict(old, **{'repair.'+k: v for k, v in FeatureRepair().state_dict().items()})
        anchor = migrated_anchor(old, initialized)
        self.assertEqual(set(anchor), set(initialized))
        with self.assertRaises(ValueError):
            migrated_anchor({}, initialized)
        incomplete = dict(loaded.state_dict()); incomplete.pop('frame_scale')
        with self.assertRaises(RuntimeError):
            loaded.load_state_dict(incomplete, strict=True)

    def test_synthetic_fit_reduces_residual_without_parent_gradients(self):
        model = FeatureRepair()
        clean = torch.randn(4, 20, 128)
        noisy = (clean + .2).requires_grad_()
        valid = length_mask(noisy, [20]*4)
        opt = torch.optim.AdamW(model.parameters(), lr=.003)
        losses = []
        for _ in range(30):
            opt.zero_grad(set_to_none=True)
            values, _ = source_feature_loss(model.auxiliary_prediction(noisy, [20]*4),
                                             clean, valid, model.frame_scale)
            loss = values.mean(); losses.append(float(loss.detach()))
            loss.backward(); opt.step()
        self.assertLess(losses[-1], losses[0]*.1)
        self.assertIsNone(noisy.grad)
        self.synthetic = dict(first=losses[0], last=losses[-1])


if __name__ == '__main__':
    unittest.main()
