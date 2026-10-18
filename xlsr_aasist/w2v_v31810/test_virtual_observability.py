"""Realized updates, matched outputs, and temporary-parameter evaluation guard."""
import copy
import json
import unittest

import torch
from torch import nn

from .calibration import CANDIDATES, CalibrationError, calibrate
from .meta import (Transaction, logit_change, restore_rng, rng_state, train_step,
                   virtual_step, virtual_update_statistics)
from .test_calibration import tasks
from .test_meta import Toy, assert_state_equal, fixture, optimizer


class VirtualObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(519)

    def test_planned_nonzero_update_can_round_to_zero_in_fp32(self):
        from .validation import evaluate_protocol
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        opt = torch.optim.SGD([dict(params=model.parameters(), name='head')], lr=.1)
        tx = Transaction(model)
        norm, factor = virtual_step(tx, [torch.full_like(model.weight, 1e-8)], .001)
        self.assertGreater(.001 * factor * norm, 0.)
        measured = virtual_update_statistics(opt, tx)
        self.assertEqual(measured['virtual_update_norm'], 0.)
        self.assertEqual(measured['module_virtual_update_norms'], {'head': 0.})
        self.assertTrue(model._v31810_virtual_active)
        with self.assertRaisesRegex(ValueError, 'temporary adapted'):
            evaluate_protocol(model, {}, {})
        with self.assertRaisesRegex(ValueError, 'temporary adapted'):
            Transaction(model)
        tx.restore()
        self.assertFalse(model._v31810_virtual_active)
        torch.testing.assert_close(model.weight, torch.ones_like(model.weight), rtol=0, atol=0)

    def test_fp32_lora_master_change_does_not_imply_bf16_forward_change(self):
        from w2v_v3183r2.model import LoRALinear
        base = nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            base.weight.zero_()
        model = LoRALinear(base, rank=1, alpha=1., dropout=0.)
        with torch.no_grad():
            model.lora_a.fill_(1.)
            model.lora_b.fill_(1.)
        opt = torch.optim.SGD([dict(params=[model.lora_a, model.lora_b], name='lora')], lr=.1)
        tx = Transaction(model)
        wave = torch.ones(1, 1, dtype=torch.bfloat16)
        before = model(wave).detach()
        virtual_step(tx, [torch.full_like(p, .001) for _, p in tx.parameters], .001)
        measured = virtual_update_statistics(opt, tx)
        self.assertGreater(measured['virtual_update_norm'], 0.)
        for (_, parameter), old in zip(tx.parameters, tx.original):
            self.assertFalse(torch.equal(parameter, old))
            torch.testing.assert_close(parameter.to(torch.bfloat16), old.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(model(wave), before, rtol=0, atol=0)
        tx.restore()
        self.assertFalse(model._v31810_virtual_active)

    def test_audit_records_actual_norm_and_outputs_without_extra_encodings(self):
        cfg, units = fixture()
        one, two = Toy(.2), Toy(.2)
        two.load_state_dict(one.state_dict())
        opt_one, opt_two = optimizer(one), optimizer(two)
        initial = rng_state()
        ordinary = train_step(one, opt_one, units, cfg, .03, .5, audit=False)
        after = rng_state()
        restore_rng(initial)
        audited = train_step(two, opt_two, units, cfg, .03, .5, audit=True)
        assert_state_equal(self, one.state_dict(), two.state_dict())
        assert_state_equal(self, opt_one.state_dict(), opt_two.state_dict())
        assert_state_equal(self, after, rng_state())
        self.assertEqual(ordinary['encoded_views'], len(units))
        self.assertEqual(audited['encoded_views'], 2 * len(units))
        self.assertEqual(two.calls, 2 * one.calls)
        self.assertIsNone(ordinary['meta']['virtual_update_norm'])
        self.assertFalse(ordinary['meta']['virtual_update_measured'])
        self.assertTrue(audited['meta']['virtual_update_measured'])
        self.assertGreater(audited['meta']['virtual_update_norm'], 0.)
        self.assertGreater(audited['meta']['planned_virtual_update_norm'], 0.)
        self.assertEqual(set(audited['meta']['module_virtual_update_norms']), {'lora', 'fusion', 'head'})
        self.assertEqual(set(audited['meta']['module_planned_virtual_update_norms']), {'lora', 'fusion', 'head'})
        for side in ('A', 'B'):
            self.assertGreater(audited['meta'][side + '_logit_change']['l2'], 0.)
        self.assertFalse(one._v31810_virtual_active)
        self.assertFalse(two._v31810_virtual_active)
        json.dumps(audited, allow_nan=False)

    def test_zero_alpha_matched_logits_are_unchanged_with_dropout(self):
        cfg, units = fixture()
        model = Toy(.25)
        result = train_step(model, optimizer(model), units, cfg, 0., .5, audit=True)
        self.assertEqual(result['meta']['virtual_update_norm'], 0.)
        for side in ('A', 'B'):
            self.assertEqual(result['meta'][side + '_logit_change'], dict(l2=0., rms=0., max_abs=0.))

    def test_failure_after_virtual_write_restores_and_clears_guard(self):
        cfg, units = fixture()
        model = Toy(.2)
        opt = optimizer(model)
        before = copy.deepcopy(model.state_dict())
        seen = []
        hook = model.register_forward_pre_hook(
            lambda module, _args: seen.append(getattr(module, '_v31810_virtual_active', False)))
        model.fail_at = 3  # A uses two microbatches; B begins at temporary weights.
        try:
            with self.assertRaisesRegex(RuntimeError, 'injected temporary'):
                train_step(model, opt, units, cfg, .03, .5)
        finally:
            hook.remove()
        self.assertEqual(seen, [False, False, True])
        self.assertFalse(model._v31810_virtual_active)
        assert_state_equal(self, before, model.state_dict())
        self.assertFalse(opt.state)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_calibration_uses_existing_forwards_and_restores_guard(self):
        cfg, _ = fixture(copies=1)
        model = Toy(.2)
        opt = optimizer(model)
        before = copy.deepcopy(model.state_dict())
        initial = rng_state()
        try:
            report = calibrate(model, opt, tasks, cfg)
        except CalibrationError as exc:
            report = exc.report
        self.assertEqual(report['encoded_views'], 12 * 8 * (1 + len(CANDIDATES)))
        self.assertEqual(len(report['records']), 12 * len(CANDIDATES))
        for row in report['records']:
            self.assertTrue(row['virtual_update_measured'])
            self.assertGreater(row['planned_virtual_update_norm'], 0.)
            self.assertGreater(row['virtual_update_norm'], 0.)
            self.assertGreater(row['A_logit_change']['l2'], 0.)
        self.assertFalse(model._v31810_virtual_active)
        assert_state_equal(self, before, model.state_dict())
        assert_state_equal(self, initial, rng_state())
        self.assertFalse(opt.state)
        json.dumps(report, allow_nan=False)

    def test_calibration_failure_clears_virtual_guard(self):
        cfg, _ = fixture(copies=1)
        model = Toy()
        model.fail_at = 2
        opt = optimizer(model)
        before = copy.deepcopy(model.state_dict())
        with self.assertRaises(CalibrationError):
            calibrate(model, opt, tasks, cfg)
        self.assertFalse(model._v31810_virtual_active)
        assert_state_equal(self, before, model.state_dict())
        self.assertFalse(opt.state)

    def test_logit_comparison_rejects_changed_source_order(self):
        before = dict(_logits=torch.ones(2, 2), _logit_sources=['a', 'b'])
        after = dict(_logits=torch.ones(2, 2), _logit_sources=['b', 'a'])
        with self.assertRaisesRegex(ValueError, 'source order'):
            logit_change(before, after)


if __name__ == '__main__':
    unittest.main()
