import copy
import json
import unittest

import torch

from .calibration import CANDIDATES, CalibrationError, calibrate, choose_alpha
from .meta import rng_state
from .test_meta import Toy, assert_state_equal, fixture, optimizer


def tasks():
    for kind in ('noise', 'rtc', 'joint'):
        for _ in range(4):
            _, units = fixture(copies=1)
            for unit in units:
                unit[0]['task_kind'] = kind
            yield dict(units=units, task_kind=kind)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_selection_is_smallest_candidate_with_exact_gates(self):
        rows = []
        drops = (.001, .004, .01, .03, .08)
        for alpha, drop in zip(CANDIDATES, drops):
            rows.extend(dict(alpha=alpha, relative_drop=drop, finite=True) for _ in range(12))
        selected, report = choose_alpha(rows)
        self.assertEqual(selected, .01)
        self.assertEqual([item['accepted'] for item in report], [False, False, True, True, False])
        rows[24]['relative_drop'] = -.21
        self.assertEqual(choose_alpha(rows)[0], .03)
        for row in rows:
            row['finite'] = False
        self.assertIsNone(choose_alpha(rows)[0])

    def test_twelve_tasks_restore_every_state_and_only_encode_A(self):
        torch.manual_seed(5)
        cfg, _ = fixture(copies=1)
        model = Toy(.2)
        opt = optimizer(model)
        before = copy.deepcopy(model.state_dict())
        moments = copy.deepcopy(opt.state_dict())
        initial_rng = rng_state()
        calls = []
        try:
            result = calibrate(model, opt, tasks, cfg, progress=lambda i, total, report: calls.append((i, total)))
        except CalibrationError as exc:
            result = exc.report
        assert_state_equal(self, before, model.state_dict())
        assert_state_equal(self, moments, opt.state_dict())
        assert_state_equal(self, initial_rng, rng_state())
        self.assertEqual(result['task_counts'], dict(noise=4, rtc=4, joint=4))
        self.assertEqual(len(result['records']), 60)
        self.assertEqual(result['encoded_views'], 12 * 8 * 6)
        self.assertEqual(calls, [(i, 12) for i in range(1, 13)])
        json.dumps(result, allow_nan=False)

    def test_calibration_error_restores_model_and_rng(self):
        torch.manual_seed(5)
        cfg, _ = fixture(copies=1)
        model = Toy(.2)
        opt = optimizer(model)
        before = copy.deepcopy(model.state_dict())
        initial_rng = rng_state()
        model.fail_at = 4
        with self.assertRaises(CalibrationError) as caught:
            calibrate(model, opt, tasks, cfg)
        self.assertEqual(caught.exception.report['status'], 'failed')
        assert_state_equal(self, before, model.state_dict())
        assert_state_equal(self, initial_rng, rng_state())
        self.assertFalse(opt.state)

    def test_calibration_rejects_wrong_task_type_budget(self):
        cfg, _ = fixture(copies=1)
        model = Toy()
        with self.assertRaises(CalibrationError) as caught:
            calibrate(model, optimizer(model), list(tasks())[:4], cfg)
        self.assertIn('four tasks', str(caught.exception))

    def test_calibration_requires_enabled_lora(self):
        cfg, _ = fixture(copies=1)
        model = Toy()
        model.set_phase(False)
        with self.assertRaisesRegex(ValueError, 'Enable full LoRA'):
            calibrate(model, optimizer(model), tasks, cfg)

    def test_local_amp_scale_retry_keeps_live_scaler_rng_and_model_unchanged(self):
        torch.manual_seed(5)
        cfg, _ = fixture(copies=1)
        model = Toy(.2)
        opt = optimizer(model)
        opt._v31810_scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        scaler_before = copy.deepcopy(opt._v31810_scaler.state_dict())
        model_before = copy.deepcopy(model.state_dict())
        initial_rng = rng_state()
        calls, identities = [], []
        forward = model.register_forward_pre_hook(
            lambda _model, args: identities.append(tuple(id(wave) for wave in args[0])))
        def initial_overflow(gradient):
            calls.append(1)
            return gradient * float('inf') if len(calls) == 1 else gradient
        hook = model.head[-1].weight.register_hook(initial_overflow)
        try:
            try:
                report = calibrate(model, opt, tasks, cfg)
            except CalibrationError as exc:
                report = exc.report
        finally:
            forward.remove(); hook.remove()
        self.assertEqual(report['task_count'], 12)
        self.assertEqual(len(report['gradient_retries']), 1)
        self.assertEqual(report['final_calibration_scale'], 64.)
        self.assertEqual(report['encoded_views'], 12 * 8 * 6 + 8)
        self.assertEqual(identities[0], identities[1])
        assert_state_equal(self, scaler_before, opt._v31810_scaler.state_dict())
        assert_state_equal(self, model_before, model.state_dict())
        assert_state_equal(self, initial_rng, rng_state())
        self.assertFalse(opt.state)

    def test_persistent_calibration_gradient_overflow_fails_after_three_retries(self):
        cfg, _ = fixture(copies=1)
        model = Toy()
        opt = optimizer(model)
        opt._v31810_scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        calls = []
        def overflow(gradient):
            calls.append(1)
            return gradient * float('inf')
        hook = model.head[-1].weight.register_hook(overflow)
        try:
            with self.assertRaises(CalibrationError) as caught:
                calibrate(model, opt, tasks, cfg)
        finally:
            hook.remove()
        self.assertIn('bounded loss-scale retries', str(caught.exception))
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(caught.exception.report['gradient_retries']), 3)
        self.assertEqual(opt._v31810_scaler.get_scale(), 128.)
        self.assertFalse(opt.state)


if __name__ == '__main__':
    unittest.main()
