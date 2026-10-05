"""Decision reachability, deployment identity, and outcome-based guard regressions."""
import copy
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from w2v_v38.test_core import bundle, dev_bundle, config as previous_config
from .metrics import measure, selection, change_audit, matched_real_recall
from .model import ResidualClassifier, CenteredStudent

torch.set_num_threads(1)


def config():
    value = previous_config()
    value.update(seed=3901, loss_temperature=2., hard_example_gain=3., hard_example_temperature=2.,
                 safety_margin=1., train_max_recall_drop=.005,
                 weight_decay=1e-4, lambda_grid=[.005, .02])
    return value


class ModelTests(unittest.TestCase):
    def test_qualified_language_arm_fits_with_controls_without_changing_base(self):
        from .fit import fit_all
        data, w, b = bundle(16)
        before = data['x'].copy()
        stages = []
        def stage(name, compute):
            stages.append(name)
            return compute()
        with redirect_stdout(io.StringIO()), patch('scipy.optimize.minimize', side_effect=AssertionError('No SciPy solve')):
            result = fit_all(data, w, b, config(), stage)
        self.assertTrue(result['language']['qualified'], result['language'])
        self.assertEqual(stages, ['language', 'calibration', 'residual_control', 'language_residual'])
        self.assertEqual(result['split']['group_overlap'], 0)
        self.assertTrue(result['split']['encoder_previously_saw_train'])
        for candidate in result['candidates']:
            self.assertEqual(candidate['status'], 'fitted')
            model = ResidualClassifier.restore(candidate['spec'])
            torch.testing.assert_close(model.weight, w, atol=0, rtol=0)
            torch.testing.assert_close(model.bias, b, atol=0, rtol=0)
            self.assertEqual(model.student is not None, candidate['name'] == 'language_residual')
            self.assertTrue(bool(torch.isfinite(model(torch.from_numpy(data['x']))).all()))
        np.testing.assert_array_equal(before, data['x'])

    def test_zero_start_is_exact_and_confident_errors_can_cross_boundary(self):
        weight, bias = torch.tensor([[1.], [0.]]), torch.zeros(2)
        x = torch.tensor([[.01], [2.], [5.3], [14.], [100.]])
        original = torch.nn.functional.linear(x, weight, bias)
        for arm in ('residual_control', 'language_residual'):
            student = CenteredStudent(1, 4, 8, torch.zeros(1), torch.ones(1)) if arm == 'language_residual' else None
            module = ResidualClassifier(weight, bias, arm=arm, mean=torch.zeros(1),
                                        scale=torch.ones(1), hidden=8, student=student)
            torch.testing.assert_close(module(x), original, atol=0, rtol=0)
            with torch.no_grad():
                module.output.bias.fill_(-10.)
            result = module(x)
            self.assertTrue(bool((result[:, 0] < result[:, 1]).all()))
            delta = result[:, 0] - result[:, 1] - x[:, 0]
            self.assertTrue(bool((delta.abs() <= 2 + x[:, 0]).all()))
            restored = ResidualClassifier.restore(module.spec())
            torch.testing.assert_close(restored(x), result, atol=0, rtol=0)
            self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
            self.assertNotIn('weight', dict(module.named_parameters()))
            with torch.no_grad():
                module.output.bias.fill_(10.)
            result = module(-x)
            self.assertTrue(bool((result[:, 0] > result[:, 1]).all()))

    def test_nonzero_cached_language_context_matches_deployed_prediction(self):
        torch.manual_seed(8)
        student = CenteredStudent(8, 4, 8, torch.zeros(8), torch.ones(8))
        with torch.no_grad():
            student.output.weight.fill_(.15)
            student.output.bias.copy_(torch.tensor([.1, -.2, .3, -.4]))
        module = ResidualClassifier(torch.randn(2, 8), torch.randn(2), arm='language_residual',
                                    mean=torch.zeros(8), scale=torch.ones(8), hidden=8, student=student)
        with torch.no_grad():
            module.output.weight.fill_(.03)
        x = torch.randn(32, 8)
        base = torch.nn.functional.linear(x, module.weight, module.bias)
        margin = base[:, 0] - base[:, 1]
        context = module.student(x)
        self.assertGreater(float(context.abs().max()), 0.)
        delta = module.adjustment(x, margin, context)
        direct = module(x)
        torch.testing.assert_close(direct, base + torch.stack((delta / 2, -delta / 2), -1), atol=0, rtol=0)
        without_context = module.adjustment(x, margin, torch.zeros_like(context))
        self.assertGreater(float((delta - without_context).abs().max()), 1e-6)
        torch.testing.assert_close(ResidualClassifier.restore(module.spec())(x), direct, atol=0, rtol=0)

    def test_old_correction_spec_cannot_silently_change_meaning(self):
        _, w, b = bundle()
        spec = ResidualClassifier(w, b).spec()
        spec.pop('correction_rule')
        with self.assertRaisesRegex(ValueError, 'adaptive-margin'):
            ResidualClassifier.restore(spec)

    def test_global_calibration_preserves_ranking(self):
        module = ResidualClassifier(torch.tensor([[1.], [0.]]), torch.zeros(2), arm='calibration')
        with torch.no_grad():
            module.raw_scale.fill_(-2.)
            module.raw_shift.fill_(3.)
        logits = module(torch.linspace(-20, 20, 1000)[:, None])
        self.assertTrue(bool((torch.diff(logits[:, 0] - logits[:, 1]) > 0).all()))

    def test_nonfinite_parameters_and_invalid_normalization_fail_closed(self):
        for scale in (torch.zeros(2), torch.tensor([1., float('nan')])):
            with self.assertRaises(ValueError):
                ResidualClassifier(torch.zeros(2, 2), torch.zeros(2), arm='residual_control',
                                   mean=torch.zeros(2), scale=scale)


class MetricTests(unittest.TestCase):
    def test_fake_and_rank_guards_still_override_headline_gain(self):
        dev, _, _ = dev_bundle()
        baseline = measure(dev['rows'], dev['logits'])
        baseline['weighted_f1'] = .95
        for group in ('online/en', 'online/zh', 'seen/en', 'seen/zh', 'heldout/en', 'heldout/zh'):
            value = copy.deepcopy(baseline)
            value['weighted_f1'] = .98
            value['groups'][group]['recall'][0] -= .01
            item = dict(name='residual_control', status='fitted', metrics=value)
            self.assertEqual(selection(baseline, [item], config()), 'baseline')
            self.assertIn(group + '_fake_recall_drop', item['guardrails'])
        value = copy.deepcopy(baseline)
        value['weighted_f1'] = .98
        value['matched']['seen/en']['real_recall'] -= .02
        item = dict(name='residual_control', status='fitted', metrics=value)
        self.assertEqual(selection(baseline, [item], config()), 'baseline')

    def test_safe_language_gain_can_be_selected_without_claiming_language_causality(self):
        dev, _, _ = dev_bundle()
        baseline = measure(dev['rows'], dev['logits'])
        baseline['weighted_f1'] = .95
        for group in ('online/en', 'seen/en', 'heldout/en'):
            baseline['groups'][group]['recall'][1] = .8
        control = copy.deepcopy(baseline)
        control['weighted_f1'] = .97
        language = copy.deepcopy(control)
        language['weighted_f1'] = .9702
        for group in ('online/en', 'seen/en', 'heldout/en'):
            language['groups'][group]['recall'][1] = .82
        items = [dict(name=n, status='fitted', metrics=copy.deepcopy(control))
                 for n in ('calibration', 'residual_control')]
        items.append(dict(name='language_residual', status='fitted', metrics=language, student_qualified=True))
        self.assertEqual(selection(baseline, items, config()), 'language_residual')
        self.assertTrue(items[-1]['eligible'])
        self.assertFalse(items[-1]['language_specific_evidence'])
        self.assertIn('no_weighted_gain_over_residual_control', items[-1]['language_attribution_reasons'])
        self.assertIn('no_en_noisy_rank_gain_at_matched_fake_recall', items[-1]['language_attribution_reasons'])

    def test_change_counts_do_not_hide_fake_errors_in_real_rescue(self):
        rows = [dict(condition='seen', language='en', label=v) for v in (0, 1)]
        old = np.asarray([[1., 0.], [1., 0.]])
        result = change_audit(rows, old, -old)['seen/en']
        self.assertEqual(result['real_rescued'], 1)
        self.assertEqual(result['new_fake_errors'], 1)

    def test_matched_recall_respects_ties(self):
        labels = np.asarray([0, 0, 0, 1, 1])
        margins = np.asarray([1., 1., 2., 1., 0.])
        value = matched_real_recall(labels, margins, .99)
        self.assertEqual(value['actual_fake_recall'], 1.)
        self.assertEqual(value['real_recall'], .5)


if __name__ == '__main__':
    unittest.main()
