"""Regression tests for useful corrections without class-prior shortcuts."""
from contextlib import redirect_stdout
from collections import defaultdict
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .fit import (objective_parts, objective, hard_example_coefficients,
                  _better, _train_metrics, _train, fit_all)
from .metrics import measure
from .model import ResidualClassifier


def fixture():
    rows, values = [], []
    for language in ('en', 'zh'):
        for label in (0, 1):
            for source in range(8):
                identity = f'{language}/{label}/{source}'
                for condition in ('offline', 'online', 'noisy_a', 'noisy_b'):
                    rows.append(dict(id=identity + '/' + condition, source_id=identity,
                        group_id=identity, language=language, label=label, condition=condition,
                        split='train', view='full'))
                    values.append([1. if label == 0 else -1.,
                                   1. if language == 'en' else -1., source / 8., .1])
    return rows, torch.tensor(values)


class ObjectiveTests(unittest.TestCase):
    def test_large_safe_confidence_change_is_not_penalized(self):
        margin = torch.tensor([10., -10.])
        delta = torch.tensor([-8., 8.])
        target = torch.tensor([1., 0.])
        mass = torch.tensor([.5, .5])
        parts = objective_parts(margin, delta, target, mass, 0., {}, torch.tensor(.5), torch.tensor(.5))
        self.assertEqual(float(parts['fake_protection'].detach()), 0.)
        self.assertEqual(float(parts['real_protection'].detach()), 0.)
        unsafe = objective_parts(margin, delta * 2, target, mass, 0., {}, torch.tensor(.5), torch.tensor(.5))
        self.assertGreater(float(unsafe['fake_protection']), 0.)
        self.assertGreater(float(unsafe['real_protection']), 0.)

    def test_wrong_baseline_examples_are_free_to_cross_boundary(self):
        margin = torch.tensor([10., -10.])
        target = torch.tensor([0., 1.])
        delta = torch.tensor([-11., 11.], requires_grad=True)
        mass = torch.tensor([.5, .5])
        parts = objective_parts(margin, delta, target, mass, 0., {}, torch.tensor(0.), torch.tensor(0.))
        self.assertEqual(float(parts['fake_protection'].detach()), 0.)
        self.assertEqual(float(parts['real_protection'].detach()), 0.)
        sum(parts.values()).backward()
        self.assertTrue(bool(torch.isfinite(delta.grad).all()))
        self.assertLess(float(sum(parts.values()).detach()), float(objective(margin, delta.detach() * 0, target,
            mass, 0., {}, torch.tensor(0.), torch.tensor(0.))))

    def test_tempering_strengthens_gradient_on_easy_examples(self):
        grads = []
        for temperature in (1., 2.):
            delta = torch.zeros(1, requires_grad=True)
            loss = objective(torch.tensor([10.]), delta, torch.ones(1), torch.ones(1), 0.,
                             dict(loss_temperature=temperature), torch.tensor(1.), torch.tensor(0.))
            loss.backward()
            grads.append(float(delta.grad.abs()))
        self.assertGreater(grads[1], grads[0] * 10)

    def test_hard_mining_preserves_every_language_class_condition_mass(self):
        rows, _ = fixture()
        mass = torch.arange(1, len(rows) + 1, dtype=torch.float32)
        mass /= mass.sum()
        margin = torch.linspace(-12, 12, len(rows), requires_grad=True)
        mined = hard_example_coefficients(rows, mass, margin, {})
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[(row['language'], row['label'], row['condition'])].append(i)
        for indices in groups.values():
            torch.testing.assert_close(mined[indices].sum(), mass[indices].sum())
            relative = mined[indices] / mass[indices]
            self.assertLessEqual(float(relative.max() / relative.min()), 4.00001)
        self.assertFalse(mined.requires_grad)
        torch.testing.assert_close(mined.sum(), mass.sum())

    def test_selection_uses_fixed_decisions_before_ce(self):
        self.assertTrue(_better(.97, .3, .96, .1))
        self.assertFalse(_better(.95, .01, .96, .1))
        self.assertTrue(_better(.96, .09, .96, .1))
        self.assertFalse(_better(.96, .11, .96, .1))

    def test_fast_train_metric_equals_deployment_metric(self):
        rows, _ = fixture()
        margin = torch.linspace(-4, 5, len(rows))
        fast = _train_metrics(rows, margin)
        exact = measure(rows, np.stack((margin.numpy(), np.zeros(len(rows))), axis=1), train_proxy=True)
        for key in ('clean_f1', 'noisy_f1', 'weighted_f1'):
            self.assertAlmostEqual(fast[key], exact[key], places=12)

    def test_fitting_rejects_dev_before_accessing_features(self):
        with self.assertRaisesRegex(ValueError, 'official Train only'):
            fit_all(dict(rows=[dict(split='dev')]), None, None, {}, None)

    def test_nonfinite_validation_is_rejected_before_trace_serialization(self):
        rows, x = fixture()
        cfg = dict(seed=39, residual_hidden=8, teacher_dim=4, residual_cap=2.,
                   learning_rate=.001, batch_rows=len(rows), epochs=1)
        with patch('w2v_v39.fit._predict_delta', return_value=torch.full((len(rows),), float('nan'))):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(FloatingPointError, 'validation correction'):
                _train('residual_control', x, rows, torch.zeros(2, 4), torch.zeros(2), cfg, .005,
                       tune=(x, rows))
        with self.assertRaisesRegex(FloatingPointError, 'selection margins'):
            _train_metrics(rows, torch.full((len(rows),), float('inf')))

    def test_high_confidence_errors_can_be_corrected_in_actual_fit(self):
        rows, x = fixture()
        weight, bias = torch.zeros(2, 4), torch.zeros(2)
        weight[0, 0], weight[1, 0] = -4., 4.
        cfg = dict(seed=39, residual_hidden=8, teacher_dim=4, residual_cap=2.,
                   learning_rate=.03, batch_rows=len(rows), epochs=24)
        with redirect_stdout(io.StringIO()):
            spec, detail = _train('residual_control', x, rows, weight, bias, cfg, .005,
                                  tune=(x, rows))
        logits = ResidualClassifier.restore(spec)(x)
        self.assertGreater(detail['weighted_f1'], .99)
        self.assertGreater(detail['best_epoch'], 0)
        self.assertTrue(all(r['recall_guards_passed'] for r in detail['trace']))
        self.assertGreater(int((logits.argmax(1) == torch.tensor([r['label'] for r in rows])).sum()), len(rows) - 1)
        self.assertEqual(detail['optimizer_steps_at_best'], detail['best_epoch'])
        self.assertIn('objective_components', detail['trace'][-1])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
