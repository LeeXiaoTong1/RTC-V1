"""Numerical, data isolation, and false-negative protection regressions."""
from contextlib import redirect_stdout
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .common import read_json, atomic_json
from .data import validate_rows, cache_path, replay_base
from .fit import fit_all, objective
from .metrics import matched_real_recall, measure, selection, change_audit
from .model import ResidualClassifier, CenteredStudent
from .student import qualified, train_student

torch.set_num_threads(1)


def config():
    return dict(device='cpu', seed=3801, holdout_fraction=.2, batch_rows=64,
        epochs=3, student_epochs=12, student_hidden=8, residual_hidden=8,
        student_learning_rate=.01, learning_rate=.003, residual_cap=2., lambda_grid=[.02, .1],
        fake_protection=2., real_protection=1., protection_slack=.1,
        min_student_r2=.02, min_student_language_accuracy=.65, min_teacher_language_accuracy=.7,
        min_gain=.001, max_clean_drop=.001, max_fake_drop=.002, max_real_drop=.005,
        max_auc_drop=.0005, min_en_real_gain=.005, min_language_control_gain=.0005,
        min_ranking_gain=.002, matched_fake_recall=.99, max_matched_real_drop=.005)


def bundle(n=10, constant=False):
    rng = np.random.default_rng(5)
    rows, x, g = [], [], []
    for language in ('en', 'zh'):
        sign = 1 if language == 'en' else -1
        for label in (0, 1):
            for source in range(n):
                key = f'{language}/{label}/{source}'
                for condition in ('offline', 'online', 'noisy_a', 'noisy_b'):
                    rows.append(dict(id=key + '/' + condition, source_id=key, group_id=key,
                        language=language, label=label, condition=condition, split='train', view='full'))
                    value = rng.normal(0, .02, 8)
                    value[0] = (1 if label == 0 else -1) * 2.
                    value[1] = sign
                    x.append(value)
                    target = np.asarray([0 if constant else sign * .4, 0., .7, 1.])
                    g.append(target / np.linalg.norm(target))
    x, g = np.asarray(x, dtype=np.float32), np.asarray(g, dtype=np.float32)
    weight, bias = torch.zeros(2, 8), torch.zeros(2)
    weight[0, 0] = .5
    weight[1, 0] = -.5
    logits = torch.nn.functional.linear(torch.from_numpy(x), weight, bias).numpy()
    return dict(x=x, lid=g, rows=rows, logits=logits), weight, bias


def dev_bundle():
    value, weight, bias = bundle()
    indices = [i for i, r in enumerate(value['rows']) if r['condition'] != 'offline']
    rows = []
    for i in indices:
        row = value['rows'][i]
        rows.append(dict(row, id='dev/' + row['id'], source_id='dev/' + row['source_id'],
            group_id='dev/' + row['group_id'], split='dev',
            condition=dict(online='online', noisy_a='seen', noisy_b='heldout')[row['condition']]))
    return dict(x=value['x'][indices], logits=value['logits'][indices], rows=rows), weight, bias


class ModelTests(unittest.TestCase):
    def test_zero_initialization_exact_restore_and_bounded_correction(self):
        data, w, b = bundle()
        x = torch.from_numpy(data['x'])
        for arm in ('baseline', 'calibration', 'residual_control', 'language_residual'):
            with self.subTest(arm=arm):
                student = CenteredStudent(8, 4, 8, torch.zeros(8), torch.ones(8)) if arm == 'language_residual' else None
                model = ResidualClassifier(w, b, arm=arm, mean=torch.zeros(8), scale=torch.ones(8), hidden=8, student=student)
                torch.testing.assert_close(model(x), torch.from_numpy(data['logits']), rtol=0, atol=0)
                self.assertNotIn('weight', dict(model.named_parameters()))
                if arm.endswith('residual') or arm == 'residual_control':
                    with torch.no_grad():
                        model.output.weight.fill_(-100)
                        model.output.bias.fill_(-100)
                    delta = model(x)[:, 0] - model(x)[:, 1] - (x @ (w[0] - w[1]))
                    self.assertLessEqual(float(delta.abs().max()), 2.000001)
                restored = ResidualClassifier.restore(model.spec())
                torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
                self.assertTrue(all(not p.requires_grad for p in restored.parameters()))

    def test_calibration_is_global_and_monotone(self):
        model = ResidualClassifier(torch.tensor([[1.], [0.]]), torch.zeros(2), arm='calibration')
        with torch.no_grad():
            model.raw_scale.fill_(-2.)
            model.raw_shift.fill_(-2.)
        x = torch.linspace(-20, 20, 1000).reshape(-1, 1)
        result = model(x)
        self.assertTrue(bool(torch.all(torch.diff(result[:, 0] - result[:, 1]) > 0)))

    def test_fake_margin_penalty_opposes_wrong_direction(self):
        cfg = config()
        margin, target, mass = torch.tensor([.5]), torch.ones(1), torch.ones(1)
        wrong = torch.tensor([-1.], requires_grad=True)
        loss = objective(margin, wrong, target, mass, 0., cfg, torch.tensor(1.), torch.tensor(1.))
        loss.backward()
        self.assertLess(float(wrong.grad), 0.)  # Gradient descent moves delta upward.
        cfg['fake_protection'] = 0.
        unprotected = objective(margin, wrong.detach(), target, mass, 0., cfg, torch.tensor(1.), torch.tensor(1.))
        self.assertGreater(float(loss.detach()), float(unprotected.detach()))

    def test_nonfinite_or_nonpositive_scale_and_excessive_cap_rejected(self):
        for kwargs in (dict(scale=torch.zeros(2)), dict(cap=20.), dict(scale=torch.tensor([1., float('nan')]))):
            value = dict(mean=torch.zeros(2), scale=torch.ones(2), arm='residual_control')
            value.update(kwargs)
            with self.assertRaises(ValueError):
                ResidualClassifier(torch.zeros(2, 2), torch.zeros(2), **value)


class DataAndTrainingTests(unittest.TestCase):
    def test_teacher_and_detector_rows_must_match_and_no_dev_overlap(self):
        train, _, _ = bundle()
        dev, _, _ = dev_bundle()
        validate_rows(train['rows'], dev['rows'], copy.deepcopy(train['rows']))
        with self.assertRaises(ValueError):
            validate_rows(train['rows'], dev['rows'], list(reversed(train['rows'])))
        rows = copy.deepcopy(dev['rows'])
        rows[0]['group_id'] = train['rows'][0]['group_id']
        with self.assertRaises(ValueError):
            validate_rows(train['rows'], rows, train['rows'])

    def test_base_replay_detects_other_classifier(self):
        data, w, b = bundle()
        replay_base(data, w, b, 'cpu')
        with self.assertRaises(ValueError):
            replay_base(data, w, b + torch.tensor([1., 0.]), 'cpu')

    def test_student_real_only_constant_control_and_informative_holdout(self):
        from w2v_v36.fit import split_sources
        data, _, _ = bundle(16)
        ids, val, _ = split_sources(data['rows'])
        x, g = torch.from_numpy(data['x']), torch.from_numpy(data['lid'])
        rows = [data['rows'][i] for i in ids]
        valid = (x[val], g[val], [data['rows'][i] for i in val])
        with redirect_stdout(io.StringIO()):
            spec, quality = train_student(x[ids], g[ids], rows, config(), validation=valid)
            corrupt_x, corrupt_g = x[ids].clone(), g[ids].clone()
            fake = torch.tensor([r['label'] == 0 for r in rows])
            corrupt_x[fake] = 1000.
            corrupt_g[fake] = -1000.
            same, _ = train_student(corrupt_x, corrupt_g, rows, config(), validation=valid)
        self.assertGreater(quality['centered_r2_vs_train_mean'], .02)
        self.assertGreater(quality['student_language_probe']['source_balanced_accuracy'], .65)
        for key in spec['state']:
            torch.testing.assert_close(spec['state'][key], same['state'][key], rtol=0, atol=0)
        self.assertTrue(qualified(quality, config())[0])
        constant, _, _ = bundle(constant=True)
        with redirect_stdout(io.StringIO()):
            _, poor = train_student(torch.from_numpy(constant['x']), torch.from_numpy(constant['lid']), constant['rows'], config())
        self.assertFalse(qualified(poor, config())[0])

    def test_complete_train_controls_are_fitted_without_dev_and_frozen_weights_unchanged(self):
        import inspect
        self.assertNotIn('dev', inspect.signature(fit_all).parameters)
        data, w, b = bundle(4, constant=True)
        before = data['x'].copy()
        with redirect_stdout(io.StringIO()), patch('scipy.optimize.minimize', side_effect=AssertionError('No L-BFGS')):
            result = fit_all(data, w, b, config(), lambda _, function: function())
        self.assertEqual(result['split']['group_overlap'], 0)
        self.assertTrue(result['split']['encoder_previously_saw_train'])
        self.assertEqual([v['name'] for v in result['candidates']], ['calibration', 'residual_control', 'language_residual'])
        self.assertEqual(result['candidates'][-1]['status'], 'skipped')
        for candidate in result['candidates'][:2]:
            restored = ResidualClassifier.restore(candidate['spec'])
            torch.testing.assert_close(restored.weight, w, atol=0, rtol=0)
        np.testing.assert_array_equal(data['x'], before)
        self.assertEqual(result['train_views'], len(data['rows']))
        data['rows'][0]['split'] = 'dev'
        with self.assertRaises(ValueError):
            fit_all(data, w, b, config(), lambda _, function: function())

    def test_reuse_prefers_complete_local_then_verified_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            borrowed = source / 'borrowed'
            borrowed.mkdir()
            atomic_json(borrowed / 'complete.json', {})
            atomic_json(source / 'feature_reuse.json', dict(train=dict(path=str(borrowed))))
            self.assertEqual(cache_path(source, 'train'), borrowed.resolve())
            (borrowed / 'complete.json').unlink()
            with self.assertRaises(FileNotFoundError):
                cache_path(source, 'train')

    def test_qualified_language_branch_is_refitted_and_trained_with_both_controls(self):
        data, w, b = bundle(16)
        with redirect_stdout(io.StringIO()), patch('scipy.optimize.minimize', side_effect=AssertionError('No SciPy solver')):
            result = fit_all(data, w, b, config(), lambda _, function: function())
        self.assertTrue(result['language']['qualified'], result['language'])
        self.assertEqual(len(result['candidates']), 3)
        for candidate in result['candidates']:
            self.assertEqual(candidate['status'], 'fitted')
            module = ResidualClassifier.restore(candidate['spec'])
            value = module(torch.from_numpy(data['x']))
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertEqual(module.student is not None, candidate['name'] == 'language_residual')
            if candidate['name'] == 'language_residual':
                self.assertGreater(candidate['train_selection']['best_epoch'], 0)
                self.assertGreater(float(module.output.weight.abs().sum()), 0.)


class MetricTests(unittest.TestCase):
    def test_matched_recall_ties_match_exhaustive_threshold_search(self):
        rng = np.random.default_rng(12)
        for _ in range(30):
            labels = np.r_[np.zeros(37), np.ones(24)]
            scores = rng.integers(-5, 6, len(labels))
            for target in (.8, .99, 1.):
                measured = matched_real_recall(labels, scores, target)
                thresholds = [t for t in np.unique(scores) if np.mean(scores[labels == 0] >= t) >= target]
                truth = max(np.mean(scores[labels == 1] < t) for t in thresholds)
                self.assertAlmostEqual(measured['real_recall'], truth)
                self.assertGreaterEqual(measured['actual_fake_recall'], target)

    def test_both_languages_fake_guard_and_rank_guard_override_headline_gain(self):
        dev, _, _ = dev_bundle()
        base = measure(dev['rows'], dev['logits'])
        # Synthetic metric perturbations isolate guard behavior, not a performance claim.
        base['weighted_f1'] = .95
        for group in ('seen/en', 'heldout/en', 'seen/zh', 'heldout/zh'):
            candidate = copy.deepcopy(base)
            candidate['weighted_f1'] = .97
            candidate['groups'][group]['recall'][0] -= .01
            item = dict(name='residual_control', status='fitted', metrics=candidate)
            self.assertEqual(selection(base, [item], config()), 'baseline')
            self.assertIn(group + '_fake_recall_drop', item['guardrails'])
        candidate = copy.deepcopy(base)
        candidate['weighted_f1'] = .97
        candidate['groups']['seen/en']['auc'] -= .002
        item = dict(name='residual_control', status='fitted', metrics=candidate)
        self.assertEqual(selection(base, [item], config()), 'baseline')
        self.assertIn('seen/en_auc_drop', item['guardrails'])

    def test_language_must_beat_controls_and_show_ranking_gain(self):
        dev, _, _ = dev_bundle()
        base = measure(dev['rows'], dev['logits'])
        base['weighted_f1'] = .95
        candidate = copy.deepcopy(base)
        candidate['weighted_f1'] = .97
        items = [dict(name=n, status='fitted', metrics=copy.deepcopy(candidate), student_qualified=True)
                 for n in ('calibration', 'residual_control', 'language_residual')]
        selected = selection(base, items, config())
        self.assertEqual(selected, 'calibration')
        self.assertIn('no_weighted_gain_over_calibration', items[-1]['guardrails'])
        self.assertIn('no_en_noisy_rank_gain_at_matched_fake_recall', items[-1]['guardrails'])

    def test_change_counts_distinguish_real_rescue_from_new_fake_errors(self):
        rows = [dict(condition='seen', language='en', label=v) for v in (0, 1)]
        old = np.asarray([[1., 0.], [1., 0.]])
        new = -old
        result = change_audit(rows, old, new)['seen/en']
        self.assertEqual(result['real_rescued'], 1)
        self.assertEqual(result['new_fake_errors'], 1)


if __name__ == '__main__':
    unittest.main()
