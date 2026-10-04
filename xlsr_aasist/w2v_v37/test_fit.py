"""Leakage, numerical replay, source-budget and deployment-guard checks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .fit import (CANDIDATES, _choose, _student_stats, balanced_cross_entropy, fit_candidates,
                  fit_language_state, logits_from_state, predict,
                  real_source_coefficients, save_fit_report, select_candidate, sources,
                  split_sources, transform_features)
from w2v_v36.metrics import evaluate


def train_rows(counts=(8, 8, 8, 8), missing_online=False):
    rows = []
    for (language, label), count in zip([('en', 0), ('en', 1), ('zh', 0), ('zh', 1)], counts):
        for i in range(count):
            source = f'{language}/{label}/{i}'
            for condition in ('offline', 'online', 'noisy_a', 'noisy_b'):
                if missing_online and condition == 'online' and i % 2 == 0:
                    continue
                rows.append(dict(source_id=source, group_id='sha:' + source, language=language,
                                 label=label, condition=condition, view='full'))
    return rows


def dev_rows():
    rows = [r for r in train_rows() if r['condition'] != 'offline']
    return [dict(r, source_id='dev/' + r['source_id'], group_id='dev/' + r['group_id'],
                 condition={'noisy_a': 'seen', 'noisy_b': 'heldout'}.get(r['condition'], r['condition']))
            for r in rows]


def mock_predict_student(x, state):
    result = np.zeros((len(x), 256), dtype=np.float32)
    result[:, 0] = np.tanh(x[:, 1])
    result[:, 1] = np.tanh(x[:, 2])
    result[:, 2] = 1.
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def mock_fit_student(x, teacher, rows, cfg):
    return {'schema': 'test-student', 'diagnostics': {'fitting_rows': len(rows)}}


def bundle(rows, weight, bias, teacher=True):
    x = np.zeros((len(rows), 512), dtype=np.float32)
    x[:, 0] = [1. if row['label'] == 0 else -1. for row in rows]
    x[:, 1] = [.2 if row['language'] == 'en' else -.2 for row in rows]
    x[:, 2] = [int(row['source_id'].rsplit('/', 1)[1]) * .03 for row in rows]
    result = dict(x=x, rows=rows, logits=predict(x, weight, bias))
    if teacher:
        result['lid'] = mock_predict_student(x, {})
    return result


class LanguageMapTests(unittest.TestCase):
    def setUp(self):
        self.rows = train_rows((4, 3, 2, 5), missing_online=True)
        rng = np.random.default_rng(81)
        self.g = rng.normal(size=(len(self.rows), 256)).astype(np.float32)
        self.g[:, -1] = 7.  # A constant coordinate must remain well behaved.
        self.x = rng.normal(size=(len(self.rows), 512)).astype(np.float32)
        self.x[:, :4] += 2 * self.g[:, :4]

    def test_real_only_equal_language_and_per_source_budgets(self):
        coefficients = real_source_coefficients(self.rows)
        totals = {'en': 0., 'zh': 0.}
        for source in sources(self.rows).values():
            mass = sum(coefficients[i] for i in source['indices'].values())
            if source['label'] == 0:
                self.assertEqual(mass, 0.)
                continue
            totals[source['language']] += mass
            ordinary = sum(coefficients[i] for name, i in source['indices'].items() if not name.startswith('noisy'))
            self.assertAlmostEqual(ordinary, mass * .5)
            self.assertAlmostEqual(coefficients[source['indices']['noisy_a']], mass * .25)
            self.assertAlmostEqual(coefficients[source['indices']['noisy_b']], mass * .25)
        self.assertAlmostEqual(totals['en'], .5)
        self.assertAlmostEqual(totals['zh'], .5)

    def test_fake_feature_changes_cannot_change_language_scaler_or_map(self):
        state = fit_language_state(self.x, self.g, self.rows, .1)
        fake = np.asarray([r['label'] == 0 for r in self.rows])
        x, g = self.x.copy(), self.g.copy()
        x[fake] = 10000.; g[fake] = -10000.
        changed = fit_language_state(x, g, self.rows, .1)
        self.assertEqual(state, changed)
        self.assertEqual(state['scale'][-1], 1.)

    def test_centered_transform_preserves_weighted_real_mean_and_alpha_zero(self):
        state = fit_language_state(self.x, self.g, self.rows, 1., alpha=.5)
        mass = real_source_coefficients(self.rows)
        transformed = transform_features(self.x, self.g, state, chunk_rows=17)
        np.testing.assert_allclose(mass @ transformed, mass @ self.x, rtol=0, atol=2e-7)
        zero = dict(state, alpha=0.)
        np.testing.assert_array_equal(transform_features(self.x, None, zero), self.x)
        weight = np.zeros((2, 512), np.float32); weight[0, 0] = 1.
        exported = dict(weight=weight.tolist(), bias=[0., 0.], language_state=zero, student_state=None)
        np.testing.assert_array_equal(logits_from_state(self.x, state=exported), predict(self.x, weight, [0., 0.]))

    def test_exported_state_and_fitting_prediction_match(self):
        language = fit_language_state(self.x, self.g, self.rows, .1, alpha=.75)
        rng = np.random.default_rng(9)
        weight = rng.normal(size=(2, 512)).astype(np.float32)
        state = dict(weight=weight.tolist(), bias=[.1, -.2], language_state=language, student_state=None)
        roundtrip = json.loads(json.dumps(state, allow_nan=False))
        direct = predict(transform_features(self.x, self.g, language, 11), weight, state['bias'], 11)
        replay = logits_from_state(self.x, self.g, roundtrip, chunk_rows=11)
        np.testing.assert_array_equal(replay, direct)

    def test_dimensions_nonfinite_and_excess_subtraction_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '256'):
            fit_language_state(self.x, self.g[:, :255], self.rows, .1)
        with self.assertRaisesRegex(ValueError, '512'):
            fit_language_state(self.x[:, :511], self.g, self.rows, .1)
        broken = self.g.copy(); broken[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            fit_language_state(self.x, broken, self.rows, .1)
        with self.assertRaisesRegex(ValueError, 'alpha'):
            fit_language_state(self.x, self.g, self.rows, .1, alpha=.8)

    def test_identical_original_and_all_views_stay_in_one_partition(self):
        rows = train_rows()
        rows += [dict(r, source_id='alias/' + r['source_id']) for r in rows if r['source_id'] == 'en/0/0']
        fit, tune, audit = split_sources(rows, .2, 3701)
        self.assertFalse({rows[i]['group_id'] for i in fit} & {rows[i]['group_id'] for i in tune})
        self.assertEqual(len(fit) + len(tune), len(rows))
        self.assertEqual(audit['group_overlap'], 0)
        self.assertTrue(audit['encoder_previously_saw_train'])


class GuardrailTests(unittest.TestCase):
    def metrics(self):
        rows = dev_rows()
        labels = np.asarray([r['label'] for r in rows])
        baseline = evaluate(rows, np.stack([1. - labels, labels], axis=1))
        baseline.update(weighted_f1=.9, noisy_f1=.9, clean_f1=.9)
        for name, group in baseline['groups'].items():
            group['recall'] = [.9, .9]
        candidate = copy.deepcopy(baseline)
        candidate.update(weighted_f1=.91, noisy_f1=.91, clean_f1=.91)
        for condition in ('online', 'seen', 'heldout'):
            candidate['groups'][condition + '/en']['recall'][1] = .92
        return baseline, dict(name='language_debias', status='converged', metrics=candidate)

    def test_each_required_fake_real_guard_overrides_overall_gain(self):
        baseline, candidate = self.metrics()
        self.assertEqual(select_candidate(baseline, [candidate], {}), 'language_debias')
        for condition in ('online', 'seen', 'heldout'):
            for language, label in [('en', 0), ('en', 1), ('zh', 1)]:
                baseline, candidate = self.metrics()
                group = condition + '/' + language
                candidate['metrics']['groups'][group]['recall'][label] = .89
                self.assertEqual(select_candidate(baseline, [candidate], {}), 'baseline')
                kind = 'fake' if label == 0 else 'real'
                self.assertIn(group + '_' + kind + '_recall_below_guardrail', candidate['guardrails'])

    def test_en_real_mean_noisy_clean_and_weighted_guards(self):
        for field, value, issue in [('noisy_f1', .899, 'noisy_below_baseline'),
                                    ('clean_f1', .898, 'clean_below_guardrail'),
                                    ('weighted_f1', .901, 'weighted_gain_below_minimum')]:
            baseline, candidate = self.metrics()
            candidate['metrics'][field] = value
            self.assertEqual(select_candidate(baseline, [candidate], {}), 'baseline')
            self.assertIn(issue, candidate['guardrails'])
        baseline, candidate = self.metrics()
        for condition in ('online', 'seen', 'heldout'):
            candidate['metrics']['groups'][condition + '/en']['recall'][1] = .901
        self.assertEqual(select_candidate(baseline, [candidate], {}), 'baseline')
        self.assertIn('en_real_mean_gain_below_minimum', candidate['guardrails'])

    def test_incomplete_groups_and_nonconverged_reject(self):
        baseline, candidate = self.metrics()
        candidate['metrics']['groups'].pop('online/en')
        self.assertEqual(select_candidate(baseline, [candidate], {}), 'baseline')
        self.assertIn('online/en_real_recall_unavailable', candidate['guardrails'])
        baseline, candidate = self.metrics()
        candidate['status'] = 'rejected'
        self.assertEqual(select_candidate(baseline, [candidate], {}), 'baseline')

    def test_strict_guards_cannot_be_loosened(self):
        baseline, candidate = self.metrics()
        with self.assertRaisesRegex(ValueError, 'stricter'):
            select_candidate(baseline, [candidate], {'min_gain': 0.})

    def test_hyperparameters_choose_balanced_ce_instead_of_f1(self):
        a = dict(status='converged', train_holdout_balanced_ce=.21, regularization=.1,
                 alpha=.5, ridge=.1, metrics={'weighted_f1': .99})
        b = dict(status='converged', train_holdout_balanced_ce=.20, regularization=1.,
                 alpha=.75, ridge=1., metrics={'weighted_f1': .90})
        self.assertIs(_choose([a, b]), b)
        c = dict(b, alpha=.25)
        self.assertIs(_choose([b, c]), c)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.weight = np.zeros((2, 512), dtype=np.float32)
        self.weight[0, 0], self.weight[1, 0] = .25, -.25
        self.bias = np.asarray([1., -1.], dtype=np.float32)
        self.train = bundle(train_rows(), self.weight, self.bias)
        self.dev = bundle(dev_rows(), self.weight, self.bias, teacher=False)
        self.cfg = dict(lambda_grid=[.01], alpha_grid=[.25], ridge_grid=[.1],
                        max_iterations=100, fit_chunk_rows=31)

    def fit(self, dev=None, cfg=None, out=None):
        with contextlib.redirect_stdout(io.StringIO()), \
                patch('w2v_v37.fit.fit_student', side_effect=mock_fit_student) as student_fit, \
                patch('w2v_v37.fit.predict_student', side_effect=mock_predict_student):
            result = fit_candidates(self.train, dev or self.dev, self.weight, self.bias,
                                    self.cfg if cfg is None else cfg, out)
        return result, student_fit.call_args_list

    def test_two_candidates_full_refit_json_export_and_student_only_fitpart(self):
        with tempfile.TemporaryDirectory() as directory:
            result, student_calls = self.fit(out=directory)
            self.assertEqual([c['name'] for c in result['candidates']], list(CANDIDATES))
            self.assertEqual(len(student_calls), 2)
            fit, tune, _ = split_sources(self.train['rows'], .2, 3701)
            seen = {r['group_id'] for r in student_calls[0].args[2]}
            heldout = {self.train['rows'][i]['group_id'] for i in tune}
            self.assertFalse(seen & heldout)
            self.assertEqual(len(student_calls[0].args[0]), len(fit))
            self.assertEqual(len(student_calls[1].args[0]), len(self.train['rows']))
            disk = json.loads((Path(directory) / 'fit_report.json').read_text(encoding='utf-8'))
            self.assertNotIn('selected_patch', disk)
            self.assertNotIn('selected_state', disk)
            self.assertEqual(disk['selected'], result['selected'])
            self.assertEqual(disk['student_audit'], result['student_audit'])
            for saved, candidate in zip(disk['candidates'], result['candidates']):
                self.assertNotIn('patch', saved)
                self.assertEqual(saved['tuning'], candidate['tuning'])
                self.assertIn('patch', candidate)
            for candidate in result['candidates']:
                self.assertEqual(candidate['status'], 'converged')
                with patch('w2v_v37.fit.predict_student', side_effect=mock_predict_student):
                    replay = logits_from_state(self.dev['x'], state=candidate['patch'], chunk_rows=31)
                with np.load(Path(directory) / result['dev_score_files'][candidate['name']], allow_pickle=False) as scores:
                    np.testing.assert_array_equal(scores['logits'], replay)
                    self.assertTrue(all(scores[key].dtype.kind != 'O' for key in scores.files))
            self.assertEqual(result['selected'], 'head_only_control')
            self.assertIn('not evidence', result['outcome'])
            self.assertIsNone(result['selected_patch']['language_state'])
            self.assertIsNone(result['selected_patch']['student_state'])

    def test_dev_labels_cannot_affect_fit_or_hyperparameters(self):
        first, _ = self.fit()
        flipped = dict(self.dev, rows=[dict(r, label=1-r['label']) for r in self.dev['rows']])
        second, _ = self.fit(dev=flipped)
        for left, right in zip(first['candidates'], second['candidates']):
            self.assertEqual(left['patch'], right['patch'])
            self.assertEqual(left['tuning'], right['tuning'])
            self.assertEqual(left['selected_regularization'], right['selected_regularization'])
        self.assertEqual(second['selected'], 'baseline')
        np.testing.assert_array_equal(second['selected_patch']['weight'], self.weight)
        np.testing.assert_array_equal(second['selected_patch']['bias'], self.bias)
        self.assertIsNone(second['selected_patch']['language_state'])

    def test_nonconvergence_and_language_numerical_failure_keep_exact_baseline_or_control(self):
        result, _ = self.fit(cfg=dict(self.cfg, max_iterations=1))
        self.assertEqual(result['selected'], 'baseline')
        self.assertTrue(all(c['status'] == 'rejected' for c in result['candidates']))
        np.testing.assert_array_equal(result['selected_patch']['weight'], self.weight)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch('w2v_v37.fit.fit_student', side_effect=FloatingPointError('bad student')):
            result = fit_candidates(self.train, self.dev, self.weight, self.bias, self.cfg)
        self.assertEqual(result['candidates'][1]['status'], 'rejected')
        self.assertEqual(result['selected'], 'head_only_control')

    def test_baseline_replay_and_teacher_validation(self):
        self.train['logits'] += .1
        with self.assertRaisesRegex(ValueError, 'do not replay'):
            self.fit()
        self.train['logits'] = predict(self.train['x'], self.weight, self.bias)
        self.train.pop('lid')
        with self.assertRaisesRegex(ValueError, 'teacher'):
            self.fit()

    def test_balanced_ce_is_finite_for_extreme_logits(self):
        logits = np.tile(np.asarray([10000., -10000.], np.float32), (len(self.train['rows']), 1))
        self.assertAlmostEqual(balanced_cross_entropy(self.train['rows'], logits), 10000.)

    def test_student_fidelity_uses_fit_constant_reference_and_reports_variance(self):
        predicted = self.train['lid']
        fitting, reference = _student_stats(predicted, predicted, self.train['rows'])
        self.assertAlmostEqual(fitting['weighted_cosine'], 1.)
        self.assertFalse(fitting['effectively_constant'])
        opposite = -predicted
        heldout, _ = _student_stats(opposite, opposite, self.train['rows'], reference)
        self.assertAlmostEqual(heldout['weighted_cosine'], 1.)
        self.assertLess(heldout['constant_fit_teacher_mean_cosine'], 0.)
        result, _ = self.fit()
        self.assertIn('holdout_fidelity', result['student_audit'])
        self.assertGreater(result['student_audit']['holdout_fidelity']['weighted_predicted_variance'], 1e-10)

    def test_effectively_constant_student_rejects_language_only(self):
        def constant(x, state):
            output = np.zeros((len(x), 256), dtype=np.float32)
            output[:, 0] = 1.
            return output
        with contextlib.redirect_stdout(io.StringIO()), \
                patch('w2v_v37.fit.fit_student', side_effect=mock_fit_student), \
                patch('w2v_v37.fit.predict_student', side_effect=constant):
            result = fit_candidates(self.train, self.dev, self.weight, self.bias, self.cfg)
        self.assertEqual(result['selected'], 'head_only_control')
        self.assertEqual(result['candidates'][0]['status'], 'converged')
        self.assertEqual(result['candidates'][1]['status'], 'rejected')
        self.assertIn('effectively constant', result['candidates'][1]['reason'])
        self.assertTrue(result['student_audit']['holdout_fidelity']['effectively_constant'])

    def test_report_export_recursively_excludes_parameters_without_mutating_result(self):
        result, _ = self.fit()
        result['selected_state'] = result['selected_patch']
        result['deployment_replay'] = {'same_decisions': True, 'audit_copy': {
            'student_state': {'w1': [[123.]], 'b1': [456.]},
            'weight': [[789.]], 'bias': [321.], 'mapping': [[654.]],
            'diagnostics': {'weighted_cosine': .81},
            'nested': [{'patch': {'weight': [[999.]]}, 'row_count': 10}]}}
        before = copy.deepcopy(result)
        selected_identity = result['selected_patch']
        banned = {'selected_patch', 'selected_state', 'patch', 'weight', 'bias',
                  'language_state', 'student_state', 'input_mean', 'input_scale',
                  'w1', 'b1', 'w2', 'b2', 'mapping'}

        def assert_diagnostics_only(value):
            if isinstance(value, dict):
                self.assertFalse(banned.intersection(value))
                for item in value.values():
                    assert_diagnostics_only(item)
            elif isinstance(value, list):
                for item in value:
                    assert_diagnostics_only(item)

        with tempfile.TemporaryDirectory() as directory:
            target = save_fit_report(directory, result)
            disk = json.loads(target.read_text(encoding='utf-8'))
            assert_diagnostics_only(disk)
            self.assertTrue(disk['deployment_replay']['same_decisions'])
            self.assertEqual(disk['deployment_replay']['audit_copy']['diagnostics'], {'weighted_cosine': .81})
            self.assertEqual(disk['deployment_replay']['audit_copy']['nested'], [{'row_count': 10}])
            self.assertEqual(disk['student_audit'], result['student_audit'])
            self.assertEqual(disk['candidates'][1]['tuning'], result['candidates'][1]['tuning'])
            self.assertFalse((Path(directory) / 'fit_report.json.tmp').exists())
        self.assertEqual(result, before)
        self.assertIs(result['selected_patch'], selected_identity)


if __name__ == '__main__':
    unittest.main()
