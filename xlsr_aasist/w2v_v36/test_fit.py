"""Numerical and data-budget checks for low-storage frozen-classifier fitting."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .fit import (fit_candidates, margin_objective, predict, source_coefficients,
                  sources, split_sources)
from .metrics import evaluate, select_candidate


def train_rows(counts=(8, 8, 8, 8), missing_online=False):
    rows = []
    for (language, label), count in zip([('en', 0), ('en', 1), ('zh', 0), ('zh', 1)], counts):
        for index in range(count):
            source = f'{language}/{label}/{index}'
            for condition in ('offline', 'online', 'noisy_a', 'noisy_b'):
                if condition == 'online' and missing_online and index % 2 == 0:
                    continue
                rows.append(dict(source_id=source, group_id='sha:' + source, language=language,
                                 label=label, condition=condition, view='full'))
    return rows


def bundle(rows, weight, bias):
    # Perfectly informative frozen feature, plus a harmless language feature.
    x = np.asarray([[1. if row['label'] == 0 else -1., .2 if row['language'] == 'en' else -.2]
                    for row in rows], dtype=np.float32)
    return dict(x=x, logits=predict(x, weight, bias), rows=rows)


def dev_rows():
    rows = train_rows()
    rows = [r for r in rows if r['condition'] != 'offline']
    return [dict(r, condition={'noisy_a': 'seen', 'noisy_b': 'heldout'}.get(r['condition'], r['condition'])) for r in rows]


class ObjectiveTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference_with_extreme_logits(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(11, 3)).astype(np.float32)
        labels = rng.integers(0, 2, len(x))
        coefficients = rng.random(len(x)); coefficients /= coefficients.sum()
        anchor = np.array([40., -15., 6., -7.])
        delta = rng.normal(size=4)
        value, grad = margin_objective(delta, x, labels, coefficients, anchor, .1, 3)
        self.assertTrue(np.isfinite(value))
        numerical = []
        for column in range(4):
            offset = np.zeros(4); offset[column] = 1e-5
            left = margin_objective(delta - offset, x, labels, coefficients, anchor, .1, 5)[0]
            right = margin_objective(delta + offset, x, labels, coefficients, anchor, .1, 5)[0]
            numerical.append((right - left) / 2e-5)
        np.testing.assert_allclose(grad, numerical, rtol=1e-6, atol=1e-7)

    def test_each_arm_source_and_missing_online_budgets(self):
        rows = train_rows((3, 2, 1, 4), missing_online=True)
        records = sources(rows)
        for arm in ('class_balanced', 'group_balanced'):
            coefficients = source_coefficients(rows, arm)
            self.assertAlmostEqual(float(coefficients.sum()), 1.)
            group_totals = {}
            for source in records.values():
                indices = source['indices']
                total = sum(coefficients[i] for i in indices.values())
                ordinary = sum(coefficients[i] for condition, i in indices.items() if not condition.startswith('noisy'))
                self.assertAlmostEqual(ordinary, total * .5)
                self.assertAlmostEqual(coefficients[indices['noisy_a']], total * .25)
                self.assertAlmostEqual(coefficients[indices['noisy_b']], total * .25)
                if 'online' not in indices:
                    self.assertAlmostEqual(coefficients[indices['offline']], total * .5)
                group = source['language'] + '/' + str(source['label'])
                group_totals[group] = group_totals.get(group, 0.) + total
            if arm == 'group_balanced':
                for total in group_totals.values():
                    self.assertAlmostEqual(total, .25)
            else:
                self.assertAlmostEqual(group_totals['en/0'], .5 * 3 / 4)
                self.assertAlmostEqual(group_totals['zh/0'], .5 / 4)
                self.assertAlmostEqual(group_totals['en/1'] + group_totals['zh/1'], .5)

    def test_duplicate_condition_and_identity_conflict_rejected(self):
        rows = train_rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            source_coefficients(rows + [rows[0]], 'group_balanced')
        conflicting = [dict(r) for r in rows]
        for row in conflicting:
            if row['source_id'] == conflicting[0]['source_id']:
                row['group_id'] = conflicting[-1]['group_id']
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            split_sources(conflicting)

    def test_same_original_audio_and_all_views_stay_in_one_split(self):
        rows = train_rows()
        # A second identifier for the identical original recording must travel with it.
        alias = [dict(r, source_id='duplicate/' + r['source_id']) for r in rows if r['source_id'] == 'en/0/0']
        rows += alias
        left, right, audit = split_sources(rows, .2, 42)
        a = {rows[i]['group_id'] for i in left}
        b = {rows[i]['group_id'] for i in right}
        self.assertFalse(a & b)
        self.assertEqual(len(left) + len(right), len(rows))
        self.assertEqual(audit['group_overlap'], 0)
        again = split_sources(rows, .2, 42)
        np.testing.assert_array_equal(left, again[0])
        np.testing.assert_array_equal(right, again[1])


class MetricTests(unittest.TestCase):
    def test_auc_ties_and_fixed_fake_tie_decision(self):
        rows = dev_rows()
        result = evaluate(rows, np.zeros((len(rows), 2)))
        self.assertTrue(result['complete'])
        self.assertEqual(result['groups']['online']['recall'], [1., 0.])
        self.assertEqual(result['groups']['seen/en']['auc'], .5)
        self.assertAlmostEqual(result['weighted_f1'], 1 / 3)

    def test_absent_online_is_not_replaced_by_offline(self):
        rows = train_rows()
        rows = [r for r in rows if r['condition'] != 'online']
        result = evaluate(rows, np.zeros((len(rows), 2)), train_proxy=True)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['weighted_f1'])
        self.assertIsNone(result['clean_f1'])

    def test_noisy_and_en_fake_guardrails_override_weighted_gain(self):
        rows = dev_rows()
        labels = np.array([r['label'] for r in rows])
        z = np.stack([1. - labels, labels], axis=1)
        baseline = evaluate(rows, z)
        baseline['weighted_f1'] = .9
        worse = evaluate(rows, z)
        worse['noisy_f1'] = .99
        candidate = dict(name='worse', status='converged', metrics=worse)
        self.assertEqual(select_candidate(baseline, [candidate], {}), 'baseline')
        self.assertIn('noisy_below_baseline', candidate['guardrails'])
        fake_bad = evaluate(rows, z)
        fake_bad['groups']['seen/en']['recall'][0] = .99
        candidate = dict(name='fake_bad', status='converged', metrics=fake_bad)
        self.assertEqual(select_candidate(baseline, [candidate], {}), 'baseline')
        self.assertIn('seen/en_fake_recall_below_guardrail', candidate['guardrails'])


class FitTests(unittest.TestCase):
    def setUp(self):
        self.weight = np.array([[.25, .05], [-.25, .05]], dtype=np.float32)
        self.bias = np.array([1., -1.], dtype=np.float32)
        self.train = bundle(train_rows((12, 12, 12, 12)), self.weight, self.bias)
        self.dev = bundle(dev_rows(), self.weight, self.bias)
        self.cfg = dict(lambda_grid=[.001, .01], max_iterations=100, fit_chunk_rows=31)

    def fit(self, dev=None, cfg=None, out=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return fit_candidates(self.train, dev or self.dev, self.weight, self.bias,
                                  self.cfg if cfg is None else cfg, out)

    def test_correcting_artificial_bias_improves_and_preserves_shared_logit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.fit(out=directory)
            self.assertTrue((Path(directory) / 'fit_report.json').is_file())
            for tag, filename in result['dev_score_files'].items():
                with np.load(Path(directory) / filename, allow_pickle=False) as scores:
                    self.assertEqual(set(scores.files), {'ids', 'source_ids', 'language', 'condition', 'labels', 'logits'})
                    for key in scores.files:
                        self.assertNotEqual(scores[key].dtype.kind, 'O')
                    self.assertEqual(scores['logits'].dtype, np.dtype('float32'))
                    np.testing.assert_array_equal(scores['labels'], [r['label'] for r in self.dev['rows']])
                    if tag == 'baseline':
                        np.testing.assert_array_equal(scores['logits'], self.dev['logits'])
                    else:
                        candidate = next(c for c in result['candidates'] if c['tag'] == tag)
                        np.testing.assert_array_equal(scores['logits'], predict(self.dev['x'], **candidate['patch']))
        self.assertNotEqual(result['selected'], 'baseline')
        selected = next(c for c in result['candidates'] if c['name'] == result['selected'])
        self.assertEqual(selected['metrics']['weighted_f1'], 1.)
        self.assertGreater(selected['metrics']['weighted_f1'], result['baseline']['weighted_f1'])
        weight = np.array(result['selected_patch']['weight'])
        bias = np.array(result['selected_patch']['bias'])
        np.testing.assert_allclose(weight.mean(0), self.weight.mean(0), atol=1e-7)
        self.assertAlmostEqual(float(bias.mean()), float(self.bias.mean()), places=7)

    def test_dev_has_no_effect_on_parameters_or_train_lambda_selection(self):
        first = self.fit()
        flipped = dict(self.dev, rows=[dict(r, label=1-r['label']) for r in self.dev['rows']])
        second = self.fit(dev=flipped)
        for a, b in zip(first['candidates'], second['candidates']):
            self.assertEqual(a['selected_regularization'], b['selected_regularization'])
            self.assertEqual(a['patch'], b['patch'])
            self.assertEqual(a['tuning'], b['tuning'])
        self.assertEqual(second['selected'], 'baseline')
        np.testing.assert_array_equal(second['selected_patch']['weight'], self.weight)
        np.testing.assert_array_equal(second['selected_patch']['bias'], self.bias)

    def test_nonconvergence_is_rejected_and_falls_back_exactly(self):
        result = self.fit(cfg=dict(self.cfg, max_iterations=1))
        self.assertEqual(result['selected'], 'baseline')
        self.assertTrue(all(c['status'] == 'rejected' for c in result['candidates']))
        np.testing.assert_array_equal(result['selected_patch']['weight'], self.weight)
        np.testing.assert_array_equal(result['selected_patch']['bias'], self.bias)

    def test_corrupted_cache_or_wrong_original_classifier_cannot_fit(self):
        original = self.train['logits']
        self.train['logits'] = original + .1
        with self.assertRaisesRegex(ValueError, 'do not replay'):
            self.fit()
        self.train['logits'] = original
        self.train['x'] = self.train['x'].astype(np.float64)
        with self.assertRaisesRegex(ValueError, 'FP32'):
            self.fit()


if __name__ == '__main__':
    unittest.main()
