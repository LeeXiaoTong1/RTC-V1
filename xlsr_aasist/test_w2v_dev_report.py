"""Tests of paired error accounting, noisy aggregation and grouped uncertainty."""
import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_w2v_report import (
    _bootstrap_samples, _validate, auc_fake, build_report, metrics,
)


def score(p):
    z = math.log(p/(1-p))
    return {'logit_fake': z, 'logit_real': 0., 'pfake': p, 'margin': z}


def fixtures():
    records, predictions = [], []
    for condition in ('online', 'offline', 'seen', 'heldout'):
        for band in ((-1,) if condition in ('online', 'offline') else range(4)):
            for source, label in (('f0', 0), ('f1', 0), ('f2', 0), ('r0', 1)):
                records.append({'condition': condition, 'source_id': source, 'band': band,
                                'label': label, 'audio_path': 'group/%s.wav' % source,
                                'snr_db': '' if band == -1 else 7.+5*band,
                                'processing_family': '' if band == -1 else 'ffmpeg',
                                'source_directory': 'group', 'processing_json': '{}',
                                'rtc_json': '{}', 'mix_id': '' if band == -1 else 'mix0'})
                predictions.append(score(.8 if label == 0 else .2))
    return records, predictions


class DevAuditReportTests(unittest.TestCase):
    def test_metrics_match_training_metric_accumulator(self):
        import torch
        from w2v_rebuild.core import Metrics
        z = torch.tensor(np.random.default_rng(17).normal(size=(120, 2)), dtype=torch.float32)
        y = torch.tensor([0]*90+[1]*30)
        accumulated = Metrics()
        accumulated.update(z[:47], y[:47])
        accumulated.update(z[47:], y[47:])
        expected = accumulated.result()
        p = z.softmax(1)[:, 0]
        scores = [{'logit_fake': float(a), 'logit_real': float(b), 'pfake': float(prob),
                   'margin': float(a-b)} for (a, b), prob in zip(z, p)]
        actual = metrics(y.tolist(), scores)
        self.assertEqual(actual['confusion'], expected['confusion'])
        self.assertEqual(actual['recall'], expected['recall'])
        self.assertEqual(actual['macro_f1'], expected['macro_f1'])
        self.assertAlmostEqual(actual['balanced_ce'], expected['balanced_ce'], places=7)

    def test_fixed_tie_fake_stable_ce_and_auc_ties(self):
        result = metrics([0, 1], [score(.5), score(.5)])
        self.assertEqual(result['confusion'], [[1, 0], [1, 0]])
        self.assertAlmostEqual(result['macro_f1'], 1/3)
        self.assertAlmostEqual(result['balanced_ce'], math.log(2))
        self.assertEqual(result['auc_fake'], .5)
        self.assertEqual(auc_fake([0, 0, 1, 1], [.9, .5, .5, .1]), .875)
        self.assertIsNone(auc_fake([1, 1], [.1, .9]))
        extreme = [dict(logit_fake=1000., logit_real=-1000., pfake=1., margin=2000.),
                   dict(logit_fake=-1000., logit_real=1000., pfake=0., margin=-2000.)]
        self.assertEqual(metrics([1, 0], extreme)['balanced_ce'], 2000.)

    def test_auc_uses_unsaturated_margin_while_predictions_keep_stored_probability(self):
        scores = [dict(logit_fake=30., logit_real=0., pfake=1., margin=30.),
                  dict(logit_fake=20., logit_real=0., pfake=1., margin=20.)]
        result = metrics([0, 1], scores)
        self.assertEqual(result['confusion'], [[1, 0], [1, 0]])
        self.assertEqual(result['auc_fake'], 1.)
        self.assertEqual(auc_fake([0, 1], [1., 1.]), .5)

    def test_identical_models_saved_outputs_have_zero_paired_deltas(self):
        records, predictions = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            result = build_report(records, predictions, predictions, tmp, bootstrap=12)
            self.assertEqual(result['delta']['robust_f1'], 0.)
            self.assertEqual(result['delta']['robust_ce'], 0.)
            self.assertIsNone(result['bootstrap']['robust_interval'])
            for intervals in result['bootstrap']['intervals'].values():
                for interval in intervals.values():
                    self.assertEqual(interval, {'lower': 0., 'upper': 0.})
            expected = {'report.md', 'summary.json', 'predictions.csv', 'changed_errors.csv',
                        'persistent_errors.csv', 'groups.csv'}
            self.assertEqual({p.name for p in Path(tmp).iterdir()}, expected)
            self.assertEqual(json.loads((Path(tmp)/'summary.json').read_text(encoding='utf-8')), result)
            self.assertTrue((Path(tmp)/'predictions.csv').read_bytes().startswith(b'\xef\xbb\xbf'))
            with (Path(tmp)/'changed_errors.csv').open(encoding='utf-8-sig', newline='') as f:
                self.assertEqual(list(csv.DictReader(f)), [])

    def test_error_transitions_and_formula_safe_strings(self):
        records, old = fixtures()
        new = copy.deepcopy(old)
        # f0 rescued, f1 newly wrong at the boundary, r0 persistently very wrong.
        old[0], new[0] = score(.2), score(.8)
        new[1] = score(.49)
        old[3], new[3] = score(.95), score(.96)
        records[0]['audio_path'] = '=HYPERLINK("bad")'
        records[0]['source_directory'] = '  +formula'
        records[0].update(noise_json='{"offset": 11}', source_samples=80000,
                          output_samples_before_crop=64640, source_sha256='1234',
                          audio_size=256, audio_mtime_ns=1759999999999999999)
        with tempfile.TemporaryDirectory() as tmp:
            result = build_report(records, old, new, tmp, bootstrap=0)
            fake, real = (result['transitions']['online'][k] for k in ('fake', 'real'))
            self.assertEqual(fake['rescued'], 1)
            self.assertEqual(fake['new_errors'], 1)
            self.assertEqual(fake['new_errors_candidate_near_boundary'], 1)
            self.assertEqual(real['persistent_errors'], 1)
            self.assertEqual(real['candidate_high_confidence_wrong'], 1)
            with (Path(tmp)/'changed_errors.csv').open(encoding='utf-8-sig', newline='') as f:
                changed = list(csv.DictReader(f))
            self.assertEqual(len(changed), 2)
            self.assertEqual(changed[0]['audio_path'][0], "'")
            self.assertEqual(changed[0]['source_directory'][0], "'")
            self.assertEqual(changed[1]['candidate_prediction'], '1')
            self.assertEqual(changed[0]['noise_json'], '{"offset": 11}')
            self.assertEqual(changed[0]['source_samples'], '80000')
            self.assertEqual(changed[0]['output_samples_before_crop'], '64640')
            self.assertEqual(changed[0]['source_sha256'], '1234')
            self.assertEqual(changed[0]['audio_size'], '256')
            self.assertEqual(changed[0]['audio_mtime_ns'], '1759999999999999999')
            self.assertEqual(changed[1]['noise_json'], '')

    def test_four_band_mean_not_pooled_f1_and_input_order_invariant(self):
        records, old = fixtures()
        new = copy.deepcopy(old)
        for i, r in enumerate(records):
            if r['condition'] == 'seen' and r['band'] in (0, 1):
                new[i] = score(.8 if r['band'] == 0 else .2)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = build_report(records, old, new, first, bootstrap=8)
            order = np.random.default_rng(13).permutation(len(records))
            b = build_report([records[i] for i in order], [old[i] for i in order],
                             [new[i] for i in order], second, bootstrap=8)
            self.assertEqual(a, b)
            for name in ('predictions.csv', 'groups.csv'):
                self.assertEqual((Path(first)/name).read_bytes(), (Path(second)/name).read_bytes())
            ix = [i for i, r in enumerate(records) if r['condition'] == 'seen']
            pooled = metrics([records[i]['label'] for i in ix], [new[i] for i in ix])['macro_f1']
            averaged = a['candidate']['seen']['macro_f1']
            self.assertAlmostEqual(averaged, (.42857142857142855+.2+1+1)/4)
            self.assertNotAlmostEqual(pooled, averaged)
            self.assertAlmostEqual(a['candidate']['robust_f1'], .3+.35*averaged+.35)

    def test_rejects_misalignment_missing_bands_labels_and_nonfinite(self):
        records, old = fixtures()
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            _validate(records, old[:-1], old)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            _validate(records+[records[0]], old+[old[0]], old+[old[0]])
        bad = copy.deepcopy(records)
        bad[-1]['label'] = 0
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            _validate(bad, old, old)
        ix = [i for i, r in enumerate(records) if (r['condition'], r['band']) != ('seen', 3)]
        with self.assertRaisesRegex(ValueError, 'all four'):
            _validate([records[i] for i in ix], [old[i] for i in ix], [old[i] for i in ix])
        bad = copy.deepcopy(old)
        bad[-1]['logit_fake'] = float('inf')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            _validate(records, old, bad)
        bad = copy.deepcopy(old)
        bad[-1]['pfake'] = .7
        with self.assertRaisesRegex(ValueError, 'Probability'):
            _validate(records, old, bad)
        bad = copy.deepcopy(records)
        bad[-1]['source_id'] = 'different-real-source'
        with self.assertRaisesRegex(ValueError, 'identical source'):
            _validate(bad, old, old)

    def test_bootstrap_keeps_eight_views_of_each_source_together(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        old = np.repeat(y[:, None], 8, axis=1)
        new = old.copy()
        new[[0, 4]] = 1-new[[0, 4]]
        zeros = np.zeros(old.shape)
        samples = _bootstrap_samples(y, old, new, zeros, zeros, 40, np.random.default_rng(3))
        self.assertGreater(np.std(samples['macro_f1'][:, 0]), 0)
        for metric, values in samples.items():
            for band in range(1, 8):
                np.testing.assert_array_equal(values[:, 0], values[:, band], err_msg=metric)
        # Every draw has the original number of real examples: recall steps are 1/4.
        np.testing.assert_allclose(samples['real_recall']*4, np.round(samples['real_recall']*4))


if __name__ == '__main__':
    unittest.main()
