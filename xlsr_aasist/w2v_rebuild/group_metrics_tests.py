"""CPU diagnostics: aggregation, raw-margin AUC and safe download packaging."""
import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch
from torch.nn import functional as F

from .group_metrics import GroupMetrics, DevGroupRecorder, auc_fake, export_language_report, tradeoff_metrics, comparable_reference


def dev_config_fixture():
    config = {'stage': 3, 'dev_protocol': '/data/dev.txt', 'dev_data_path': '/data/wav/dev',
              'dev_noisy_cache': '/cache/seen', 'dev_heldout_cache': '/cache/heldout',
              'ssl_path': '/pretrained', 'eval_microbatch': 4, 'eval_batch': 16}
    config['data_fingerprints'] = {name: 'a'*64 for name in ('/data/dev.txt', '/cache/seen/config.json',
        '/cache/seen/manifest.jsonl', '/cache/heldout/config.json', '/cache/heldout/manifest.jsonl',
        '/pretrained/preprocessor_config.json')}
    return config


def reference_fixture(dev):
    config = dev_config_fixture()
    return {'source_run': 'prior_run', 'epoch': 1, 'metrics': {'dev': dev}, 'available': True,
            'dev_inputs': {key: config[key] for key in ('dev_protocol', 'dev_data_path', 'dev_noisy_cache',
                                                       'dev_heldout_cache', 'ssl_path')},
            'data_fingerprints': config['data_fingerprints'], 'eval_microbatch': 4, 'eval_batch': 16}


class GroupDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_chunked_groups_match_direct_unweighted_metrics(self):
        z = torch.tensor([[2., -1.], [-1., 1.], [2., 0.], [0., 2.], [0., 0.], [-2., 1.]])
        y = torch.tensor([0, 1, 1, 0, 1, 1])
        language = torch.tensor([0, 0, 0, 1, 1, 1])
        meter = GroupMetrics()
        # Training update must not perform even an implicit device-to-host copy.
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('update must stay on device')):
            meter.update(z[:3], y[:3], language[:3])
            meter.update(z[3:], y[3:], language[3:])
        result = meter.result()
        for lang, name in enumerate(('en', 'zh')):
            for label, cls_name in enumerate(('fake', 'real')):
                mask = (language == lang) & (y == label)
                m = result[name+'-'+cls_name]
                self.assertEqual(m['count'], int(mask.sum()))
                self.assertAlmostEqual(m['mean_ce'], float(F.cross_entropy(z[mask], y[mask])))
                self.assertAlmostEqual(m['mean_fake_score'], float(z[mask].softmax(1)[:, 0].mean()))
                self.assertAlmostEqual(m['recall'], float(((z[mask].softmax(1)[:, 0] < .5).long() == y[mask]).float().mean()))
        self.assertEqual(result['zh-real']['recall'], .5)  # At p_fake=.5, the decision is fake.

    def test_missing_group_has_zero_count_and_null_metrics(self):
        meter = GroupMetrics()
        meter.update(torch.empty((0, 2)), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        for values in meter.result().values():
            self.assertEqual(values, {'count': 0, 'recall': None, 'mean_ce': None, 'mean_fake_score': None})

    def test_auc_ties_single_class_and_saturated_probabilities(self):
        self.assertEqual(auc_fake([0, 0, 1, 1], [2., 1., 1., 0.]), .875)
        self.assertEqual(auc_fake([1, 0, 1, 0], [1., 1., 1., 1.]), .5)
        self.assertEqual(auc_fake([0, 1], [100., 90.]), 1.)
        self.assertEqual(auc_fake([0, 1], [-100., -90.]), 0.)
        self.assertIsNone(auc_fake([0], [1.]))
        self.assertIsNone(auc_fake([], []))
        with self.assertRaises(ValueError):
            auc_fake([0, 1], [float('nan'), 0.])

    def test_recorder_preserves_predictions_and_noisy_band_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scores.jsonl'
            z = torch.tensor([[10., 0.], [0., 10.], [1., 0.], [0., 1.]])
            y = torch.tensor([0, 1, 0, 1])
            ids = ['offline/en/fake/a.wav', 'offline/en/real/b.wav',
                   'offline/zh/fake/c.wav', 'offline/zh/real/d.wav']
            with DevGroupRecorder(path) as recorder:
                recorder.update('seen', z, y, ids, [0, 0, 1, 1])
                recorder.update('online', z, y, ids)
                result = recorder.result()
                self.assertFalse(path.exists())
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[0], {'condition': 'seen', 'source_id': ids[0], 'label': 0,
                                       'language': 'en', 'p_fake': float(z.softmax(1)[0, 0]), 'margin': 10., 'band': 0})
            self.assertEqual(result['conditions']['online']['auc_by_language'], {'en': 1., 'zh': 1.})
            self.assertEqual(result['noisy_bands']['seen']['0']['en-real']['count'], 1)
            self.assertEqual(result['noisy_bands']['seen']['0']['zh-real']['count'], 0)
            self.assertEqual(result['conditions']['offline']['groups']['en-real']['count'], 0)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_recording_is_optional_and_invalid_ids_fail_closed(self):
        z, y = torch.zeros((1, 2)), torch.tensor([0])
        with DevGroupRecorder() as recorder:
            recorder.update('offline', z, y, ['offline/en/fake/a.wav'])
        self.assertEqual(recorder.result()['conditions']['offline']['groups']['en-fake']['count'], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scores.jsonl'
            with self.assertRaises(ValueError):
                with DevGroupRecorder(path) as recorder:
                    recorder.update('online', z, y, ['online/unknown/a.wav'])
            self.assertEqual(list(Path(directory).iterdir()), [])
            path.write_text('preserve me', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                with DevGroupRecorder(path):
                    pass
            self.assertEqual(path.read_text(encoding='utf-8'), 'preserve me')

    def test_export_contains_only_diagnostics_and_copies_download(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/'new_run'
            stage = run/'stage3'
            stage.mkdir(parents=True)
            dev = {'online': {'macro_f1': .96, 'recall': [.99, .92]},
                   'seen': {'macro_f1': .94, 'bands': [{'recall': [.98, .9]}]*4},
                   'heldout': {'macro_f1': .95, 'bands': [{'recall': [.96, .94]}]*4},
                   'robust_f1': .9495, 'robust_ce': .4}
            inputs = {'baseline_dev.json': dev, 'config.json': dev_config_fixture(),
                      'completed.json': {'status': 'no_eligible_improvement'},
                      'epoch_001_evaluation.json': {'epoch': 1, 'dev': dev, 'train_groups': {'all': GroupMetrics().result()}},
                      'language_budget.json': {'en_real': .35, 'en_fake': .4},
                      'launcher_status.json': {'training_complete': True, 'returncode': 0}}
            for name, value in inputs.items():
                (stage/name).write_text(json.dumps(value), encoding='utf-8')
            (stage/'baseline_scores.jsonl').write_text('{}\n', encoding='utf-8')
            (stage/'epoch_001_scores.jsonl').write_text('{}\n', encoding='utf-8')
            (stage/'best_model.pt').write_bytes(b'never package weights')
            (stage/'sample.wav').write_bytes(b'never package audio')
            (stage/'arbitrary.json').write_text('private unrelated file', encoding='utf-8')
            (run/'en_plan.json').write_text('{}', encoding='utf-8')
            reference = reference_fixture(dev)
            result = export_language_report(stage, reference, Path(directory)/'download')
            summary = json.loads(Path(result['summary']).read_text(encoding='utf-8'))
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['reference']['metrics'], reference['metrics'])
            self.assertTrue(summary['reference']['comparability']['verified'])
            self.assertEqual(len(summary['reference']['comparability']['checked_dev_fingerprints']), 6)
            self.assertEqual(len(summary['tradeoff_comparison']), 3)
            for row in summary['tradeoff_comparison']:
                self.assertAlmostEqual(row['online_real_recall'], .92)
                self.assertAlmostEqual(row['noisy_fake_recall'], .97)
                self.assertAlmostEqual(row['noisy_real_recall'], .92)
                self.assertAlmostEqual(row['noisy_f1'], .945)
            with zipfile.ZipFile(result['archive']) as package:
                names = package.namelist()
                self.assertIn('stage3/epoch_001_scores.jsonl', names)
                self.assertIn('en_plan.json', names)
                self.assertNotIn('stage3/best_model.pt', names)
                self.assertNotIn('stage3/sample.wav', names)
                self.assertNotIn('stage3/arbitrary.json', names)
                self.assertIsNone(package.testzip())
            self.assertEqual(Path(result['archive']).read_bytes(), Path(result['download_archive']).read_bytes())
            text = Path(result['report']).read_text(encoding='utf-8')
            self.assertIn('Prior reference E1', text)
            self.assertIn('not clean independent sources', text)
            self.assertIn('Mean noisy fake recall', text)
            self.assertEqual((stage/'best_model.pt').read_bytes(), b'never package weights')

    def test_reference_requires_matching_current_dev_hashes_and_shapes(self):
        reference = reference_fixture({'robust_f1': .9})
        original = copy.deepcopy(reference)
        config = dev_config_fixture()
        config['data_fingerprints']['/unrelated/train.txt'] = 'b'*64
        self.assertTrue(comparable_reference(reference, config)['available'])
        cases = []
        changed = copy.deepcopy(config)
        changed['data_fingerprints']['/cache/seen/manifest.jsonl'] = 'b'*64
        cases.append(changed)
        missing = copy.deepcopy(config)
        del missing['data_fingerprints']['/data/dev.txt']
        cases.append(missing)
        wrong_shape = copy.deepcopy(config)
        wrong_shape['eval_microbatch'] = 8
        cases.append(wrong_shape)
        for changed in cases:
            with self.subTest(config=changed):
                checked = comparable_reference(reference, changed)
                self.assertFalse(checked['available'])
                self.assertTrue(checked['reason'])
                self.assertEqual(checked['metrics'], reference['metrics'])
        self.assertEqual(reference, original)
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)/'stage3'
            stage.mkdir()
            (stage/'config.json').write_text(json.dumps(cases[0]))
            result = export_language_report(stage, reference)
            summary = json.loads(Path(result['summary']).read_text(encoding='utf-8'))
            self.assertFalse(summary['reference']['available'])
            self.assertEqual(summary['tradeoff_comparison'], [])
            report = Path(result['report']).read_text(encoding='utf-8')
            self.assertIn('excluded from comparable-model tables', report)
            self.assertIn('dev_noisy_cache/manifest.jsonl', report)

    def test_tradeoffs_require_full_band_coverage(self):
        partial = {'online': {'recall': [.99, .9]},
                   'seen': {'macro_f1': .9, 'bands': [{'recall': [.8, .7]}]*3},
                   'heldout': {'macro_f1': .8, 'bands': [{'recall': [.6, .5]}]*4}}
        result = tradeoff_metrics(partial)
        self.assertEqual(result['online_real_recall'], .9)
        self.assertIsNone(result['noisy_fake_recall'])
        self.assertIsNone(result['noisy_real_recall'])
        self.assertAlmostEqual(result['noisy_f1'], .85)

    def test_failed_or_partial_export_is_not_labelled_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)/'stage3'
            stage.mkdir()
            result = export_language_report(stage)
            self.assertEqual(json.loads(Path(result['summary']).read_text())['status'], 'incomplete')
            (stage/'completed.json').write_text('{}')
            (stage/'launcher_status.json').write_text(json.dumps({'failed': True, 'returncode': 1}))
            result = export_language_report(stage)
            self.assertEqual(json.loads(Path(result['summary']).read_text())['status'], 'failed')
            self.assertIn('partial/failed-run', Path(result['report']).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
