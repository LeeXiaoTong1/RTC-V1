"""Completed checkpoint replay must identify its weights and preserve the run."""
from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from w2v_v39.common import atomic_json, digest, read_json
from . import train as training, validate as validation
from .state import load_selected
from .test_workflow import WorkflowFixture


class ValidationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def fixture(folder, fallback=False):
        f = WorkflowFixture(folder)
        f.cfg.update(source_fingerprints={}, data_fingerprints={})
        run = f.run_dir('complete')
        with f.patched():
            if fallback:
                with patch.object(training, 'acceptance', return_value=(False, ['fixture_guard'])):
                    training.run_experiment(f.cfg, run)
            else:
                training.run_experiment(f.cfg, run)
        return f, run

    @staticmethod
    @contextmanager
    def canonical(f):
        @contextmanager
        def bundles(cfg):
            yield dict(rows=[]), dict(rows=f.dev)
        with patch.object(validation, 'bundles', bundles), \
             patch.object(validation, 'verify_parent', return_value=None), \
             redirect_stdout(io.StringIO()):
            yield

    @staticmethod
    def logits(rows):
        z = np.zeros((len(rows), 2), dtype=np.float32)
        z[np.arange(len(rows)), [r['label'] for r in rows]] = 2.
        # One EN-real error keeps the test sensitive to labels and condition routing.
        index = next(i for i, r in enumerate(rows) if r['condition'] == 'seen'
                     and r['language'] == 'en' and r['label'] == 1)
        z[index] = (2., 0.)
        return z

    def test_all_selectors_replay_exact_weights_and_leave_run_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            f, run = self.fixture(folder)
            before = {p.name: digest(p) for p in run.iterdir() if p.is_file()}
            for kind in ('best_weighted', 'best_guarded', 'last'):
                with self.subTest(kind=kind):
                    selected, done = load_selected(run, kind)

                    def infer(model, rows, cfg, label):
                        self.assertFalse(model.training)
                        self.assertFalse(any(p.requires_grad for p in model.parameters()))
                        self.assertFalse(torch.is_grad_enabled())
                        for name, value in selected['candidate']['state'].items():
                            torch.testing.assert_close(dict(model.named_parameters())[name], value, atol=0, rtol=0)
                        self.assertEqual(rows, f.dev)
                        return self.logits(rows), None

                    out = Path(folder)/kind
                    with self.canonical(f), patch.object(validation, 'load_model', f.load_model), \
                         patch.object(validation, 'load_reference', side_effect=AssertionError('Unexpected fallback')), \
                         patch.object(validation, 'infer', infer):
                        metrics = validation.validate(run, out, 'cpu', 0, kind)
                    meta = read_json(out/'validation_meta.json')
                    self.assertEqual(meta['selected'], done['selected'])
                    self.assertEqual(meta['checkpoint_kind'], kind)
                    self.assertFalse(meta['baseline_fallback'])
                    self.assertFalse(meta['selection_updated'])
                    self.assertEqual(metrics, read_json(out/'metrics.json'))
                    self.assertEqual(metrics['groups']['seen/en']['recall'][1], 0.)
                    self.assertAlmostEqual(metrics['weighted_f1'], .3*metrics['clean_f1']+.7*metrics['noisy_f1'])
                    for name in ('ap', 'auc', 'eer', 'macro_f1', 'recall'):
                        self.assertIn(name, metrics['groups']['online/en'])
                    with np.load(out/'dev_scores.npz', allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['logits'], self.logits(f.dev))
                        self.assertEqual(saved['ids'].tolist(), [r['id'] for r in f.dev])
                    self.assertLess(sum(p.stat().st_size for p in out.iterdir()), 100_000)
            after = {p.name: digest(p) for p in run.iterdir() if p.is_file()}
            self.assertEqual(before, after)

    def test_guarded_fallback_loads_and_names_historical_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            f, run = self.fixture(folder, fallback=True)
            out = Path(folder)/'fallback'
            with self.canonical(f), patch.object(validation, 'load_reference', f.load_model), \
                 patch.object(validation, 'load_model', side_effect=AssertionError('Do not load fresh weights')), \
                 patch.object(validation, 'infer', return_value=(self.logits(f.dev), None)):
                validation.validate(run, out, 'cpu', 0)
            meta = read_json(out/'validation_meta.json')
            self.assertEqual(meta['selected'], 'starting_parent')
            self.assertTrue(meta['baseline_fallback'])
            self.assertEqual(meta['actual_model'], 'historical_parent')
            self.assertEqual(meta['parent_selected_tag'], f.cfg['parent_selected_tag'])

    def test_refuses_changed_dev_labels_before_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            f, run = self.fixture(folder)
            changed = read_json(run/'dev_rows.json')
            changed[0]['label'] = 1 - changed[0]['label']
            atomic_json(run/'dev_rows.json', changed)
            with self.canonical(f), patch.object(validation, 'load_model') as loader:
                with self.assertRaisesRegex(ValueError, 'Fixed Dev row identity'):
                    validation.validate(run, Path(folder)/'out', 'cpu', 0)
                loader.assert_not_called()

    def test_refuses_input_change_during_replay_before_publishing_results(self):
        with tempfile.TemporaryDirectory() as folder:
            f, run = self.fixture(folder)
            out = Path(folder)/'out'

            def infer(*args):
                atomic_json(run/'completed.json', dict(read_json(run/'completed.json'), modified=True))
                return self.logits(f.dev), None

            with self.canonical(f), patch.object(validation, 'load_model', f.load_model), \
                 patch.object(validation, 'infer', infer):
                with self.assertRaisesRegex(ValueError, 'Pinned input changed'):
                    validation.validate(run, out, 'cpu', 0)
            self.assertFalse(out.exists())

    def test_refuses_unfinished_run_and_writing_inside_run_or_existing_results(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)/'unfinished'
            run.mkdir()
            with self.assertRaisesRegex(ValueError, 'completed V3.16'):
                validation.validate(run, Path(folder)/'out', 'cpu', 0)
            with self.assertRaisesRegex(ValueError, 'outside the training run'):
                validation.validate(run, run/'out', 'cpu', 0)
            out = Path(folder)/'existing'
            out.mkdir()
            (out/'keep.txt').write_text('keep', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                validation.validate(run, out, 'cpu', 0)
            self.assertEqual((out/'keep.txt').read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()
