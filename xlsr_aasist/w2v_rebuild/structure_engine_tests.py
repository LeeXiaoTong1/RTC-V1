"""CPU integration: opt-in structure loss, old checkpoint and selection contracts.

Controlled Dev outcomes exercise promotion mechanics, not accuracy claims.
"""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from . import train as engine
from .control import AdaptationControl, NoisyAdaptationControl
from .core import load_checkpoint, sha256
from .group_metrics import export_language_report
from . import refinement_engine_tests as fixtures
from .refinement_engine_tests import FixtureBundle, build_fixture_detector, validation_report
from .selection import candidate_decision, selection_key


def report(clean, noisy):
    result = validation_report(*clean)
    other = validation_report(*noisy)
    for kind in ('seen', 'heldout'):
        result[kind] = other[kind]
    result['robust_f1'] = .3*result['online']['macro_f1'] + .7*other['robust_f1']
    result['robust_ce'] = .3*result['online']['balanced_ce'] + .7*other['robust_ce']
    return result


class StructureFixtureBundle(FixtureBundle):
    def __init__(self, args):
        super().__init__(args)
        for stream in self.train:
            for batch in stream:
                batch['features'] = batch['features'].repeat(1, 2, 1)
                batch['mask'] = batch['mask'].repeat(1, 2)
                size = len(batch['labels'])
                batch['languages'] = (torch.arange(size)//2) % 2
                batch['language_weights'] = torch.ones(size)
                batch['source_ids'] = ['offline/en/fake/'+str(i) for i in range(size)]
                if 'pairs' in batch:
                    # Identical source content with a small, time-varying distortion.
                    n = batch['pairs']
                    batch['features'][n:] = batch['features'][:n]*.97
                    batch['features'][n:, ::3, :10] += .02


class StructureEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_noisy_promotion_requires_clean_and_weighted_protection_not_each_recall(self):
        baseline = report((90, 90), (90, 90))
        improved = report((90, 90), (98, 89))
        accepted, reason = candidate_decision(improved, selection_key(baseline, True), baseline, 'noisy')
        self.assertTrue(accepted, reason)
        self.assertIn('seen_real_recall_decreased', reason['warnings'])
        self.assertFalse(candidate_decision(improved, selection_key(baseline), baseline)[0])
        clean_regression = report((89, 89), (99, 99))
        accepted, reason = candidate_decision(clean_regression, selection_key(baseline, True), baseline, 'noisy')
        self.assertFalse(accepted)
        self.assertIn('online_f1_below_baseline', reason['reasons'])
        self.assertFalse(candidate_decision(baseline, selection_key(baseline, True), baseline, 'noisy')[0])

    def test_candidate_keeps_strongest_noisy_with_serializable_progress(self):
        baseline = report((90, 90), (90, 90))
        first = report((99, 99), (92, 92))
        second = report((90, 90), (93, 93))
        self.assertLess(second['robust_f1'], first['robust_f1'])
        control = NoisyAdaptationControl(baseline)
        self.assertTrue(control.observe(first, 1, False)[0])
        self.assertTrue(control.observe(second, 2, False)[0])
        restored = NoisyAdaptationControl(baseline)
        restored.load_state_dict(json.loads(json.dumps(control.state_dict())))
        self.assertEqual(restored.state_dict(), control.state_dict())
        legacy = AdaptationControl(baseline)
        legacy.observe(first, 1, False)
        self.assertFalse(legacy.observe(second, 2, False)[0])

    def test_real_engine_preserves_checkpoint_architecture_freezing_and_reports(self):
        helper = fixtures.RefinementEngineTests()
        baseline_dev, improved = report((90, 90), (90, 90)), report((90, 90), (98, 89))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.manual_seed(541)
            baseline, original = helper.make_baseline(root)
            digest = sha256(baseline)
            out = root/'structure'/'stage3'
            arguments = helper.arguments(out, 1) + ['--finetune_from', str(baseline),
                '--adaptation_control', '--noisy_selection', '--local_structure_weight', '.02',
                '--local_structure_warmup_steps', '1']

            def run(argv, reports):
                with patch.object(engine, 'DataBundle', StructureFixtureBundle), \
                     patch.object(engine.Detector, 'load', side_effect=build_fixture_detector), \
                     patch.object(engine, 'source_hashes', return_value={}), \
                     patch.object(engine, 'validate', side_effect=copy.deepcopy(reports)), \
                     patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                    engine.main()

            run(arguments, [baseline_dev, improved])
            self.assertEqual(sha256(baseline), digest)
            best, last = (load_checkpoint(out/name) for name in ('best_model.pt', 'last.pt'))
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(set(best['model']), set(original))
            self.assertEqual(last['best_key'], selection_key(improved, True))
            self.assertTrue(last['config']['local_structure_config'])
            self.assertEqual(last['config']['local_structure_warmup_steps'], 1)
            self.assertTrue((out/'candidate_best.pt').is_file())
            for name, value in original.items():
                if name.startswith('backbone.feature_projection.') or (
                    name.startswith('backbone.encoder.layers.') and int(name.split('.')[3]) < 20):
                    torch.testing.assert_close(last['model'][name], value, rtol=0, atol=0)
            for prefix in ['head.'] + [f'backbone.encoder.layers.{i}.' for i in range(20, 24)]:
                self.assertTrue(any(not torch.equal(last['model'][k], v)
                                    for k, v in original.items() if k.startswith(prefix)))
            row = json.loads((out/'metrics.jsonl').read_text())
            self.assertEqual(row['last_batch']['noisy_weight'], 0.)
            self.assertEqual(row['last_batch']['rtc_weight'], .1)
            self.assertAlmostEqual(row['mean_batch']['structure_weight'], .02, places=6)
            self.assertIn('structure_en_real_pairs', row['mean_batch'])
            self.assertTrue(row['selection']['accepted'])
            for value in row['mean_batch'].values():
                self.assertIsInstance(value, (float, int))
            exported = export_language_report(out, download_dir=root/'download')
            with zipfile.ZipFile(exported['download_archive']) as package:
                summary = json.loads(package.read('summary.json'))
                self.assertEqual(summary['status'], 'complete')
                self.assertEqual(len(summary['structure_training']), 1)
                self.assertIn('Local-structure robustness report', package.read('report.md').decode())
                self.assertFalse(any(p.endswith(('.pt', '.wav', '.npy')) for p in package.namelist()))
            resumed = arguments.copy()
            index = resumed.index('--finetune_from')
            resumed[index:index+2] = ['--resume', str(out/'last.pt')]
            before = {name: sha256(out/name) for name in ('last.pt', 'best_model.pt', 'metrics.jsonl')}
            run(resumed, [])
            self.assertEqual(before, {name: sha256(out/name) for name in before})
            with self.assertRaisesRegex(ValueError, 'local_structure_weight'):
                run(resumed + ['--local_structure_weight', '.01'], [])
            self.assertEqual(sha256(baseline), digest)


if __name__ == '__main__':
    unittest.main()
