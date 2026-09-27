"""Exercise guarded adaptation and checkpoint recovery with a small CPU model.

Validation reports are controlled to test model selection, not model quality.
The real engine still performs forward/backward passes, optimizer updates,
gradient audits, checkpoint serialization, early stopping, and resume checks.
No pretrained downloads, speech files, or GPU are used.
"""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import SCHEMA
from . import train as engine
from .core import Metrics, atomic_save, load_checkpoint, sha256
from .model import Detector
from .tests import ToyEncoder


class FixtureBundle:
    def __init__(self, args):
        self.steps, self.completed = 4, 0
        self.weights = torch.tensor([.6, 2.5])
        self.counts = torch.tensor([80, 20])
        self.base_fingerprints = {'fixture': 'unchanged'}
        self.fingerprints = {**self.base_fingerprints, 'cache': 'fixed'}
        generator = torch.Generator().manual_seed(222)

        def batch(size, paired=False):
            item = {'features': torch.randn(size, 24, 160, generator=generator),
                    'mask': torch.ones(size, 24, dtype=torch.long),
                    'labels': torch.arange(size) % 2}
            if paired:
                item['pairs'] = size // 2
            else:
                item['ids'] = ['online/' + str(i) for i in range(size)]
            return item

        self.train = [[batch(24) for _ in range(self.steps)],
                      [batch(8, True) for _ in range(self.steps)],
                      [batch(8, True) for _ in range(self.steps)]]

    def begin(self, epoch):
        self.epoch = epoch

    def end(self, steps):
        if steps != self.steps:
            raise AssertionError('Fixture epoch was not fully consumed')
        self.completed += 1

    def sampler_state(self):
        return {'completed': self.completed}

    def load_sampler_state(self, state):
        self.completed = state['completed']


def build_fixture_detector(*args, **kwargs):
    model = Detector(ToyEncoder(24, checkpointing=False))
    model.backbone.config.to_dict = lambda: {'hidden_size': 16, 'num_hidden_layers': 24}
    return model


def validation_report(fake_correct=90, real_correct=90):
    """Use actual metric accumulation to construct internally consistent reports."""
    labels = torch.tensor([0] * 100 + [1] * 100)
    predictions = torch.tensor([0] * fake_correct + [1] * (100 - fake_correct)
                               + [1] * real_correct + [0] * (100 - real_correct))
    logits = torch.zeros(200, 2)
    logits[torch.arange(200), predictions] = 2.
    metrics = Metrics()
    metrics.update(logits, labels)
    result = metrics.result()
    report = {kind: copy.deepcopy(result) for kind in ('all', 'online', 'offline')}
    for kind in ('seen', 'heldout'):
        report[kind] = {'macro_f1': result['macro_f1'], 'balanced_ce': result['balanced_ce'],
                        'bands': [copy.deepcopy(result) for _ in range(4)]}
    report.update(robust_f1=result['macro_f1'], robust_ce=result['balanced_ce'])
    return report


class RefinementEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(234)

    def make_baseline(self, root):
        path = root / 'original' / 'best_model.pt'
        path.parent.mkdir()
        model = build_fixture_detector()
        state = {'schema': SCHEMA, 'kind': 'weights', 'stage': 3, 'epoch': 12,
                 'model': model.state_dict(), 'model_config': model.backbone.config.to_dict(),
                 'data_fingerprints': {'fixture': 'unchanged', 'cache': 'old'},
                 'config': {}, 'source_hashes': {}, 'dev': validation_report()}
        atomic_save(state, path)
        return path, state['model']

    def arguments(self, target, epochs):
        args = ['test', '--device', 'cpu', '--amp', 'none', '--microbatch', '4',
                '--num_workers', '0', '--stage', '3', '--out', str(target),
                '--epochs', str(epochs), '--trainable_encoder_layers', '4',
                '--real_ce_weight', '1.25', '--guard_baseline', '--warmup_epochs', '.25',
                '--patience', '1', '--earlystop', '2', '--encoder_lr', '2e-5',
                '--head_lr', '1e-4']
        for name in ('train_data_path', 'dev_data_path', 'train_protocol', 'dev_protocol',
                     'rtc_pairs', 'train_noise_manifest', 'train_noisy_cache',
                     'dev_noisy_cache', 'dev_heldout_cache', 'ssl_path'):
            args += ['--' + name, 'fixture']
        return args

    def run_engine(self, args, reports):
        captured = io.StringIO()
        with patch.object(engine, 'DataBundle', FixtureBundle), \
                patch.object(engine.Detector, 'load', side_effect=build_fixture_detector), \
                patch.object(engine, 'source_hashes', return_value={}), \
                patch.object(engine, 'validate', side_effect=copy.deepcopy(reports)) as validate, \
                patch.object(sys, 'argv', args), redirect_stdout(captured):
            engine.main()
        self.assertEqual(validate.call_count, len(reports))
        return captured.getvalue()

    def assert_model_identical(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_rejects_real_regression_and_stops_without_replacing_original(self):
        baseline_dev = validation_report(90, 90)
        misleading = validation_report(100, 89)
        regressed = validation_report(85, 80)
        self.assertGreater(misleading['robust_f1'], baseline_dev['robust_f1'])
        self.assertLess(misleading['online']['recall'][1], baseline_dev['online']['recall'][1])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, original = self.make_baseline(root)
            baseline_digest = sha256(baseline)
            target = root / 'rejected'
            args = self.arguments(target, 3) + ['--finetune_from', str(baseline)]
            self.run_engine(args, [baseline_dev, misleading, regressed])

            self.assertEqual(sha256(baseline), baseline_digest)
            best = load_checkpoint(target / 'best_model.pt')
            self.assertEqual(best['epoch'], 0)
            self.assert_model_identical(best['model'], original)
            last = load_checkpoint(target / 'last.pt')
            self.assertEqual((last['epoch'], last['global_step'], last['best_epoch']), (2, 8, 0))
            self.assertEqual(last['baseline_dev'], baseline_dev)
            self.assertEqual(last['schedule']['warmup_steps'], 1)
            self.assertEqual(last['config']['real_ce_weight'], 1.25)
            self.assertEqual(last['config']['trainable_encoder_layers'], 4)
            self.assertTrue(last['config']['guard_baseline'])

            frozen = ['backbone.feature_projection.'] + [f'backbone.encoder.layers.{i}.' for i in range(20)]
            for name in original:
                if any(name.startswith(prefix) for prefix in frozen):
                    self.assertTrue(torch.equal(last['model'][name], original[name]), name)
            for prefix in ['head.'] + [f'backbone.encoder.layers.{i}.' for i in range(20, 24)]:
                self.assertTrue(any(not torch.equal(last['model'][name], original[name])
                                    for name in original if name.startswith(prefix)), prefix)
            audit = json.loads((target / 'gradient_epoch_002.json').read_text())
            self.assertTrue(audit['feature_projection']['frozen'])
            self.assertTrue(all(audit[f'layer_{i:02d}']['frozen'] for i in range(20)))
            self.assertTrue(all(audit[f'layer_{i:02d}']['sampled_update_norm'] > 0 for i in range(20, 24)))
            self.assertGreater(audit['head']['sampled_update_norm'], 0)

            rows = [json.loads(line) for line in (target / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['epoch'] for row in rows], [1, 2])
            self.assertTrue(all(not row['best'] and not row['selection']['accepted'] for row in rows))
            self.assertIn('online_real_recall_below_baseline', rows[0]['selection']['reasons'])
            self.assertTrue(all('ce' in row['mean_batch'] and 'ce_noisy_processed' in row['mean_batch'] for row in rows))
            self.assertFalse((target / 'epoch_003_evaluation.json').exists())
            completed = json.loads((target / 'completed.json').read_text())
            self.assertEqual(completed['status'], 'no_eligible_improvement')
            self.assertEqual(completed['best_epoch'], 0)

    def test_accepts_improvement_and_completed_resume_preserves_guard_state(self):
        baseline_dev, improved = validation_report(90, 90), validation_report(92, 92)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, original = self.make_baseline(root)
            baseline_digest = sha256(baseline)
            target = root / 'accepted'
            common = self.arguments(target, 1)
            self.run_engine(common + ['--finetune_from', str(baseline)], [baseline_dev, improved])
            best = load_checkpoint(target / 'best_model.pt')
            last = load_checkpoint(target / 'last.pt')
            self.assertEqual((best['epoch'], last['epoch'], last['best_epoch']), (1, 1, 1))
            self.assertEqual(last['baseline_dev'], baseline_dev)
            self.assertEqual(tuple(last['best_key']), (improved['robust_f1'], -improved['robust_ce']))
            self.assert_model_identical(best['model'], last['model'])
            self.assertTrue(any(not torch.equal(best['model'][name], original[name]) for name in original))
            self.assertEqual(sha256(baseline), baseline_digest)
            row = json.loads((target / 'metrics.jsonl').read_text().strip())
            self.assertTrue(row['best'] and row['selection']['accepted'])

            before = {name: sha256(target / name) for name in ('best_model.pt', 'last.pt', 'metrics.jsonl')}
            self.run_engine(common + ['--resume', str(target / 'last.pt')], [])
            self.assertEqual(before, {name: sha256(target / name) for name in before})
            self.assertEqual(sha256(baseline), baseline_digest)
            completed = json.loads((target / 'completed.json').read_text())
            self.assertEqual(completed['status'], 'improved')
            self.assertEqual(completed['best_epoch'], 1)


if __name__ == '__main__':
    unittest.main()
