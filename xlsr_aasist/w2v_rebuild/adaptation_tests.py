"""Regression tests for the reported two-epoch failure and limited noise mixing."""
import copy
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from .control import AdaptationControl
from .selection import candidate_decision, noisy_metrics
from .core import sha256, load_checkpoint
from .storage import GIB, checkpoint_headroom
from . import refinement_engine_tests as fixtures
from .train import parser
from rtc_noisy_v2.sampling import RotatingViewBatchSampler
from start_w2v_adapt import adapt_config, check_diverse_cache
from recover_w2v_storage import training_command


def screenshot_reports():
    baseline = fixtures.validation_report()
    baseline.update(robust_f1=.95483, robust_ce=.463848)
    baseline['online'].update(macro_f1=.96878, recall=[6060/6068, 1268/1397])
    baseline['offline']['recall'][1] = .918044
    baseline['seen']['macro_f1'], baseline['heldout']['macro_f1'] = .94513, .95258
    for kind, values in [('seen', [88.774, 89.325, 90.702, 90.565]),
                         ('heldout', [90.152, 91.529, 92.011, 92.631])]:
        for band, value in zip(baseline[kind]['bands'], values):
            band['recall'][1] = value/100
    one, two = copy.deepcopy(baseline), copy.deepcopy(baseline)
    one.update(robust_f1=.95544, robust_ce=.376111)
    one['online'].update(macro_f1=.97261, recall=[6057/6068, 1287/1397])
    one['seen']['macro_f1'], one['heldout']['macro_f1'] = .94449, .95167
    for kind, values in [('seen', [89.945, 91.185, 91.942, 91.667]),
                         ('heldout', [91.116, 92.562, 92.7, 93.526])]:
        for band, value in zip(one[kind]['bands'], values):
            band['recall'][1] = value/100
    two.update(robust_f1=.95537, robust_ce=.421459)
    two['online'].update(macro_f1=.96902, recall=[6060/6068, 1269/1397])
    two['seen']['macro_f1'], two['heldout']['macro_f1'] = .94598, .95306
    # Screenshot reports a decrease but not its magnitude. This is a fixture,
    # not an estimate of the actual server Offline metric.
    two['offline']['recall'][1] -= .002
    for kind, values in [('seen', [88.774, 89.394, 90.978, 90.702]),
                         ('heldout', [90.289, 91.391, 92.218, 92.7])]:
        for band, value in zip(two[kind]['bands'], values):
            band['recall'][1] = value/100
    return baseline, one, two


class BalancedSources:
    def __init__(self, steps=100):
        self.steps = steps

    def __len__(self):
        return self.steps

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        for _ in range(self.steps):
            yield [0, 1, 2, 3]  # two fake and two real


class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_screenshot_candidates_are_retained_and_offline_is_diagnostic(self):
        baseline, one, two = screenshot_reports()
        key = (baseline['robust_f1'], -baseline['robust_ce'])
        control = AdaptationControl(baseline)
        self.assertTrue(control.observe(one, 1, False)[0])
        accepted, report = candidate_decision(one, key, baseline, 'targeted')
        self.assertFalse(accepted)
        self.assertEqual(report['reasons'], ['noisy_f1_below_baseline'])
        self.assertFalse(control.observe(two, 2, True)[0])
        self.assertEqual(control.candidate_epoch, 1)
        accepted, report = candidate_decision(two, key, baseline, 'targeted')
        self.assertTrue(accepted)
        self.assertIn('offline_real_recall_decreased', report['warnings'])
        self.assertFalse(candidate_decision(two, key, baseline)[0])  # legacy unchanged
        self.assertEqual(control.stale, 0)  # new noisy F1 progress despite candidate score below epoch 1

    def test_targeted_promotion_still_protects_real_and_noisy(self):
        baseline, _, two = screenshot_reports()
        key = (baseline['robust_f1'], -baseline['robust_ce'])
        for kind in ('online', 'seen', 'heldout'):
            changed = copy.deepcopy(two)
            if kind == 'online':
                changed[kind]['recall'][1] = .7
            else:
                for b in changed[kind]['bands']:
                    b['recall'][1] = .7
            self.assertFalse(candidate_decision(changed, key, baseline, 'targeted')[0])
        ce_only = copy.deepcopy(baseline)
        ce_only['robust_ce'] -= .1
        self.assertFalse(candidate_decision(ce_only, key, baseline, 'targeted')[0])

    def test_reduction_gets_one_epoch_and_control_state_survives_resume(self):
        baseline, _, _ = screenshot_reports()
        controller = AdaptationControl(baseline)
        worse = copy.deepcopy(baseline)
        worse['robust_ce'] += .1
        _, progress = controller.observe(worse, 1, True)
        self.assertEqual(progress['stale'], 1)
        self.assertFalse(controller.should_stop(1, 1))
        restored = AdaptationControl(baseline)
        restored.load_state_dict(json.loads(json.dumps(controller.state_dict())))
        self.assertEqual(restored.state_dict(), controller.state_dict())
        restored.observe(worse, 2, False)
        self.assertTrue(restored.should_stop(2, 1))
        with self.assertRaises(ValueError):
            restored.load_state_dict({})

    def sampler(self, banks=2, fraction=.2, warmup=100):
        sources = [{'offline': f'offline/{i}.wav', 'label': i//2} for i in range(4)]
        sampler = RotatingViewBatchSampler(BalancedSources(), sources, 48, banks, 'cycle')
        sampler.configure_mixture(fraction, warmup)
        return sampler

    def test_mixture_quota_class_symmetry_band_coverage_and_replay(self):
        sampler = self.sampler()
        visits = {(i, bank): [] for i in range(4) for bank in range(2)}
        for epoch, expected_extra in ((1, 10), (2, 20), (3, 20)):
            sampler.set_epoch(epoch)
            first = list(sampler)
            self.assertEqual(first, list(sampler))  # prefetch/reiteration cannot advance visits
            self.assertEqual(sum(batch[0][1] == 1 for batch in first), expected_extra)
            self.assertEqual(sampler.plan_summary()['source_views_by_bank'],
                             [(100-expected_extra)*4, expected_extra*4])
            for batch in first:
                self.assertEqual(len({bank for _, bank, _ in batch}), 1)
                self.assertEqual([i//2 for i, _, _ in batch].count(1), 2)
                for i, bank, band in batch:
                    visits[i, bank].append(band)
            sampler.commit_epoch(100)
        for history in visits.values():
            for start in range(0, len(history)-3, 4):
                self.assertEqual(set(history[start:start+4]), {0, 1, 2, 3})
        restored = self.sampler()
        restored.load_state_dict(sampler.state_dict())
        sampler.set_epoch(4)
        restored.set_epoch(4)
        self.assertEqual(list(sampler), list(restored))
        self.assertEqual(sampler.plan_summary(), restored.plan_summary())
        restored.discard_epoch()
        restored.set_epoch(4)
        self.assertEqual(list(sampler), list(restored))
        with self.assertRaises(ValueError):
            self.sampler(fraction=.3).load_state_dict(sampler.state_dict())

    def test_three_banks_share_only_the_extra_quota(self):
        sampler = self.sampler(banks=3, warmup=0)
        sampler.set_epoch(1)
        self.assertEqual(sampler.plan_summary()['source_views_by_bank'], [320, 40, 40])
        with self.assertRaises(ValueError):
            self.sampler(banks=1)

    def test_candidate_space_is_reserved_even_before_first_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(checkpoint_headroom(root, 2*GIB, 6*GIB, ('candidate_best.pt',)), 17*GIB)
            (root/'best_model.pt').touch()
            (root/'last.pt').touch()
            self.assertEqual(checkpoint_headroom(root, 2*GIB, 6*GIB, ('candidate_best.pt',)), 9*GIB)
            (root/'candidate_best.pt').touch()
            self.assertEqual(checkpoint_headroom(root, 2*GIB, 6*GIB, ('candidate_best.pt',)), 7*GIB)

    def test_real_engine_retains_epoch1_candidate_promotes_epoch2_and_uses_lower_lr(self):
        baseline_dev, one, two = screenshot_reports()
        harness = fixtures.RefinementEngineTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, original = harness.make_baseline(root)
            original_hash = sha256(baseline)
            out = root/'new'
            common = harness.arguments(out, 3) + ['--adaptation_control']
            harness.run_engine(common + ['--finetune_from', str(baseline)],
                               [baseline_dev, one, two, two])
            candidate = load_checkpoint(out/'candidate_best.pt')
            best = load_checkpoint(out/'best_model.pt')
            last = load_checkpoint(out/'last.pt')
            self.assertEqual((candidate['epoch'], best['epoch'], last['epoch']), (1, 2, 3))
            self.assertEqual(sha256(baseline), original_hash)
            self.assertTrue(any(not torch.equal(candidate['model'][n], best['model'][n])
                                for n in best['model'] if n.startswith('head.')))
            for name, weight in original.items():
                if name.startswith('backbone.feature_projection.') or any(
                        name.startswith(f'backbone.encoder.layers.{i}.') for i in range(20)):
                    self.assertTrue(torch.equal(last['model'][name], weight), name)
            rows = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]['lr_scale_next'], .5)
            self.assertAlmostEqual(rows[2]['used_lr']['head/decay'], rows[1]['used_lr']['head/decay']*.5)
            completed = json.loads((out/'completed.json').read_text())
            self.assertEqual((completed['status'], completed['best_epoch'], completed['candidate_epoch']),
                             ('improved', 2, 1))
            saved_hashes = {name: sha256(out/name) for name in ('candidate_best.pt', 'best_model.pt', 'last.pt')}
            harness.run_engine(common + ['--resume', str(out/'last.pt')], [])
            self.assertEqual(saved_hashes, {name: sha256(out/name) for name in saved_hashes})

    def test_launcher_reuses_only_train_cache_and_preserves_existing_files(self):
        import start_w2v_adapt as launcher
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root/'original'/'stage3'/'best_model.pt'
            original.parent.mkdir(parents=True)
            original.write_bytes(b'original best fixture')
            source = root/'exp'/'previous'
            (source/'stage3').mkdir(parents=True)
            last = source/'stage3'/'last.pt'
            last.write_bytes(b'keep epoch2')
            bank = root/'cache'/'train_g1'
            bank.mkdir(parents=True)
            metadata = {'role': 'train', 'generation': 1, 'processing': {'profile': 'diverse'}}
            (bank/'config.json').write_text(json.dumps(metadata))
            (bank/'manifest.jsonl').write_text('{}\n')
            config = {'stage': 3, 'adaptation': True, 'baseline_path': str(original),
                      'init_sha256': sha256(original), 'train_noisy_cache': str(root/'train_g0'),
                      'dev_noisy_cache': str(root/'cache'/'dev_seen'), 'feature_cache': str(root/'features')}
            config_path = source/'stage3'/'config.json'
            config_path.write_text(json.dumps(config))
            keep = {p: p.read_bytes() for p in (original, last, config_path, bank/'config.json', bank/'manifest.jsonl')}
            def child(command, **kwargs):
                out = Path(command[command.index('--out')+1])
                out.mkdir()
                (out/'completed.json').write_text(json.dumps({'status': 'candidate_only',
                    'best_model': str(out/'best_model.pt'), 'candidate_model': str(out/'candidate_best.pt')}))
            argv = ['test', '--from-run', str(source)]
            with patch.object(launcher, 'ROOT', root), patch.object(launcher, 'recovery_lock', nullcontext), \
                    patch.object(launcher.subprocess, 'run', side_effect=child) as child_run, redirect_stdout(io.StringIO()):
                with patch.object(sys, 'argv', argv):
                    launcher.main()
                child_run.assert_not_called()
                with patch.object(sys, 'argv', argv+['--run']):
                    launcher.main()
                self.assertEqual(child_run.call_count, 1)
                command = child_run.call_args.args[0]
                self.assertNotIn('--preflight', command)
                self.assertNotIn('--resume', command)
                self.assertIn('--adaptation_control', command)
                self.assertEqual(command[command.index('--noisy_extra_fraction')+1], '0.2')
                self.assertEqual(command[command.index('--finetune_from')+1], str(original.resolve()))
                self.assertEqual(command[command.index('--extra_train_noisy_cache')+1], str(bank.resolve()))
            self.assertTrue(all(p.read_bytes() == value for p, value in keep.items()))
            metadata['role'] = 'dev_seen'
            (bank/'config.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                check_diverse_cache(bank)
            with self.assertRaises(FileNotFoundError):
                check_diverse_cache(root/'absent')


if __name__ == '__main__':
    unittest.main(verbosity=2)
