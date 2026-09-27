"""Refinement safety, loss gradients and automatic best-selection regressions."""
import copy
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from .core import objective, Schedule, build_optimizer, sha256
from .model import Detector
from .tests import ToyEncoder
from .train import train_step, parser
from .selection import candidate_decision
from start_w2v_refine import refine_config
from recover_w2v_storage import training_command


def dev_fixture():
    part = {'macro_f1': .95483, 'balanced_ce': .463848, 'recall': [.9987, .9077]}
    return {'online': copy.deepcopy(part), 'offline': copy.deepcopy(part),
            'seen': {**part, 'bands': [copy.deepcopy(part) for _ in range(4)]},
            'heldout': {**part, 'bands': [copy.deepcopy(part) for _ in range(4)]},
            'robust_f1': part['macro_f1'], 'robust_ce': part['balanced_ce']}


class Refinement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_cost_all_ce_groups_and_gradient_direction(self):
        z = torch.zeros(40, 2, requires_grad=True)
        y = torch.arange(40) % 2
        h = torch.randn(40, 16)
        loss, stats = objective(z, h, y, 24, 4, 4, torch.ones(2), 0, 0, real_ce_weight=1.25)
        loss.backward()
        for fake, real in ((0, 1), (24, 25), (32, 33), (36, 37)):
            self.assertAlmostEqual(float(z.grad[real, 0] / -z.grad[fake, 0]), 1.25, places=5)
        self.assertAlmostEqual(float(loss), float(torch.log(torch.tensor(2.))), places=6)
        self.assertAlmostEqual(sum(stats['coefficients'].values()), 1.)

    def test_inverse_frequency_only_applies_to_ordinary(self):
        torch.manual_seed(9)
        z, h, y = torch.randn(40, 2), torch.randn(40, 16), torch.arange(40) % 2
        w = torch.tensor([.617, 2.64])
        _, parts = objective(z, h, y, 24, 4, 4, w, 0, 0, real_ce_weight=1.25)
        ce = F.cross_entropy(z, y, reduction='none')
        cost = torch.where(y == 1, 1.25, 1.)
        ow = w[y[:24]] * cost[:24]
        self.assertAlmostEqual(parts['ce_ordinary'], float((ce[:24]*ow).sum()/ow.sum()), places=6)
        self.assertAlmostEqual(parts['ce_real_pair'], float((ce[24:32]*cost[24:32]).sum()/cost[24:32].sum()), places=6)

    def test_unit_cost_preserves_legacy_loss_and_gradients_exactly(self):
        torch.manual_seed(7)
        z, h = torch.randn(40, 2, requires_grad=True), torch.randn(40, 16, requires_grad=True)
        y, w = torch.arange(40) % 2, torch.tensor([.617, 2.64])
        a, sa = objective(z, h, y, 24, 4, 4, w, .1, .1)
        b, sb = objective(z, h, y, 24, 4, 4, w, .1, .1, real_ce_weight=1.)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(sa, sb)
        ga, gb = torch.autograd.grad(a, (z, h), retain_graph=True), torch.autograd.grad(b, (z, h))
        for left, right in zip(ga, gb):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_original_baseline_anchors_scheduler_and_survives_resume(self):
        p = torch.nn.Parameter(torch.tensor(1.))
        opt = torch.optim.AdamW([{'params': [p], 'lr': 1e-4, 'base_lr': 1e-4}])
        s = Schedule(opt, 10, patience=1)
        s.anchor(.463848)
        self.assertFalse(s.validate(.6, 5))
        self.assertTrue(s.validate(.558575, 10))
        self.assertEqual(s.scale, .5)
        t = Schedule(opt, 10, patience=1)
        t.load_state_dict(s.state_dict())
        self.assertEqual(t.best, .463848)
        self.assertFalse(t.validate(.54, 20))  # Still worse than baseline; cooldown only.
        self.assertEqual(t.best, .463848)

    def test_guard_rejects_f1_gain_with_real_or_noisy_regression(self):
        baseline = dev_fixture()
        key = (baseline['robust_f1'], -baseline['robust_ce'])
        for kind, field in (('online', 'recall'), ('offline', 'recall'), ('seen', 'macro_f1'), ('heldout', 'macro_f1')):
            dev = copy.deepcopy(baseline)
            dev['robust_f1'] += .002
            if field == 'recall':
                dev[kind][field][1] -= .01
            else:
                dev[kind][field] -= .01
            accepted, report = candidate_decision(dev, key, baseline)
            self.assertFalse(accepted, (kind, field))
            self.assertTrue(any('below_baseline' in x for x in report['reasons']))
        for kind in ('seen', 'heldout'):
            dev = copy.deepcopy(baseline)
            dev['robust_f1'] += .002
            dev[kind]['bands'][0]['recall'][1] -= .01
            self.assertFalse(candidate_decision(dev, key, baseline)[0])

    def test_guard_accepts_improvement_and_records_per_band_deltas(self):
        baseline = dev_fixture()
        key = (baseline['robust_f1'], -baseline['robust_ce'])
        self.assertFalse(candidate_decision(baseline, key, baseline)[0])
        dev = copy.deepcopy(baseline)
        dev['robust_ce'] -= .01
        self.assertFalse(candidate_decision(dev, key, baseline)[0])
        dev['robust_f1'] += .002
        accepted, report = candidate_decision(dev, key, baseline)
        self.assertTrue(accepted)
        self.assertIn('heldout_band_3_real_recall', report['baseline_deltas'])

    def test_partial_model_first_step_audit_and_frozen_weights(self):
        torch.manual_seed(19)
        model = Detector(ToyEncoder(24, checkpointing=True)).configure_trainable_layers(4)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        opt = build_optimizer(model, 1e-7, 2e-6, 1e-4)
        sched = Schedule(opt, 0)
        args = SimpleNamespace(device='cpu', amp='none', microbatch=2, stage=3, grad_clip=1., real_ce_weight=1.25)
        b = {'features': torch.randn(10, 24, 160), 'mask': torch.ones(10, 24, dtype=torch.long),
             'labels': torch.arange(10) % 2}
        stats, _, _ = train_step(model, opt, sched, b, (2, 2, 2), torch.tensor([.617, 2.64]), args, 0, 10, True)
        for name, value in frozen.items():
            p = dict(model.named_parameters())[name]
            torch.testing.assert_close(p, value, rtol=0, atol=0)
            self.assertIsNone(p.grad)
        audit = stats['gradient_audit']
        self.assertTrue(audit['feature_projection']['frozen'])
        self.assertTrue(audit['layer_19']['frozen'])
        for name in ('layer_20', 'layer_21', 'layer_22', 'layer_23', 'head'):
            self.assertGreater(audit[name]['sampled_update_norm'], 0.)

    def test_launcher_rolls_back_recipe_keeps_paths_and_never_resumes_rejected_last(self):
        config = {'stage': 3, 'adaptation': True, 'baseline_path': '/old/stage3/best_model.pt',
                  'ordinary_sampling': 'balanced', 'extra_train_noisy_cache': ['/new/train_g1'],
                  'consistency_weight': .02, 'feature_cache': '/cache/features',
                  'train_noisy_cache': '/old/train_g0', 'dev_noisy_cache': '/new/dev_seen',
                  'dev_heldout_cache': '/new/dev_heldout', 'resume': '/rejected/last.pt'}
        before = copy.deepcopy(config)
        refined = refine_config(config)
        self.assertEqual(config, before)
        self.assertEqual(refined['ordinary_sampling'], 'legacy')
        self.assertEqual(refined['extra_train_noisy_cache'], [])
        self.assertEqual(refined['consistency_weight'], 0.)
        self.assertEqual(refined['feature_cache'], config['feature_cache'])
        self.assertEqual(refined['dev_noisy_cache'], config['dev_noisy_cache'])
        cmd = training_command(refined, '/new/run/stage3', config['baseline_path'], False, True)
        self.assertNotIn('--resume', cmd)
        self.assertNotIn('--preflight', cmd)
        self.assertNotIn('--extra_train_noisy_cache', cmd)
        self.assertIn('--guard_baseline', cmd)
        self.assertIn(config['baseline_path'], cmd)
        self.assertEqual(cmd[cmd.index('--trainable_encoder_layers')+1], '4')

    def test_launcher_checks_original_hash_and_launches_once_without_preflight(self):
        import start_w2v_refine as launcher
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root/'original'/'stage3'/'best_model.pt'
            baseline.parent.mkdir(parents=True)
            baseline.write_bytes(b'original weights fixture')
            source = root/'exp'/'rejected'
            (source/'stage3').mkdir(parents=True)
            config = {'stage': 3, 'adaptation': True, 'baseline_path': str(baseline),
                      'init_sha256': sha256(baseline), 'feature_cache': str(root/'features')}
            config_path = source/'stage3'/'config.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            digest = sha256(baseline)
            def child(command, **kwargs):
                out = Path(command[command.index('--out')+1])
                out.mkdir()
                (out/'completed.json').write_text(json.dumps({'status': 'no_eligible_improvement',
                                                             'best_model': str(out/'best_model.pt')}))
            argv = ['test', '--from-run', str(source)]
            with patch.object(launcher, 'ROOT', root), patch.object(launcher, 'recovery_lock', nullcontext), \
                    patch.object(launcher.subprocess, 'run', side_effect=child) as run, redirect_stdout(io.StringIO()):
                with patch.object(sys, 'argv', argv):
                    launcher.main()
                run.assert_not_called()
                with patch.object(sys, 'argv', argv+['--run']):
                    launcher.main()
                self.assertEqual(run.call_count, 1)
                command = run.call_args.args[0]
                self.assertNotIn('--preflight', command)
                self.assertNotIn('--resume', command)
                self.assertEqual(sha256(baseline), digest)
                self.assertEqual(json.loads(config_path.read_text()), config)
                config['init_sha256'] = '0'*64
                config_path.write_text(json.dumps(config), encoding='utf-8')
                with patch.object(sys, 'argv', argv+['--run']), self.assertRaisesRegex(ValueError, 'SHA256'):
                    launcher.main()
                self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
