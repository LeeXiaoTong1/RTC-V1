"""Actual V5.1 CE/PCL, exact update equivalence and rollback without snapshots."""
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from w2v_v52.test_contrastive import pair_fixture
from .test_fusion import configuration, detector
from .model import optimizer_for
from .step import train_step


def fixture():
    settings, units = pair_fixture()
    cfg = dict(configuration(), **settings)
    cfg.update(consistency='dual', microbatch=24, pcl_noise_weight=.15,
        pcl_platform_weight=.045, pcl_ramp_updates=1, pcl_variance_min_sources=1,
        pcl_variance_min_phones=1, pcl_variance_weight=.01, pcl_variance_target=.5,
        contrastive_weight=0., classification_ce_weights=dict(online=.25, clean_rtc=.25, noisy=.5),
        robust_arm='baseline', _candidate_diagnostics=False, pcl_min_frames=1)
    for unit in units:
        row = unit[0]
        row.pop('contrastive_enabled', None)
        row.pop('contrastive_group_budget', None)
        row.pop('robust_candidates', None)
        if row['role'] == 'noisy' and row['meta_side'] == 'A':
            row.update(pcl_noisy_index=0, pcl_platform_budget=1, pcl_scheduled=('noise', 'platform'),
                pcl_pairs=dict(noise=[[0, 2000, 0, 2000]], platform=[[0, 2000, 0, 2000]]))
            row['pcl_refs']['clean']['wave'] += np.tile(np.array([.15,-.3,.1,.2], np.float32), 1000)
    model = detector(cfg)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.pcl_scale.observe(torch.randn(3,128), 'train-a')
    model.pcl_scale.observe(torch.randn(3,128), 'train-b')
    model.pcl_scale.freeze()
    return cfg, units, model


class JointStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(530)

    def test_actual_joint_step_retains_all_losses_and_encodes_each_view_once(self):
        cfg, units, model = fixture()
        opt = optimizer_for(model, cfg)
        calls = []
        hook = model.encoder.register_forward_hook(lambda module, args, out: calls.append(out.shape[0]))
        before = model.fusion.conditioner[-1].weight.detach().clone()
        stats = train_step(model, opt, units, cfg, audit=True)
        hook.remove()
        self.assertEqual(stats['encoded_views'], 28)
        self.assertEqual(sum(calls), 28)
        self.assertEqual(stats['adaptive_fusion']['encoded_views'], 28)
        self.assertGreater(stats['adaptive_fusion']['conditioner_update_norm'], 0.)
        self.assertEqual(stats['adaptive_fusion']['extra_encoder_passes'], 0)
        self.assertNotIn('contrastive', stats)
        self.assertFalse(hasattr(model, '_v52_contrastive_bank'))
        self.assertGreater(stats['execution']['parameter_snapshot_bytes'], 0)
        for kind in ('noise', 'platform'):
            self.assertGreater(stats['pcl_losses'][kind]['sources'], 0)
            self.assertGreater(stats['pcl_losses'][kind]['weighted_loss'], 0.)
        total = (stats['classification_loss']
                 + sum(v['weighted_loss'] for v in stats['pcl_losses'].values())
                 + stats['pcl_variance']['weighted_loss'] + stats['en_real_confidence']['weighted_loss'])
        self.assertAlmostEqual(stats['loss'], total, places=6)
        self.assertFalse(torch.equal(before, model.fusion.conditioner[-1].weight))
        self.assertEqual(model.fusion._observations, [])
        self.assertEqual(stats['adaptive_fusion']['mean_absolute_change'], 0.)  # Forward before first update.
        second = train_step(model, opt, units, cfg, audit=True)
        self.assertGreater(second['adaptive_fusion']['input_weight_std'], 0.)
        self.assertGreater(second['adaptive_fusion']['mean_absolute_change'], 0.)
        json.dumps(second, allow_nan=False)

    def test_failed_amp_step_without_parameter_snapshots_restores_rng_and_buffers(self):
        cfg, units, model = fixture()
        opt = optimizer_for(model, cfg)
        scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        from w2v_v3191.meta import rng_state
        from w2v_v3191.test_step import assert_rng_equal
        with patch('w2v_v53.step.scaler_for', return_value=scaler):
            train_step(model, opt, units, cfg)
            snapshot = copy.deepcopy(model.state_dict())
            rng = rng_state()
            adam = copy.deepcopy(opt.state_dict())
            hook = model.fusion.conditioner[-1].weight.register_hook(lambda g: g*float('nan'))
            stats = train_step(model, opt, units, cfg, audit=False)
            hook.remove()
            self.assertTrue(stats['amp']['skipped_update'])
            for name, value in snapshot.items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            assert_rng_equal(self, rng, rng_state())
            for key,state in adam['state'].items():
                for name,value in state.items():
                    torch.testing.assert_close(opt.state_dict()['state'][key][name], value, rtol=0, atol=0)
            self.assertEqual(model.fusion._observations, [])

    def test_ordinary_update_matches_v51_objective_and_optimizer_exactly(self):
        from w2v_v3191.step import train_step as v51_step
        from w2v_v3191.meta import rng_state, restore_rng
        from w2v_v3191.test_step import assert_rng_equal
        cfg, units, model = fixture()
        other = copy.deepcopy(model)
        opt, other_opt = optimizer_for(model,cfg), optimizer_for(other,cfg)
        rng = rng_state()
        expected = v51_step(other, other_opt, units, cfg)
        expected_rng = rng_state()
        restore_rng(rng)
        actual = train_step(model, opt, units, cfg)
        self.assertEqual(actual['execution']['parameter_snapshot_bytes'], 0)
        self.assertEqual(actual['encoded_views'], expected['encoded_views'])
        self.assertEqual(actual['loss'], expected['loss'])
        self.assertEqual(actual['classification_loss'], expected['classification_loss'])
        assert_rng_equal(self, expected_rng, rng_state())
        for name,value in other.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
        for key,state in other_opt.state_dict()['state'].items():
            for name,value in state.items():
                torch.testing.assert_close(opt.state_dict()['state'][key][name],value,rtol=0,atol=0)

    def test_unused_official_references_do_not_run_encoder_or_aasist(self):
        cfg, units, model = fixture()
        cfg['consistency'] = 'none'
        calls = []
        hook = model.head.register_forward_pre_hook(lambda m,args: calls.append(len(args[0])))
        result = train_step(model, optimizer_for(model,cfg), units,cfg)
        hook.remove()
        self.assertEqual(result['encoded_views'], 24)
        self.assertEqual(result['auxiliary_reference_views'], 0)
        self.assertEqual(sum(calls), 24)


if __name__ == '__main__':
    unittest.main()
