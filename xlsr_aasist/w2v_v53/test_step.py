"""Exercise the actual V5.2 CE/PCL/contrast code through adaptive Omni+AASIST."""
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from w2v_v52.test_contrastive import pair_fixture
from w2v_v52.contrastive import bank_for
from .test_fusion import configuration, detector
from .model import optimizer_for
from .step import train_step


def fixture():
    settings, units = pair_fixture()
    cfg = dict(configuration(), **settings)
    cfg.update(consistency='dual', microbatch=24, pcl_noise_weight=.03,
        pcl_platform_weight=.045, pcl_ramp_updates=1, pcl_variance_min_sources=1,
        pcl_variance_min_phones=1, pcl_variance_weight=.01, pcl_variance_target=.5,
        v52_parent_pcl_weights=dict(noise=.15, platform=.045), pcl_min_frames=1)
    for unit in units:
        row = unit[0]
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
        self.assertGreater(stats['contrastive']['active_pairs'], 0)
        self.assertGreater(stats['contrastive']['weighted_loss'], 0.)
        self.assertEqual(stats['contrastive']['bank_entries_after'], 8)
        for kind in ('noise', 'platform'):
            self.assertGreater(stats['pcl_losses'][kind]['sources'], 0)
            self.assertGreater(stats['pcl_losses'][kind]['weighted_loss'], 0.)
        total = (stats['classification_loss'] + stats['contrastive']['weighted_loss']
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

    def test_failed_amp_step_restores_new_parameters_and_does_not_commit_bank(self):
        cfg, units, model = fixture()
        opt = optimizer_for(model, cfg)
        scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        with patch('w2v_v52.step.scaler_for', return_value=scaler):
            train_step(model, opt, units, cfg)
            snapshot = copy.deepcopy(model.state_dict())
            bank = copy.deepcopy(bank_for(model, cfg))
            hook = model.fusion.conditioner[-1].weight.register_hook(lambda g: g*float('nan'))
            stats = train_step(model, opt, units, cfg, audit=True)
            hook.remove()
            self.assertTrue(stats['amp']['skipped_update'])
            self.assertFalse(stats['adaptive_fusion']['committed'])
            self.assertEqual(stats['adaptive_fusion']['conditioner_update_norm'], 0.)
            for name, value in snapshot.items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            for a,b in zip(bank['entries'], bank_for(model,cfg)['entries']):
                self.assertEqual(a['source'], b['source'])
                torch.testing.assert_close(a['feature'], b['feature'], rtol=0, atol=0)
            self.assertEqual(model.fusion._observations, [])


if __name__ == '__main__':
    unittest.main()
