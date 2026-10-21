"""Real AASIST and small Omni-shaped encoder: gradients, budget and state."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from w2v_v53.test_step import fixture as parent_fixture
from w2v_v53.test_fusion import detector
from w2v_v53.common import partial_state as parent_state
from .model import Detector, optimizer_for, parent_anchor
from .common import partial_state, apply_partial
from .schedule import schedule
from .step import train_step
from .objective import paired_losses


def fixture(arm='paired'):
    cfg, units, old = parent_fixture()
    cfg.update(repair_arm=arm, repair_seed=54001, repair_feature_weight=.05, repair_kd_weight=.02,
        repair_temperature=3., repair_teacher_confidence=.9, repair_teacher_advantage=.02,
        repair_min_frames=4, repair_lr=1e-4, repair_gate_lr=5e-5,
        repair_warmup_max=2, repair_warmup_fraction=.1, repair_feature_ramp=1,
        repair_kd_ramp=1, epochs=1, pcl_variance_reference_weight=.15,
        pcl_noise_weight=.15 if arm=='baseline' else 0., lr_floor=.5)
    # Identical small frozen encoder assets, just as production reloads Omni.
    torch.manual_seed(985)
    model = detector(cfg, Detector)
    old_params = old.state_dict()
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if name in old_params:
                value.copy_(old_params[name])
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.pcl_scale.sync_state()
    model.repair.scale_ready.fill_(True)
    for unit in units:
        row = unit[0]
        if row['role'] == 'noisy':
            n = (len(row['wave'])-400)//320+1
            c = (len(row['pcl_refs']['clean']['wave'])-400)//320+1
            row['repair_pair'] = dict(shift_frames=0., clean_valid=np.ones(c,bool),
                noisy_valid=np.ones(n,bool), delay=dict(valid=True,offset=0.,reason='ok'))
            row['v5_origin_split'] = 'dev'  # Auxiliary remains valid without CTC.
    return cfg, units, model, old


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(54)

    def test_initial_real_aasist_matches_v53_and_reload_uses_trained_repair(self):
        cfg, units, model, old = fixture()
        anchor = parent_anchor(model, parent_state(old))
        apply_partial(model, anchor)
        waves = [unit[0]['wave'] for unit in units[:2]]
        model.eval(); old.eval()
        with torch.no_grad():
            original = old(waves)
            torch.testing.assert_close(model(waves), original, rtol=0, atol=0)
            model.repair.out.bias.add_(torch.linspace(-.3,.5,128))
            changed = model(waves)
        self.assertFalse(torch.equal(original, changed))
        saved = partial_state(model)
        apply_partial(model, anchor)
        apply_partial(model, saved)
        with torch.no_grad():
            torch.testing.assert_close(model(waves), changed, rtol=0, atol=0)
        # Anchor always uses zero initial repair, never the trained live branch.
        self.assertEqual(int(parent_anchor(model, parent_state(old))['repair.out.bias'].count_nonzero()), 0)
        bad = dict(saved); bad.pop('repair.out.bias')
        with self.assertRaises(ValueError):
            apply_partial(model, bad)

    def test_warmup_only_repair_then_joint_all_losses_and_one_encoder_per_view(self):
        cfg, units, model, old = fixture()
        opt = optimizer_for(model, cfg)
        before = partial_state(model)
        calls = []
        hook = model.encoder.register_forward_hook(lambda m,a,o: calls.append(o.shape[0]))
        schedule(opt, cfg, 0, 0, 20)
        first = train_step(model, opt, units, cfg, audit=True)
        self.assertEqual(first['repair_stage'], 'repair_warmup')
        self.assertGreater(first['repair_losses']['feature'], 0)
        self.assertEqual(first['repair_losses']['kd'], 0)
        self.assertTrue(first['hard_examples']['validated'])
        self.assertEqual(sum(calls), first['encoded_views'])
        for name, p in model.named_parameters():
            if name in before and not name.startswith('repair.'):
                torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        self.assertGreater(first['repair_update']['processor'], 0)
        # Force reliable teacher gate on for this numerical integration test.
        joint_cfg = dict(cfg, repair_teacher_confidence=0., repair_teacher_advantage=-1.)
        calls.clear(); schedule(opt, cfg, 0, 3, 20)
        second = train_step(model, opt, units, joint_cfg, audit=True)
        hook.remove()
        self.assertEqual(second['repair_stage'], 'joint')
        self.assertEqual(sum(calls), second['encoded_views'])
        self.assertGreater(second['repair_losses']['feature'], 0)
        self.assertGreater(second['repair_losses']['kd'], 0)
        self.assertEqual(second['pcl_losses']['noise']['weighted_loss'], 0)
        self.assertGreater(second['pcl_losses']['platform']['weighted_loss'], 0)
        self.assertTrue(second['hard_examples']['validated'])
        self.assertTrue(any(p.requires_grad for p in model.fusion.parameters()))
        total = (second['classification_loss'] + sum(x['weighted_loss'] for x in second['pcl_losses'].values())
            + second['pcl_variance']['weighted_loss'] + second['en_real_confidence']['weighted_loss']
            + second['repair_losses']['feature'] + second['repair_losses']['kd'])
        self.assertAlmostEqual(total, second['loss'], places=6)

    def test_baseline_matches_original_v53_update_and_keeps_platform_variance(self):
        from w2v_v53.step import train_step as previous_step
        from w2v_v53.model import optimizer_for as previous_optimizer
        from w2v_v3191.meta import rng_state, restore_rng
        cfg, units, model, old = fixture('baseline')
        opt = optimizer_for(model, cfg)
        old_opt = previous_optimizer(old, cfg)
        state = rng_state()
        expected = previous_step(old, old_opt, units, cfg)
        restore_rng(state)
        actual = train_step(model, opt, units, cfg)
        self.assertEqual(actual['loss'], expected['loss'])
        self.assertEqual(actual['encoded_views'], expected['encoded_views'])
        for key, value in parent_state(old).items():
            torch.testing.assert_close(partial_state(model)[key], value, rtol=0, atol=0)

    def test_lf_gradient_stops_at_processor_and_does_not_change_encoder_gradient(self):
        cfg, units, model, _ = fixture()
        from .side import _layout
        selected = [u for u in units if u[0]['role']=='noisy' and u[0]['meta_side']=='A'][:1]
        rows, slots = _layout(selected[0], dict(noise=0, platform=0))
        x = model.encode([r['wave'] for r in rows])
        processed = model.processed_sequences(x)
        lf, kd, info = paired_losses(model, selected, [(0,slots)], x, processed, [0], [1], [1]*4,
            dict(cfg, current_update=1, _repair_warm_steps=2))
        lf.backward()
        self.assertTrue(all(p.grad is None for p in model.fusion.parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.repair.gate.parameters()))
        self.assertGreater(float(model.repair.out.weight.grad.norm()), 0)

    def test_failed_amp_attempt_rolls_back_buffers_rng_and_does_not_advance_schedule(self):
        from w2v_v3191.meta import rng_state
        from w2v_v3191.test_step import assert_rng_equal
        cfg, units, model, _ = fixture()
        opt = optimizer_for(model,cfg)
        schedule(opt,cfg,0,0,20)
        scaler = torch.amp.GradScaler('cpu',init_scale=128.)
        before, rng = copy.deepcopy(model.state_dict()), rng_state()
        hook = model.repair.out.weight.register_hook(lambda g:g*float('nan'))
        with patch('w2v_v54.step.scaler_for',return_value=scaler):
            result = train_step(model,opt,units,cfg)
        hook.remove()
        self.assertTrue(result['amp']['skipped_update'])
        self.assertEqual(opt._v3189_update,1)
        self.assertEqual(opt.state_dict()['state'],{})
        for name,value in before.items():
            torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
        assert_rng_equal(self,rng,rng_state())


if __name__ == '__main__':
    unittest.main()
