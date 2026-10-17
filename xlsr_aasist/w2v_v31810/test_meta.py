"""Numerical and transactional tests, requiring neither CUDA nor Omni assets."""
import copy
import json
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .meta import (CELLS, Transaction, cell, rng_state, restore_rng, train_step,
                   validate_units, virtual_step)


class Toy(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.lora = nn.Linear(4, 5, bias=False)
        self.fusion = nn.Linear(5, 6)
        self.head = nn.Sequential(nn.Tanh(), nn.Dropout(dropout), nn.Linear(6, 2))
        self.frozen = nn.Parameter(torch.randn(27), requires_grad=False)
        self.register_buffer('counter', torch.tensor(0))
        self.joint = True
        self.calls = 0
        self.fail_at = None

    def set_phase(self, joint):
        self.joint = joint
        self.lora.weight.requires_grad_(joint)

    def forward(self, waves):
        self.calls += 1
        self.counter += 1
        if self.fail_at == self.calls:
            torch.rand(7)
            random.random()
            np.random.random()
            raise RuntimeError('injected temporary failure')
        x = torch.stack([torch.as_tensor(wave[:4]) for wave in waves])
        return self.head(self.fusion(self.lora(x)))


def fixture(copies=3, varied=False):
    cfg = dict(device='cpu', backend_precision='fp32', microbatch=12,
               frame_budget=9000, label_smoothing=.02, max_grad_norm=1.)
    rng = np.random.default_rng(147)
    units = []
    for side in ('A', 'B'):
        for i, (role, language, label) in enumerate(CELLS):
            count = copies + (i % 2 if varied else 0)
            for j in range(count):
                identity = f'{side}-{role}-{language}-{label}-{j}'
                wave = np.tile(rng.normal(size=4).astype(np.float32), 130)
                units.append([dict(id=identity, source_id=identity, group_id=identity,
                                   canonical_source=identity, meta_side=side, role=role,
                                   language=language, label=label, split='train', wave=wave,
                                   task_kind='noise')])
    return cfg, units


def optimizer(model):
    return torch.optim.AdamW([
        dict(params=list(model.lora.parameters()), name='lora', lr=.002),
        dict(params=list(model.fusion.parameters()), name='fusion', lr=.004),
        dict(params=list(model.head.parameters()), name='head', lr=.003)], weight_decay=.001)


def balanced_loss(model, units, smoothing=.02):
    cells = {key: [unit[0] for unit in units if cell(unit[0]) == key] for key in CELLS}
    return sum(F.cross_entropy(model([row['wave'] for row in rows]),
                               torch.tensor([row['label'] for row in rows]),
                               label_smoothing=smoothing) for rows in cells.values()) / 8


def assert_state_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_state_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_state_equal(test, a, b)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        test.assertEqual(left, right)


class MetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)
        np.random.seed(32)
        random.seed(71)

    def test_zero_alpha_matches_weighted_ordinary_adamw(self):
        cfg, units = fixture(varied=True)
        one, two = Toy(), Toy()
        two.load_state_dict(one.state_dict())
        opt_one, opt_two = optimizer(one), optimizer(two)
        stats = train_step(one, opt_one, units, cfg, alpha=0., beta=.5)
        sides = validate_units(units)
        loss = (balanced_loss(two, sides['A']) + .5 * balanced_loss(two, sides['B'])) / 1.5
        loss.backward()
        torch.nn.utils.clip_grad_norm_(two.parameters(), 1.)
        opt_two.step()
        for a, b in zip(one.parameters(), two.parameters()):
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
        self.assertAlmostEqual(stats['loss'], float(loss.detach()), places=6)
        self.assertEqual(stats['encoded_views'], len(units))
        json.dumps(stats, allow_nan=False)

    def test_microbatch_normalization_unequal_cell_sizes(self):
        cfg, units = fixture(varied=True)
        one, two = Toy(), Toy()
        two.load_state_dict(one.state_dict())
        a = train_step(one, optimizer(one), units, dict(cfg, microbatch=1), .03, .5)
        b = train_step(two, optimizer(two), units, dict(cfg, microbatch=17), .03, .5)
        for p, q in zip(one.parameters(), two.parameters()):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-6)
        self.assertAlmostEqual(a['loss'], b['loss'], places=6)
        self.assertTrue(all(value['views'] in (6, 8) for value in a['cells'].values()))

    def test_first_order_raw_A_clipped_inner_outer_at_original(self):
        cfg, units = fixture()
        one, two = Toy(), Toy()
        two.load_state_dict(one.state_dict())
        opt_one, opt_two = optimizer(one), optimizer(two)
        alpha, beta = .08, .5
        stats = train_step(one, opt_one, units, dict(cfg, max_grad_norm=.005), alpha, beta, audit=True)
        sides = validate_units(units)
        parameters = [p for p in two.parameters() if p.requires_grad]
        originals = [p.detach().clone() for p in parameters]
        balanced_loss(two, sides['A']).backward()
        raw_a = [p.grad.clone() for p in parameters]
        torch.nn.utils.clip_grad_norm_(parameters, .005)
        with torch.no_grad():
            for p in parameters:
                p.add_(p.grad, alpha=-alpha)
        opt_two.zero_grad(set_to_none=True)
        balanced_loss(two, sides['B']).backward()
        with torch.no_grad():
            for p, old, ga in zip(parameters, originals, raw_a):
                p.copy_(old)
                p.grad = (ga + beta * p.grad) / (1 + beta)
        torch.nn.utils.clip_grad_norm_(parameters, .005)
        opt_two.step()
        for p, q in zip(one.parameters(), two.parameters()):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-6)
        self.assertLess(stats['meta']['inner_clip_factor'], 1.)
        self.assertEqual(set(stats['meta']['module_outer_update_norms']), {'lora', 'fusion', 'head'})
        self.assertTrue(all(value > 0 for value in stats['meta']['module_outer_update_norms'].values()))
        self.assertEqual(stats['encoded_views'], 2 * len(units))

    def test_exception_restores_parameters_buffers_rng_and_optimizer(self):
        cfg, units = fixture()
        model = Toy(dropout=.3)
        opt = optimizer(model)
        train_step(model, opt, units, cfg, .03, .5)
        before = copy.deepcopy(model.state_dict())
        moments = copy.deepcopy(opt.state_dict())
        rng = rng_state()
        model.fail_at = model.calls + 3  # first B forward after temporary SGD
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            train_step(model, opt, units, cfg, .03, .5)
        assert_state_equal(self, before, model.state_dict())
        assert_state_equal(self, moments, opt.state_dict())
        assert_state_equal(self, rng, rng_state())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_audit_uses_same_random_state_and_does_not_change_update(self):
        cfg, units = fixture()
        a, b = Toy(.2), Toy(.2)
        b.load_state_dict(a.state_dict())
        oa, ob = optimizer(a), optimizer(b)
        state = rng_state()
        sa = train_step(a, oa, units, cfg, .03, .5, audit=False)
        after = rng_state()
        restore_rng(state)
        sb = train_step(b, ob, units, cfg, .03, .5, audit=True)
        assert_state_equal(self, a.state_dict(), b.state_dict())
        assert_state_equal(self, oa.state_dict(), ob.state_dict())
        assert_state_equal(self, after, rng_state())
        self.assertAlmostEqual(sa['loss'], sb['loss'], places=7)
        self.assertIsNotNone(sb['meta']['delta_B'])

    def test_zero_beta_counts_B_but_only_A_backward(self):
        cfg, units = fixture()
        one, two = Toy(), Toy()
        two.load_state_dict(one.state_dict())
        o1, o2 = optimizer(one), optimizer(two)
        stats = train_step(one, o1, units, cfg, .1, 0.)
        balanced_loss(two, validate_units(units)['A']).backward()
        torch.nn.utils.clip_grad_norm_(two.parameters(), 1.)
        o2.step()
        for p, q in zip(one.parameters(), two.parameters()):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-6)
        self.assertEqual(stats['views'], 48)
        self.assertEqual(stats['encoded_views'], 48)

    def test_warmup_uses_all_cells_full_task_and_freezes_lora(self):
        cfg, units = fixture(varied=True)
        one, two = Toy(), Toy()
        two.load_state_dict(one.state_dict())
        initial_lora = one.lora.weight.detach().clone()
        stats = train_step(one, optimizer(one), units, cfg, .1, .5, joint=False)
        two.set_phase(False)
        opt = optimizer(two)
        loss = balanced_loss(two, units)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(two.parameters(), 1.)
        opt.step()
        for p, q in zip(one.parameters(), two.parameters()):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(initial_lora, one.lora.weight, rtol=0, atol=0)
        self.assertEqual(stats['phase'], 'head_warmup')
        self.assertEqual(stats['encoded_views'], len(units))

    def test_amp_cpu_scaler_overflow_skips_whole_task_and_backs_off(self):
        cfg, units = fixture()
        model = Toy()
        opt = optimizer(model)
        opt._v31810_scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        before = copy.deepcopy(model.state_dict())
        rng = rng_state()
        hook = model.head[-1].weight.register_hook(lambda grad: grad * float('inf'))
        stats = train_step(model, opt, units, cfg, .03, .5)
        hook.remove()
        assert_state_equal(self, before, model.state_dict())
        assert_state_equal(self, rng, rng_state())
        self.assertFalse(opt.state)
        self.assertTrue(stats['amp']['skipped_update'])
        self.assertEqual(stats['amp']['scale'], 64.)
        json.dumps(stats, allow_nan=False)
        # No stale unscale state survives; the next ordinary task can commit.
        recovered = train_step(model, opt, units, cfg, .03, .5)
        self.assertFalse(recovered['amp']['skipped_update'])

    def test_first_forward_finite_guard_skips_with_initialized_scaler(self):
        cfg, units = fixture()
        model = Toy()
        opt = optimizer(model)
        opt._v31810_scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        def fail(_module, _arguments, _outputs):
            raise FloatingPointError('Nonfinite window ASP logits')
        hook = model.register_forward_hook(fail)
        stats = train_step(model, opt, units, cfg, .03, .5)
        hook.remove()
        self.assertTrue(stats['amp']['skipped_update'])
        self.assertEqual(stats['amp']['scale'], 64.)
        self.assertFalse(opt.state)

    def test_B_overflow_restores_virtual_parameters_and_preserves_moments(self):
        cfg, units = fixture()
        model = Toy()
        opt = optimizer(model)
        opt._v31810_scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        train_step(model, opt, units, cfg, .03, .5)
        before = copy.deepcopy(model.state_dict())
        moments = copy.deepcopy(opt.state_dict())
        state = rng_state()
        calls = []
        def overflow_only_B(gradient):
            calls.append(1)
            return gradient if len(calls) <= 2 else gradient * float('inf')
        hook = model.head[-1].weight.register_hook(overflow_only_B)
        result = train_step(model, opt, units, cfg, .03, .5)
        hook.remove()
        self.assertTrue(result['amp']['skipped_update'])
        self.assertEqual(result['amp']['scale'], 64.)
        assert_state_equal(self, before, model.state_dict())
        assert_state_equal(self, moments, opt.state_dict())
        assert_state_equal(self, state, rng_state())
        self.assertFalse(train_step(model, opt, units, cfg, .03, .5)['amp']['skipped_update'])

    def test_enabled_amp_zero_alpha_matches_unscaled_same_task(self):
        cfg, units = fixture()
        a, b = Toy(), Toy()
        b.load_state_dict(a.state_dict())
        oa, ob = optimizer(a), optimizer(b)
        ob._v31810_scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        train_step(a, oa, units, cfg, 0., .5)
        train_step(b, ob, units, cfg, 0., .5)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-6)

    def test_exact_resume_includes_optimizer_scaler_and_rng(self):
        cfg, units = fixture()
        a = Toy(.25)
        oa = optimizer(a)
        oa._v31810_scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        train_step(a, oa, units, cfg, .03, .5)
        checkpoint = copy.deepcopy(dict(model=a.state_dict(), optimizer=oa.state_dict(),
                                        scaler=oa._v31810_scaler.state_dict(), rng=rng_state()))
        expected = train_step(a, oa, units, cfg, .03, .5)
        b = Toy(.25)
        b.load_state_dict(checkpoint['model'])
        ob = optimizer(b)
        ob.load_state_dict(checkpoint['optimizer'])
        ob._v31810_scaler = torch.amp.GradScaler('cpu')
        ob._v31810_scaler.load_state_dict(checkpoint['scaler'])
        restore_rng(checkpoint['rng'])
        actual = train_step(b, ob, units, cfg, .03, .5)
        assert_state_equal(self, a.state_dict(), b.state_dict())
        assert_state_equal(self, oa.state_dict(), ob.state_dict())
        self.assertEqual(expected, actual)

    def test_source_isolation_uses_canonical_union_identity(self):
        _, units = fixture()
        units[-1][0]['canonical_source'] = units[0][0]['canonical_source']
        with self.assertRaisesRegex(ValueError, 'repeats'):
            validate_units(units)

    def test_transaction_never_copies_frozen_base(self):
        model = Toy()
        tx = Transaction(model)
        self.assertNotIn('frozen', [name for name, _ in tx.parameters])
        self.assertEqual(sum(p.numel() for p in tx.original),
                         sum(p.numel() for p in model.parameters() if p.requires_grad))

    def test_actual_window_aasist_checkpointed_fusion_joint_transaction(self):
        from w2v_v3189.test_model import tiny_cfg
        from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayout
        from .model import Detector, optimizer_for
        cfg, units = fixture(copies=1)
        settings = tiny_cfg()
        settings.update(cfg, warm_epochs=1, lora_dropout=0.)
        for unit in units:
            unit[0]['wave'] = np.tile(unit[0]['wave'], 10)
        model = Detector(TinyFrontend(), TinyEncoder(), settings, TinyLayout)
        opt = optimizer_for(model, settings)
        before_frozen = {name: p.detach().clone() for name, p in model.named_parameters()
                         if not p.requires_grad and not name.endswith(('.lora_a', '.lora_b'))}
        warm = train_step(model, opt, units, settings, 0., 0., joint=False)
        joint = train_step(model, opt, units, settings, .003, .5, joint=True, audit=True)
        self.assertEqual(model.head.aasist.out_layer.in_features, 384)
        self.assertFalse(hasattr(model, 'pcl_scale'))
        self.assertTrue(np.isfinite(joint['loss']))
        self.assertEqual(warm['encoded_views'], 16)
        self.assertEqual(joint['encoded_views'], 32)
        for name, p in model.named_parameters():
            if name in before_frozen:
                torch.testing.assert_close(before_frozen[name], p, rtol=0, atol=0)
        self.assertTrue(any('lora' in key and value > 0
                            for key, value in joint['meta']['module_outer_update_norms'].items()))
        self.assertFalse(any(layer._forward_hooks for layer in model.encoder.layers))


if __name__ == '__main__':
    unittest.main()
