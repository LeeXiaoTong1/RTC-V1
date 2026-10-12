"""Behavioral acceptance for the all-layer reader and staged LoRA adaptation."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .fusion import LayerFusion
from .model import LoRALinear, optimizer_for
from .test_support import tiny_cfg, tiny_model


def reference(fusion, states):
    """Materialized mathematical reference, independent of streaming/checkpoint code."""
    projected = []
    for state in states:
        normalized = F.layer_norm(state, (state.shape[-1],), eps=fusion.input_norm.eps)
        z = F.linear(normalized, fusion.projection.weight, fusion.projection.bias)
        gate = F.linear(z, fusion.gating.gate.weight, fusion.gating.gate.bias)
        value = F.linear(z, fusion.gating.value.weight, fusion.gating.value.bias)
        projected.append(F.silu(gate) * value)
    weights = fusion.layer_logits.softmax(0)
    return (torch.stack(projected) * weights[:, None, None, None]).sum(0)


class TestFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(8318)

    def test_checkpoint_stream_matches_full_reader_and_all_gradients(self):
        stream = LayerFusion(16, 6, projection=8, checkpointing=True).train()
        plain = copy.deepcopy(stream)
        states = [torch.randn(2, 19, 16, requires_grad=i >= 4) for i in range(6)]
        reference_states = [x.detach().clone().requires_grad_(x.requires_grad) for x in states]
        weights = stream.weights()
        actual = sum(stream.project_layer(x, i + 1, weights) for i, x in enumerate(states))
        expected = reference(plain, reference_states)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        target = torch.randn_like(actual)
        (actual * target).sum().backward()
        (expected * target).sum().backward()
        for (name, actual_param), (other_name, expected_param) in zip(stream.named_parameters(), plain.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(actual_param.grad, name)
            self.assertTrue(torch.isfinite(actual_param.grad).all(), name)
            torch.testing.assert_close(actual_param.grad, expected_param.grad, atol=3e-5, rtol=3e-5, msg=name)
        self.assertTrue((stream.layer_logits.grad.abs() > 1e-7).all())
        for actual_state, expected_state in zip(states, reference_states):
            if actual_state.requires_grad:
                torch.testing.assert_close(actual_state.grad, expected_state.grad, atol=3e-6, rtol=3e-5)
            else:
                self.assertIsNone(actual_state.grad)
                self.assertIsNone(expected_state.grad)

    def test_uniform_initial_weights_and_unrestricted_early_layer_selection(self):
        fusion = LayerFusion(16, 60, projection=8, checkpointing=False).eval()
        self.assertEqual(tuple(fusion.layers), tuple(range(1, 61)))
        torch.testing.assert_close(fusion.weights(), torch.full((60,), 1 / 60))
        with torch.no_grad():
            fusion.layer_logits.fill_(-30)
            fusion.layer_logits[0] = 30
        weights = fusion.weights()
        self.assertGreater(float(weights[0]), .999999)
        self.assertLess(float(weights[-1]), 1e-20)
        states = [torch.randn(1, 4, 16) for _ in range(60)]
        actual = sum(fusion.project_layer(x, i + 1, weights) for i, x in enumerate(states))
        expected = fusion.gating(fusion.projection(fusion.input_norm(states[0])))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_initial_swiglu_is_identity_but_parameters_receive_gradients(self):
        fusion = LayerFusion(16, 3, projection=8, checkpointing=True)
        z = torch.randn(2, 7, 8, requires_grad=True)
        out = fusion.gating(z)
        torch.testing.assert_close(out, z, atol=2e-6, rtol=2e-6)
        (out * torch.randn_like(out)).sum().backward()
        for name, parameter in fusion.gating.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    @staticmethod
    def waves():
        return [np.random.default_rng(i).normal(0, .05, n).astype(np.float32)
                for i, n in enumerate((9000, 14000))]

    def test_warm_keeps_inherited_lora_active_and_freezes_only_its_updates(self):
        cfg = tiny_cfg()
        cfg['lora_dropout'] = 0.
        model = tiny_model(cfg)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.lora_b.normal_(0, .15)
        model.set_phase(False)
        model.eval()
        adapters = [m for m in model.modules() if isinstance(m, LoRALinear)]
        self.assertTrue(all(m.enabled for m in adapters))
        self.assertTrue(all(not p.requires_grad for m in adapters for p in (m.lora_a, m.lora_b)))
        with torch.no_grad():
            enabled = model.encode(self.waves())
            for module in adapters:
                module.enabled = False
            disabled = model.encode(self.waves())
        self.assertTrue(any(not torch.allclose(a, b, atol=1e-6, rtol=1e-6) for a, b in zip(enabled, disabled)))
        model.set_phase(False)
        optimizer = optimizer_for(model, cfg)
        lora_before = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith(('.lora_a', '.lora_b'))}
        model.train()
        F.cross_entropy(model(self.waves()), torch.tensor([0, 1])).backward()
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if n in lora_before))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fusion.parameters()))
        optimizer.step()
        self.assertTrue(all(torch.equal(p, lora_before[n]) for n, p in model.named_parameters() if n in lora_before))
        optimizer.zero_grad(set_to_none=True)
        model.set_phase(True)
        F.cross_entropy(model(self.waves()), torch.tensor([0, 1])).backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(p, lora_before[n]) for n, p in model.named_parameters() if n in lora_before))

    def test_bn_updates_in_both_phases_and_frozen_base_never_gets_gradients(self):
        model = tiny_model(tiny_cfg())
        norm = model.head.aasist.first_bn
        for joint in (False, True):
            model.zero_grad(set_to_none=True)
            model.set_phase(joint)
            model.train()
            self.assertTrue(norm.training)
            self.assertAlmostEqual(norm.momentum, .01)
            previous = norm.num_batches_tracked.detach().clone()
            F.cross_entropy(model(self.waves()), torch.tensor([0, 1])).backward()
            self.assertGreater(int(norm.num_batches_tracked), int(previous))
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
            if joint:
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for n, p in model.named_parameters() if n.endswith('.lora_b')))
            self.assertTrue(all(not layer._forward_hooks for layer in model.encoder.layers))

    def test_padding_invariance_and_hook_cleanup_after_failure(self):
        model = tiny_model(tiny_cfg())
        model.set_phase(True)
        model.eval()
        waves = self.waves()
        with torch.no_grad():
            together = model(waves)
            separate = torch.cat([model([x]) for x in waves])
        torch.testing.assert_close(together, separate, atol=3e-5, rtol=3e-5)
        with patch.object(model.fusion, 'project_layer', side_effect=RuntimeError('injected fusion failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected fusion failure'):
                model(waves)
        self.assertTrue(all(not layer._forward_hooks for layer in model.encoder.layers))
        with torch.no_grad():
            recovered = model(waves)
        torch.testing.assert_close(recovered, together, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
