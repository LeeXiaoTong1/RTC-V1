"""Real detector suffix adapters, full-depth reader and checkpoint efficiency."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayer, TinyLayout
from w2v_v3189.test_model import tiny_cfg
from w2v_v3183r2.model import LoRALinear, RecomputedLayer
from .common import apply_partial, partial_state
from .model import Detector, adapter_inventory, optimizer_for


class ObservedLayer(TinyLayer):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.output_requires_grad = []

    def forward(self, x, layout, bias_cache):
        self.calls += 1
        output = super().forward(x, layout, bias_cache)
        self.output_requires_grad.append(output.requires_grad)
        return output


def detector(depth=4, count=2, *, checkpoints=True, warm=False):
    cfg = tiny_cfg(depth)
    cfg.update(lora_layers=count, checkpoint_layers=count if checkpoints else 0,
               fusion_checkpointing=checkpoints, lora_dropout=0.,
               backend_precision='fp32', warm_epochs=1 if warm else 0)
    encoder = TinyEncoder()
    encoder.layers = nn.ModuleList(ObservedLayer() for _ in range(depth))
    return Detector(TinyFrontend(), encoder, cfg, TinyLayout)


def waves(frames=(12, 17)):
    rng = np.random.default_rng(3181016)
    return [rng.normal(0, .1, 400 + 320 * (count - 1)).astype(np.float32) for count in frames]


def loss_from(sequences):
    # Nonuniform channels avoid LayerNorm symmetries masking adapter gradients.
    return sum((value * torch.linspace(.1, 1.1, 128)).square().mean() for value in sequences)


class PartialLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(3181016)

    def test_actual_sixty_layer_inventory_selects_only_indices_44_through_59(self):
        model = detector(depth=60, count=16)
        audit = adapter_inventory(model)
        self.assertEqual(audit['lora_layer_indices_zero_based'], list(range(44, 60)))
        self.assertEqual(audit['frozen_prefix_indices_zero_based'], list(range(44)))
        self.assertEqual(audit['fusion_layers_one_based'], list(range(1, 61)))
        self.assertEqual(audit['recomputed_layer_indices_zero_based'], list(range(44, 60)))
        self.assertEqual(audit['lora_parameters'], 16 * 4 * 2 * 16 * model.cfg['lora_rank'])
        optimizer = optimizer_for(model, model.cfg)
        groups = [group for group in optimizer.param_groups if group['name'].startswith('lora_')]
        self.assertEqual({group['name'] for group in groups}, {'lora_late'})
        self.assertEqual(sum(parameter.numel() for group in groups for parameter in group['params']),
                         audit['lora_parameters'])
        self.assertTrue(all(group['initial_lr'] == model.cfg['lora_lr'] for group in groups))

    def test_warmup_trains_reader_from_every_layer_without_adapter_or_encoder_graph(self):
        model = detector(warm=True)
        outputs, indices = {}, []
        original = model.fusion.project_layer
        def capture(hidden, index, weights):
            indices.append(index)
            result = original(hidden, index, weights)
            outputs[index] = result
            self.assertFalse(hidden.requires_grad)
            return result
        with patch.object(model.fusion, 'project_layer', side_effect=capture):
            sequences = model.encode(waves())
        self.assertEqual(indices, [1, 2, 3, 4])
        prefix_grad = torch.autograd.grad(outputs[1].square().mean(), model.fusion.projection.weight,
                                          retain_graph=True)[0]
        self.assertGreater(float(prefix_grad.abs().sum()), 0.)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                  if name.endswith(('.lora_a', '.lora_b'))}
        optimizer = optimizer_for(model, model.cfg)
        loss_from(sequences).backward(); optimizer.step()
        for name, parameter in model.named_parameters():
            if name in before:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        self.assertEqual(adapter_inventory(model)['active_lora_parameters'], 0)
        for layer in model.encoder.layers:
            inner = layer.layer if isinstance(layer, RecomputedLayer) else layer
            self.assertEqual(inner.calls, 1)
            self.assertFalse(any(inner.output_requires_grad))

    def test_joint_keeps_prefix_frozen_without_recompute_and_updates_late_adapters(self):
        model = detector()
        base = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                if name.startswith(('frontend.', 'encoder.')) and not name.endswith(('.lora_a', '.lora_b'))}
        before_b = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                    if name.endswith('.lora_b')}
        seen = []
        original = model.fusion.project_layer
        def capture(hidden, index, weights):
            seen.append((index, hidden.requires_grad))
            return original(hidden, index, weights)
        optimizer = optimizer_for(model, model.cfg)
        with patch.object(model.fusion, 'project_layer', side_effect=capture):
            loss_from(model.encode(waves())).backward()
        self.assertEqual(seen, [(1, False), (2, False), (3, True), (4, True)])
        self.assertGreater(float(model.fusion.projection.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.fusion.layer_logits.grad.abs().sum()), 0.)
        b_gradients = [parameter.grad for name, parameter in model.named_parameters() if name in before_b]
        self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in b_gradients))
        self.assertGreater(sum(float(value.abs().sum()) for value in b_gradients), 0.)
        optimizer.step()
        self.assertTrue(any(not torch.equal(parameter, before_b[name])
                            for name, parameter in model.named_parameters() if name in before_b))
        for name, parameter in model.named_parameters():
            if name in base:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, base[name], rtol=0, atol=0)
        self.assertEqual([layer.calls for layer in model.encoder.layers[:2]], [1, 1])
        self.assertTrue(all(layer.layer.calls == 2 for layer in model.encoder.layers[2:]))
        self.assertFalse(any(layer._forward_hooks for layer in model.encoder.layers))
        optimizer.zero_grad(set_to_none=True)
        loss_from(model.encode(waves())).backward()
        self.assertFalse(any(layer._forward_hooks for layer in model.encoder.layers))

    def test_checkpointing_changes_recomputation_not_outputs_or_gradients(self):
        checkpointed = detector()
        direct = copy.deepcopy(checkpointed)
        direct.cfg = dict(direct.cfg, checkpoint_layers=0)
        direct.fusion.checkpointing = False
        direct.set_phase(True)
        left, right = checkpointed.encode(waves()), direct.encode(waves())
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        loss_from(left).backward(); loss_from(right).backward()
        for (a_name, a), (b_name, b) in zip(checkpointed.named_parameters(), direct.named_parameters()):
            self.assertEqual(a_name, b_name)
            if a.grad is None:
                self.assertIsNone(b.grad)
            else:
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)
        self.assertEqual([layer.layer.calls for layer in checkpointed.encoder.layers[2:]], [2, 2])
        self.assertEqual([layer.layer.calls for layer in direct.encoder.layers[2:]], [1, 1])

    def test_full_waveform_and_overlapping_statistics_readout_are_retained(self):
        model = detector(checkpoints=False).eval()
        with torch.no_grad():
            sequences = model.encode(waves((12, 193)))
            self.assertEqual([len(value) for value in sequences], [12, 193])
            logits, weights = model.head(sequences, return_weights=True)
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(model.head.aasist.out_layer.in_features, 384)
        self.assertEqual([len(value) for value in weights], [1, 3])
        ranges, _ = model.head.window_plan(193, torch.device('cpu'))
        covered = set(index for begin, end in ranges for index in range(begin, end))
        self.assertEqual(covered, set(range(193)))
        self.assertTrue(any(ranges[i + 1][0] < ranges[i][1] for i in range(len(ranges) - 1)))

    def test_checkpoint_restore_rejects_wrong_adapter_depth_before_any_tensor_changes(self):
        partial, full = detector(depth=60, count=16), detector(depth=60, count=60)
        for source, target in ((partial, full), (full, partial)):
            old = partial_state(target)
            with self.assertRaisesRegex(ValueError, 'partial keys differ'):
                apply_partial(target, partial_state(source))
            for name, value in partial_state(target).items():
                torch.testing.assert_close(value, old[name], rtol=0, atol=0)
        saved = partial_state(partial)
        with torch.no_grad():
            for module in partial.encoder.modules():
                if isinstance(module, LoRALinear):module.lora_b.add_(.1)
        apply_partial(partial, saved)
        for name, value in partial_state(partial).items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
