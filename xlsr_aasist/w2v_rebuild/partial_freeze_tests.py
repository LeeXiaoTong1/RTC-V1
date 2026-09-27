"""Partial-backbone regression tests using a small random real HF encoder.

No pretrained files, downloads or speech data are needed. These tests validate
optimization and checkpoint compatibility, not model quality on the task.
"""
import copy
import io
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from .model import Detector


def small_detector(checkpointing=False):
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
    config = Wav2Vec2BertConfig(
        hidden_size=16, num_hidden_layers=3, num_attention_heads=2,
        intermediate_size=32, feature_projection_input_dim=160,
        conv_depthwise_kernel_size=3, layerdrop=0., apply_spec_augment=False,
        hidden_dropout=.1, activation_dropout=.1, attention_dropout=.1,
        feat_proj_dropout=.1, conformer_conv_dropout=.1)
    backbone = Wav2Vec2BertModel(config)
    if checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        backbone.gradient_checkpointing_disable()
    return Detector(backbone)


class PartialFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(1204)
        self.features = torch.randn(2, 24, 160)
        self.mask = torch.ones(2, 24, dtype=torch.long)
        self.labels = torch.tensor([0, 1])

    def assert_frozen(self, module):
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
        self.assertTrue(all(not child.training for child in module.modules()))

    def test_invalid_count_does_not_modify_model(self):
        model = small_detector()
        for value in (-1, 4, 1.5, True, '1', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model.configure_trainable_layers(value)
        self.assertIsNone(model.trainable_encoder_layers)

    def test_modes_and_full_unfreeze(self):
        model = small_detector(True)
        self.assertIs(model.configure_trainable_layers(1), model)
        for mode in (True, False, True, False, True):
            self.assertIs(model.train(mode), model)
            self.assertEqual(model.training, mode)
            self.assertEqual(model.backbone.training, mode)
            self.assertEqual(model.backbone.encoder.training, mode)
            self.assert_frozen(model.backbone.feature_projection)
            for layer in model.backbone.encoder.layers[:-1]:
                self.assert_frozen(layer)
            self.assertFalse(model.backbone.encoder.dropout.training)
            self.assertTrue(all(m.training == mode for m in model.backbone.encoder.layers[-1].modules()))
            self.assertTrue(all(m.training == mode for m in model.head.modules()))
        model.configure_trainable_layers(0)
        self.assert_frozen(model.backbone)
        self.assertTrue(model.head.training)
        model.configure_trainable_layers(3)
        self.assertTrue(all(m.training for m in model.modules()))
        for name, p in model.backbone.named_parameters():
            self.assertEqual(p.requires_grad, name != 'masked_spec_embed')
        model.eval().configure_trainable_layers(1)
        self.assertFalse(any(m.training for m in model.modules()))

    def test_frozen_parameters_unchanged_and_tail_updates(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                model = small_detector(checkpointing).configure_trainable_layers(1)
                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                frozen_outputs = []
                handle = model.backbone.encoder.layers[1].register_forward_hook(
                    lambda module, args, out: frozen_outputs.append(out[0].requires_grad))
                optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
                logits, _ = model(self.features, self.mask)
                F.cross_entropy(logits, self.labels).backward()
                handle.remove()
                self.assertTrue(frozen_outputs)
                self.assertFalse(any(frozen_outputs))
                tail = model.backbone.encoder.layers[-1]
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in tail.parameters()))
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        self.assertIsNone(p.grad, name)
                optimizer.step()
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
                self.assertTrue(any(not torch.equal(p, before[name]) for name, p in model.named_parameters()
                                    if name.startswith('backbone.encoder.layers.2.')))
                self.assertFalse(torch.equal(model.head.classifier.weight, before['head.classifier.weight']))

    def test_head_only_updates_and_frozen_stale_grad_is_cleared(self):
        model = small_detector(True)
        for p in model.backbone.parameters():
            p.grad = torch.ones_like(p)
        model.configure_trainable_layers(0)
        self.assert_frozen(model.backbone)
        before = copy.deepcopy(model.backbone.state_dict())
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        logits, _ = model(self.features, self.mask)
        F.cross_entropy(logits, self.labels).backward()
        self.assertIsNotNone(model.head.classifier.weight.grad)
        optimizer.step()
        self.assert_frozen(model.backbone)
        for name, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_legacy_keys_and_evaluation_roundtrip(self):
        model = small_detector(True).eval()
        keys = tuple(model.state_dict())
        with torch.no_grad():
            expected = model(self.features, self.mask)
        for count in (0, 1, 3):
            model.configure_trainable_layers(count).eval()
            self.assertEqual(tuple(model.state_dict()), keys)
            with torch.no_grad():
                actual = model(self.features, self.mask)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        clone = small_detector().configure_trainable_layers(1).eval()
        clone.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        with torch.no_grad():
            actual = clone(self.features, self.mask)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_default_behavior_and_explicit_full_training_agree(self):
        default = small_detector(True).train()
        explicit = copy.deepcopy(default).configure_trainable_layers(3)
        self.assertIsNone(default.trainable_encoder_layers)
        for name, p in default.backbone.named_parameters():
            self.assertEqual(p.requires_grad, name != 'masked_spec_embed')
        torch.manual_seed(317)
        expected = default(self.features, self.mask)[0]
        F.cross_entropy(expected, self.labels).backward()
        torch.manual_seed(317)
        actual = explicit(self.features, self.mask)[0]
        F.cross_entropy(actual, self.labels).backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for (name, a), (_, b) in zip(default.named_parameters(), explicit.named_parameters()):
            if a.requires_grad:
                self.assertIsNotNone(a.grad, name)
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_checkpointing_preserves_partial_gradients(self):
        plain = small_detector(False).configure_trainable_layers(1)
        checked = copy.deepcopy(plain)
        checked.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        for model in (plain, checked):
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.
            logits, _ = model(self.features, self.mask)
            F.cross_entropy(logits, self.labels).backward()
        for (name, a), (_, b) in zip(plain.named_parameters(), checked.named_parameters()):
            if a.requires_grad:
                self.assertIsNotNone(a.grad, name)
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)

    def test_no_forced_detach_of_input_gradients(self):
        model = small_detector(True).configure_trainable_layers(1)
        features = self.features.clone().requires_grad_()
        logits, _ = model(features, self.mask)
        F.cross_entropy(logits, self.labels).backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.backbone.encoder.layers[0].parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
