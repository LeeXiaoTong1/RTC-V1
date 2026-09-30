"""Small real-HF and numerical tests; no external data or pretrained download."""
import copy
from dataclasses import asdict
import math
import unittest
from unittest.mock import patch
import torch
from torch import nn
from .model import Detector, HeadConfig, MultiConvHead, diversity_cka, microbatches
from .step import ActivationOffload, loss_function, predict, supervised_step

torch.set_num_threads(1)


def small_head(dropout=0.):
    return HeadConfig(input_dim=16, projection=8, expansion=32, kernels=(3, 5, 7, 9),
                      merge_kernel=5, dropout=dropout)


def tiny_detector(checkpointing=True, dropout=0.):
    from transformers import Wav2Vec2BertConfig
    cfg = Wav2Vec2BertConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=32, feature_projection_input_dim=160,
                           conv_depthwise_kernel_size=7, hidden_dropout=dropout,
                           attention_dropout=dropout, activation_dropout=dropout,
                           feat_proj_dropout=0., layerdrop=0., apply_spec_augment=False,
                           num_conv_pos_embedding_groups=2)
    return Detector.from_config(cfg.to_dict(), asdict(small_head(dropout)), checkpointing)


def examples(dim=160, lengths=(13, 15, 13, 15, 17, 17)):
    return [dict(features=torch.randn(1, length, dim), mask=torch.ones(1, length).long(),
                 label=i % 2, noisy=i >= 3) for i, length in enumerate(lengths)]


class ToyDetector(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(5, 16) for _ in range(3)])
        self.head = MultiConvHead(small_head(dropout))
        self.calls = 0

    def forward(self, features, mask):
        self.calls += 1
        return self.head([layer(features) for layer in self.layers], mask)


class ModelStepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(129)

    def test_global_cka_gradient_matches_single_full_batch(self):
        for cka_weight in (.17, 0.):
            with self.subTest(cka_weight=cka_weight):
                reference = ToyDetector()
                actual = copy.deepcopy(reference)
                ex = examples(5, (13,) * 8)
                labels = torch.tensor([e['label'] for e in ex])
                noisy = torch.tensor([e['noisy'] for e in ex])
                ow, nw = torch.tensor([.7, 1.4]), torch.tensor([.55, 2.2])
                z, h = reference(torch.cat([e['features'] for e in ex]), torch.cat([e['mask'] for e in ex]))
                loss, _ = loss_function(z, h, labels, noisy, ow, .45, cka_weight, nw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
                torch.optim.SGD(reference.parameters(), lr=.07).step()
                stats, scores = supervised_step(actual, ex, torch.optim.SGD(actual.parameters(), lr=.07),
                    ow, 'cpu', amp='none', noisy_weight=.45, cka_weight=cka_weight,
                    noisy_class_weights=nw, microbatch=2, frame_budget=26)
                self.assertEqual(actual.calls, 4)
                self.assertAlmostEqual(stats['loss'], float(loss), places=6)
                torch.testing.assert_close(scores, z.detach(), atol=2e-6, rtol=2e-5)
                for a, b in zip(reference.parameters(), actual.parameters()):
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_variable_lengths_global_gradient_and_prediction_order(self):
        reference = ToyDetector()
        actual = copy.deepcopy(reference)
        ex = examples(5)
        z, h = zip(*(reference(e['features'], e['mask']) for e in ex))
        loss, _ = loss_function(torch.cat(z), torch.cat(h), torch.tensor([e['label'] for e in ex]),
            torch.tensor([e['noisy'] for e in ex]), torch.tensor([.7, 1.4]), .4, .2, torch.tensor([1., 2.]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        torch.optim.SGD(reference.parameters(), lr=.03).step()
        stats, scores = supervised_step(actual, ex, torch.optim.SGD(actual.parameters(), lr=.03),
            torch.tensor([.7, 1.4]), 'cpu', cka_weight=.2, noisy_weight=.4,
            noisy_class_weights=torch.tensor([1., 2.]), microbatch=4)
        self.assertEqual(actual.calls, 3)
        torch.testing.assert_close(scores, torch.cat(z).detach(), atol=2e-6, rtol=2e-5)
        for a, b in zip(reference.parameters(), actual.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        predicted = predict(actual.eval(), ex, 'cpu', amp='none')
        expected = torch.cat([actual(e['features'], e['mask'])[0] for e in ex])
        torch.testing.assert_close(predicted, expected, atol=2e-6, rtol=2e-5)

    def test_dropout_uses_one_forward_and_preserves_rng(self):
        reference = ToyDetector(.2).train()
        actual = copy.deepcopy(reference)
        ex = examples(5, (13, 14, 15, 16, 17, 18))
        torch.manual_seed(999)
        z, h = zip(*(reference(e['features'], e['mask']) for e in ex))
        loss, _ = loss_function(torch.cat(z), torch.cat(h), torch.tensor([e['label'] for e in ex]),
            torch.tensor([e['noisy'] for e in ex]), torch.ones(2))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        torch.optim.SGD(reference.parameters(), lr=.03).step()
        rng = torch.get_rng_state()
        torch.manual_seed(999)
        supervised_step(actual, ex, torch.optim.SGD(actual.parameters(), lr=.03), torch.ones(2), 'cpu')
        self.assertEqual(actual.calls, len(ex))
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for a, b in zip(reference.parameters(), actual.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_real_hf_freeze_then_joint_checkpointing(self):
        model = tiny_detector(checkpointing=True, dropout=.1)
        ex = examples()
        for count in (0, 1):
            with self.subTest(trainable_layers=count):
                model.configure_trainable_layers(count).train()
                before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
                self.assertFalse(model.backbone.encoder.layers[0].training)
                self.assertEqual(model.backbone.encoder.layers[-1].training, bool(count))
                calls = []
                hook = model.backbone.register_forward_hook(lambda *_: calls.append(1))
                layer_calls = [0, 0]
                layer_hooks = []
                for index, layer in enumerate(model.backbone.encoder.layers):
                    def record(_module, _args, i=index):
                        layer_calls[i] += 1
                    layer_hooks.append(layer.register_forward_pre_hook(record))
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
                stats, _ = supervised_step(model, ex, opt, torch.ones(2), 'cpu', microbatch=4)
                hook.remove()
                for layer_hook in layer_hooks:
                    layer_hook.remove()
                self.assertEqual(len(calls), 3)
                self.assertEqual(layer_calls, [3, 6 if count else 3])
                self.assertTrue(math.isfinite(stats['grad_norm']))
                if count:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.backbone.encoder.layers[-1].parameters()))
                for name, p in model.named_parameters():
                    if name in before:
                        torch.testing.assert_close(p, before[name], rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_original_encoder_only(self):
        model = tiny_detector().eval()
        state = {'model': model.state_dict(), **model.architecture()}
        restored = Detector.from_checkpoint(state).eval()
        features, mask = torch.randn(1, 19, 160), torch.ones(1, 19).long()
        torch.testing.assert_close(model(features, mask)[0], restored(features, mask)[0])
        original = {'schema': 'rtc_w2v_rebuild_v1', 'model_config': state['model_config'],
                    'model': {**{k: v for k, v in state['model'].items() if k.startswith('backbone.')},
                              'head.this_is_not_multiconv': torch.ones(1)}}
        copied = Detector.from_original(original, asdict(small_head()))
        for key, value in model.backbone.state_dict().items():
            torch.testing.assert_close(copied.backbone.state_dict()[key], value, rtol=0, atol=0)
        self.assertFalse(torch.equal(copied.head.projection.weight, model.head.projection.weight))

    def test_masked_head_padding_and_encoder_rejects_padding(self):
        head = MultiConvHead(small_head()).eval()
        hidden = [torch.randn(1, 13, 16) for _ in range(3)]
        expected = head(hidden, torch.ones(1, 13).long())
        actual = head([torch.cat((h, torch.randn(1, 11, 16) * 20), 1) for h in hidden],
                      torch.cat((torch.ones(1, 13), torch.zeros(1, 11)), 1).long())
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        with self.assertRaisesRegex(ValueError, 'exact-length'):
            tiny_detector()(torch.randn(1, 15, 160), torch.cat((torch.ones(1, 13), torch.zeros(1, 2)), 1).long())

    def test_cka_gradients_and_independent_class_costs(self):
        blocks = torch.randn(8, 4, 6, requires_grad=True)
        diversity_cka(blocks).backward()
        self.assertGreater(float(blocks.grad.abs().sum()), .001)
        logits = torch.zeros(8, 2, requires_grad=True)
        labels = torch.tensor([0, 1] * 4)
        noisy = torch.tensor([False] * 4 + [True] * 4)
        loss, _ = loss_function(logits, blocks, labels, noisy, torch.tensor([.5, 2.]), .5, 0., torch.tensor([1., 3.]))
        grad, = torch.autograd.grad(loss, logits)
        self.assertAlmostEqual(float(abs(grad[1, 1] / grad[0, 0])), 4.)
        self.assertAlmostEqual(float(abs(grad[5, 1] / grad[4, 0])), 3.)

    def test_nonfinite_aborts_without_optimizer_update(self):
        model = ToyDetector()
        before = copy.deepcopy(model.state_dict())
        ex = examples(5)
        ex[-1]['features'][0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            supervised_step(model, ex, torch.optim.SGD(model.parameters(), lr=.1), torch.ones(2), 'cpu')
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_ram_budget_and_exact_length_grouping(self):
        with patch('w2v_v3.step.available_host_bytes', return_value=8 * 1024**3):
            context = ActivationOffload(ToyDetector(), pin_memory=False)
            self.assertEqual(context.limit, 4 * 1024**3)
        with patch('w2v_v3.step.available_host_bytes', return_value=0):
            with self.assertRaises(MemoryError):
                ActivationOffload(ToyDetector(), pin_memory=False)
        batches = list(microbatches(examples(5), size=4, frame_budget=26))
        self.assertEqual([len(x[0]) for x in batches], [2, 1, 1, 1, 1])

    def test_offload_bypass_detaches_without_copying_storage(self):
        model = ToyDetector()
        with patch('w2v_v3.step.available_host_bytes', return_value=8 * 1024**3):
            context = ActivationOffload(model, pin_memory=False)
        parameter = next(model.parameters())
        _, packed = context.pack_hook(parameter)
        self.assertFalse(packed.requires_grad)
        self.assertIsNot(packed, parameter)
        self.assertEqual(packed.untyped_storage().data_ptr(), parameter.untyped_storage().data_ptr())
        x = torch.randn(3, requires_grad=True)
        with context:
            loss = (x.square() * 3).sum()
        loss.backward()
        torch.testing.assert_close(x.grad, 6 * x.detach())
        self.assertEqual(context.bytes, 0)

    def test_cpu_bf16_head_loss_and_gradients_are_finite(self):
        model = ToyDetector()
        ex = examples(5)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            stats, scores = supervised_step(model, ex, torch.optim.SGD(model.parameters(), lr=.01),
                                            torch.ones(2), 'cpu', amp='none')
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(math.isfinite(stats['grad_norm']))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA host offload requires GPU hardware')
    def test_cuda_activation_offload_matches_device_graph(self):
        reference = ToyDetector().cuda()
        actual = copy.deepcopy(reference)
        ex = examples(5)
        expected, z = supervised_step(reference, ex, torch.optim.SGD(reference.parameters(), lr=.03),
            torch.ones(2), 'cuda', amp='none', offload_activations=False)
        stats, scores = supervised_step(actual, ex, torch.optim.SGD(actual.parameters(), lr=.03),
            torch.ones(2), 'cuda', amp='none', activation_budget_gib=.5)
        self.assertGreater(stats['activation_offload_gib'], 0)
        self.assertAlmostEqual(stats['loss'], expected['loss'], places=5)
        torch.testing.assert_close(scores, z, rtol=2e-5, atol=2e-6)
        for a, b in zip(reference.parameters(), actual.parameters()):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
