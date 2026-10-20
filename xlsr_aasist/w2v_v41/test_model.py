import unittest
import numpy as np
import torch
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertConfig, Wav2Vec2BertModel
from .model import Detector, MultiHeadStatistics, adapter_inventory, optimizer_for


def tiny_model(checkpointing=True, depth=2):
    config = Wav2Vec2BertConfig(hidden_size=16, num_hidden_layers=depth, num_attention_heads=2,
        intermediate_size=32, feature_projection_input_dim=160, conv_depthwise_kernel_size=3,
        position_embeddings_type='relative_key', left_max_position_embeddings=4,
        right_max_position_embeddings=2, hidden_dropout=0., attention_dropout=0.,
        conformer_conv_dropout=.1, layerdrop=.1, apply_spec_augment=False, add_adapter=False)
    cfg = dict(device='cpu', encoder_dim=16, encoder_layers=depth, lora_layers=depth, lora_rank=4,
        lora_alpha=8., lora_dropout=.05, warm_epochs=1, mlp_dims=[24, 16, 8],
        checkpointing=checkpointing, checkpoint_layers=depth, fusion_checkpointing=True,
        whole_max_frames=3000, backend_precision='fp32', classifier_dropout=0., pool_heads=4,
        lora_lr=1e-3, head_lr=1e-3, classifier_lr=1e-3, layer_mix_lr=1e-3, weight_decay=.01)
    return Detector(Wav2Vec2BertModel(config), cfg, SeamlessM4TFeatureExtractor()), cfg


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_hf_forward_backward_and_base_freeze(self):
        torch.manual_seed(17)
        model, cfg = tiny_model()
        optimizer = optimizer_for(model, cfg)
        waves = [np.random.default_rng(n).normal(0, .1, n).astype(np.float32) for n in (3920, 7000)]
        base = {n: p.detach().clone() for n, p in model.backbone.named_parameters() if not n.endswith(('.lora_a', '.lora_b'))}
        for joint in (False, True):
            model.set_phase(joint)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(waves)
            self.assertEqual(tuple(logits.shape), (2, 2))
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
            active = [p for n, p in model.named_parameters() if n.endswith('.lora_b')]
            self.assertEqual(any(p.grad is not None and bool(p.grad.abs().sum()) for p in active), joint)
            optimizer.step()
            self.assertTrue(all(not layer._forward_hooks for layer in model.encoder.layers))
        for n, p in model.backbone.named_parameters():
            if n in base:
                torch.testing.assert_close(p, base[n], rtol=0, atol=0)
                self.assertIsNone(p.grad)
        self.assertEqual(adapter_inventory(model)['lora_projections'], 8)

    def test_eval_padding_invariance_and_no_base_dropout(self):
        model, _ = tiny_model(False)
        model.set_phase(True)
        model.train()
        self.assertFalse(model.encoder.layers[0].layer.conv_module.dropout.training)
        self.assertTrue(model.encoder.layers[0].layer.self_attn.linear_q.drop.training)
        self.assertEqual(model.backbone.config.layerdrop, 0.)
        model.eval()
        short = np.random.default_rng(9).normal(size=7000).astype(np.float32)
        long = np.random.default_rng(10).normal(size=12000).astype(np.float32)
        with torch.no_grad():
            a, b = model([short]), model([short, long])
        torch.testing.assert_close(a[0], b[0], rtol=2e-4, atol=2e-5)

    def test_mhasp_ignores_padding_and_keeps_all_channels(self):
        pool = MultiHeadStatistics(16, 4)
        for p in pool.parameters():
            torch.nn.init.zeros_(p)
        x = torch.arange(48, dtype=torch.float32).reshape(1, 3, 16)
        mask = torch.tensor([[1, 1, 0]])
        first = pool(x, mask)
        x[:, -1] = 999999
        torch.testing.assert_close(first, pool(x, mask), rtol=0, atol=0)
        pieces = x[0, :2].reshape(2, 4, 4)
        expected = torch.cat([v for i in range(4) for v in (pieces[:, i].mean(0), pieces[:, i].std(0, correction=0))])
        torch.testing.assert_close(first[0], expected)
        self.assertEqual(tuple(first.shape), (1, 32))

    def test_fusion_initial_mass_and_partial_state(self):
        from .model import LayerFusion
        from .common import partial_state, apply_partial
        fusion = LayerFusion(16, 24)
        self.assertAlmostEqual(float(fusion.weights()[-1].detach()), .5, places=6)
        torch.testing.assert_close(fusion.weights()[:-1], torch.full((23,), .5/23))
        model, _ = tiny_model()
        state = partial_state(model)
        self.assertTrue(all(k.startswith(('head.', 'fusion.')) or k.endswith(('.lora_a', '.lora_b')) for k in state))
        self.assertFalse(any('.base.' in k for k in state))
        apply_partial(model, state)

    def test_checkpoint_recomputation_preserves_values_and_gradients(self):
        left, _ = tiny_model(True)
        right, _ = tiny_model(False)
        right.load_state_dict(left.state_dict())
        waves = [np.random.default_rng(n).normal(0, .1, n).astype(np.float32) for n in (6000, 8500)]
        results = []
        for model in (left, right):
            model.set_phase(True)
            model.train()
            torch.manual_seed(99)
            value = model(waves)
            value.square().sum().backward()
            results.append((value.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        self.assertEqual(set(results[0][1]), set(results[1][1]))
        for name, gradient in results[0][1].items():
            torch.testing.assert_close(gradient, results[1][1][name], rtol=1e-6, atol=1e-7)

    def test_every_one_of_24_layers_receives_adapter_gradients(self):
        model, _ = tiny_model(True, depth=24)
        model.set_phase(True)
        waves = [np.random.default_rng(n).normal(0, .1, n).astype(np.float32) for n in (4500, 6200)]
        logits = model(waves)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        gradients = [p.grad for n, p in model.named_parameters() if n.endswith('.lora_b')]
        self.assertEqual(len(gradients), 96)
        self.assertTrue(all(g is not None and bool(g.abs().sum()) and bool(torch.isfinite(g).all()) for g in gradients))
        self.assertIsNotNone(model.fusion.layer_logits.grad)
        self.assertEqual(model.fusion.layer_logits.grad.numel(), 24)


if __name__ == '__main__':
    unittest.main()
