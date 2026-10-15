"""Full-depth LoRA, sample-local normalization and genuine temporal evidence."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayer, TinyLayout
from w2v_v3185.test_support import tiny_cfg as old_cfg
from w2v_v318.regions import RegionHead, windows
from w2v_v3183r2.model import RecomputedLayer
from .regions import StatisticsHead, moments
from .normalization import stable_std
from .aasist import SSLAASIST
from .model import Detector, optimizer_for


def tiny_cfg(depth=2):
    cfg = old_cfg()
    cfg.update(version='3.18.7', warm_epochs=0, epochs=20, classifier_lr=.001,
        pooling='asp', local_evidence=True, whole_max_frames=3000, temporal_stats=True,
        lora_layers=depth, fusion_layers=list(range(1, depth+1)), checkpoint_layers=depth,
        lora_early_lr=.00025, lora_middle_lr=.0005, lora_lr=.001)
    return cfg


def tiny_model(cfg):
    encoder = TinyEncoder()
    encoder.layers = nn.ModuleList([TinyLayer() for _ in cfg['fusion_layers']])
    return Detector(TinyFrontend(), encoder, cfg, TinyLayout)


class RecordingAASIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_layer = nn.Linear(160, 2)
        self.shapes, self.source_batches = [], []

    def evidence(self, x):
        self.shapes.append(tuple(x.shape))
        self.source_batches.append(x[:, 0, 0].tolist())
        mean = x.mean(1)
        return torch.cat((mean, mean[:, :32]), -1)

    def evidence_with_temporal(self, x):
        return self.evidence(x), stable_std(x[:, :, :64], 1)

    def classify(self, x):
        self.classifier_input = x
        return self.out_layer(x)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_coverage_no_padding_and_single_backend_pass(self):
        backend = RecordingAASIST()
        head = StatisticsHead(RegionHead(backend, True, 128, 64, 3))
        sequences = [torch.randn(n, 128) for n in (140, 25, 201)]
        output, weights = head(sequences, return_weights=True)
        self.assertEqual(output.shape, (3, 2))
        self.assertEqual(backend.out_layer.in_features, 384)
        self.assertEqual([len(w) for w in weights], [2, 1, 3])
        self.assertEqual(backend.shapes, [(1, 25, 128), (3, 128, 128), (2, 128, 128)])
        for seq, weight in zip(sequences, weights):
            self.assertAlmostEqual(weight.sum().item(), 1., places=6)
            ranges, _ = head.window_plan(len(seq), seq.device)
            self.assertEqual(ranges[-1][1], len(seq))
            self.assertEqual(ranges, tuple(windows(len(seq), 128, 64)))

    def test_single_window_retains_real_temporal_variation(self):
        backend = RecordingAASIST()
        head = StatisticsHead(RegionHead(backend, True, 128, 64, 3))
        seq = torch.arange(25.).reshape(-1, 1).expand(-1, 128).clone().requires_grad_()
        head([seq]); pooled = backend.classifier_input[0]
        torch.testing.assert_close(pooled[160:320], torch.zeros(160), rtol=0, atol=0)
        torch.testing.assert_close(pooled[320:], stable_std(seq[:, :64], 0))
        self.assertTrue(bool((pooled[320:] > 0).all()))
        pooled[320:].sum().backward()
        self.assertTrue(bool(torch.isfinite(seq.grad).all()))
        self.assertGreater(float(seq.grad.abs().sum()), 0.)

    def test_temporal_aggregation_uses_same_window_weights(self):
        backend = RecordingAASIST()
        head = StatisticsHead(RegionHead(backend, True, 16, 8, 3))
        with torch.no_grad(): head.attention_delta[-1].weight.normal_(0, .2)
        seq = torch.randn(40, 128)
        _, weights = head([seq], return_weights=True)
        ranges, _ = head.window_plan(len(seq), seq.device)
        evidence, temporal = [], []
        for a, b in ranges:
            mean = seq[a:b].mean(0)
            evidence.append(torch.cat((mean, mean[:32])))
            temporal.append(stable_std(seq[a:b, :64], 0))
        expected = torch.cat((*moments(torch.stack(evidence), weights[0]),
            (torch.stack(temporal) * weights[0][:, None]).sum(0)))
        torch.testing.assert_close(backend.classifier_input[0], expected)

    def test_temporal_ablation_is_320_without_extra_computation(self):
        backend = RecordingAASIST()
        head = StatisticsHead(RegionHead(backend, True, 128, 64, 3), temporal_stats=False)
        with patch.object(backend, 'evidence_with_temporal', side_effect=AssertionError('Unexpected temporal computation')):
            head([torch.randn(25, 128)])
        self.assertEqual(backend.classifier_input.shape, (1, 320))
        self.assertEqual(len(backend.shapes), 1)

    def test_no_batchnorm_or_dead_preactivation(self):
        model = tiny_model(tiny_cfg())
        self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()))
        self.assertEqual(model.head.aasist.first_bn.num_groups, 1)
        self.assertEqual(model.head.aasist.first_bn1.num_groups, 8)
        self.assertIsInstance(model.head.aasist.GAT_layer_T.bn, nn.LayerNorm)
        self.assertFalse(any('.encoder.' in n and '.bn1.' in n for n, _ in model.head.aasist.named_parameters()))

    def test_train_eval_and_batch_composition_invariance_without_dropout(self):
        torch.manual_seed(23)
        model = tiny_model(dict(tiny_cfg(), checkpoint_layers=0))
        for module in model.modules():
            if isinstance(module, nn.Dropout): module.p = 0.
        target, other, short = torch.randn(25, 128), torch.randn(25, 128) * 7 + 8, torch.randn(12, 128)
        values = []
        with torch.no_grad():
            for training in (True, False):
                model.train(training)
                solo = model.classify_sequences([target])[0]
                mixed = model.classify_sequences([short, target, other])[1]
                torch.testing.assert_close(solo, mixed, rtol=3e-4, atol=3e-5)
                values.append(solo)
        torch.testing.assert_close(values[0], values[1], rtol=1e-5, atol=1e-6)

    def test_true_e_t_statistics_with_one_graph_pass(self):
        backend = SSLAASIST(128).eval(); backend.LL = nn.Identity()
        captures = []
        hook = backend.GAT_layer_T.register_forward_pre_hook(lambda module, args: captures.append(args[0]))
        with torch.no_grad(): evidence, deviation = backend.evidence_with_temporal(torch.randn(2, 25, 128))
        hook.remove()
        self.assertEqual(len(captures), 1)
        self.assertEqual(evidence.shape, (2, 160))
        self.assertEqual(captures[0].shape, (2, 8, 64))
        torch.testing.assert_close(deviation, stable_std(captures[0], 1))

    def test_full_depth_groups_and_optimizer_coverage_receive_gradients(self):
        cfg = dict(tiny_cfg(3), window=32, hop=16, window_batch=5)
        model = tiny_model(cfg)
        self.assertEqual(set(model.lora_inventory), {'0', '1', '2'})
        self.assertTrue(all(len(names) == 4 for names in model.lora_inventory.values()))
        with torch.no_grad():
            model.head.attention_delta[-1].weight.normal_(0, .01)
            for name, p in model.named_parameters():
                if name.endswith('.lora_b'): p.normal_(0, .01)
        optimizer = optimizer_for(model, cfg)
        expected = {'lora_early', 'lora_middle', 'lora_late', 'aasist', 'classifier', 'evidence', 'pooling', 'fusion', 'layer_mix'}
        self.assertEqual({g['name'] for g in optimizer.param_groups}, expected)
        for group in optimizer.param_groups:
            if group['name'].startswith('lora_'):
                self.assertEqual(group['role'], 'lora')
                self.assertEqual(group['lr'], {'lora_early': .00025, 'lora_middle': .0005, 'lora_late': .001}[group['name']])
        logits = model([torch.randn(18000), torch.randn(25000)])
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        for name in expected:
            grads = [p.grad for g in optimizer.param_groups if g['name'] == name for p in g['params'] if p.grad is not None]
            self.assertTrue(grads, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, name)
        self.assertFalse(any(p.grad is not None for p in model.parameters() if not p.requires_grad))

    def test_references_skip_classifier_and_reencoding(self):
        model = tiny_model(tiny_cfg()).eval()
        waves = [torch.randn(n) for n in (18000, 25000, 18000)]
        with torch.no_grad(), patch.object(model, 'encode', wraps=model.encode) as encode:
            sequences = model.encode(waves)
            self.assertEqual(model.classify_sequences(sequences[:2]).shape, (2, 2))
            self.assertEqual(encode.call_count, 1)
            self.assertEqual(len(sequences), 3)

    def test_frontend_guard_and_structure_contract(self):
        with self.assertRaisesRegex(ValueError, 'overlapping-window ASP'): tiny_model(dict(tiny_cfg(), pooling='whole'))
        model = tiny_model(dict(tiny_cfg(), whole_max_frames=12))
        with self.assertRaisesRegex(ValueError, 'Frontend frame limit exceeded'): model.encode([torch.randn(18000)])

    def test_fresh_backbone_complete_classifier_and_frozen_warmup(self):
        frontend, encoder = TinyFrontend(), TinyEncoder()
        before = {n: p.detach().clone() for n, p in frontend.named_parameters()}
        model = Detector(frontend, encoder, dict(tiny_cfg(), warm_epochs=1, fusion_checkpointing=False), TinyLayout)
        self.assertFalse(model.joint); self.assertFalse(model.fusion.checkpointing)
        self.assertEqual(model.head.aasist.out_layer.in_features, 384)
        for a, b in ((0, 160), (160, 320), (320, 384)):
            self.assertGreater(float(model.head.aasist.out_layer.weight[:, a:b].detach().abs().sum()), 0.)
        for name, value in model.frontend.named_parameters(): torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for name, value in model.named_parameters():
            if name.endswith('.lora_b'):
                self.assertEqual(float(value.abs().sum()), 0.); self.assertFalse(value.requires_grad)
        wrappers = [m for m in model.encoder.layers if isinstance(m, RecomputedLayer)]
        self.assertEqual(len(wrappers), 2); self.assertFalse(any(m.enabled for m in wrappers))
        model.set_phase(True); self.assertTrue(all(m.enabled for m in wrappers))

    def test_source_interleaving(self):
        backend = RecordingAASIST()
        StatisticsHead(RegionHead(backend, True, 128, 64, 3))([torch.full((400,128), float(i)) for i in range(3)])
        self.assertTrue(all(batch == [0.,1.,2.] for batch in backend.source_batches))

    def test_all_layer_checkpoint_preserves_logits_and_gradients(self):
        left = tiny_model(dict(tiny_cfg(), checkpoint_layers=0))
        right = tiny_model(dict(tiny_cfg(), checkpoint_layers=2))
        right.load_state_dict(left.state_dict()); left.eval(); right.eval()
        waves = [torch.randn(8000), torch.randn(12000)]
        a = left(waves); b = right(waves)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        a.square().sum().backward(); b.square().sum().backward()
        for (name, p), (_, q) in zip(left.named_parameters(), right.named_parameters()):
            if p.grad is not None:
                self.assertIsNotNone(q.grad, name)
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)

    def test_constant_singleton_std_has_zero_value_and_finite_gradient(self):
        for shape in ((1, 1, 64), (3, 8, 64)):
            values = torch.ones(shape, requires_grad=True); deviation = stable_std(values, 1)
            torch.testing.assert_close(deviation, torch.zeros_like(deviation), rtol=0, atol=0)
            deviation.sum().backward()
            self.assertTrue(bool(torch.isfinite(values.grad).all()))
            self.assertEqual(float(values.grad.abs().sum()), 0.)


if __name__ == '__main__': unittest.main()
