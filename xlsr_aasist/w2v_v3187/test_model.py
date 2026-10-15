"""Parent-exact window statistics, one encoder pass, and optimizer coverage."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayout
from w2v_v3185.test_support import tiny_cfg as old_cfg
from w2v_v3184_aug5.model import Detector as ParentDetector
from .regions import StatisticsHead
from w2v_v318.regions import RegionHead, windows
from .model import Detector, optimizer_for


def tiny_cfg():
    cfg = old_cfg()
    cfg.update(version='3.18.7', warm_epochs=0, epochs=20, classifier_lr=.001,
               pooling='asp', local_evidence=True, whole_max_frames=3000)
    return cfg


class RecordingAASIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_layer = nn.Linear(160, 2)
        self.shapes = []

    def evidence(self, x):
        self.shapes.append(tuple(x.shape))
        mean = x.mean(1)
        return torch.cat((mean, mean[:, :32]), -1)

    def classify(self, x):
        return self.out_layer(x)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_overlapping_windows_and_unpadded_short_audio(self):
        backend = RecordingAASIST()
        head = StatisticsHead(RegionHead(backend, True, 128, 64, 3))
        sequences = [torch.randn(n, 128) for n in (140, 25, 201)]
        output, weights = head(sequences, return_weights=True)
        self.assertEqual(output.shape, (3, 2))
        self.assertEqual(backend.out_layer.in_features, 320)
        self.assertEqual([len(w) for w in weights], [2, 1, 3])
        self.assertEqual(backend.shapes, [(1, 25, 128), (3, 128, 128), (2, 128, 128)])
        for seq, weight in zip(sequences, weights):
            self.assertAlmostEqual(weight.sum().item(), 1., places=6)
            ranges, _ = head.window_plan(len(seq), seq.device)
            self.assertEqual(ranges[-1][1], len(seq))
            self.assertEqual(ranges, tuple(windows(len(seq), 128, 64)))

    def test_full_parent_state_reproduces_multiwindow_predictions_and_bn(self):
        cfg = tiny_cfg()
        torch.manual_seed(713)
        parent = ParentDetector(TinyFrontend(), TinyEncoder(), cfg, TinyLayout)
        parent.set_phase(True)
        with torch.no_grad():
            for p in parent.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * .01)
            for module in parent.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.running_mean.normal_(0, .05)
                    module.running_var.uniform_(.8, 1.2)
                    module.num_batches_tracked.fill_(137)
        current = Detector(TinyFrontend(), TinyEncoder(), cfg, TinyLayout)
        current.load_state_dict(parent.state_dict(), strict=True)
        parent.eval(); current.eval()
        waves = [torch.randn(400 + 320 * (n - 1)) for n in (140, 25, 201)]
        with torch.no_grad():
            expected = parent(waves)
            with patch.object(current, 'encode', wraps=current.encode) as encode:
                actual = current(waves)
                self.assertEqual(encode.call_count, 1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(bool(current.head.aasist.out_layer.weight[:, 160:].abs().sum() > 0))
        for key, value in parent.state_dict().items():
            torch.testing.assert_close(current.state_dict()[key], value, rtol=0, atol=0)

    def test_all_restored_parameter_groups_receive_gradients(self):
        cfg = tiny_cfg()
        cfg.update(window=32, hop=16, window_batch=5)
        model = Detector(TinyFrontend(), TinyEncoder(), cfg, TinyLayout)
        self.assertTrue(model.joint)
        self.assertEqual((model.head.window, model.head.hop, model.head.window_batch), (32, 16, 5))
        self.assertEqual(model.head.aasist.out_layer.in_features, 320)
        with torch.no_grad():
            model.head.attention_delta[-1].weight.normal_(0, .01)
            model.head.aasist.out_layer.weight[:, 160:].normal_(0, .01)
            for name, p in model.named_parameters():
                if name.endswith('.lora_b'):
                    p.normal_(0, .01)
        optimizer = optimizer_for(model, cfg)
        expected = {'lora', 'aasist', 'classifier', 'evidence', 'pooling', 'fusion', 'layer_mix'}
        self.assertEqual({group['name'] for group in optimizer.param_groups}, expected)
        logits = model([torch.randn(18000), torch.randn(25000)])
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        for name in expected:
            grads = [p.grad for group in optimizer.param_groups if group['name'] == name
                     for p in group['params'] if p.grad is not None]
            self.assertTrue(grads, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, name)
        self.assertFalse(any(p.grad is not None for p in model.parameters() if not p.requires_grad))

    def test_pcl_references_skip_classifier_and_reencoding(self):
        model = Detector(TinyFrontend(), TinyEncoder(), tiny_cfg(), TinyLayout).eval()
        waves = [torch.randn(n) for n in (18000, 25000, 18000)]
        with torch.no_grad(), patch.object(model, 'encode', wraps=model.encode) as encode:
            sequences = model.encode(waves)
            # The third view is a clean RTC PCL reference. Only two main views
            # enter classification, using already encoded fused features.
            output = model.classify_sequences(sequences[:2])
            self.assertEqual(output.shape, (2, 2))
            self.assertEqual(encode.call_count, 1)
            self.assertEqual(len(sequences), 3)

    def test_frontend_guard_and_structure_contract(self):
        cfg = tiny_cfg()
        with self.assertRaisesRegex(ValueError, 'overlapping-window ASP'):
            Detector(TinyFrontend(), TinyEncoder(), dict(cfg, pooling='whole'), TinyLayout)
        model = Detector(TinyFrontend(), TinyEncoder(), dict(cfg, whole_max_frames=12), TinyLayout)
        with self.assertRaisesRegex(ValueError, 'Frontend frame limit exceeded'):
            model.encode([torch.randn(18000)])

    def test_fresh_preserves_backbone_and_initializes_complete_classifier(self):
        frontend, encoder = TinyFrontend(), TinyEncoder()
        frozen_before = {name: value.detach().clone() for name, value in frontend.named_parameters()}
        cfg = dict(tiny_cfg(), warm_epochs=1, checkpoint_layers=1, fusion_checkpointing=False)
        model = Detector(frontend, encoder, cfg, TinyLayout)
        self.assertFalse(model.joint)
        self.assertFalse(model.fusion.checkpointing)
        self.assertGreater(float(model.head.aasist.out_layer.weight[:, 160:].abs().sum()), 0)
        for name, value in model.frontend.named_parameters():
            torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)
        for name, value in model.named_parameters():
            if name.endswith('.lora_b'):
                self.assertEqual(float(value.abs().sum()), 0.)
                self.assertFalse(value.requires_grad)
        from w2v_v3183r2.model import RecomputedLayer
        wrappers = [m for m in model.encoder.layers if isinstance(m, RecomputedLayer)]
        self.assertEqual(sum(m.enabled for m in wrappers), 0)
        model.set_phase(True)
        self.assertEqual(sum(m.enabled for m in wrappers), 1)
        self.assertTrue(wrappers[-1].enabled)

    def test_window_batches_interleave_independent_sources(self):
        class RecordingSources(RecordingAASIST):
            def evidence(self, x):
                self.source_batches = getattr(self, 'source_batches', []) + [x[:, 0, 0].tolist()]
                return super().evidence(x)
        backend = RecordingSources()
        head = StatisticsHead(RegionHead(backend, True, 128, 64, 3))
        head([torch.full((400,128), float(i)) for i in range(3)])
        self.assertTrue(all(batch == [0.,1.,2.] for batch in backend.source_batches))

    def test_selective_checkpoint_preserves_logits_and_gradients(self):
        cfg=tiny_cfg()
        left=Detector(TinyFrontend(),TinyEncoder(),dict(cfg,checkpoint_layers=0),TinyLayout)
        right=Detector(TinyFrontend(),TinyEncoder(),dict(cfg,checkpoint_layers=1),TinyLayout)
        right.load_state_dict(left.state_dict())
        left.eval();right.eval()
        waves=[torch.randn(8000),torch.randn(12000)]
        a=left(waves);b=right(waves)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        a.square().sum().backward();b.square().sum().backward()
        for (name,p),(_,q) in zip(left.named_parameters(),right.named_parameters()):
            if p.grad is not None:
                self.assertIsNotNone(q.grad,name)
                torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)


if __name__ == '__main__':
    unittest.main()
