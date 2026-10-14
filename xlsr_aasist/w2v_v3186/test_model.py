"""Whole-utterance geometry, unpadded head batching and optimizer inventory."""
import unittest
import torch
from torch import nn
from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayout
from w2v_v3185.test_support import tiny_cfg as old_cfg
from .model import Detector, optimizer_for
from .regions import WholeUtteranceHead


def tiny_cfg():
    cfg = old_cfg()
    cfg.update(version='3.18.6', warm_epochs=0, epochs=6, classifier_lr=.001,
               whole_max_frames=3000, whole_graph_budget=1_000_000, whole_batch=8)
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

    def test_equal_lengths_only_no_windows_or_padding(self):
        backend = RecordingAASIST()
        head = WholeUtteranceHead(backend)
        sequences = [torch.randn(n, 128) for n in (140, 25, 140, 26)]
        output = head(sequences)
        self.assertEqual(output.shape, (4, 2))
        self.assertEqual(backend.shapes, [(2, 140, 128), (1, 25, 128), (1, 26, 128)])
        expected = torch.stack([backend.classify(backend.evidence(s[None]))[0] for s in sequences])
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_graph_budget_splits_batch_but_never_an_utterance(self):
        backend = RecordingAASIST()
        head = WholeUtteranceHead(backend, max_frames=50, graph_budget=200, batch_size=8)
        head([torch.randn(30, 128) for _ in range(5)])
        self.assertEqual([shape[0] for shape in backend.shapes], [2, 2, 1])
        with self.assertRaisesRegex(ValueError, 'no crop fallback'):
            head([torch.randn(60, 128)])

    def test_real_aasist_whole_utterance_and_all_parameter_groups(self):
        cfg = tiny_cfg()
        model = Detector(TinyFrontend(), TinyEncoder(), cfg, TinyLayout)
        self.assertTrue(model.joint)
        self.assertEqual(model.head.aasist.out_layer.in_features, 160)
        self.assertFalse(any('contribution' in name or 'attention_delta' in name for name, _ in model.named_parameters()))
        optimizer = optimizer_for(model, cfg)
        self.assertEqual({group['name'] for group in optimizer.param_groups},
                         {'lora', 'aasist', 'classifier', 'fusion', 'layer_mix'})
        logits = model([torch.randn(6000), torch.randn(7900)])
        self.assertEqual(logits.shape, (2, 2))
        logits.square().sum().backward()
        self.assertIsNotNone(model.fusion.projection.weight.grad)
        self.assertTrue(all(p.grad is not None for name, p in model.named_parameters() if name.endswith('.lora_b')))


if __name__ == '__main__':
    unittest.main()
