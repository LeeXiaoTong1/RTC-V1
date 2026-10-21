"""Real AASIST + small Omni-shaped encoder: migration, gradients and invariance."""
import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayer, TinyLayout
from w2v_v3189.test_model import tiny_cfg
from w2v_v3191.model import Detector as OldDetector
from w2v_v3183r2.fusion import LayerFusion
from w2v_v5.weights import average_states, invariant_keys
from .common import partial_state, apply_partial
from .fusion import AdaptiveLayerFusion, acoustic_summary
from .model import Detector, parent_anchor, optimizer_for, adapter_inventory


def configuration(depth=3, mode='dynamic'):
    return dict(tiny_cfg(depth), retained_lora_layers=depth, lora_layers=min(2, depth),
        encoder_layers=depth, adaptive_fusion_mode=mode, adaptive_fusion_width=8,
        adaptive_fusion_seed=53001, lora_dropout=0.)


def detector(cfg, cls=Detector):
    encoder = TinyEncoder()
    encoder.layers = nn.ModuleList([TinyLayer() for _ in range(cfg['encoder_layers'])])
    return cls(TinyFrontend(), encoder, cfg, TinyLayout)


def transferred(cfg):
    torch.manual_seed(813)
    old = detector(cfg, OldDetector)
    with torch.no_grad():
        for name,p in old.named_parameters():
            if name.endswith('.lora_b'):
                p.normal_(0., .04)
        old.fusion.layer_logits.copy_(torch.linspace(-1, 1, cfg['encoder_layers']))
    rng = torch.get_rng_state()
    torch.manual_seed(813)
    new = detector(cfg)
    apply_partial(new, parent_anchor(new, partial_state(old)))
    torch.set_rng_state(rng)
    return old, new


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(137)

    def test_masked_population_statistics_ignore_padding_and_batch_companions(self):
        x = torch.tensor([[[1., 2.], [3., 6.], [float('nan'), float('nan')]],
                          [[10., -2.], [12., 0.], [14., 2.]]])
        expected = torch.tensor([[2., 4., 1., 2.], [12., 0., (8/3)**.5, (8/3)**.5]])
        torch.testing.assert_close(acoustic_summary(x, [2,3]), expected)
        torch.testing.assert_close(acoustic_summary(x[:1,:2], [2]), expected[:1])
        with self.assertRaises(ValueError):
            acoustic_summary(x, [0,3])

    def test_zero_init_preserves_actual_parent_predictions_and_frozen_adapter_contributions(self):
        cfg = configuration()
        old, new = transferred(cfg)
        waves = [torch.randn(n) for n in (8400, 4240, 10000)]
        old.eval(); new.eval()
        calls = []
        hook = new.encoder.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            expected = old(waves)
            actual = new(waves)
        hook.remove()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(calls, [1])
        self.assertEqual(adapter_inventory(new)['trainable_layers'], [2,3])
        self.assertEqual(adapter_inventory(new)['retained_layers'], [1,2,3])
        for k,v in partial_state(old).items():
            torch.testing.assert_close(partial_state(new)[k], v, rtol=0, atol=0)

    def test_constructor_does_not_change_existing_random_stream(self):
        base = LayerFusion(4, 3, 2, False)
        rng = torch.get_rng_state()
        AdaptiveLayerFusion(copy.deepcopy(base), seed=7)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        a = AdaptiveLayerFusion(copy.deepcopy(base), seed=7)
        b = AdaptiveLayerFusion(copy.deepcopy(base), seed=7)
        for name,p in a.conditioner.state_dict().items():
            torch.testing.assert_close(p, b.conditioner.state_dict()[name], rtol=0, atol=0)

    def test_input_adaptive_weights_are_normalized_and_sample_local(self):
        fusion = AdaptiveLayerFusion(LayerFusion(4, 3, 2, False), width=8)
        with torch.no_grad():
            fusion.conditioner[-1].weight.normal_(0, .2)
        x = torch.randn(2, 12, 4)
        x[1] = x[1]*2 + torch.tensor([2., -1., 0., 3.])
        weights = fusion.weights(x, [6,12])
        torch.testing.assert_close(weights.sum(-1), torch.ones(2))
        self.assertGreater(float((weights[0]-weights[1]).abs().sum().detach()), 1e-5)
        torch.testing.assert_close(weights[:1], fusion.weights(x[:1,:6], [6]), rtol=1e-6, atol=1e-7)
        changed = x.clone(); changed[0,6:] = 1e8
        torch.testing.assert_close(weights, fusion.weights(changed, [6,12]))

    def test_all_60_layers_stream_match_materialized_reference_and_checkpoint_backward(self):
        cfg = configuration(60)
        model = detector(cfg).eval()
        with torch.no_grad():
            model.fusion.conditioner[-1].weight.normal_(0., .2)
        waves = [torch.randn(4240), torch.randn(5200)]
        captured, frontend, final = {}, [], []
        handles = []
        for i, layer in enumerate(model.encoder.layers[:-1], 1):
            handles.append(layer.register_forward_hook(lambda m,a,o,i=i: captured.update({i:o})))
        handles.append(model.encoder.register_forward_pre_hook(lambda m,a: frontend.append(a[0])))
        handles.append(model.encoder.register_forward_hook(lambda m,a,o: final.append(o)))
        streamed = model.encode(waves)
        for handle in handles:
            handle.remove()
        material = model.fusion(final[0], captured, condition=frontend[0], lengths=[13,16])
        for i, seq in enumerate(streamed):
            torch.testing.assert_close(seq, material[i,:len(seq)], rtol=0, atol=0)
        sum(seq.square().mean() for seq in streamed).backward()
        self.assertGreater(float(model.fusion.conditioner[-1].weight.grad.norm()), 0.)
        self.assertTrue(all(len(layer._forward_hooks) == 0 for layer in model.encoder.layers))

    def test_optimizer_updates_conditioner_then_hidden_layer_without_unfreezing_prefix(self):
        old, model = transferred(configuration())
        model.eval()
        opt = optimizer_for(model, model.cfg)
        waves = [torch.randn(6800), torch.randn(5200)]
        output_before = model.fusion.conditioner[-1].weight.detach().clone()
        for index in range(2):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(waves), torch.tensor([0,1]))
            loss.backward()
            self.assertGreater(float(model.fusion.conditioner[-1].weight.grad.norm()), 0.)
            if index == 1:
                self.assertGreater(float(model.fusion.conditioner[1].weight.grad.norm()), 0.)
            self.assertTrue(all(p.grad is None for n,p in model.named_parameters()
                                if n.startswith('frontend.') or (n.startswith('encoder.layers.0.') )))
            opt.step()
        self.assertFalse(torch.equal(output_before, model.fusion.conditioner[-1].weight))

    def test_static_control_freezes_conditioner_and_restores_fixed_ablation(self):
        _, model = transferred(configuration(mode='static'))
        opt = optimizer_for(model, model.cfg)
        ids = {id(p) for g in opt.param_groups for p in g['params']}
        self.assertTrue(all(not p.requires_grad and id(p) not in ids for p in model.fusion.conditioner.parameters()))
        x = torch.randn(2, 10, 16)
        expected = model.fusion.weights(x, [10,8])
        with model.fusion.fixed([0., 1., 0.]):
            torch.testing.assert_close(model.fusion.weights(x, [10,8]), torch.tensor([[0.,1.,0.]]).expand(2,-1))
        torch.testing.assert_close(model.fusion.weights(x, [10,8]), expected, rtol=0, atol=0)

    def test_parent_anchor_after_training_remains_zero_and_resume_rejects_missing_gate(self):
        old, model = transferred(configuration())
        before = parent_anchor(model, partial_state(old))
        with torch.no_grad():
            model.fusion.conditioner[-1].weight.add_(1.)
            model.fusion.conditioner[1].weight.add_(1.)
        anchor = parent_anchor(model, partial_state(old))
        for name in anchor:
            torch.testing.assert_close(before[name], anchor[name], rtol=0, atol=0)
        averaged = average_states([anchor, partial_state(model)], [.5,.5], constant_keys=invariant_keys(model, anchor))
        torch.testing.assert_close(averaged['fusion.conditioner.3.weight'], torch.full_like(averaged['fusion.conditioner.3.weight'], .5))
        with self.assertRaises(ValueError):
            apply_partial(model, partial_state(old))
        bad = partial_state(old); bad.pop('fusion.layer_logits')
        with self.assertRaisesRegex(ValueError, 'beyond'):
            parent_anchor(model, bad)

    def test_reader_rng_optimization_preserves_dropout_outputs_gradients_and_rng(self):
        from torch.utils.checkpoint import checkpoint as original_checkpoint
        cfg = dict(configuration(), lora_dropout=.2, fusion_checkpointing=True)
        _, model = transferred(cfg)
        model.train()
        with torch.no_grad():
            model.fusion.conditioner[-1].weight.normal_(0., .1)
        reference = copy.deepcopy(model)
        waves = [torch.randn(4240), torch.randn(5200)]
        rng = torch.get_rng_state()
        def legacy_reader(function, *args, **kwargs):
            kwargs['preserve_rng_state'] = True
            return original_checkpoint(function, *args, **kwargs)
        with patch('w2v_v53.fusion.checkpoint', side_effect=legacy_reader):
            expected = reference(waves)
            expected.square().mean().backward()
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        actual = model(waves)
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for (name,p), (_,other) in zip(model.named_parameters(), reference.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,other.grad,rtol=0,atol=0,msg=name)


if __name__ == '__main__':
    unittest.main()
