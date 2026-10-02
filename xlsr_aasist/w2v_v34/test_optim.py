"""Real tiny-HF encoder tests: block unfreezing, fusion gradients, and resume."""
import copy
import io
import unittest
import torch
from torch.nn import functional as F
from w2v_v3.test_model_step import tiny_detector
from w2v_v3.model import Detector
from w2v_v33.model import install_runtime
from .optim import configure_model, optimizer_for, apply_learning_rates, UpdateDiagnostics

torch.set_num_threads(1)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3403)
        self.cfg = dict(trainable_layers=2, encoder_lr=5e-7, layer_decay=.9,
                        joint_head_lr=2e-6, weight_decay=1e-4)

    def model(self):
        return install_runtime(tiny_detector(checkpointing=True)).train()

    @staticmethod
    def backward(model):
        x = torch.randn(2, 15, 160)
        z, _, frames, _ = model.forward_training(x, torch.ones(2, 15).long())
        loss = F.cross_entropy(z, torch.tensor([0, 1])) + .01 * frames.square().mean()
        loss.backward()

    def test_real_hf_all_blocks_and_head_update_frozen_projection_unchanged(self):
        model = self.model()
        optimizer = optimizer_for(model, self.cfg)
        origin = {n: p.detach().clone() for n, p in model.named_parameters()}
        diagnostic = UpdateDiagnostics(model)
        self.backward(model)
        for layer in model.backbone.encoder.layers:
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in layer.parameters()))
        diagnostic.record_gradients(model)
        optimizer.step()
        for i, layer in enumerate(model.backbone.encoder.layers):
            self.assertTrue(any(not torch.equal(p, origin[f'backbone.encoder.layers.{i}.{n}'])
                                for n, p in layer.named_parameters()))
        for name, parameter in model.backbone.named_parameters():
            if not name.startswith('encoder.layers.'):
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, origin['backbone.' + name], atol=0, rtol=0)
        report = diagnostic.report(model)['groups']
        self.assertEqual(report['backbone.frozen']['sampled_delta_l2'], 0.)
        for name in ('encoder.layer_00', 'encoder.layer_01', 'head'):
            self.assertGreater(report[name]['sampled_delta_l2'], 0.)
            self.assertGreater(report[name]['sampled_gradient_l2'], 0.)

    def test_last_n_includes_only_eligible_parameters_and_preserves_ratios(self):
        model = self.model()
        cfg = dict(self.cfg, trainable_layers=1)
        optimizer = optimizer_for(model, cfg)
        actual = [id(p) for g in optimizer.param_groups for p in g['params']]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), expected)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.encoder.layers[0].parameters()))
        self.assertEqual({g['name'] for g in optimizer.param_groups},
            {'encoder.layer_01.decay', 'encoder.layer_01.no_decay', 'head.decay', 'head.no_decay'})
        optimizer = optimizer_for(model, self.cfg)
        rates = apply_learning_rates(optimizer, .3)
        self.assertAlmostEqual(rates['encoder.layer_00.decay'], 5e-7 * .9 * .3)
        self.assertAlmostEqual(rates['encoder.layer_01.no_decay'], 5e-7 * .3)
        self.assertAlmostEqual(rates['head.decay'], 2e-6 * .3)
        for group in optimizer.param_groups:
            self.assertEqual(group['weight_decay'], 0. if group['name'].endswith('.no_decay') else 1e-4)

    def test_twenty_four_real_hf_blocks_receive_gradients_and_update(self):
        architecture = tiny_detector().architecture()
        architecture['model_config']['num_hidden_layers'] = 24
        model = install_runtime(Detector.from_config(architecture['model_config'],
                                architecture['head_config'], checkpointing=True)).train()
        optimizer = optimizer_for(model, dict(self.cfg, trainable_layers=24))
        diagnostics = UpdateDiagnostics(model)
        self.backward(model)
        diagnostics.record_gradients(model)
        optimizer.step()
        report = diagnostics.report(model)['groups']
        for i in range(24):
            group = report[f'encoder.layer_{i:02d}']
            self.assertGreater(group['sampled_gradient_l2'], 0., f'block {i} has no sampled gradient')
            self.assertGreater(group['sampled_delta_l2'], 0., f'block {i} did not update')
        rates = apply_learning_rates(optimizer, 1.)
        self.assertAlmostEqual(rates['encoder.layer_00.decay'], 5e-7 * .9 ** 23)
        self.assertAlmostEqual(rates['encoder.layer_23.decay'], 5e-7)
        self.assertEqual(report['backbone.frozen']['sampled_delta_l2'], 0.)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         '24-block BF16/checkpoint/CPU-spill integration requires BF16 CUDA')
    def test_cuda_twenty_four_blocks_bf16_checkpointed_pair_step_with_forced_cpu_spill(self):
        from w2v_v33.step import supervised_step
        from w2v_v33.test_step import examples
        architecture = tiny_detector().architecture()
        architecture['model_config']['num_hidden_layers'] = 24
        model = install_runtime(Detector.from_config(architecture['model_config'],
                                architecture['head_config'], checkpointing=True)).cuda().train()
        optimizer = optimizer_for(model, dict(self.cfg, trainable_layers=24))
        diagnostics = UpdateDiagnostics(model)
        frozen = {name: parameter.detach().cpu().clone() for name, parameter in model.backbone.named_parameters()
                  if not name.startswith('encoder.layers.')}
        # Three independent sources retain real/fake labels, complete/short
        # views, two noisy families and one source with a missing online pair.
        rows = examples(count=3)
        dtypes = []
        hook = model.head.projection.register_forward_hook(lambda _m, _args, output: dtypes.append(output.dtype))
        try:
            stats, scores = supervised_step(model, rows, optimizer, torch.ones(2, device='cuda'),
                'cuda', amp='bf16', noisy_weight=.5, cka_weight=.01, pair_weight=.02,
                aux_max_tokens=12, microbatch=4, frame_budget=64,
                offload_activations=True, activation_budget_gib=.25,
                gpu_activation_gib=0., gpu_reserve_gib=0., diagnose_aux_grad=True)
        finally:
            hook.remove()
        self.assertTrue(model.backbone.is_gradient_checkpointing)
        self.assertIn(torch.bfloat16, dtypes)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.isfinite(torch.tensor(stats['loss'])))
        self.assertEqual(stats['source_count'], 3)
        self.assertEqual(stats['pair_valid_count'], 8)
        self.assertGreater(stats['activation_offload_gib'], 0.)
        self.assertEqual(stats['gpu_saved_activations_gib'], 0.)
        self.assertGreater(stats['weighted_pair_shared_feature_grad_norm'], 0.)
        diagnostics.record_gradients(model)
        report = diagnostics.report(model)['groups']
        for i, layer in enumerate(model.backbone.encoder.layers):
            group = report[f'encoder.layer_{i:02d}']
            self.assertGreater(group['sampled_gradient_l2'], 0., f'block {i} lacks gradients')
            self.assertGreater(group['sampled_delta_l2'], 0., f'block {i} did not update')
            for parameter in layer.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(report['head']['sampled_delta_l2'], 0.)
        for name, parameter in model.backbone.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter.detach().cpu(), frozen[name], atol=0, rtol=0)
        self.assertEqual(report['backbone.frozen']['sampled_delta_l2'], 0.)

    def test_fusion_propagates_to_early_state_without_final_layer_path(self):
        model = self.model()
        configure_model(model, self.cfg)
        output = model.backbone(input_features=torch.randn(2, 13, 160),
                                attention_mask=torch.ones(2, 13).long(),
                                output_hidden_states=True, return_dict=True)
        # Detach every hidden state except the first block's direct fusion input.
        states = [h if i == 1 else h.detach() for i, h in enumerate(output.hidden_states)]
        logits, _, _ = model.head.forward_training(states, torch.ones(2, 13).long())
        F.cross_entropy(logits, torch.tensor([0, 1])).backward()
        early = model.backbone.encoder.layers[0]
        late = model.backbone.encoder.layers[1]
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in early.parameters()))
        self.assertTrue(all(p.grad is None for p in late.parameters()))

    def test_optimizer_and_origin_resume_preserve_rates_and_exact_next_update(self):
        model = self.model()
        optimizer = optimizer_for(model, self.cfg)
        diagnostics = UpdateDiagnostics(model)
        self.backward(model)
        optimizer.step()
        diagnostics.record_gradients(model)
        apply_learning_rates(optimizer, .7)
        saved = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        diagnostics=diagnostics.state_dict()), saved)
        saved.seek(0)
        state = torch.load(saved, weights_only=True)
        restored = self.model()
        restored.load_state_dict(state['model'])
        resumed_optimizer = optimizer_for(restored, self.cfg)
        resumed_optimizer.load_state_dict(state['optimizer'])
        resumed_diagnostics = UpdateDiagnostics(restored)
        resumed_diagnostics.load_state_dict(state['diagnostics'])
        self.assertEqual(diagnostics.report(model), resumed_diagnostics.report(restored))
        self.assertEqual(apply_learning_rates(optimizer, .4), apply_learning_rates(resumed_optimizer, .4))
        for left, right in zip(model.parameters(), restored.parameters()):
            left.grad = torch.full_like(left, .02) if left.requires_grad else None
            right.grad = torch.full_like(right, .02) if right.requires_grad else None
        optimizer.step()
        resumed_optimizer.step()
        for left, right in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_diagnostics_are_small_rng_free_and_origin_mismatch_rejected(self):
        model = self.model()
        configure_model(model, self.cfg)
        rng = torch.get_rng_state().clone()
        diagnostic = UpdateDiagnostics(model, samples_per_parameter=8)
        diagnostic.report(model)
        diagnostic.record_gradients(model)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        self.assertLess(sum(v.numel() for v in diagnostic.origin.values()), sum(p.numel() for p in model.parameters()))
        self.assertEqual(diagnostic.report(model)['groups']['encoder.layer_00']['sampled_delta_l2'], 0.)
        altered = copy.deepcopy(diagnostic.state_dict())
        altered['layout'][0]['shape'] = [100]
        with self.assertRaises(ValueError):
            diagnostic.load_state_dict(altered)

    def test_invalid_configuration_and_missing_resume_metadata_fail(self):
        for change in (dict(trainable_layers=0), dict(trainable_layers=3), dict(trainable_layers=True),
                       dict(encoder_lr=0), dict(layer_decay=1.1), dict(layer_decay=float('nan'))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                optimizer_for(self.model(), dict(self.cfg, **change))
        optimizer = optimizer_for(self.model(), self.cfg)
        for scale in (-.1, float('nan')):
            with self.assertRaises(ValueError):
                apply_learning_rates(optimizer, scale)
        del optimizer.param_groups[0]['base_lr']
        with self.assertRaises(ValueError):
            apply_learning_rates(optimizer, .5)


if __name__ == '__main__':
    unittest.main()
