"""Check the faster runtime against the existing source-normalized objective."""
import copy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch

from w2v_v3.model import microbatches
from w2v_v3.test_model_step import ToyDetector, tiny_detector
from w2v_v31.step import supervised_step as previous_step
from w2v_v31.test_data_step import expanded_examples
from .model import Detector as RuntimeDetector
from .step import supervised_step


torch.set_num_threads(1)


class RuntimeStepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91037)

    @staticmethod
    def optimizer(model):
        # Use the deployed head learning rate. A much larger artificial rate
        # magnifies roundoff on nearly zero first-step Adam gradients; gradient
        # tensors and both moments are compared independently below.
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=2e-6, weight_decay=.01)

    def assert_parameters_equal(self, expected, actual, *, exact=False):
        self.assertEqual(set(expected.state_dict()), set(actual.state_dict()))
        for name, value in expected.state_dict().items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(actual.state_dict()[name], value,
                                           rtol=0 if exact else 3e-5,
                                           atol=0 if exact else 3e-6)

    def test_real_encoder_logits_gradients_and_adam_update_match_previous(self):
        # The varying view lengths require several physical microbatches. CKA
        # must still use the complete set of full views, with no short views.
        for position_mode, cka_weight in ((p, c) for p in ('relative_key', 'relative', 'rotary')
                                         for c in (0., .01)):
            with self.subTest(cka_weight=cka_weight, position_mode=position_mode):
                original = tiny_detector(checkpointing=True).configure_trainable_layers(1).train()
                original_config = original.backbone.config.to_dict()
                original_config['position_embeddings_type'] = position_mode
                original = type(original).from_config(original_config, asdict(original.head.config), True)
                original.configure_trainable_layers(1).train()
                # Compare arithmetic and optimization, not stochastic dropout
                # schedules (which can differ with execution organization).
                for module in original.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = 0.
                actual = RuntimeDetector.from_checkpoint(
                    {**original.architecture(), 'model': original.state_dict()}, checkpointing=True)
                actual.configure_trainable_layers(1).train()
                for module in actual.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = 0.
                examples = expanded_examples(160)
                previous_optimizer, actual_optimizer = self.optimizer(original), self.optimizer(actual)
                weights, noisy_weights = torch.tensor([.73, 1.48]), torch.tensor([.58, 2.04])
                kwargs = dict(amp='none', noisy_weight=.43, cka_weight=cka_weight,
                              noisy_class_weights=noisy_weights, microbatch=2,
                              frame_budget=26, offload_activations=False)
                observed = []
                hook = actual.backbone.register_forward_pre_hook(
                    lambda _m, _args, kw: observed.append(len(kw['input_features'])),
                    with_kwargs=True)
                previous_stats, previous_logits = previous_step(
                    original, examples, previous_optimizer, weights, 'cpu', **kwargs)
                actual_stats, actual_logits = supervised_step(
                    actual, examples, actual_optimizer, weights, 'cpu', **kwargs)
                hook.remove()
                self.assertEqual(sum(observed), len(examples))
                self.assertLess(len(observed), len(list(microbatches(examples, 2, 26))))
                torch.testing.assert_close(actual_logits.cpu(), previous_logits.cpu(), rtol=3e-5, atol=3e-6)
                for key in ('ordinary_ce', 'noisy_ce', 'full_ce_contribution',
                            'short_ce_contribution', 'ce', 'cka', 'loss', 'grad_norm'):
                    self.assertAlmostEqual(actual_stats[key], previous_stats[key], places=5, msg=key)
                for key in ('ordinary_sources', 'noisy_sources', 'full_views',
                            'short_views', 'cka_full_views'):
                    self.assertEqual(actual_stats[key], previous_stats[key], key)
                self.assertEqual(actual_stats['encoder_forward_microbatches'], len(observed))
                self.assert_parameters_equal(original, actual)
                for (name, left), (_, right) in zip(original.named_parameters(), actual.named_parameters()):
                    if left.grad is None:
                        self.assertIsNone(right.grad, name)
                    else:
                        torch.testing.assert_close(right.grad, left.grad, rtol=4e-5, atol=4e-6)
                # Adam moments must be the same too; matching weights alone can
                # hide a different gradient that affects subsequent updates.
                for left, right in zip(previous_optimizer.state.values(), actual_optimizer.state.values()):
                    for key in ('step', 'exp_avg', 'exp_avg_sq'):
                        torch.testing.assert_close(right[key], left[key], rtol=4e-5, atol=4e-6)

    def test_long_variable_waveforms_and_poisoned_padding_preserve_update(self):
        from . import step as runtime_step
        model = tiny_detector(checkpointing=True)
        model.configure_trainable_layers(1).train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        actual = RuntimeDetector.from_checkpoint(
            {**model.architecture(), 'model': model.state_dict()}, checkpointing=True)
        actual.configure_trainable_layers(1).train()
        for module in actual.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        examples = []
        for source, length in enumerate((111, 257, 163, 283)):
            common = dict(id=str(source), source_group=str(source),
                          label=source % 2, noisy=source >= 2)
            for view, frames, weight in (('full', length, .7), ('short', 83+source*7, .3)):
                examples.append({**common, 'view': view, 'view_weight': weight,
                                 'features': torch.randn(1, frames, 160),
                                 'mask': torch.ones(1, frames, dtype=torch.long)})
        base_microbatches = runtime_step.microbatches
        poison_count = [0]

        def poisoned(*args, **kwargs):
            for indices, features, mask in base_microbatches(*args, **kwargs):
                invalid = ~mask.bool()
                poison_count[0] += int(invalid.sum())
                yield indices, features.masked_fill(invalid.unsqueeze(-1), 98765.), mask

        kwargs = dict(amp='none', noisy_weight=.37, cka_weight=.13, microbatch=4,
                      frame_budget=1200, offload_activations=False)
        stats, logits = previous_step(model, examples, self.optimizer(model), torch.ones(2), 'cpu', **kwargs)
        with patch.object(runtime_step, 'microbatches', side_effect=poisoned):
            actual_stats, actual_logits = supervised_step(
                actual, examples, self.optimizer(actual), torch.ones(2), 'cpu', **kwargs)
        self.assertGreater(poison_count[0], 0)
        torch.testing.assert_close(actual_logits.cpu(), logits.cpu(), rtol=3e-5, atol=3e-6)
        self.assertAlmostEqual(actual_stats['loss'], stats['loss'], places=5)
        self.assert_parameters_equal(model, actual)
        for (name, left), (_, right) in zip(model.named_parameters(), actual.named_parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad, name)
            else:
                torch.testing.assert_close(right.grad, left.grad, rtol=4e-5, atol=4e-6)

    def test_nonfinite_later_microbatch_does_not_update_model_or_optimizer(self):
        for cka_weight in (0., .1):
            for failing_output in (0, 1):
                with self.subTest(cka_weight=cka_weight, output=failing_output):
                    model = ToyDetector()
                    before = copy.deepcopy(model)
                    optimizer = self.optimizer(model)
                    calls = [0]

                    def corrupt_second(_model, _inputs, outputs):
                        calls[0] += 1
                        if calls[0] != 2:
                            return outputs
                        values = list(outputs)
                        values[failing_output] = values[failing_output] * float('nan')
                        return tuple(values)

                    handle = model.register_forward_hook(corrupt_second)
                    try:
                        with self.assertRaises((FloatingPointError, RuntimeError)):
                            supervised_step(model, expanded_examples(), optimizer, torch.ones(2),
                                            'cpu', amp='none', cka_weight=cka_weight,
                                            microbatch=1, frame_budget=26)
                    finally:
                        handle.remove()
                    self.assertGreaterEqual(calls[0], 2)
                    self.assert_parameters_equal(before, model, exact=True)
                    self.assertEqual(optimizer.state, {})

    def test_nonfinite_gradient_rejected_before_adam_state_exists(self):
        model = ToyDetector()
        before = copy.deepcopy(model)
        optimizer = self.optimizer(model)
        parameter = next(p for p in model.parameters() if p.requires_grad)
        handle = parameter.register_hook(lambda grad: grad * float('nan'))
        try:
            with self.assertRaises((FloatingPointError, RuntimeError)):
                supervised_step(model, expanded_examples(), optimizer, torch.ones(2),
                                'cpu', amp='none', cka_weight=.1, microbatch=2)
        finally:
            handle.remove()
        self.assert_parameters_equal(before, model, exact=True)
        self.assertEqual(optimizer.state, {})

    def test_invalid_source_budget_rejected_before_any_forward(self):
        model = ToyDetector()
        examples = expanded_examples()
        examples[1]['view_weight'] = .9
        optimizer = self.optimizer(model)
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            supervised_step(model, examples, optimizer, torch.ones(2), 'cpu', amp='none')
        self.assertEqual(model.calls, 0)
        self.assertEqual(optimizer.state, {})


if __name__ == '__main__':
    unittest.main()
