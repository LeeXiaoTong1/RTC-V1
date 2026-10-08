"""Real tiny w2v-BERT tests for Online bridge gradients and triplet execution."""
import copy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import numpy as np
import torch

from w2v_v3.test_model_step import tiny_detector, small_head
from w2v_v32.model import install_runtime, Detector as RuntimeDetector
from w2v_v315.step import train_step as previous_step
from w2v_v3151.data import validate_batch
from w2v_v3151.objectives import TFCL, fusion_frames
from w2v_v3151.step import train_step, triplet_batches


def examples():
    result = []
    for i, (language, label) in enumerate((('en', 0), ('en', 1), ('zh', 0), ('zh', 1))):
        # Online has an independent length; RTC reference/noisy share duration.
        for role, mass, length in (('online', .5, 17 + i),
                                   ('reference', .1, 14 + i),
                                   ('noisy', .4, 14 + i)):
            valid = np.ones(length, dtype=bool)
            if role != 'online':
                valid[3:5] = False
            result.append(dict(id=f'{i}:{role}', source_id=str(i), group_id=str(i),
                source_sha256=str(i), language=language, label=label, role=role,
                condition=role if role == 'online' else 'rtc_' + role, split='train',
                pair_occurrence=str(i), pair_eligible=True, online_pair_eligible=True,
                noise_pair_eligible=True, ce_weight=mass / 4,
                features=torch.randn(1, length, 160),
                mask=torch.ones(1, length, dtype=torch.long), aux_valid=valid))
    return result


def config(**changes):
    return dict(dict(source_batch=4, microbatch=12, frame_budget=1000,
        device='cpu', amp='none', tfcl_time_weight=.15, tfcl_structure_weight=.045,
        tfcl_bridge_mass=.5, tfcl_matched_mass=.5, max_grad_norm=1.), **changes)


def model_fixture():
    model = install_runtime(tiny_detector(checkpointing=False))
    model.configure_trainable_layers(1)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    return model


def optimizer(model, auxiliary):
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad]
                           + list(auxiliary.parameters()), lr=.02)


class ObservedTFCL(TFCL):
    """Inspect the auxiliary graph itself, before supervised gradients mix in."""
    def __init__(self):
        super().__init__(8, 2, 21)
        self.observations = []

    def forward_batch(self, left, right, lm, rm):
        t, s, valid = super().forward_batch(left, right, lm, rm)
        gradients = torch.autograd.grad(t.sum() + s.sum(), left + right,
                                        retain_graph=True, allow_unused=True)
        self.observations.append(dict(left=[x.detach() for x in left],
            right=[x.detach() for x in right], lm=[m.clone() for m in lm],
            rm=[m.clone() for m in rm],
            gradients=[g.detach() if g is not None else None for g in gradients]))
        return t, s, valid


class StepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(315100)
        torch.set_num_threads(1)

    def assert_module_close(self, expected, actual):
        for (name, a), (other, b) in zip(expected.named_parameters(), actual.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5, msg=name)
            if a.grad is None:
                self.assertIsNone(b.grad, name)
            else:
                torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=4e-5, msg=name)

    def test_native_triplets_preserve_full_samples_masks_and_ce_budget(self):
        rows = examples()
        groups = list(validate_batch(rows, 4).values())
        seen = []
        for batch, x, mask in triplet_batches(groups, 6, 120):
            self.assertEqual(len(batch) % 3, 0)
            self.assertLessEqual(len(batch), 6)
            for i, row in enumerate(batch):
                n = row['features'].shape[1]
                torch.testing.assert_close(x[i, :n], row['features'][0], atol=0, rtol=0)
                self.assertEqual(mask[i].sum().item(), n)
                self.assertEqual(x[i, n:].count_nonzero().item(), 0)
                seen.append(row['id'])
        self.assertCountEqual(seen, [r['id'] for r in rows])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertAlmostEqual(sum(r['ce_weight'] for r in rows), 1.)
        # A long native triplet is emitted intact even above the nominal budget.
        batch, x, _ = next(triplet_batches(groups[:1], 3, 1))
        self.assertEqual(len(batch), 3)
        self.assertEqual(x.shape[1], 17)

    def test_pre_padded_or_malformed_triplet_input_is_rejected(self):
        rows = examples()
        groups = list(validate_batch(rows, 4).values())
        groups[0][0]['mask'][0, -1] = 0
        with self.assertRaisesRegex(ValueError, 'valid native'):
            list(triplet_batches(groups, 6, 120))
        groups[0][0]['mask'][0, -1] = 1
        groups[0][0]['features'] = groups[0][0]['features'][0]
        with self.assertRaisesRegex(ValueError, 'matching'):
            list(triplet_batches(groups, 6, 120))
        for size, budget in ((True, 120), (2, 120), (6, 0), (6, True)):
            with self.assertRaises(ValueError):
                list(triplet_batches([], size, budget))

    def test_bridge_survives_invalid_noisy_pair_and_reaches_online_tail(self):
        rows = examples()
        for row in rows:
            row['pair_eligible'] = row['noise_pair_eligible'] = False
            if row['role'] == 'noisy':
                row['aux_valid'][:] = False
        model, aux = model_fixture(), ObservedTFCL()
        stats = train_step(model, aux, optimizer(model, aux), rows, config(), 1., audit=True)
        self.assertEqual(stats['bridge_pairs'], 4)
        self.assertEqual(stats['matched_pairs'], 0)
        self.assertGreater(stats['weighted_time_loss'], 0.)
        # Losing a matched edge must not double the bridge's source budget.
        self.assertAlmostEqual(stats['time_loss'], .5 * stats['bridge_time_loss'], places=6)
        self.assertAlmostEqual(stats['structure_loss'], .5 * stats['bridge_structure_loss'], places=6)
        self.assertTrue(all(v > 0 for v in stats['aux_gradient_audit'].values()))
        observed_edges = 0
        for observation in aux.observations:
            count = len(observation['left'])
            observed_edges += count
            for i, (a, am) in enumerate(zip(observation['left'], observation['lm'])):
                self.assertTrue(am.all())  # Simulated erasures do not mask Online.
                self.assertGreater(observation['gradients'][i][-1].norm().item(), 0.)
                self.assertEqual(len(a), len(am))
            for i, mask in enumerate(observation['rm']):
                grad = observation['gradients'][count + i]
                self.assertGreater(grad[-1].norm().item(), 0.)
                self.assertEqual(grad[~mask].count_nonzero().item(), 0)
        self.assertEqual(observed_edges, 4)

    def test_grouping_padding_losses_gradients_and_one_forward_per_view(self):
        left = model_fixture()
        right = copy.deepcopy(left)
        la, ra = TFCL(8, 2, 21), None
        ra = copy.deepcopy(la)
        rows = examples()
        calls = []
        handle = right.backbone.register_forward_pre_hook(
            lambda _m, _args, kw: calls.append(len(kw['input_features'])), with_kwargs=True)
        first = train_step(left, la, optimizer(left, la), rows,
                           config(microbatch=3, frame_budget=30), 1.)
        original_batches = triplet_batches
        poisoned = []

        def poisoned_batches(*args, **kwargs):
            for batch, x, mask in original_batches(*args, **kwargs):
                poisoned.append((~mask.bool()).sum().item())
                yield batch, x.masked_fill(~mask.bool()[..., None], 98765.), mask

        try:
            with patch('w2v_v3151.step.triplet_batches', side_effect=poisoned_batches):
                second = train_step(right, ra, optimizer(right, ra), rows, config(), 1.)
        finally:
            handle.remove()
        self.assertGreater(sum(poisoned), 0)
        self.assertLess(second['physical_forwards'], first['physical_forwards'])
        self.assertEqual(len(calls), second['physical_forwards'])
        self.assertEqual(sum(calls), len(rows))  # No separate Online/TFCL SSL pass.
        for key in ('classification_loss', 'time_loss', 'structure_loss', 'total_loss',
                    'weighted_time_loss', 'weighted_structure_loss'):
            self.assertAlmostEqual(first[key], second[key], places=6, msg=key)
        self.assertEqual((second['bridge_pairs'], second['matched_pairs']), (4, 4))
        self.assert_module_close(left, right)
        self.assert_module_close(la, ra)

    def test_supervised_objective_matches_previous_execution_with_same_view_weights(self):
        new_model = model_fixture()
        old_model = copy.deepcopy(new_model)
        new_aux = TFCL(8, 2, 21)
        old_aux = copy.deepcopy(new_aux)
        rows = examples()
        old_rows = [{**row, 'role': 'ordinary' if row['role'] == 'online' else row['role']}
                    for row in rows]
        # warm=0 isolates preservation of existing CE semantics from new edges.
        actual = train_step(new_model, new_aux, optimizer(new_model, new_aux), rows,
                            config(), 0.)
        # Old validation hardcodes its historical noisy .5-.6 schedule. Validate
        # the new rows above, then compare the old execution with the SAME weights;
        # classification arithmetic and all model operations remain unmodified.
        def same_budget_validation(previous_rows, count):
            current = [{**row, 'role': 'online' if row['role'] == 'ordinary' else row['role']}
                       for row in previous_rows]
            validate_batch(current, count)
            groups = {}
            for row in previous_rows:
                groups.setdefault(row['pair_occurrence'], []).append(row)
            return groups

        with patch('w2v_v315.step.validate_batch', side_effect=same_budget_validation):
            expected = previous_step(old_model, old_aux, optimizer(old_model, old_aux),
                                     old_rows, config(microbatch=4), 0.)
        self.assertAlmostEqual(actual['classification_loss'], expected['classification_loss'], places=6)
        self.assertEqual(actual['weighted_time_loss'], 0.)
        self.assertEqual(actual['weighted_structure_loss'], 0.)
        self.assert_module_close(old_model, new_model)

    def test_nonfinite_loss_or_gradient_never_updates_parameters(self):
        for kind in ('loss', 'gradient'):
            with self.subTest(kind=kind):
                model, aux = model_fixture(), TFCL(8, 2, 21)
                opt = optimizer(model, aux)
                before = {k: v.clone() for k, v in model.state_dict().items()}
                if kind == 'loss':
                    handle = model.head.classifier.register_forward_hook(
                        lambda _m, _args, output: output * float('nan'))
                else:
                    handle = model.head.projection.weight.register_hook(
                        lambda grad: torch.full_like(grad, float('nan')))
                try:
                    with patch.object(opt, 'step', wraps=opt.step) as step_call:
                        with self.assertRaisesRegex(FloatingPointError, 'Nonfinite'):
                            train_step(model, aux, opt, examples(), config(), 1.)
                        step_call.assert_not_called()
                finally:
                    handle.remove()
                for k, v in model.state_dict().items():
                    torch.testing.assert_close(v, before[k], atol=0, rtol=0)
                self.assertEqual(len(model.head.blocks[0]._forward_pre_hooks), 0)

    def test_runtime_full_wave_fusion_and_logits_match_exact_lengths(self):
        base = tiny_detector(checkpointing=False)
        for position in ('relative_key', 'relative', 'rotary'):
            with self.subTest(position=position):
                cfg = base.backbone.config.to_dict()
                cfg.update(position_embeddings_type=position, conformer_conv_dropout=0.)
                model = RuntimeDetector.from_config(cfg, asdict(small_head()), False)
                model.configure_trainable_layers(1).train()
                waves = [torch.randn(1, n, 160) for n in (31, 14, 14)]
                exact_logits, exact_fusion = [], []
                for wave in waves:
                    with fusion_frames(model) as captured:
                        exact_logits.append(model(wave, torch.ones(1, wave.shape[1]).long())[0])
                        exact_fusion.append(captured[0].clone())
                x = torch.nn.utils.rnn.pad_sequence([w[0] for w in waves], batch_first=True)
                mask = torch.arange(31)[None] < torch.tensor([31, 14, 14])[:, None]
                x = x.masked_fill(~mask[..., None], 76543.)
                with fusion_frames(model) as captured:
                    z, _ = model(x, mask.long())
                    fusion = captured[0]
                    torch.testing.assert_close(z, torch.cat(exact_logits), atol=3e-6, rtol=4e-5)
                    for i, wave in enumerate(waves):
                        torch.testing.assert_close(fusion[i, :wave.shape[1]], exact_fusion[i][0],
                                                   atol=3e-6, rtol=4e-5)
                    self.assertEqual(fusion[~mask].count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
