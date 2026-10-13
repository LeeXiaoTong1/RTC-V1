"""Native tiny w2v-BERT: physical batching, group budgets and update safety."""
import copy
import unittest
from unittest.mock import patch

import torch

from w2v_v3151.test_step import examples as old_examples, model_fixture
from .objectives import TFCL
from .step import train_step, triplet_batches, validate_examples


def examples():
    rows = old_examples()
    masses = dict(online=.3, reference=.1, noisy=.6)
    for row in rows:
        row.update(source_weight=.25, ce_weight=.25*masses[row['role']], official_online=True)
    return rows


def config(**changes):
    cfg = dict(source_batch=4, group_quotas=(1, 1, 1, 1), microbatch=12, frame_budget=1000, device='cpu', amp='none',
        tfcl_time_weight=.15, tfcl_structure_weight=.045, tfcl_bridge_mass=.2,
        tfcl_matched_mass=.8, max_grad_norm=1.)
    cfg.update(changes)
    return cfg


def optimizer(model):
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.02)


class StepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(319)
        torch.set_num_threads(1)

    def test_grouping_preserves_losses_gradients_and_each_view_forward_once(self):
        a, rows = model_fixture(), examples()
        b = copy.deepcopy(a)
        aux = TFCL(8, 2, 13)
        first = train_step(a, aux, optimizer(a), rows, config(microbatch=3), audit=True)
        calls = []
        handle = b.backbone.register_forward_pre_hook(
            lambda _m, _args, kw: calls.append(len(kw['input_features'])), with_kwargs=True)
        try:
            second = train_step(b, aux, optimizer(b), rows, config())
        finally:
            handle.remove()
        self.assertEqual(sum(calls), len(rows))
        self.assertLess(second['physical_forwards'], first['physical_forwards'])
        for key in ('classification_loss', 'time_loss', 'structure_loss', 'total_loss'):
            self.assertAlmostEqual(first[key], second[key], places=6, msg=key)
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5, msg=name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5, msg=name)
        self.assertEqual((first['bridge_pairs'], first['matched_pairs']), (4, 4))
        self.assertEqual(first['aux_parameter_grad_norm'], 0.)
        for value in first['gradient_alignment'].values():
            self.assertGreater(value['auxiliary_norm'], 0.)
            self.assertGreater(value['ce_norm'], 0.)
            self.assertGreaterEqual(value['cosine'], -1.00001)
            self.assertLessEqual(value['cosine'], 1.00001)

    def test_missing_official_bridge_does_not_renormalize_main_budget(self):
        rows = examples()
        for row in rows:
            row['online_pair_eligible'] = False
            row['official_online'] = False
        model, aux = model_fixture(), TFCL(8, 2, 13)
        stats = train_step(model, aux, optimizer(model), rows, config())
        self.assertEqual(stats['bridge_pairs'], 0)
        self.assertEqual(stats['matched_pairs'], 4)
        self.assertEqual(stats['bridge_skipped_missing_online'], 4)
        self.assertAlmostEqual(stats['time_loss'], .8*stats['matched_time_loss'], places=6)
        self.assertAlmostEqual(stats['structure_loss'], .8*stats['matched_structure_loss'], places=6)

    def test_disabled_edges_and_structure_do_not_compute_unused_losses(self):
        model, aux = model_fixture(), TFCL(8, 2, 13)
        with patch.object(aux, '_indices', side_effect=AssertionError('Disabled bridge was computed')), \
                patch.object(aux, '_structure', side_effect=AssertionError('Disabled CKA was computed')):
            stats = train_step(model, aux, optimizer(model), examples(),
                config(tfcl_bridge_mass=0., tfcl_matched_mass=1., tfcl_structure_weight=0.))
        self.assertEqual(stats['bridge_pairs'], 0)
        self.assertEqual(stats['matched_pairs'], 4)
        self.assertEqual(stats['structure_loss'], 0.)
        with patch.object(aux, 'forward_batch', side_effect=AssertionError('CE-only computed consistency')):
            stats = train_step(model, aux, optimizer(model), examples(),
                config(tfcl_time_weight=0., tfcl_structure_weight=0.))
        self.assertEqual(stats['time_loss'], 0.)

    def test_unequal_quotas_receive_equal_group_ce_and_pair_mass(self):
        base, rows = examples(), []
        quotas = [4, 3, 5, 4]
        for group_index, count in enumerate(quotas):
            for j in range(count):
                for row in base[group_index*3:group_index*3+3]:
                    value = dict(row, pair_occurrence=f'{group_index}:{j}', source_id=f'{group_index}:{j}')
                    value['source_weight'] = .25/count
                    value['ce_weight'] = row['ce_weight']/count
                    rows.append(value)
        cfg = config(source_batch=16, group_quotas=quotas)
        grouped = validate_examples(rows, cfg)
        self.assertEqual(len(grouped), 16)
        for index in range(4):
            occurrences = [v for k, v in grouped.items() if k.startswith(str(index)+':')]
            self.assertAlmostEqual(sum(v[0]['source_weight'] for v in occurrences), .25)
        rows[0]['ce_weight'] *= 2
        with self.assertRaisesRegex(ValueError, 'budget'):
            validate_examples(rows, cfg)

    def test_nonfinite_loss_or_gradient_never_calls_optimizer(self):
        for kind in ('loss', 'gradient'):
            with self.subTest(kind=kind):
                model, aux = model_fixture(), TFCL(8, 2, 13)
                opt = optimizer(model)
                before = {k: v.clone() for k, v in model.state_dict().items()}
                if kind == 'loss':
                    handle = model.head.classifier.register_forward_hook(
                        lambda _m, _args, output: output*float('nan'))
                else:
                    handle = model.head.projection.weight.register_hook(
                        lambda gradient: torch.full_like(gradient, float('nan')))
                try:
                    with patch.object(opt, 'step', wraps=opt.step) as update:
                        with self.assertRaisesRegex(FloatingPointError, 'Nonfinite'):
                            train_step(model, aux, opt, examples(), config())
                        update.assert_not_called()
                finally:
                    handle.remove()
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name], atol=0, rtol=0)
                self.assertEqual(len(model.head.blocks[0]._forward_pre_hooks), 0)


if __name__ == '__main__':
    unittest.main()
