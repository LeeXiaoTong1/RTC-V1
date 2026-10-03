"""Risk gradients, global four-group budgets, absent Online and bounded execution."""
import copy
import unittest
import torch
from w2v_v3.test_model_step import tiny_detector
from w2v_v33.model import install_runtime
from .losses import group_weights, source_components, objective
from .step import supervised_step, source_rows

torch.set_num_threads(1)


def examples(count=4):
    rows = []
    for source in range(count):
        for condition in ('offline', 'online', 'noisy_a', 'noisy_b'):
            if source == 1 and condition == 'online':
                continue
            length = 13 + source * 2
            rows.append(dict(source_id=str(source), condition=condition, view='full',
                language='en' if source % 4 < 2 else 'zh', label=source % 2,
                features=torch.randn(1, length, 160), mask=torch.ones(1, length, dtype=torch.long)))
    return rows


def model():
    result = install_runtime(tiny_detector(checkpointing=True)).configure_trainable_layers(2).train()
    result.backbone.config.conformer_conv_dropout = 0.
    for module in result.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    return result


class RiskTests(unittest.TestCase):
    def test_four_groups_have_equal_epoch_mass_including_last_partial_batch(self):
        counts = {'en/0': 7, 'en/1': 2, 'zh/0': 4, 'zh/1': 1}
        weights = group_weights(counts)
        totals = {key: count * weights[key] / 4 for key, count in counts.items()}
        self.assertTrue(all(value == sum(counts.values()) / 16 for value in totals.values()))
        with self.assertRaises(ValueError):
            group_weights({**counts, 'en/1': 0})

    def test_difficult_noisy75_easy25_and_symmetric_tie(self):
        rows = examples(1)
        parts = source_components(rows, {key: 1. for key in ('en/0', 'en/1', 'zh/0', 'zh/1')})
        values = torch.tensor([1., 2., 4., 1.], requires_grad=True)
        loss, _ = objective(values, parts)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([.1, .3, .45, .15]))
        tied = torch.tensor([1., 2., 3., 3.], requires_grad=True)
        objective(tied, parts)[0].backward()
        torch.testing.assert_close(tied.grad, torch.tensor([.1, .3, .3, .3]))

    def test_missing_online_renormalizes_source_without_duplicate_classweight(self):
        rows = [row for row in examples(2) if row['source_id'] == '1']
        parts = source_components(rows, {'en/0': 1., 'en/1': 2., 'zh/0': 1., 'zh/1': 1.},
                                  source_denominator=4)
        values = torch.ones(3, requires_grad=True)
        loss, _ = objective(values, parts)
        loss.backward()
        self.assertAlmostEqual(float(loss), .5)
        torch.testing.assert_close(values.grad, torch.tensor([.1, .3, .3]) / .7 * .5)

    def test_invalid_source_rejected_before_any_update(self):
        rows = examples(1)[:-1]
        detector = model()
        opt = torch.optim.AdamW(detector.parameters(), lr=1e-3)
        calls = []
        hook = detector.backbone.register_forward_pre_hook(lambda *_: calls.append(1))
        with self.assertRaisesRegex(ValueError, 'both full noisy'):
            supervised_step(detector, rows, opt, group_weights(dict.fromkeys(('en/0', 'en/1', 'zh/0', 'zh/1'), 1)), 'cpu')
        hook.remove()
        self.assertFalse(calls)
        self.assertFalse(opt.state)

    def test_worker_padded_source_chunks_reuse_identical_tensor_storage(self):
        from w2v_v32.batching import PreparedBatch
        from w2v_v32.model import microbatches
        rows = examples(2)
        batches = []
        for source in ('0', '1'):
            indices = [i for i, row in enumerate(rows) if row['source_id'] == source]
            subset = [rows[i] for i in indices]
            batches.extend(([indices[i] for i in ids], features, mask)
                           for ids, features, mask in microbatches(subset, 4, 1000))
        prepared = PreparedBatch(rows, batches, 4, 1000)
        indices = [i for i, row in enumerate(rows) if row['source_id'] == '1']
        chosen = source_rows(prepared, indices, 4, 1000)
        self.assertIsInstance(chosen, PreparedBatch)
        self.assertEqual(len(chosen.batches), 1)
        self.assertEqual(chosen.batches[0][1].data_ptr(), batches[-1][1].data_ptr())
        self.assertEqual(chosen.batches[0][2].data_ptr(), batches[-1][2].data_ptr())
        self.assertEqual(chosen.batches[0][0], [0, 1, 2])

    def test_bounded_source_backward_matches_combined_graph_one_forward_each_view(self):
        torch.manual_seed(9)
        left = model(); right = copy.deepcopy(left)
        rows = examples(4)
        weights = group_weights(dict.fromkeys(('en/0', 'en/1', 'zh/0', 'zh/1'), 1))
        opt_left = torch.optim.SGD(left.parameters(), lr=.001)
        opt_right = torch.optim.SGD(right.parameters(), lr=.001)
        seen = []
        hook = left.backbone.register_forward_pre_hook(lambda module, args, kwargs: seen.append(kwargs['input_features'].shape[0]), with_kwargs=True)
        a, scores_a = supervised_step(left, rows, opt_left, weights, 'cpu', amp='none',
            source_microbatch=1, microbatch=1, frame_budget=1000)
        hook.remove()
        b, scores_b = supervised_step(right, rows, opt_right, weights, 'cpu', amp='none',
            source_microbatch=4, microbatch=1, frame_budget=1000)
        self.assertEqual(sum(seen), len(rows))
        self.assertAlmostEqual(a['loss'], b['loss'], places=6)
        torch.testing.assert_close(scores_a, scores_b, atol=2e-6, rtol=2e-6)
        for name, parameter in left.named_parameters():
            torch.testing.assert_close(parameter, dict(right.named_parameters())[name], atol=2e-7, rtol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA BF16 required')
    def test_cuda_bf16_checkpoint_cpu_spill_and_ema_complete_step(self):
        from .optim import optimizer_for, EMA
        detector = model().cuda()
        optimizer = optimizer_for(detector, dict(trainable_layers=2), 'joint')
        ema = EMA(detector)
        rows = examples(4)
        weights = group_weights(dict.fromkeys(('en/0', 'en/1', 'zh/0', 'zh/1'), 1))
        before = {name: p.detach().clone() for name, p in detector.named_parameters()}
        stats, logits = supervised_step(detector, rows, optimizer, weights, 'cuda', amp='bf16',
            source_microbatch=2, microbatch=2, frame_budget=200,
            gpu_activation_gib=0., gpu_reserve_gib=0., activation_budget_gib=1.)
        ema.update(detector)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(stats['activation_offload_gib'], 0.)
        for prefix in ('backbone.encoder.layers.0.', 'backbone.encoder.layers.1.', 'head.'):
            self.assertTrue(any(not torch.equal(p, before[name]) for name, p in detector.named_parameters()
                                if name.startswith(prefix)))
        for name, parameter in detector.named_parameters():
            if name.startswith('backbone.') and not name.startswith('backbone.encoder.layers.'):
                torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
