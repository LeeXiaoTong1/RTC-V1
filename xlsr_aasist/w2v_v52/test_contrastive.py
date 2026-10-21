"""Numerical contrast, real gradient paths, bounded memory and failed commits."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .contrastive import (empty_bank, paired_loss, next_bank, validate_bank, bank_for,
                          classify_with_features, effective_weight)
from .step import train_step
from .side import side_pass
from .test_ce import fixture, optimizer, Detector
from w2v_v3191.meta import rng_state
from w2v_v3191.test_step import assert_rng_equal


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.aasist = nn.Module()
        self.aasist.drop = nn.Dropout(.0)
        self.aasist.out_layer = nn.Linear(5, 2)

    def forward(self, features):
        return self.aasist.out_layer(self.aasist.drop(features))


class PairDetector(Detector):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Module()
        self.fusion.projection = nn.Linear(4, 5)
        self.head = Head()

    def encode(self, waves):
        self.counter += 1
        self.encode_counts.append((self.training, len(waves)))
        return [self.fusion.projection(torch.as_tensor(wave[:4])).repeat(12, 1) for wave in waves]


def pair_fixture(copies=1):
    cfg, units = fixture(copies)
    cfg.update(contrastive_weight=.01, contrastive_ramp_updates=1, microbatch=12)
    for unit in units:
        row = unit[0]
        if row['role'] == 'noisy' and row['meta_side'] == 'A':
            row.update(contrastive_enabled=True, contrastive_group_budget=copies)
            row['pcl_refs']['official'] = dict(row, role='official_ref',
                wave=row['wave']*.4 + np.tile(np.array([.2, -.1, .4, .1], np.float32), 1000))
    return cfg, units


def pair(source, label, language='en', budget=1):
    row = dict(canonical_source=source, split='train', language=language, label=label,
               contrastive_group_budget=budget)
    return (torch.randn(5, requires_grad=True), torch.randn(5, requires_grad=True), row)


class ContrastiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(59)

    def test_symmetric_infonce_matches_manual_and_reaches_both_views(self):
        pairs = [pair('real', 1), pair('fake', 0)]
        cfg = dict(contrastive_temperature=.2)
        actual, stats, pending = paired_loss(pairs, empty_bank(cfg), cfg, torch.tensor(0.))
        manual = []
        for i, (noisy, official, _) in enumerate(pairs):
            for a, b, c in ((noisy, official, pairs[1-i][1]), (official, noisy, pairs[1-i][0])):
                positive = F.cosine_similarity(a, b, dim=0)/.2
                negative = F.cosine_similarity(a, c, dim=0)/.2
                manual.append(F.softplus(negative-positive)*.25/2)
        torch.testing.assert_close(actual, torch.stack(manual).sum())
        self.assertEqual(stats['negative_comparisons'], 4)
        self.assertEqual(stats['active_pairs'], 2)
        actual.backward()
        for noisy, official, _ in pairs:
            self.assertGreater(float(noisy.grad.norm()), 0.)
            self.assertGreater(float(official.grad.norm()), 0.)
        self.assertTrue(all(not entry['feature'].requires_grad for entry in pending))

    def test_singleton_uses_only_opposite_label_same_language_other_source(self):
        cfg = {}
        current = pair('current', 1)
        entries = []
        for source, language, label in (('valid', 'en', 0), ('same_class', 'en', 1),
                                        ('other_language', 'zh', 0), ('current', 'en', 0)):
            for view in ('noisy', 'official'):
                entries.append(dict(source=source, language=language, label=label, view=view,
                                    split='train', feature=F.normalize(torch.randn(5), dim=0)))
        bank = next_bank(empty_bank(cfg), entries, cfg)
        loss, stats, _ = paired_loss([current], bank, cfg, torch.tensor(0.))
        self.assertEqual(stats['directions'], 2)
        self.assertEqual(stats['negative_comparisons'], 2)
        self.assertEqual(stats['queue_negatives'], 2)
        loss.backward()
        self.assertGreater(float(current[0].grad.norm()), 0.)
        # Missing negatives are skipped and still queued for future updates.
        loss, stats, pending = paired_loss([pair('a', 1), pair('b', 1)], empty_bank(cfg), cfg, torch.tensor(0.))
        self.assertEqual(float(loss), 0.)
        self.assertEqual(stats['skipped_no_negative'], 4)
        self.assertEqual(len(pending), 4)

    def test_bank_is_bounded_deduplicated_train_only_and_nonmutating(self):
        cfg = dict(contrastive_bank_per_group=2)
        bank = empty_bank(cfg)
        _, _, pending = paired_loss([pair(str(i), 1) for i in range(5)], bank, cfg, torch.tensor(0.))
        other = next_bank(bank, pending, cfg)
        self.assertEqual(bank['entries'], [])
        self.assertEqual(len(other['entries']), 4)
        self.assertEqual({r['source'] for r in other['entries']}, {'3', '4'})
        again = next_bank(other, pending[-2:], cfg)
        self.assertEqual(len(again['entries']), 4)
        with self.assertRaisesRegex(ValueError, 'non-Train'):
            next_bank(bank, [dict(pending[0], split='dev')], cfg)
        with self.assertRaises(ValueError):
            validate_bank(other, dict(contrastive_bank_per_group=3))
        bad = pair('dev', 0)
        bad[2]['split'] = 'dev'
        with self.assertRaises(ValueError):
            paired_loss([bad], bank, cfg, torch.tensor(0.))
        with self.assertRaises(ValueError):
            paired_loss([pair('same', 0), pair('same', 1)], bank, cfg, torch.tensor(0.))

    def test_feature_capture_is_actual_classifier_input_and_hook_is_removed(self):
        model = PairDetector()
        sequences = [torch.randn(12, 5, requires_grad=True) for _ in range(3)]
        logits, readouts = classify_with_features(model, sequences)
        expected = torch.stack([s.mean(0) for s in sequences])
        torch.testing.assert_close(readouts, expected, rtol=0, atol=0)
        torch.testing.assert_close(logits, model.head.aasist.out_layer(expected))
        self.assertEqual(len(model.head.aasist.drop._forward_pre_hooks), 0)
        readouts.square().sum().backward()
        self.assertTrue(all(s.grad is not None for s in sequences))
        with patch.object(model, 'classify_sequences', side_effect=RuntimeError('broken')):
            with self.assertRaises(RuntimeError):
                classify_with_features(model, sequences)
        self.assertEqual(len(model.head.aasist.drop._forward_pre_hooks), 0)

    def test_production_aasist_pooling_exposes_384d_with_backward_and_no_parameter_changes(self):
        from w2v_v3189.test_model import tiny_cfg, tiny_model
        model = tiny_model(dict(tiny_cfg(), checkpoint_layers=0))
        model.eval()
        keys = set(model.state_dict())
        sequences = [torch.randn(n, 128, requires_grad=True) for n in (25, 140)]
        logits, features = classify_with_features(model, sequences)
        self.assertEqual(features.shape, (2, 384))
        torch.testing.assert_close(logits, model.head.aasist.out_layer(features), rtol=0, atol=0)
        features.square().mean().backward()
        for sequence in sequences:
            self.assertTrue(bool(torch.isfinite(sequence.grad).all()))
            self.assertGreater(float(sequence.grad.norm()), 0.)
        self.assertEqual(keys, set(model.state_dict()))

    def test_full_step_has_no_duplicate_encoder_views_and_loss_reconciles(self):
        cfg, units = pair_fixture()
        model = PairDetector()
        stats = train_step(model, optimizer(model), units, cfg, audit=True)
        self.assertEqual(stats['encoded_views'], 28)  # 16 main + 8 clean + 4 official.
        self.assertEqual(sum(n for _, n in model.encode_counts), 28)
        self.assertEqual(stats['views'], 24)  # Official references add no CE mass.
        self.assertEqual(stats['selection']['scoring_encoded_views'], 0)
        self.assertEqual(stats['contrastive']['active_pairs'], 4)
        self.assertEqual(stats['contrastive']['bank_entries_after'], 8)
        self.assertGreater(stats['contrastive']['weighted_loss'], 0.)
        self.assertGreater(stats['auxiliary_gradient_audit']['norms']['contrastive'], 0.)
        self.assertAlmostEqual(stats['loss'], stats['classification_loss'] + stats['contrastive']['weighted_loss'], places=6)
        json.dumps(stats, allow_nan=False)
        self.assertFalse(any('contrast' in name for name in model.state_dict()))

    def test_singleton_microbatch_bank_warmup_is_explicit_then_active(self):
        cfg, units = pair_fixture()
        cfg['microbatch'] = 3
        model = PairDetector()
        opt = optimizer(model)
        first = train_step(model, opt, units, cfg)
        self.assertEqual(first['contrastive'].get('active_pairs', 0), 0)
        self.assertEqual(first['contrastive']['skipped_no_negative'], 8)
        second = train_step(model, opt, units, cfg)
        self.assertEqual(second['contrastive']['active_pairs'], 4)
        self.assertEqual(second['contrastive']['queue_negatives'], 8)

    def test_amp_failure_restores_rng_parameters_buffers_optimizer_and_bank(self):
        cfg, units = pair_fixture()
        model = PairDetector()
        opt = optimizer(model)
        scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        with patch('w2v_v52.step.scaler_for', return_value=scaler):
            train_step(model, opt, units, cfg)
            state, saved_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
            bank = copy.deepcopy(bank_for(model, cfg))
            rng = rng_state()
            handle = model.head.aasist.out_layer.weight.register_hook(lambda gradient: gradient * float('nan'))
            failed = train_step(model, opt, units, cfg)
            handle.remove()
            self.assertTrue(failed['amp']['skipped_update'])
            self.assertEqual(scaler.get_scale(), 64.)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
            for a, b in zip(bank['entries'], bank_for(model, cfg)['entries']):
                self.assertEqual(a['source'], b['source'])
                torch.testing.assert_close(a['feature'], b['feature'], rtol=0, atol=0)
            for key, values in saved_optimizer['state'].items():
                for name, value in values.items():
                    torch.testing.assert_close(opt.state_dict()['state'][key][name], value, rtol=0, atol=0)
            assert_rng_equal(self, rng, rng_state())
            self.assertFalse(train_step(model, opt, units, cfg)['amp']['skipped_update'])

    def test_new_pair_loss_changes_actual_update_and_ramp_is_monotonic(self):
        cfg, units = pair_fixture()
        first = PairDetector()
        second = copy.deepcopy(first)
        train_step(first, optimizer(first), units, cfg)
        train_step(second, optimizer(second), units, dict(cfg, contrastive_weight=0.))
        self.assertGreater(float((first.fusion.projection.weight-second.fusion.projection.weight).norm()), 0.)
        self.assertEqual([effective_weight(dict(current_update=u)) for u in (0, 50, 100, 150)], [0., .005, .01, .01])

    def test_noise_pcl_and_its_variance_scale_by_point_two_platform_unchanged(self):
        cfg, units = pair_fixture()
        cfg.update(contrastive_weight=0., consistency='dual', microbatch=96,
            pcl_noise_weight=.15, pcl_platform_weight=.045, pcl_ramp_updates=1,
            v52_parent_pcl_weights=dict(noise=.15, platform=.045),
            pcl_variance_min_sources=1, pcl_variance_min_phones=1,
            pcl_variance_target=4., pcl_variance_weight=.01)
        units = [u for u in units if u[0]['meta_side'] == 'A']
        for unit in units:
            row = unit[0]
            if row['role'] == 'noisy':
                row.update(contrastive_enabled=False, pcl_noisy_index=0, pcl_platform_budget=1,
                    pcl_pairs=dict(noise=[[0, 2000, 0, 2000]], platform=[[0, 2000, 0, 2000]]))
        model = PairDetector()
        model.pcl_scale = SimpleNamespace(is_frozen=True, scale=torch.ones(5))
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        original = side_pass(model, units, cfg, scaler, backward=False)
        reduced = side_pass(model, units, dict(cfg, pcl_noise_weight=.03), scaler, backward=False)
        noise_only = side_pass(model, units, dict(cfg, pcl_platform_weight=0.), scaler, backward=False)
        platform_only = side_pass(model, units, dict(cfg, pcl_noise_weight=0.), scaler, backward=False)
        self.assertGreater(original['pcl_losses']['noise']['weighted_loss'], 0.)
        self.assertAlmostEqual(reduced['pcl_losses']['noise']['weighted_loss'],
                               .2*original['pcl_losses']['noise']['weighted_loss'], places=7)
        self.assertEqual(reduced['pcl_losses']['platform']['weighted_loss'],
                         original['pcl_losses']['platform']['weighted_loss'])
        self.assertGreater(platform_only['pcl_variance']['weighted_loss'], 0.)
        self.assertAlmostEqual(reduced['pcl_variance']['weighted_loss'],
            .2*noise_only['pcl_variance']['weighted_loss']+platform_only['pcl_variance']['weighted_loss'], places=7)


if __name__ == '__main__':
    unittest.main()
