"""CPU numerical and state-isolation tests, without audio/model downloads."""
import copy
import json
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from w2v_v3191.meta import CELLS, rng_state
from .step import train_step
from .ce import CE_WEIGHTS


class Detector(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.fusion = nn.Linear(4, 5)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(5, 2))
        self.register_buffer('counter', torch.tensor(0))
        self.joint = True
        self.encode_counts = []

    def encode(self, waves):
        self.counter += 1
        self.encode_counts.append((self.training, len(waves)))
        # Exercise all random streams even when the stochastic head is off.
        if not self.training:
            random.random()
            np.random.random()
            torch.rand(1)
        return [self.fusion(torch.as_tensor(wave[:4])).repeat(12, 1) for wave in waves]

    def classify_sequences(self, sequences):
        return self.head(torch.stack([sequence.mean(0) for sequence in sequences]))

    def set_phase(self, joint):
        self.joint = joint


def fixture(copies=3):
    rng = np.random.default_rng(41)
    units = []
    for side in ('A', 'B'):
        for role, language, label in CELLS:
            for j in range(copies):
                identity = f'{side}-{role}-{language}-{label}-{j}'
                wave = np.tile(rng.normal(size=4).astype(np.float32), 1000)
                row = dict(id=identity, source_id=identity, group_id=identity,
                           canonical_source=identity, meta_side=side, role=role,
                           language=language, label=label, split='train', wave=wave,
                           batch_group_quotas=(copies,)*4)
                if role == 'noisy':
                    candidates = []
                    for k, mechanism in enumerate(('legacy', 'm1', 'm2', 'm3')):
                        candidate = dict(row, wave=wave*(1+k*.45)+k*.15,
                                         robust_mechanism=mechanism,
                                         robust_quality=dict(passed=True, reasons=[]))
                        candidate['pcl_refs'] = {'clean': dict(row, wave=wave*.7,
                                                               role='clean_ref')}
                        candidates.append([candidate])
                    row = dict(candidates[0][0], robust_candidates=candidates)
                units.append([row])
    cfg = dict(device='cpu', backend_precision='fp32', microbatch=8,
               frame_budget=9000, label_smoothing=.02, max_grad_norm=1.,
               group_quotas=(copies,)*4, consistency='none', seed=3191,
               robust_arm='baseline', robust_hard_fraction=.5, current_update=3,
               classification_ce_weights=dict(CE_WEIGHTS), contrastive_weight=0.)
    return cfg, units


def optimizer(model):
    return torch.optim.AdamW([dict(params=model.parameters(), name='detector', lr=.002)],
                             weight_decay=.001)


def direct_loss(model, units, smoothing=.02):
    # Independent definition: three separate four-class macro means.
    means = {}
    for view in ('online', 'clean_rtc', 'noisy'):
        groups = []
        for language in ('en', 'zh'):
            for label in (0, 1):
                values = []
                for unit in units:
                    row = unit[0]
                    role = 'online' if view == 'online' else 'noisy'
                    if (row['role'], row['language'], row['label']) != (role, language, label):
                        continue
                    example = row['pcl_refs']['clean'] if view == 'clean_rtc' else row
                    logits = model.classify_sequences(model.encode([example['wave']]))
                    values.append(F.cross_entropy(logits, torch.tensor([label]),
                                                  label_smoothing=smoothing))
                groups.append(torch.stack(values).mean())
        means[view] = torch.stack(groups).mean()
    return .6*means['online'] + .2*means['clean_rtc'] + .2*means['noisy']


def assert_rng_equal(test, a, b):
    test.assertEqual(a['python'], b['python'])
    test.assertEqual(a['numpy'][0], b['numpy'][0])
    np.testing.assert_array_equal(a['numpy'][1], b['numpy'][1])
    test.assertEqual(a['numpy'][2:], b['numpy'][2:])
    torch.testing.assert_close(a['torch'], b['torch'], rtol=0, atol=0)


class StepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)
        random.seed(33)
        np.random.seed(46)

    def test_paired_balanced_accumulation_matches_explicit_objective(self):
        cfg, units = fixture()
        one, two = Detector(), Detector()
        two.load_state_dict(one.state_dict())
        a, b = optimizer(one), optimizer(two)
        result = train_step(one, a, units, cfg)
        loss_a = direct_loss(two, [u for u in units if u[0]['meta_side'] == 'A'])
        loss_b = direct_loss(two, [u for u in units if u[0]['meta_side'] == 'B'])
        loss = (loss_a + loss_b)/2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(two.parameters(), 1.)
        b.step()
        for p, q in zip(one.parameters(), two.parameters()):
            torch.testing.assert_close(p, q, rtol=3e-6, atol=2e-7)
        self.assertAlmostEqual(result['loss'], float(loss.detach()), places=6)
        self.assertEqual(result['classification_source_views'], len(units))
        self.assertEqual(result['views'], len(units)*3//2)
        self.assertEqual(result['training_encoded_views'], result['views'])
        json.dumps(result, allow_nan=False)

    def test_microbatch_changes_do_not_change_source_weights(self):
        cfg, units = fixture()
        one, two = Detector(), Detector()
        two.load_state_dict(one.state_dict())
        train_step(one, optimizer(one), units, dict(cfg, microbatch=2))
        train_step(two, optimizer(two), units, dict(cfg, microbatch=16))
        for p, q in zip(one.parameters(), two.parameters()):
            torch.testing.assert_close(p, q, rtol=3e-6, atol=2e-7)
        self.assertTrue(all(count <= 2 for _, count in one.encode_counts))

    def test_unequal_group_counts_match_explicit_weighted_sgd_update(self):
        cfg, units = fixture()
        quotas = (1, 2, 3, 1)
        groups = [('en', 0), ('en', 1), ('zh', 0), ('zh', 1)]
        units = [u for u in units if int(u[0]['id'].rsplit('-', 1)[1]) <
                 quotas[groups.index((u[0]['language'], u[0]['label']))]]
        for unit in units:
            unit[0]['batch_group_quotas'] = quotas
            for candidate in [unit] if unit[0]['role'] == 'noisy' else []:
                candidate[0]['batch_group_quotas'] = quotas
        cfg.update(group_quotas=quotas, max_grad_norm=1e6, microbatch=3)
        one, two = Detector(), Detector()
        two.load_state_dict(one.state_dict())
        a = torch.optim.SGD([dict(params=one.parameters(), name='detector')], lr=.03)
        b = torch.optim.SGD(two.parameters(), lr=.03)
        result = train_step(one, a, units, cfg)
        loss = direct_loss(two, units)
        loss.backward()
        b.step()
        self.assertAlmostEqual(result['loss'], float(loss.detach()), places=6)
        for p, q in zip(one.parameters(), two.parameters()):
            torch.testing.assert_close(p, q, rtol=2e-6, atol=2e-8)

    def test_exact_view_mass_and_separate_clean_noisy_statistics(self):
        class ViewDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.tensor([
                    [1., -1.], [-1., 1.],       # Online correct
                    [2., -2.], [-2., 2.],       # Clean-RTC correct
                    [-3., 3.], [3., -3.]]))     # Noisy wrong

            def encode(self, waves):
                return [self.logits[int(w[0])].unsqueeze(0) for w in waves]

            def classify_sequences(self, sequences):
                return torch.stack([s.mean(0) for s in sequences])

        cfg, units = fixture(copies=1)
        cfg.update(max_grad_norm=1e6, microbatch=3)
        for unit in units:
            row = unit[0]
            if row['role'] == 'online':
                row['wave'] = np.full(4000, row['label'], dtype=np.float32)
            else:
                for candidate in [[row]]:
                    candidate[0]['wave'] = np.full(4000, 4+row['label'], dtype=np.float32)
                    candidate[0]['pcl_refs']['clean']['wave'] = np.full(
                        4000, 2+row['label'], dtype=np.float32)
        model = ViewDetector()
        expected = model.logits.detach().clone().requires_grad_()
        labels = torch.tensor([0, 1])
        losses = [F.cross_entropy(expected[i:i+2], labels, label_smoothing=.02)
                  for i in (0, 2, 4)]
        loss = .6*losses[0] + .2*losses[1] + .2*losses[2]
        loss.backward()
        opt = torch.optim.SGD([dict(params=model.parameters(), name='detector')], lr=.1)
        result = train_step(model, opt, units, cfg)
        torch.testing.assert_close(model.logits, expected.detach()-.1*expected.grad,
                                   rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(result['classification_loss'], float(loss.detach()), places=6)
        self.assertEqual(result['classification_ce_weights'],
                         {'online': .6, 'clean_rtc': .2, 'noisy': .2})
        for i, (view, weight) in enumerate((('online', .6), ('clean_rtc', .2), ('noisy', .2))):
            cells = [v for k, v in result['cells'].items() if k.startswith(view+'/')]
            self.assertEqual(sum(c['views'] for c in cells), 8)
            self.assertEqual(sum(c['correct'] for c in cells), 0 if view == 'noisy' else 8)
            self.assertAlmostEqual(sum(c['objective'] for c in cells),
                                   weight*float(losses[i].detach()), places=6)

    def test_missing_or_old_weights_fail_before_model_update(self):
        cfg, units = fixture()
        for weights in (None, {'online': .5, 'clean_rtc': .25, 'noisy': .25}):
            model = Detector()
            opt = optimizer(model)
            before = copy.deepcopy(model.state_dict())
            with self.assertRaisesRegex(ValueError, 'start a new run'):
                train_step(model, opt, units, dict(cfg, classification_ce_weights=weights))
            self.assertFalse(model.encode_counts)
            self.assertFalse(opt.state)
            for key, value in before.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_clean_reference_is_encoded_once_for_ce_and_pcl(self):
        cfg, units = fixture()
        model = Detector()
        model.pcl_scale = SimpleNamespace(is_frozen=True, scale=torch.ones(5))
        for unit in units:
            for candidate in [unit] if unit[0]['role'] == 'noisy' else []:
                candidate[0]['pcl_noisy_index'] = 0
                candidate[0]['pcl_pairs'] = {'noise': [[0, 2000, 0, 2000]]}
        result = train_step(model, optimizer(model), units,
                            dict(cfg, consistency='dual', pcl_variance_weight=0.))
        self.assertEqual(result['training_encoded_views'], 72)
        self.assertEqual(result['paired_classification_views'], 24)
        self.assertEqual(result['auxiliary_reference_views'], 0)
        self.assertEqual(sum(v['valid_sources'] for k, v in result['pcl_coverage'].items()
                             if k.startswith('noise/')), 12)
        self.assertGreater(result['pcl_losses']['noise']['weighted_loss'], 0.)



if __name__ == '__main__':
    unittest.main()
