"""CPU tests for language coefficients and loss/gradient preservation."""
import json
import unittest

import torch
from torch.nn import functional as F

from .core import objective, pair_loss
from .language import LanguageBudget, language_id


def make_pool(counts):
    ids, labels = [], []
    for label, row in enumerate(counts):
        for language, count in enumerate(row):
            ids += [f'offline/{("en", "zh")[language]}/{label}_{i}.wav' for i in range(count)]
            labels += [label] * count
    return ids, labels


class LanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(456)

    def test_directory_parser(self):
        cases = {'offline/en/en_voice.wav': 0, 'offline/zh/en.wav': 1,
                 r'C:\data\offline\EN\a.wav': 0, '/data/zh/x.wav': 1}
        for source, expected in cases.items():
            self.assertEqual(language_id(source), expected)
        for source in ('a_en.wav', 'english/a.wav', 'en/zh/a.wav', 'en/../a.wav', 'en', 'zh/', '', None):
            with self.subTest(source=source), self.assertRaises(ValueError):
                language_id(source)

    def test_expected_class_budgets_and_serializable_report(self):
        ids, labels = make_pool([[2, 8], [1, 4]])
        budget = LanguageBudget(ids, labels)
        self.assertEqual(budget.lookup.dtype, torch.float32)
        self.assertEqual(budget.lookup.device.type, 'cpu')
        languages = torch.tensor([language_id(x) for x in ids])
        y = torch.tensor(labels)
        coefficients = budget.coefficients(y, languages)
        for label, q in ((0, .40), (1, .35)):
            mask = y == label
            self.assertAlmostEqual(float(coefficients[mask].mean()), 1., places=6)
            english = mask & (languages == 0)
            self.assertAlmostEqual(float(coefficients[english].sum()/coefficients[mask].sum()), q, places=6)
        restored = json.loads(json.dumps(budget.report))
        self.assertEqual(restored['counts'], [[2, 8], [1, 4]])
        self.assertEqual(restored['source_count'], 15)

    def test_audited_branch_pool_counts(self):
        pools = {'ordinary': [[14632, 46800], [2922, 11431]],
                 'rtc_pair': [[7133, 23102], [1422, 5468]],
                 'noisy_pair': [[7499, 23698], [1500, 5963]]}
        lookups = []
        for name, count in pools.items():
            with self.subTest(branch=name):
                budget = LanguageBudget(*make_pool(count))
                lookups.append(budget.lookup)
                expected = torch.tensor([[q*sum(row)/row[0], (1-q)*sum(row)/row[1]]
                                         for row, q in zip(count, [.4, .35])])
                torch.testing.assert_close(budget.lookup, expected)
                self.assertEqual(budget.report['counts'], count)
        self.assertFalse(torch.equal(lookups[0], lookups[1]))
        self.assertFalse(torch.equal(lookups[1], lookups[2]))

    def test_budget_invalid_inputs(self):
        ids, labels = make_pool([[1, 1], [1, 1]])
        bad = [([], []), (ids, labels[:-1]), (ids + [ids[0]], labels + [0]),
               (ids[:3], labels[:3]), (['unknown/a.wav'] + ids[1:], labels),
               (ids, [0, 2, 1, 1]), (ids, [0., 0., 1., 1.]),
               (ids, [False, False, True, True]), (ids, [[0], [0], [1], [1]])]
        for pool_ids, pool_labels in bad:
            with self.subTest(ids=pool_ids, labels=pool_labels), self.assertRaises(ValueError):
                LanguageBudget(pool_ids, pool_labels)
        duplicate = ids + [ids[0].replace('/', '\\')]
        with self.assertRaises(ValueError):
            LanguageBudget(duplicate, labels + [0])
        for target in (-1., 0., 1., 1.1, float('nan'), float('inf'), True, None, '.35'):
            with self.subTest(target=target), self.assertRaises(ValueError):
                LanguageBudget(ids, labels, en_real=target)

    def test_coefficient_tensor_validation(self):
        budget = LanguageBudget(*make_pool([[1, 1], [1, 1]]))
        y, lang = torch.tensor([0, 1]), torch.tensor([0, 1])
        for a, b in ((y, lang[:1]), (y.float(), lang), (y, lang.bool()),
                     (torch.tensor([0, 2]), lang), (y, torch.tensor([-1, 0])),
                     ([0, 1], lang), (y, [0, 1])):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                budget.coefficients(a, b)
        self.assertEqual(budget.coefficients(y[:0], lang[:0]).shape, (0,))

    def batch(self, stage):
        n, r, s = 4, (2 if stage >= 2 else 0), (2 if stage == 3 else 0)
        y = torch.tensor([0, 0, 1, 1] + [0, 1]*r + [0, 1]*s)
        z, h = torch.randn(len(y), 2), torch.randn(len(y), 8)
        return z, h, y, n, r, s

    def run_loss(self, z, h, y, n, r, s, language, real_cost):
        z, h = z.clone().requires_grad_(), h.clone().requires_grad_()
        loss, stats = objective(z, h, y, n, r, s, torch.tensor([.7, 1.8]), .13, .07,
                                real_ce_weight=real_cost, language_weights=language)
        grads = torch.autograd.grad(loss, (z, h), allow_unused=True)
        return loss.detach(), stats, grads

    def test_none_and_all_ones_preserve_loss_and_gradients_all_stages(self):
        for stage in (1, 2, 3):
            for real_cost in (1., 1.25):
                with self.subTest(stage=stage, real_cost=real_cost):
                    batch = self.batch(stage)
                    old = self.run_loss(*batch, None, real_cost)
                    ones = self.run_loss(*batch, torch.ones_like(batch[2], dtype=torch.float32), real_cost)
                    self.assertTrue(torch.equal(old[0], ones[0]))
                    self.assertEqual(old[1], ones[1])
                    for ga, gb in zip(old[2], ones[2]):
                        if ga is None:
                            self.assertIsNone(gb)
                        else:
                            self.assertTrue(torch.equal(ga, gb))

    def test_all_branches_use_original_denominators(self):
        z, h, y, n, r, s = self.batch(3)
        language = torch.tensor([1.7, .8, 1.8, .7, 1.6, .9, 1.6, .9, 1.4, .8, 1.4, .8])
        for real_cost in (1., 1.25):
            loss, stats, _ = self.run_loss(z, h, y, n, r, s, language, real_cost)
            ce = F.cross_entropy(z, y, reduction='none')
            cost = torch.where(y == 1, real_cost, 1.)
            base = torch.tensor([.7, 1.8])[y[:n]] * cost[:n]
            expected = {'ordinary': (ce[:n]*language[:n]*base).sum()/base.sum()}
            for key, start, stop in [('real_pair', 4, 8), ('noisy_reference', 8, 10), ('noisy_processed', 10, 12)]:
                expected[key] = (ce[start:stop]*language[start:stop]*cost[start:stop]).sum()/cost[start:stop].sum()
            for key, value in expected.items():
                self.assertAlmostEqual(stats['ce_'+key], float(value), places=6)
            classification = sum(expected[key]*weight for key, weight in stats['coefficients'].items())
            reference = classification + .13*pair_loss(h[4:6], h[6:8], y[4:6]) + .07*pair_loss(h[8:10], h[10:12], y[8:10])
            torch.testing.assert_close(loss, reference)

    def test_single_language_branch_weight_is_not_cancelled(self):
        batch = self.batch(3)
        before = self.run_loss(*batch, None, 1.25)
        after = self.run_loss(*batch, torch.full_like(batch[2], 2., dtype=torch.float32), 1.25)
        for key in ('ce', 'ce_ordinary', 'ce_real_pair', 'ce_noisy_reference', 'ce_noisy_processed'):
            self.assertAlmostEqual(after[1][key], 2*before[1][key], places=6)
        for key in ('rtc', 'noisy_pair'):
            self.assertEqual(before[1][key], after[1][key])
        self.assertTrue(torch.equal(before[2][1], after[2][1]))
        torch.testing.assert_close(after[2][0], before[2][0]*2)

    def test_consistency_is_not_language_weighted(self):
        z, h, y, n, r, s = self.batch(3)
        z[8:10] = torch.tensor([[6., -6.], [-6., 6.]])
        args = (z, h, y, n, r, s, torch.tensor([.7, 1.8]), .13, .07)
        _, a = objective(*args, consistency_weight=.02)
        _, b = objective(*args, consistency_weight=.02, language_weights=torch.full((len(y),), 2.))
        self.assertEqual(a['consistency'], b['consistency'])
        self.assertEqual(a['consistency_accepted'], b['consistency_accepted'])

    def test_objective_invalid_language_weights(self):
        batch = self.batch(3)
        size = len(batch[2])
        bad = [torch.ones(size, 1), torch.ones(size-1), torch.ones(size, dtype=torch.long),
               torch.zeros(size), -torch.ones(size), torch.full((size,), float('nan')),
               torch.full((size,), float('inf')), torch.ones(size, device='meta'),
               torch.full((size,), 1e99, dtype=torch.float64),
               torch.full((size,), 1e-99, dtype=torch.float64), [1.] * size]
        for language in bad:
            with self.subTest(language=language), self.assertRaises(ValueError):
                self.run_loss(*batch, language, 1.25)

    def test_original_pair_label_guards_remain_active(self):
        z, h, y, n, r, s = self.batch(3)
        for index in (6, 10):
            bad = y.clone()
            bad[index] = 1-bad[index]
            with self.assertRaises(ValueError):
                self.run_loss(z, h, bad, n, r, s, torch.ones(len(y)), 1.25)


if __name__ == '__main__':
    unittest.main()
