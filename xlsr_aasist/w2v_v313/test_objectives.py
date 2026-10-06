"""Tests of update direction, valid correspondence and bounded objectives."""
import unittest

import torch
from torch.nn import functional as F

from .objectives import loss_terms


def paired_rows(language='en', label=1, source='source', occurrence=None,
                conditions=('offline', 'online'), weight=None):
    result = [dict(label=label, language=language, condition=condition,
                   source_id=source, pair_occurrence=occurrence or source, split='train')
              for condition in conditions]
    if weight is not None:
        for row in result:
            row['ce_weight'] = weight
    return result


def leaves(scores, vectors=None):
    scores = torch.tensor(scores, dtype=torch.float32)
    logits = torch.stack((-scores / 2, scores / 2), dim=-1).requires_grad_()
    if vectors is None:
        vectors = [[1., 0., 0.] for _ in scores]
    return logits, torch.tensor(vectors, dtype=torch.float32, requires_grad=True)


class PairedObjectiveTests(unittest.TestCase):
    def test_correct_teacher_repairs_wrong_real_and_fake_without_teacher_gradient(self):
        for label, scores in ((1, [3., -1.]), (0, [-3., 1.])):
            z, h = leaves(scores)
            terms = loss_terms(z, h, paired_rows(label=label), {'pair_feature_weight': 0.})
            grad = torch.autograd.grad(terms['pair'], z)[0]
            self.assertEqual(float(grad[0].abs().sum()), 0.)
            self.assertLess(float(grad[1, label]), 0.)
            self.assertGreater(float(grad[1, 1-label]), 0.)
            self.assertEqual(float(terms['pair_eligible_sources']), 1.)
            self.assertEqual(float(terms['pair_feature_sources']), 0.)

    def test_wrong_both_and_uncertain_teacher_cannot_distill_an_error(self):
        for scores in ([-3., -2.], [.2, -.1]):
            z, h = leaves(scores, [[1., 0., 0.], [0., 1., 0.]])
            terms = loss_terms(z, h, paired_rows(), {})
            self.assertEqual(float(terms['pair']), 0.)
            gradients = torch.autograd.grad(terms['pair'], (z, h))
            self.assertTrue(all(float(g.abs().sum()) == 0. for g in gradients))
            ce_grad = torch.autograd.grad(terms['classification'], z)[0]
            self.assertTrue(bool((ce_grad[:, 1] < 0).all()))

    def test_capped_target_does_not_reward_logit_inflation(self):
        z, h = leaves([10000., 4.1])
        terms = loss_terms(z, h, paired_rows(), {'pair_feature_weight': 0.})
        self.assertEqual(float(terms['pair']), 0.)
        z, h = leaves([10000., -10000.])
        terms = loss_terms(z, h, paired_rows(), {})
        gradient = torch.autograd.grad(terms['pair'], z)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertLessEqual(float(gradient.abs().max()), 1.)

    def test_feature_consistency_is_scale_invariant_and_teacher_detached(self):
        vectors = [[2., 0., 0.], [1., 2., 0.]]
        z, h = leaves([3., 2.], vectors)
        terms = loss_terms(z, h, paired_rows(), {'pair_margin_weight': 0., 'pair_feature_weight': 1.})
        gradient = torch.autograd.grad(terms['pair'], h)[0]
        self.assertEqual(float(gradient[0].abs().sum()), 0.)
        self.assertGreater(float(gradient[1].abs().sum()), 0.)
        updated = h.detach() - .05 * gradient
        before = F.cosine_similarity(h[0:1], h[1:2])
        after = F.cosine_similarity(updated[0:1], updated[1:2])
        self.assertGreater(float(after), float(before))
        scaled = loss_terms(z, h * 100., paired_rows(), {'pair_margin_weight': 0., 'pair_feature_weight': 1.})
        torch.testing.assert_close(scaled['pair'], terms['pair'])
        torch.testing.assert_close((gradient * h).sum(-1), torch.zeros(2), atol=1e-7, rtol=0)

    def test_ce_preserves_fractional_global_mass_and_detaches_difficulty(self):
        rows = paired_rows(weight=.025) + paired_rows(label=0, source='fake', weight=.1)
        z, h = leaves([-1., 1., -5., -4.])
        terms = loss_terms(z, h, rows, {})
        self.assertAlmostEqual(float(terms['ce_weight_mass']), .25, places=6)
        self.assertLessEqual(float(terms['hard_weight_max_observed']), 2.)
        labels = torch.tensor([1, 1, 0, 0])
        per = F.cross_entropy(z, labels, reduction='none')
        factors = torch.ones(4)
        for indices in ([0, 1], [2, 3]):
            difficulty = per[indices].detach().mean()
            factors[indices] = 1 + difficulty / (1 + difficulty)
        expected_weights = torch.tensor([.025, .025, .1, .1]) * factors
        expected_weights *= .25 / expected_weights.sum()
        expected = (per * expected_weights).sum()
        actual_grad = torch.autograd.grad(terms['classification'], z, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, z)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_no_hard_emphasis_exactly_matches_provided_ce_weights(self):
        rows = paired_rows(weight=.05) + paired_rows(label=0, source='fake', weight=.075)
        z, h = leaves([1., 2., 3., -2.])
        terms = loss_terms(z, h, rows, {'hard_weight_strength': 0.})
        expected = (F.cross_entropy(z, torch.tensor([1, 1, 0, 0]), reduction='none') * torch.tensor([.05, .05, .075, .075])).sum()
        torch.testing.assert_close(terms['classification'], expected)

    def test_repeated_source_occurrences_never_cross_pair_teachers(self):
        rows = paired_rows(occurrence='draw1') + paired_rows(occurrence='draw2')
        z, h = leaves([5., 5., -1., -2.])
        terms = loss_terms(z, h, rows, {'pair_feature_weight': 0.})
        self.assertEqual(float(terms['pair']), 0.)
        self.assertEqual(float(terms['pair_eligible_sources']), 1.)

    def test_rejects_partial_or_mislabeled_correspondence_and_non_train_data(self):
        z, h = leaves([1., 1.])
        variations = [
            [paired_rows()[0]],
            [paired_rows()[0], dict(paired_rows()[1], label=0)],
            [paired_rows()[0], dict(paired_rows()[1], source_id='other')],
            [paired_rows()[0], dict(paired_rows()[1], condition='offline')],
            [dict(row, split='dev') for row in paired_rows()],
        ]
        for rows in variations:
            with self.assertRaises(ValueError):
                loss_terms(z[:len(rows)], h[:len(rows)], rows, {})


class HardNegativeTests(unittest.TestCase):
    def test_real_fake_ranking_moves_scores_in_opposite_correct_directions(self):
        rows = paired_rows() + paired_rows(label=0, source='fake')
        z, h = leaves([-1., -.5, 1., .5])
        terms = loss_terms(z, h, rows, {'ranking_feature_weight': 0.})
        gradient = torch.autograd.grad(terms['ranking'], z)[0]
        self.assertTrue(bool((gradient[:2, 1] < 0).all()))
        self.assertTrue(bool((gradient[2:, 1] > 0).all()))
        self.assertEqual(float(terms['ranking_anchors']), 2.)
        self.assertEqual(float(terms['ranking_candidate_edges']), 2.)

    def test_only_matched_language_condition_and_other_sources_are_negatives(self):
        real = paired_rows()
        cases = [paired_rows(language='zh', label=0, source='fake'),
                 paired_rows(label=0, source='fake', conditions=('noisy_a', 'noisy_b')),
                 paired_rows(label=0, source='source', occurrence='conflicting_source_label')]
        for fake in cases:
            z, h = leaves([-1., -1., 1., 1.])
            terms = loss_terms(z, h, real + fake, {})
            self.assertEqual(float(terms['ranking']), 0.)
            self.assertEqual(float(terms['ranking_anchors']), 0.)

    def test_actual_feature_update_separates_matched_fake_and_keeps_source_positive(self):
        rows = paired_rows() + paired_rows(label=0, source='fake')
        z, h = leaves([-.5, .5, .5, -.5], [[1., 0., 0.], [.7, .7, 0.], [1., .1, 0.], [.7, .8, 0.]])
        terms = loss_terms(z, h, rows, {'ranking_logit_weight': 0.})
        gradient = torch.autograd.grad(terms['ranking'], h)[0]
        self.assertGreater(float(gradient.abs().sum()), 0.)
        updated = h.detach() - .05 * gradient
        after = loss_terms(z, updated, rows, {'ranking_logit_weight': 0.})
        self.assertLess(float(after['ranking']), float(terms['ranking']))
        self.assertEqual(float(terms['prototype']), 0.)

    def test_original_sha_aliases_are_excluded_and_repeated_negative_draws_are_one_candidate(self):
        real = [dict(row, group_id='real_sha') for row in paired_rows()]
        same_original = [dict(row, group_id='real_sha')
                         for row in paired_rows(label=0, source='alias_of_real')]
        z, h = leaves([-1., -1., 1., 1.])
        invalid = loss_terms(z, h, real + same_original, {})
        self.assertEqual(float(invalid['ranking_anchors']), 0.)
        fake_first = [dict(row, group_id='fake_sha')
                      for row in paired_rows(label=0, source='fake_first')]
        fake_repeat = [dict(row, group_id='fake_sha')
                       for row in paired_rows(label=0, source='fake_alias')]
        z, h = leaves([-1., -1., .1, .1, 2., 2.])
        terms = loss_terms(z, h, real + fake_first + fake_repeat,
                           {'ranking_feature_weight': 0.})
        self.assertEqual(float(terms['ranking_candidate_edges']), 2.)
        self.assertEqual(float(terms['ranking_candidate_views']), 4.)
        gradient = torch.autograd.grad(terms['ranking'], z)[0]
        self.assertEqual(float(gradient[2:4].abs().sum()), 0.)
        self.assertTrue(bool((gradient[4:, 1] > 0).all()))

    def test_distant_real_modes_are_not_pulled_to_a_shared_centre(self):
        rows = paired_rows(source='real_a') + paired_rows(source='real_b')
        z, h = leaves([4., 4., 4., 4.], [[1., 0., 0.], [1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]])
        terms = loss_terms(z, h, rows, {})
        self.assertEqual(float(terms['pair_feature']), 0.)
        self.assertEqual(float(terms['ranking_feature']), 0.)
        self.assertEqual(float(terms['prototype']), 0.)

    def test_huge_finite_scores_remain_finite_and_ranking_has_no_clamp_dead_zone(self):
        rows = paired_rows() + paired_rows(label=0, source='fake')
        z, h = leaves([-1000., -1000., 1000., 1000.], [[0., 0., 0.]] * 4)
        terms = loss_terms(z, h, rows, {})
        self.assertTrue(all(bool(torch.isfinite(value)) for value in terms.values()))
        gradient = torch.autograd.grad(terms['ranking_logit'], z)[0]
        self.assertTrue(bool((gradient.abs().sum(-1) > 0).all()))
        self.assertLess(float(terms['ranking_logit']), 2.3)

    def test_auxiliary_accumulation_requires_complete_groups_and_scales_by_source_fraction(self):
        rows = paired_rows() + paired_rows(label=0, source='fake')
        z, h = leaves([1., -.5, -.5, .5], [[1., 0., 0.], [0., 1., 0.], [.8, .2, 0.], [.2, .8, 0.]])
        first = loss_terms(z, h, rows, {})
        repeated = [dict(row, source_id=row['source_id'] + '_second', pair_occurrence=row['pair_occurrence'] + '_second') for row in rows]
        second = loss_terms(z, h, repeated, {})
        accumulated = .5 * (first['pair'] + first['ranking']) + .5 * (second['pair'] + second['ranking'])
        torch.testing.assert_close(accumulated, first['pair'] + first['ranking'])


if __name__ == '__main__':
    unittest.main()
