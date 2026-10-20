import unittest

import numpy as np

from .selection import (balanced_source_subset, choose_candidate, plan_tail_updates, score)


def records(count=4):
    rows, keys = [], []
    for language in ('en', 'zh'):
        for label in (0, 1):
            for source in range(count):
                key = f'{language}:{label}:{source}'
                for condition in ('online', 'seen', 'heldout'):
                    rows.append(dict(id=key + ':' + condition, condition=condition,
                        language=language, label=label))
                    keys.append(key)
    return rows, keys


class ScoreTests(unittest.TestCase):
    def test_perfect_fixed_threshold_and_four_groups(self):
        rows, _ = records(2)
        logits = np.asarray([[2., -2.] if r['label'] == 0 else [-2., 2.] for r in rows])
        value = score(rows, logits)
        self.assertEqual(value['weighted_f1'], 1.)
        self.assertEqual(set(value['four_groups']), {'en-real', 'en-fake', 'zh-real', 'zh-fake'})
        self.assertEqual(value['four_groups']['en-real']['noisy']['count'], 4)
        self.assertEqual(value['threshold'], .5)

    def test_noisy_metric_pools_rows_instead_of_averaging_panel_metrics(self):
        rows, _ = records(1)
        # Expand one condition's support to make a mean-of-panels visibly wrong.
        rows += [dict(row, id=row['id'] + ':copy') for row in rows if row['condition'] == 'seen'] * 3
        logits = np.asarray([[2., -2.] if row['label'] == 0 else [-2., 2.] for row in rows])
        for i, row in enumerate(rows):
            if row['condition'] == 'heldout':
                logits[i] *= -1
        result = score(rows, logits)
        self.assertAlmostEqual(result['clean_f1'], 1.)
        self.assertAlmostEqual(result['noisy_f1'], .8)
        self.assertAlmostEqual(result['weighted_f1'], .86)

    def test_exact_half_is_fake_and_no_class_guard_changes_score(self):
        rows, _ = records(1)
        value = score(rows, np.zeros((len(rows), 2)))
        self.assertAlmostEqual(value['weighted_f1'], 1 / 3)
        self.assertEqual(value['four_groups']['en-real']['clean']['recall'], 0.)

    def test_bad_logits_or_missing_class_panel_rejected(self):
        rows, _ = records(1)
        with self.assertRaisesRegex(ValueError, 'N x 2'):
            score(rows, np.zeros((len(rows), 1)))
        with self.assertRaisesRegex(ValueError, 'finite'):
            score(rows, np.full((len(rows), 2), np.nan))
        filtered = [r for r in rows if r['condition'] != 'online' or r['label'] == 0]
        with self.assertRaisesRegex(ValueError, 'both Real and Fake'):
            score(filtered, np.zeros((len(filtered), 2)))


class SourceSubsetTests(unittest.TestCase):
    def test_equal_source_budget_all_panels_and_reproducible(self):
        rows, keys = records(7)
        first = balanced_source_subset(rows, range(len(rows)), 3, 42, source_keys=keys)
        second = balanced_source_subset(rows, reversed(range(len(rows))), 3, 42, source_keys=keys)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4 * 3 * 3)
        selected = {keys[i] for i in first}
        for source in selected:
            self.assertEqual({rows[i]['condition'] for i in first if keys[i] == source},
                {'online', 'seen', 'heldout'})
        self.assertNotEqual(first, balanced_source_subset(rows, range(len(rows)), 3, 43, source_keys=keys))

    def test_allowed_partition_is_source_complete_and_never_escaped(self):
        rows, keys = records(5)
        allowed = [i for i, key in enumerate(keys) if not key.endswith(':4')]
        chosen = balanced_source_subset(rows, allowed, 2, 42, source_keys=keys)
        self.assertTrue(set(chosen).issubset(allowed))
        with self.assertRaisesRegex(ValueError, 'split a canonical source'):
            balanced_source_subset(rows, allowed[1:], 2, 42, source_keys=keys)

    def test_unpaired_sources_allowed_and_smallest_group_sets_equal_budget(self):
        rows, keys = records(3)
        keep = [i for i, row in enumerate(rows) if not (
            keys[i].endswith(':1') and row['condition'] == 'online') and
            not (row['language'] == 'en' and row['label'] == 1 and keys[i].endswith(':2'))]
        rows = [rows[i] for i in keep]
        keys = [keys[i] for i in keep]
        chosen = balanced_source_subset(rows, range(len(rows)), 9, 42, source_keys=keys)
        counts = {}
        for language in ('en', 'zh'):
            for label in (0, 1):
                counts[language, label] = len({keys[i] for i in chosen
                    if rows[i]['language'] == language and rows[i]['label'] == label})
        self.assertEqual(set(counts.values()), {2})
        # Both classes in both aggregate panels still exist.
        score([rows[i] for i in chosen], np.zeros((len(chosen), 2)))

    def test_identity_conflicts_duplicate_indices_and_missing_keys_rejected(self):
        rows, keys = records(2)
        with self.assertRaisesRegex(ValueError, 'exact Dev'):
            balanced_source_subset(rows, range(len(rows)), 1, 42, source_keys=keys[:-1])
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            balanced_source_subset(rows, [0, 0], 1, 42, source_keys=keys)
        bad = list(keys)
        bad[-1] = keys[0]
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            balanced_source_subset(rows, range(len(rows)), 1, 42, source_keys=bad)


class CandidateSelectionTests(unittest.TestCase):
    def test_weighted_score_can_trade_clean_or_group_recall_for_noisy(self):
        candidates = [
            dict(name='identity', metrics=dict(weighted_f1=.94, clean_f1=.99)),
            dict(name='other', metrics=dict(weighted_f1=.95, clean_f1=.97,
                en_real_recall=.8)),
        ]
        self.assertIs(choose_candidate(candidates), candidates[1])

    def test_exact_tie_identity_then_conservative_then_stable_input_order(self):
        other = dict(name='filter', metrics=dict(weighted_f1=.94), conservative_rank=0)
        identity = dict(name='identity', metrics=dict(weighted_f1=.94), conservative_rank=99)
        self.assertIs(choose_candidate([other, identity]), identity)
        strong = dict(name='strong', metrics=dict(weighted_f1=.94), conservative_rank=2)
        self.assertIs(choose_candidate([strong, other]), other)
        equal = dict(other)
        self.assertIs(choose_candidate([equal, other]), equal)

    def test_invalid_scores_do_not_silently_select_a_candidate(self):
        with self.assertRaises(ValueError):
            choose_candidate([])
        for invalid in (None, float('nan'), float('inf'), 94.):
            with self.assertRaises(ValueError):
                choose_candidate([dict(name='bad', metrics=dict(weighted_f1=invalid))])

    def test_tail_schedule_uses_committed_updates_only_and_keeps_endpoint(self):
        self.assertEqual(plan_tail_updates(100), [76, 84, 92, 100])
        self.assertEqual(plan_tail_updates(4), [4])
        self.assertEqual(plan_tail_updates(1), [1])
        self.assertEqual(plan_tail_updates(10, snapshots=100), [8, 9, 10])
        self.assertEqual(plan_tail_updates(100, snapshots=1), [100])
        for total in (1, 10, 1403):
            planned = plan_tail_updates(total)
            self.assertEqual(planned, sorted(set(planned)))
            self.assertEqual(planned[-1], total)
            self.assertTrue(all(i > total * .75 for i in planned))
        for args in ((0,), (3, 0), (5, 4, 1), (5, 4, -.1)):
            with self.assertRaises(ValueError):
                plan_tail_updates(*args)


if __name__ == '__main__':
    unittest.main()
