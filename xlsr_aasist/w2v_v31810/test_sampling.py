import copy
import csv
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from itertools import islice
from w2v_v3186.test_sampling import rows_for
from w2v_v3185.common import GROUPS
from .identity import canonical_sources
from .sampling import Plan, quota_schedule, validate_units, CATEGORIES, FAMILIES, TASKS


def cfg(**extra):
    return dict(source_draws_per_stream=(113, 89, 89, 89), consistency='none', **extra)


class SamplingTests(unittest.TestCase):
    def test_exact_default_budget_global_source_uniqueness_and_all_cells(self):
        rows = rows_for((23, 19, 35, 23)); plan = Plan(rows)
        self.assertEqual(plan.steps, 1403)
        self.assertEqual(sum(plan.schedule[-1]), 20)
        counts = Counter(); task_counts = Counter(); directions = Counter()
        for batch in plan.batches(0):
            self.assertEqual(len(batch), len({t['canonical_source'] for t in batch}))
            units = [[dict(rows[t['index']], **{k: v for k, v in t.items() if k != 'index'},
                           role='online' if t['kind'] == 'online' else 'noisy')] for t in batch]
            validate_units(units, batch[0]['batch_group_quotas'])
            task_counts[batch[0]['task_kind']] += 1; directions[batch[0]['task_direction']] += 1
            for t in batch: counts[t['kind'], (rows[t['index']]['language'], rows[t['index']]['label'])] += 1
        for kind in ('online', 'offline'):
            self.assertEqual([counts[kind, group] for group in GROUPS], [9668, 8000, 8000, 8000])
        self.assertLessEqual(max(task_counts.values()) - min(task_counts.values()), 1)
        self.assertLessEqual(max(directions.values()) - min(directions.values()), 1)

    def test_rearrangement_preserves_every_cycle_variant_and_epoch_resume(self):
        rows = rows_for((17, 17, 23, 19)); plan = Plan(rows, cfg=cfg())
        for epoch in range(4):
            batch = list(plan.batches(epoch))
            for kind in ('online', 'offline'):
                for group in GROUPS:
                    actual = Counter((t['index'], t['base_view_id'], t['source_cycle']) for bb in batch for t in bb
                        if t['kind'] == kind and (rows[t['index']]['language'], rows[t['index']]['label']) == group)
                    expected = Counter((i, s['base_view_id'], c) for i, s, c in plan.view_stream(kind, group, epoch))
                    self.assertEqual(actual, expected)
                    chronological = {}
                    for bb in batch:
                        for t in bb:
                            if t['kind'] != kind: continue
                            chronological.setdefault(t['index'], []).append(t['source_cycle'])
                    self.assertTrue(all(cycles == sorted(cycles) for cycles in chronological.values()))
        self.assertEqual(list(plan.batches(3)), list(Plan(rows, cfg=cfg()).batches(3)))
        i = plan.pools['online'][('en', 1)][0]
        augmented = [plan.view_spec(i, 'online', c) for c in range(20)]
        self.assertGreater(max(t['base_generation'] for t in augmented), 0)

    def test_e0_e2_identical_and_e1_only_reassigns_sides(self):
        rows = rows_for((23, 23, 23, 23))
        controls = {x: list(Plan(rows, cfg=cfg(experiment=x)).batches(2)) for x in ('E0', 'E1', 'E2')}
        self.assertEqual(controls['E0'], controls['E2'])
        changed = 0
        for mixed, held in zip(controls['E1'], controls['E2']):
            self.assertEqual([{k: v for k, v in t.items() if k != 'meta_side'} for t in mixed],
                [{k: v for k, v in t.items() if k != 'meta_side'} for t in held])
            changed += sum(a['meta_side'] != b['meta_side'] for a, b in zip(mixed, held))
            for side in ('A', 'B'):
                self.assertEqual({t['condition_block'] for t in mixed if t['meta_side'] == side}, {'X', 'Y'})
        self.assertGreater(changed, 0)

    def test_declared_domain_holdout_and_matched_controls_do_not_read_label(self):
        rows = rows_for((23, 23, 23, 23)); plan = Plan(rows, cfg=cfg())
        for batch in plan.batches(0):
            first = batch[0]; task = first['task_kind']; hold = first['heldout_condition']
            noisy = [t for t in batch if t['kind'] == 'offline']
            for t in noisy:
                d = t['domain_override']; block = t['condition_block']
                if task != 'rtc': self.assertEqual(d['scene'] == hold['scene'], block == 'Y')
                if task != 'noise': self.assertEqual(d['family'] == hold['family'], block == 'Y')
            # The common rank slots have identical domains and nuisance controls
            # in all four groups; only rotating integer remainder slots differ.
            for block in ('X', 'Y'):
                for rank in (0, 1):
                    copies = [t for t in noisy if t['condition_block'] == block and t['condition_rank'] == rank]
                    self.assertEqual(len(copies), 4)
                    self.assertEqual(len({str(t['domain_override']) for t in copies}), 1)
                    self.assertEqual(len({t['nuisance_seed'] for t in copies}), 1)
            for rank in (0, 1):
                copies = [t for t in noisy if t['condition_rank'] == rank]
                self.assertEqual(len({t['nuisance_seed'] for t in copies}), 1)

    def test_calibration_is_fixed_train_only_four_tasks_each_and_no_cursor_changes(self):
        plan = Plan(rows_for((23, 23, 23, 23)), cfg=cfg())
        before = list(plan.batches(2)); tasks = plan.calibration_batches()
        self.assertEqual(Counter(b[0]['task_kind'] for b in tasks), dict(noise=4, rtc=4, joint=4))
        self.assertTrue(all(t['calibration'] and t['occurrence'].startswith('alpha-calibration:') for b in tasks for t in b))
        self.assertEqual(tasks, plan.calibration_batches())
        self.assertEqual(before, list(plan.batches(2)))

    def test_every_held_condition_visits_both_learning_directions(self):
        plan = Plan(rows_for((23, 23, 23, 23)))
        directions = {}
        for batch in islice(plan.batches(0), 108):
            t = batch[0]; held = t['heldout_condition']
            key = t['task_kind'], held['scene'], held['family']
            directions.setdefault(key, set()).add(t['task_direction'])
        self.assertEqual(len(directions), 54)
        self.assertTrue(all(d == {'X_to_Y', 'Y_to_X'} for d in directions.values()))

    def test_explicit_offline_online_pairs_unify_distinct_manifest_ids(self):
        rows = rows_for((23, 23, 23, 23)); pairs = []
        for j in range(0, len(rows), 2):
            pairs.append((rows[j]['id'], rows[j+1]['id']))
            rows[j+1]['source_id'] += '-online-source'
            rows[j+1]['group_id'] += '-online-group'
            rows[j+1]['source_sha256'] += '-online-sha'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pairs.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                w = csv.writer(stream); w.writerow(('offline_id', 'online_id')); w.writerows(pairs)
            keys, _ = canonical_sources(rows, path, require_online_pairs=True)
            self.assertTrue(all(keys[j] == keys[j+1] for j in range(0, len(rows), 2)))
            plan = Plan(rows, cfg=cfg(train_pairs=str(path)))
            self.assertTrue(all(len(b) == len({t['canonical_source'] for t in b}) for b in plan.batches(0)))
        with self.assertRaisesRegex(ValueError, 'verified Offline'): Plan(rows, cfg=cfg())

    def test_insufficient_sources_conflicts_and_undersized_tails_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, 'insufficient'): Plan(rows_for((9, 9, 9, 9)), cfg=cfg())
        rows = rows_for((23, 23, 23, 23)); rows[1]['label'] = 1
        with self.assertRaisesRegex(ValueError, 'conflicting'): Plan(rows, cfg=cfg())
        schedule = quota_schedule((100, 80, 80, 78))
        self.assertEqual(tuple(map(sum, zip(*schedule))), (100, 80, 80, 78))
        self.assertTrue(all(min(q) >= 2 for q in schedule))


if __name__ == '__main__': unittest.main()
