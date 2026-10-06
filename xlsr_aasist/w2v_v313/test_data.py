"""True source pairing, balanced loss mass, transport and deterministic resume."""
from collections import Counter
import copy
import hashlib
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from .data import (SourcePlan, PairedAudioDataset, paired_microgroups, loss_weights,
                   heldout_pairs, probe_split, loader)


def records(count=24, missing_online=False):
    result = []
    for language in ('en', 'zh'):
        for label in (0, 1):
            n = count if isinstance(count, int) else count[(language, label)]
            for index in range(n):
                source = f'offline/{language}/{label}_{index}.wav'
                original_hash = hashlib.sha256(source.encode()).hexdigest()
                for condition in ('offline', 'online', 'noisy_a', 'noisy_b'):
                    if missing_online and language == 'en' and condition == 'online':
                        continue
                    result.append(dict(source_id=source, group_id=original_hash,
                        source_sha256=original_hash, id=source if condition != 'online'
                        else f'online/{language}/{label}_{index}.wav',
                        language=language, label=label, condition=condition,
                        split='train', view='full', full_length=True,
                        noisy=condition.startswith('noisy'), output_samples=9600,
                        audio=condition + '/' + source,
                        audio_sha256=original_hash if condition == 'offline'
                        else hashlib.sha256((source + condition).encode()).hexdigest()))
    return result


def examples(plan, tickets):
    return [dict(plan.rows[ticket.index], pair_occurrence=ticket.pair_occurrence,
        pair_position=ticket.pair_position, pair_kind=ticket.pair_kind,
        requested_pair_kind=ticket.requested_pair_kind, microgroup=ticket.microgroup,
        ce_weight=ticket.ce_weight, pair_weight=ticket.pair_weight) for ticket in tickets]


class SourcePlanTests(unittest.TestCase):
    def test_exact_two_views_balanced_mass_unique_occurrences_even_with_repeats(self):
        plan = SourcePlan(records(1), 32, 0)
        data = examples(plan, plan.batches(0)[0])
        self.assertEqual(len(data), 64)
        self.assertEqual(len({r['source_id'] for r in data}), 4)
        self.assertEqual(len({r['pair_occurrence'] for r in data}), 32)
        ce, adv = loss_weights(data)
        self.assertAlmostEqual(sum(ce), 1.)
        self.assertAlmostEqual(sum(adv), 1.)
        for language in ('en', 'zh'):
            for label in (0, 1):
                self.assertAlmostEqual(sum(w for w, r in zip(ce, data)
                    if (r['language'], r['label']) == (language, label)), .25)
        groups = paired_microgroups(data)
        self.assertEqual([len(group) for group in groups], [16] * 4)
        for group in groups:
            for a, b in zip(group[::2], group[1::2]):
                self.assertEqual((a['source_id'], a['source_sha256']), (b['source_id'], b['source_sha256']))
                self.assertNotEqual(a['condition'], b['condition'])
        audit = plan.coverage()
        self.assertEqual(audit['actual_pair_kinds'],
            dict(official_offline_online=16, offline_noisy=8, online_noisy=8))
        self.assertEqual(audit['original_audio_cache_bytes_added'], 0)

    def test_resume_recreates_views_not_just_source_ids_and_rotates_conditions(self):
        plan = SourcePlan(records(24), 16, 31301)
        for epoch in (0, 1, 3):
            complete = plan.batches(epoch)
            self.assertEqual(plan.batches(epoch, 2), complete[2:])
            self.assertEqual(plan.batches(epoch, plan.steps), [])
            seen = {plan.rows[t.index]['condition'] for batch in complete for t in batch}
            self.assertEqual(seen, {'offline', 'online', 'noisy_a', 'noisy_b'})
            self.assertEqual(pickle.loads(pickle.dumps(complete)), complete)
        self.assertNotEqual(plan.batches(0), plan.batches(1))
        for group in paired_microgroups(examples(plan, plan.batches(0)[0])):
            for language in ('en', 'zh'):
                for label in (0, 1):
                    self.assertEqual(len({r['group_id'] for r in group
                        if (r['language'], r['label']) == (language, label)}), 2)

    def test_source_pool_continues_before_repeating_majority_and_audits_minority_repeats(self):
        counts = {('en', 0): 101, ('en', 1): 4, ('zh', 0): 101, ('zh', 1): 4}
        plan = SourcePlan(records(counts), 16, 31301)
        ids = []
        for epoch in range(2):
            for batch in plan.batches(epoch):
                ids.extend(plan.rows[item.index]['source_id'] for item in batch
                    if item.pair_position == 0 and plan.rows[item.index]['language'] == 'en'
                    and plan.rows[item.index]['label'] == 0)
        self.assertEqual(len(set(ids[:101])), 101)
        audit = plan.coverage()['source_groups']
        self.assertEqual(audit['en/0']['draws'], audit['en/1']['draws'])
        self.assertGreater(audit['en/1']['maximum_repeats'], 1)

    def test_missing_online_has_explicit_fallback_and_preserves_source_budget(self):
        plan = SourcePlan(records(1, missing_online=True), 32, 0)
        data = examples(plan, plan.batches(0)[0])
        ce, _ = loss_weights(data)
        self.assertAlmostEqual(sum(w for w, r in zip(ce, data) if r['language'] == 'en'), .5)
        self.assertTrue(all(r['condition'] != 'online' for r in data if r['language'] == 'en'))
        self.assertEqual({r['condition'] for r in data if r['language'] == 'en'},
                         {'offline', 'noisy_a', 'noisy_b'})
        audit = plan.coverage()
        self.assertEqual(audit['missing_online_sources'], 2)
        self.assertEqual(audit['actual_pair_kinds']['missing_online_offline_noisy'], 12)

    def test_rejects_conflicting_or_forged_pair_manifests(self):
        base = records(1)
        changes = [('source_sha256', 'f' * 64), ('language', 'zh'), ('label', 1),
                   ('view', 'prefix'), ('split', 'dev'), ('full_length', False)]
        for field, value in changes:
            changed = copy.deepcopy(base)
            changed[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                SourcePlan(changed, 4, 0, micro_sources=4)
        changed = copy.deepcopy(base)
        changed[0]['audio_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'verified original'):
            SourcePlan(changed, 4, 0, micro_sources=4)
        changed = copy.deepcopy(base)
        changed[5]['id'] = changed[1]['id']
        with self.assertRaisesRegex(ValueError, 'multiple sources'):
            SourcePlan(changed, 4, 0, micro_sources=4)

    def test_missing_endpoints_cross_source_pairs_and_local_renormalization_rejected(self):
        plan = SourcePlan(records(1), 16, 0)
        data = examples(plan, plan.batches(0)[0])
        with self.assertRaises(ValueError):
            paired_microgroups(data[:-1])
        bad = copy.deepcopy(data)
        bad[1]['source_id'] = bad[2]['source_id']
        with self.assertRaisesRegex(ValueError, 'original source'):
            paired_microgroups(bad)
        with self.assertRaisesRegex(ValueError, 'full logical'):
            loss_weights(data[:16])
        with self.assertRaises(ValueError):
            plan.batches(0, -1)


class HoldoutAndTransportTests(unittest.TestCase):
    def test_two_holdouts_are_disjoint_by_original_hash_and_keep_all_views(self):
        data = records(40)
        cfg = dict(seed=31301, pair_probe_sources_per_group=4, probe_sources_per_language=4)
        # Duplicate IDs with the same original waveform must also be held out together.
        _, preliminary, _ = heldout_pairs(data, cfg)
        target = next(r['group_id'] for r in preliminary if r['language'] == 'en' and r['label'] == 0)
        duplicate = [dict(row, source_id=row['source_id'] + '_alias',
                          id=row['id'] + '_alias') for row in data if row['group_id'] == target]
        training, held, report = heldout_pairs(data + duplicate, cfg)
        self.assertEqual(sum(r['source_id'].endswith('_alias') for r in held), 4)
        self.assertFalse(any(r['source_id'].endswith('_alias') for r in training))
        remaining, probe, _, probe_report = probe_split(training, cfg)
        a = {r['group_id'] for r in remaining}
        b = {r['group_id'] for r in held}
        c = {r['group_id'] for r in probe}
        self.assertFalse(a & b or a & c or b & c)
        self.assertEqual(len(b), 16)
        for group in b:
            self.assertEqual({r['condition'] for r in held if r['group_id'] == group},
                             {'offline', 'online', 'noisy_a', 'noisy_b'})
        self.assertEqual(report['overlap'], 0)
        self.assertIn('V3.13', probe_report['scope'])
        self.assertEqual(Counter((r['language'], r['label']) for r in held if r['condition'] == 'offline')
                         [('en', 1)], 4)

    def test_verified_audio_receives_integer_ticket_and_preserves_waveform_and_weights(self):
        plan = SourcePlan(records(1), 4, 0, micro_sources=4)
        dataset = PairedAudioDataset(plan.rows, training=False, max_seconds=0., rawboost=0)
        wave = np.linspace(-.8, .8, 9600, dtype=np.float32)
        ticket = plan.batches(0)[0][0]
        with patch('w2v_aasist.data.read_wave', return_value=wave):
            sample = dataset[ticket]
        np.testing.assert_array_equal(sample['wave'], wave)
        self.assertEqual(sample['pair_occurrence'], ticket.pair_occurrence)
        self.assertEqual(sample['ce_weight'], 1. / 8)
        with self.assertRaises(TypeError):
            dataset[0]

    def test_loader_keeps_numpy_worker_transport_unpinned_and_no_audio_mutation(self):
        plan = SourcePlan(records(1), 4, 0, micro_sources=4)
        cfg = dict(seed=31301, ssl_path='local/ssl', workers=2)
        with patch('w2v_v313.data.DataLoader', side_effect=lambda **kwargs: kwargs):
            args = loader(plan, 1, 0, cfg)
        self.assertFalse(args['pin_memory'])
        self.assertEqual(args['multiprocessing_context'], 'spawn')
        self.assertEqual(args['prefetch_factor'], 1)
        self.assertFalse(args['dataset'].training)
        self.assertEqual(args['dataset'].max_samples, 0)
        self.assertEqual(args['dataset'].rawboost, 0)


if __name__ == '__main__':
    unittest.main()
