"""Actual V5.2 source planning and data path, with only waveform RTC mocked."""
import copy
from contextlib import ExitStack
import unittest
from unittest.mock import patch
import numpy as np
from .sampling import Plan
from .data import Waves, microbatches
from .side import _layout
from w2v_v5.sampling import Plan as ParentPlan
from w2v_v5.test_integration import inventory, ticket, fake_augment, fake_matched
from w2v_v31810.identity import build_registry, registry_config


class DataTests(unittest.TestCase):
    def plan(self, rows, cfg, cls=Plan):
        return cls(rows, cfg['stream_sources'], cfg['sampling_seed'],
                   cfg['group_quotas'], cfg['view_factors'], cfg=cfg)

    def test_sampler_preserves_every_classification_draw_and_admits_dev_pairs_without_pcl(self):
        rows, cfg = inventory()
        old = list(self.plan(rows, cfg, ParentPlan).batches(0))
        new = list(self.plan(rows, cfg).batches(0))
        self.assertEqual([[t['index'] for t in b] for b in old], [[t['index'] for t in b] for b in new])
        dev_pairs = 0
        for batch in new:
            self.assertEqual(len(batch), len({t['canonical_source'] for t in batch}))
            for t in batch:
                if 'contrastive_online_index' in t:
                    row, online = rows[t['index']], rows[t['contrastive_online_index']]
                    self.assertEqual(row['canonical_source'], online['canonical_source'])
                    self.assertEqual(t['meta_side'], 'A')
                    self.assertEqual(t['kind'], 'offline')
                    self.assertLessEqual(t['contrastive_group_budget'], 3)
                    if row['v5_origin_split'] == 'dev':
                        self.assertFalse(t['pcl_enabled'])
                        self.assertNotIn('pcl_online_index', t)
                        dev_pairs += 1
        self.assertGreater(dev_pairs, 0)
        disabled = self.plan(rows, dict(cfg, contrastive_weight=0.))
        self.assertTrue(all('contrastive_online_index' not in t for b in disabled.batches(0) for t in b))

    def test_missing_official_pair_keeps_source_classification(self):
        rows, cfg = inventory()
        missing = rows[1]['canonical_source']
        rows.pop(1)
        cfg['source_registry'] = registry_config(build_registry(rows, [], {}))
        old = list(self.plan(rows, cfg, ParentPlan).batches(0))
        new = list(self.plan(rows, cfg).batches(0))
        self.assertEqual([[t['index'] for t in b] for b in old], [[t['index'] for t in b] for b in new])
        tickets = [t for batch in new for t in batch if t['canonical_source'] == missing]
        self.assertTrue(tickets)
        self.assertTrue(all('contrastive_online_index' not in t for t in tickets))

    def data(self, rows, cfg):
        data = Waves(rows, cfg)
        self.reads = []
        wave = np.random.default_rng(9).normal(0, .1, 3200).astype(np.float32)
        def original(row):
            self.reads.append(row['id'])
            return wave.copy()
        data.original = original
        data.bank, data.engines = object(), object()
        return data

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch('w2v_v5.data.make_recipe', return_value={}))
        stack.enter_context(patch('w2v_v5.data.augment', side_effect=fake_augment))
        stack.enter_context(patch('w2v_v5.data.generate_matched_context', side_effect=fake_matched))
        stack.enter_context(patch('w2v_v5.data.estimate_delay', return_value=dict(valid=True, offset=0.)))
        stack.enter_context(patch('w2v_v5.data.reliable_spans', side_effect=lambda a, b, spans, *args: [True]*len(spans)))
        return stack

    def test_final_dev_pair_requires_no_phones_and_counts_all_three_views(self):
        rows, cfg = inventory()
        index = next(i for i, r in enumerate(rows) if r['v5_origin_split'] == 'dev' and r['condition'] == 'offline')
        data = self.data(rows, cfg)
        data._phones = lambda row: self.fail('Dev pair incorrectly requested CTC phonemes')
        t = dict(ticket(index), contrastive_online_index=index+1, contrastive_group_budget=2)
        with self.patches():
            unit = data[t]
        row = unit[0]
        self.assertTrue(row['contrastive_enabled'])
        self.assertEqual(row['pcl_pairs'], dict(noise=[], platform=[]))
        self.assertEqual(len(self.reads), 2)
        views, slots = _layout(unit, dict(noise=.03, platform=.045))
        self.assertEqual(len(views), 3)
        self.assertEqual(slots['contrastive'], (0, 2))
        self.assertTrue(all(len(v['wave']) == 48000 for v in views))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            list(microbatches([unit], 2, 9000))
        self.assertEqual(len(list(microbatches([unit], 3, 9000))), 1)
        row['pcl_refs']['official']['canonical_source'] = 'wrong-source'
        with self.assertRaisesRegex(ValueError, 'canonical'):
            _layout(unit, dict(noise=0., platform=0.))

    def test_existing_pcl_official_wave_is_reused_without_second_read(self):
        rows, cfg = inventory()
        data = self.data(rows, cfg)
        phones = [dict(instance=0, token=1, start_sample=800, end_sample=1600)]
        data._phones = lambda row: copy.deepcopy(phones)
        t = dict(ticket(0, paired=True), contrastive_online_index=1, contrastive_group_budget=2)
        with self.patches():
            unit = data[t]
        self.assertEqual(self.reads, [rows[0]['id'], rows[1]['id']])
        views, slots = _layout(unit, dict(noise=.03, platform=.045))
        self.assertEqual(len(views), 3)
        self.assertEqual(slots['contrastive'][1], slots['platform'][0])
        self.assertTrue(unit[0]['pcl_pairs']['platform'])

    def test_mismatched_pair_fails_and_invalid_receiver_pair_only_skips_contrast(self):
        rows, cfg = inventory()
        data = self.data(rows, cfg)
        t = dict(ticket(0), contrastive_online_index=3, contrastive_group_budget=2)
        with self.patches(), self.assertRaisesRegex(ValueError, 'disagrees'):
            data[t]
        def invalid(*args, **kwargs):
            clean, noisy, meta = fake_matched(*args, **kwargs)
            return clean, noisy, dict(meta, pair_eligible=False)
        with self.patches(), patch('w2v_v5.data.generate_matched_context', side_effect=invalid):
            unit = data[dict(t, contrastive_online_index=1)]
        self.assertEqual(unit[0]['contrastive_skip'], 'invalid_receiver_pair')
        self.assertEqual(len(_layout(unit, dict(noise=0., platform=0.))[0]), 2)


if __name__ == '__main__':
    unittest.main()
