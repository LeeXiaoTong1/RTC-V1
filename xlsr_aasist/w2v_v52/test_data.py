"""Actual V5.2 source planning and data path, with only waveform RTC mocked."""
import copy
from collections import Counter
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .sampling import Plan
from .data import Waves, microbatches
from .side import _layout
from .step import train_step
from .test_ce import fixture, optimizer
from .test_contrastive import PairDetector
from w2v_v5.sampling import Plan as ParentPlan
from w2v_v5.test_integration import inventory, ticket, fake_augment, fake_matched
from w2v_v31810.identity import build_registry, registry_config


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

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
        self.assertEqual(slots['scheduled'], [])
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

    def test_contrastive_reference_does_not_schedule_unselected_train_platform_pcl(self):
        rows, cfg = inventory()
        data = self.data(rows, cfg)
        data._phones = lambda row: [dict(instance=0, token=1, start_sample=800, end_sample=1600)]
        t = dict(ticket(0, paired=True), contrastive_online_index=1, contrastive_group_budget=2)
        t.pop('pcl_online_index')  # Train source selected for noise PCL only.
        with self.patches():
            unit = data[t]
        views, slots = _layout(unit, dict(noise=.03, platform=.045))
        self.assertEqual(slots['scheduled'], ['noise'])
        self.assertIn('noise', slots)
        self.assertIn('contrastive', slots)
        self.assertEqual(len(views), 3)

    def test_rejected_phone_matches_still_count_as_scheduled_pcl(self):
        rows, cfg = inventory()
        data = self.data(rows, cfg)
        data._phones = lambda row: []
        with self.patches():
            unit = data[ticket(0, paired=True)]
        views, slots = _layout(unit, dict(noise=.03, platform=.045))
        self.assertEqual(slots['scheduled'], ['noise', 'platform'])
        self.assertNotIn('noise', slots)
        self.assertNotIn('platform', slots)
        self.assertEqual(len(views), 2)  # Keep both CE views, skip unused Online.

    def test_real_final_plan_runs_dual_pcl_and_contrast_on_mixed_train_dev_batches(self):
        rows, data_cfg = inventory()
        cfg, _ = fixture()
        cfg.update(data_cfg, microbatch=48, frame_budget=9000,
                   contrastive_weight=.01, contrastive_ramp_updates=1,
                   pcl_noise_weight=.03, pcl_platform_weight=.045, pcl_ramp_updates=1,
                   pcl_variance_weight=0.)
        data = self.data(rows, cfg)
        def phones(row):
            self.assertEqual(row['v5_origin_split'], 'train')
            return [dict(instance=0, token=1, start_sample=800, end_sample=1600)]
        data._phones = phones
        def matched(*args, **kwargs):
            clean, noisy, meta = fake_matched(*args, **kwargs)
            for name in ('clean_activity', 'rtc_activity'):
                meta[name].update(profile='off', enabled=False, suppressed_samples=0,
                                  samples=len(clean))
            return clean, noisy, meta
        model = PairDetector()
        model.pcl_scale = SimpleNamespace(is_frozen=True, scale=torch.ones(5))
        opt = optimizer(model)
        dev_contrast = pcl_sources = active_contrast = 0
        for tickets in self.plan(rows, cfg).batches(0):
            with self.patches(), patch('w2v_v5.data.generate_matched_context', side_effect=matched):
                units = [data[t] for t in tickets]
            expected = Counter()
            for t, unit in zip(tickets, units):
                row = unit[0]
                if row['v5_origin_split'] == 'dev' and row.get('contrastive_enabled'):
                    self.assertEqual(row['pcl_platform_budget'], 0)
                    dev_contrast += 1
                if t.get('pcl_enabled'):
                    for kind, scheduled in (('noise', t.get('pcl_noise')),
                                            ('platform', 'pcl_online_index' in t)):
                        if scheduled:
                            expected[f'{kind}/{row["language"]}/{row["label"]}'] += 1
            stats = train_step(model, opt, units, cfg, audit=True)
            self.assertFalse(stats['amp']['skipped_update'])
            self.assertEqual({k: v['scheduled_sources'] for k, v in stats['pcl_coverage'].items()},
                             dict(expected))
            self.assertEqual(stats['views'], len(tickets) + sum(t['kind'] == 'offline' for t in tickets))
            self.assertEqual(stats['classification_source_views'], len(tickets))
            self.assertAlmostEqual(stats['loss'], stats['classification_loss']
                + sum(v['weighted_loss'] for v in stats['pcl_losses'].values())
                + stats['contrastive']['weighted_loss'], places=6)
            pcl_sources += sum(v['sources'] for v in stats['pcl_losses'].values())
            active_contrast += stats['contrastive'].get('active_pairs', 0)
        self.assertGreater(dev_contrast, 0)
        self.assertGreater(pcl_sources, 0)
        self.assertGreater(active_contrast, 0)

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
