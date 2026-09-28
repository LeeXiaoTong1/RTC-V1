"""CPU checks for content selection, fixed-budget view plans and replay safety."""
from collections import Counter, defaultdict
import copy
import json
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.coverage_crop import select_window
from utils.data_utils import SpoofAudioDataset
from utils.rtc_data import BalancedPairBatchSampler
from rtc_noisy_v2.cache import RotatingNoisyDataset
from .coverage import CoverageViewBatchSampler, CoverageMeter
from .data import SeededDataset, SourceTaggedDataset, FeatureCollator, combine_batches
from .language_data_tests import BundleFixture


def make_sampler(steps=401, seed=42, en_real=.35, en_fake=.40):
    banks = [[], []]
    for label in (0, 1):
        for language in ('en', 'zh'):
            for i in range(18):
                source = f'offline/{language}/{label}_{i}.wav'
                for bank in range(2):
                    for band in range(4):
                        row = {'source': source, 'label': label, 'band': band, 'audio': 'unused'}
                        if bank:
                            row['processing'] = {'family': 'webrtc' if (i+band) % 2 else 'ffmpeg'}
                        banks[bank].append(row)
    dataset = RotatingNoisyDataset(banks, 'unused')
    source = BalancedPairBatchSampler(dataset.sources, 4, steps, seed)
    result = CoverageViewBatchSampler(source, dataset, seed+1, en_real, en_fake)
    result.configure_mixture(.2, steps)
    return result, dataset


class CropTests(unittest.TestCase):
    def test_tail_is_used_without_changing_length_or_augmentation_rng(self):
        wave = np.linspace(-.2, .2, 4*64600, dtype=np.float32)
        random.seed(10)
        np.random.seed(10)
        torch.manual_seed(10)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        proposals = [select_window(wave, seed=seed) for seed in range(200)]
        self.assertTrue(any(p['start'] > 64600 for p in proposals))
        self.assertTrue(70 < sum(p['kind'] == 'random' for p in proposals) < 130)
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_equal(np.random.get_state(), before[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertEqual(select_window(wave, seed=100), select_window(wave, seed=100))

    def test_silent_proposals_fall_back_and_short_files_keep_original_repeat(self):
        self.assertEqual(select_window(np.zeros(150000), seed=1, prefix_probability=0)['kind'], 'fallback_prefix')
        self.assertEqual(select_window(np.ones(100), seed=1)['kind'], 'short')
        with self.assertRaises(ValueError):
            select_window(np.array([np.nan]))
        with self.assertRaises(ValueError):
            select_window(np.ones(100000), prefix_probability=float('nan'))

    def test_real_dataset_crops_after_rawboost_before_noise_and_dev_stays_prefix(self):
        source = 'offline/en/test.wav'
        wave = np.arange(64600*3, dtype=np.float32)/np.float32(1000000)
        args = SimpleNamespace(coverage_training=True, coverage_prefix_probability=0.)
        train = SpoofAudioDataset([source], '.', {source: 1}, args=args, use_rawboost=True)
        seen = []
        train.env_noise = lambda x, rate: seen.append(x.copy()) or x
        seeded = SeededDataset(train, 42)
        seeded.epoch[0] = 1
        def raw(x, *unused):
            self.assertEqual(len(x), len(wave))
            return x+.1
        with patch('utils.data_utils.librosa.load', return_value=(wave, 16000)), \
             patch('utils.data_utils.process_rawboost_feature', side_effect=raw):
            row = seeded[0]
            same = seeded[0]
            start = row[-1]['crop']['start']
            self.assertGreater(start, 0)
            self.assertEqual(row[0].shape, (64600,))
            np.testing.assert_array_equal(seen[0], (wave+.1)[start:start+64600])
            torch.testing.assert_close(row[0], same[0], rtol=0, atol=0)
            seeded.epoch[0] = 2
            self.assertNotEqual(seeded[0][-1]['crop']['start'], start)
            dev = SpoofAudioDataset([source], '.', {source: 1}, args=args, use_rawboost=False)
            self.assertEqual(len(dev[0]), 3)
            np.testing.assert_array_equal(dev[0][0], wave[:64600])
        with patch('utils.data_utils.librosa.load', return_value=(np.ones(100, np.float32), 16000)), \
             patch('utils.data_utils.process_rawboost_feature', side_effect=lambda x, *a: x):
            self.assertEqual(seeded[0][0].shape, (64600,))
            self.assertEqual(seeded[0][-1]['crop']['kind'], 'short')

    def test_prefix_path_and_disabled_path_preserve_original_augmentation_draws(self):
        source = 'offline/en/r.wav'
        wave = np.ones(80000, dtype=np.float32)*.1
        results = []
        for enabled in (False, True):
            args = SimpleNamespace(coverage_training=enabled, coverage_prefix_probability=1.)
            dataset = SpoofAudioDataset([source], '.', {source: 1}, args=args, use_rawboost=True)
            dataset.env_noise = lambda x, sr: x + np.random.uniform(0, .01)
            with patch('utils.data_utils.librosa.load', return_value=(wave, 16000)), \
                 patch('utils.data_utils.process_rawboost_feature', side_effect=lambda x, *a: x+random.random()):
                results.append(SeededDataset(dataset, 52)[0][0])
        torch.testing.assert_close(*results, rtol=0, atol=0)


class CoverageSamplerTests(unittest.TestCase):
    def test_exact_compute_class_symmetry_and_cell_quotas(self):
        sampler, dataset = make_sampler(steps=3158)
        sampler.set_epoch(1)
        observed, condition_counts = Counter(), Counter()
        for batch in sampler:
            self.assertEqual(len(batch), 4)
            self.assertEqual(len({i for i, _, _ in batch}), 4)
            labels = [dataset.sources[i]['label'] for i, _, _ in batch]
            self.assertEqual(Counter(labels), {0: 2, 1: 2})
            conditions = {(b, sampler.families[i, b, s], s) for i, b, s in batch}
            self.assertEqual(len(conditions), 1)
            condition = next(iter(conditions))
            condition_counts[condition] += 1
            for i, _, _ in batch:
                observed[(*condition, sampler.labels[i], sampler.languages[i])] += 1
        for condition, count in condition_counts.items():
            for label in (0, 1):
                self.assertLess(abs(observed[(*condition, label, 0)]-2*count*sampler.targets[label]), 1.00001)
        summary = sampler.plan_summary()
        self.assertEqual(summary['source_views_by_bank'], [11372, 1260])
        self.assertEqual(sum(row['views'] for row in summary['cells']), 3158*4)
        for bank in range(2):
            counts = [n for (b, _, _), n in condition_counts.items() if b == bank]
            self.assertLessEqual(max(counts)-min(counts), 1)
        self.assertTrue(all(row['views'] > 0 and row['unique_sources'] > 0 for row in summary['cells']))

    def test_replay_resume_and_prefetch_do_not_advance_committed_state(self):
        first, _ = make_sampler()
        initial = first.state_dict()
        first.set_epoch(1)
        plan = list(first)
        self.assertEqual(initial, first.state_dict())
        self.assertEqual(plan, list(first))
        with self.assertRaises(ValueError):
            first.commit_epoch(len(first)-1)
        first.discard_epoch()
        first.set_epoch(1)
        self.assertEqual(plan, list(first))
        first.commit_epoch(len(first))
        state = json.loads(json.dumps(first.state_dict()))
        first.set_epoch(2)
        resumed, _ = make_sampler()
        resumed.load_state_dict(state)
        resumed.set_epoch(2)
        self.assertEqual(list(first), list(resumed))
        self.assertEqual(first.plan_summary(), resumed.plan_summary())
        changed, _ = make_sampler(en_real=.45)
        with self.assertRaisesRegex(ValueError, 'distribution'):
            changed.load_state_dict(state)
        corrupt = copy.deepcopy(state)
        corrupt['history']['pool_draws']['not-a-cell'] = 1
        with self.assertRaisesRegex(ValueError, 'history'):
            make_sampler()[0].load_state_dict(corrupt)

    def test_missing_cell_fails_before_training_instead_of_biased_fallback(self):
        _, dataset = make_sampler()
        for source in dataset.sources:
            if '/en/' in source['offline'] and source['label'] == 1:
                for row in dataset.banks[1][source['offline']].values():
                    row['processing']['family'] = 'ffmpeg'
        source = BalancedPairBatchSampler(dataset.sources, 4, 20)
        with self.assertRaisesRegex(ValueError, 'cell needs'):
            CoverageViewBatchSampler(source, dataset)

    def test_within_cell_visits_cover_pool_before_reusing(self):
        sampler, _ = make_sampler(steps=160)
        sampler.set_epoch(1)
        per_cell = defaultdict(list)
        for batch in sampler:
            for i, b, s in batch:
                key = f'{b}|{sampler.families[i,b,s]}|{s}|{sampler.labels[i]}|{sampler.languages[i]}'
                per_cell[key].append(i)
        for key, indices in per_cell.items():
            take = min(len(indices), len(sampler.pools[key]))
            self.assertEqual(len(set(indices[:take])), take)


class CoverageDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_noisy_ce_is_not_double_weighted_and_actual_counts_match_plan(self):
        fixture = BundleFixture()
        args = fixture.args(True)
        args.coverage_training, args.noisy_bank_policy = True, 'cycle'
        for row in fixture.caches[str(fixture.root/'extra')]:
            row['processing'] = {'family': 'webrtc'}
        with fixture.active():
            from .data import DataBundle
            bundle = DataBundle(args)
            bundle.rotation.configure_mixture(.2, bundle.steps)
            for loader in [*bundle.train, *bundle.dev.values()]:
                loader.collate_fn.extractor = SimpleNamespace(sampling_rate=16000)
            bundle.begin(1)
            meter = CoverageMeter()
            ordinary_seen = set()
            for batches in zip(*bundle.train):
                noisy = batches[2]
                torch.testing.assert_close(noisy['language_weights'], torch.ones(8))
                self.assertTrue(any(float(x) != 1. for x in batches[0]['language_weights']))
                self.assertEqual(noisy['source_ids'][:4], noisy['source_ids'][4:])
                merged, layout = combine_batches(batches)
                self.assertEqual(layout[1:], (4, 4))
                self.assertEqual(len(merged['coverage_rows']['noisy_pair']), 4)
                ordinary_seen.update(batches[0]['source_ids'])
                # Synthetic ordinary fixture does not crop; attach explicit records
                # to exercise actual-count validation separately from crop I/O.
                merged['coverage_rows']['ordinary'] = [
                    {'source_id': sid, 'label': int(y), 'language': int(lang),
                     'crop': {'start': 0, 'kind': 'prefix'}} for sid, y, lang in
                    zip(batches[0]['source_ids'], batches[0]['labels'], batches[0]['languages'])]
                meter.update(merged['coverage_rows'])
            meter.verify(bundle.rotation.plan_summary(), len(fixture.ids))
            self.assertEqual(ordinary_seen, set(fixture.ids))
            self.assertEqual(meter.result()['noisy_processed_views'], bundle.steps*4)
            self.assertEqual(bundle.language_budgets['noisy_pair']['coefficients'], [[1., 1.], [1., 1.]])
            self.assertTrue(all('coverage_rows' not in b for loader in bundle.dev.values() for b in loader))


if __name__ == '__main__':
    unittest.main()
