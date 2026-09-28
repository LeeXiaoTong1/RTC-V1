"""Real DataBundle integration; only protocol/cache I/O and features are mocked."""
from contextlib import ExitStack, contextmanager, redirect_stdout
import copy
import io
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import DataBundle, FeatureCollator, SourceTaggedDataset, combine_batches
from .language import language_id


class SyntheticOrdinary(Dataset):
    def __init__(self, ids, labels, markers, augment):
        self.ids, self.labels, self.markers = ids, labels, markers
        self.augment, self.env_noise = augment, object()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        source = self.ids[index]
        value = self.markers[source]
        if self.augment:
            value += random.random() + np.random.rand() + float(torch.rand(()))
        return torch.full((64600,), value), self.labels[source], source


def synthetic_features(_collator, waves):
    features = torch.stack([w[:640] for w in waves]).reshape(len(waves), 4, 160)
    return features, torch.ones((len(waves), 4), dtype=torch.long)


class BundleFixture:
    def __init__(self):
        self.root = Path('language-data-fixture').absolute()
        self.noisy_counts, self.pair_counts = [[3, 6], [4, 7]], [[2, 3], [2, 4]]
        self.ordinary_counts = [[5, 9], [6, 11]]
        self.offline, self.pairs, self.labels, self.markers = [], [], {}, {}
        for label, counts in enumerate(self.noisy_counts):
            for language, count in enumerate(counts):
                for index in range(count):
                    source = f'offline/{("en", "zh")[language]}/{label}_{index}.wav'
                    self.offline.append(source)
                    self.labels[source] = label
                    self.markers[source] = float(len(self.markers) + 1)
                    if index < self.pair_counts[label][language]:
                        online = source.replace('offline/', 'online/', 1)
                        self.labels[online] = label
                        self.markers[online] = float(len(self.markers) + 101)
                        self.pairs.append({'offline': source, 'online': online, 'label': label})
        self.ids = list(self.labels)
        random.Random(95).shuffle(self.ids)
        random.Random(96).shuffle(self.pairs)
        self.caches, self.cached_values = {}, {}
        for bank, name in enumerate(('old', 'extra', 'seen', 'heldout')):
            rows = []
            for source in self.offline:
                for band in range(4):
                    key = f'{name}:{source}:{band}'
                    self.cached_values[key] = float(1000*(bank + 1) + 10*self.markers[source] + band)
                    rows.append({'source': source, 'label': self.labels[source], 'band': band, 'audio': key})
            self.caches[str(self.root/name)] = list(reversed(rows))

    def args(self, enabled):
        return SimpleNamespace(
            stage=3, seed=34, language_weighting=enabled, en_real_budget=.35, en_fake_budget=.40,
            ordinary_sampling='legacy', noisy_bank_policy='mixed', device='cpu', num_workers=0,
            feature_cache=None, eval_batch=7, algo=5, ssl_path=str(self.root/'model'),
            train_noise_manifest=str(self.root/'noise.jsonl'), rtc_pairs=str(self.root/'pairs.jsonl'),
            train_protocol=str(self.root/'train.txt'), dev_protocol=str(self.root/'dev.txt'),
            train_data_path=str(self.root/'audio'/'train'), dev_data_path=str(self.root/'audio'/'dev'),
            train_noisy_cache=str(self.root/'old'), extra_train_noisy_cache=[str(self.root/'extra')],
            dev_noisy_cache=str(self.root/'seen'), dev_heldout_cache=str(self.root/'heldout'))

    def build_dataset(self, _protocol, _directory, mode, **_kwargs):
        return SyntheticOrdinary(self.ids, self.labels, self.markers, mode == 'train'), self.ids, self.labels

    def load_cache(self, folder, _role, _protocol, _root):
        return copy.deepcopy(self.caches[str(folder)]), {'noise': {'manifest_sha256': 'fixture-sha'}}

    def read_wave(self, path):
        source = Path(path).relative_to(self.root/'audio'/'train').as_posix()
        return np.full(64600, self.markers[source], dtype=np.float32)

    def read_cached(self, key):
        return torch.full((64600,), self.cached_values[key])

    @contextmanager
    def active(self):
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ))
            stack.enter_context(patch('utils.data_utils.build_dataset_from_protocol', side_effect=self.build_dataset))
            stack.enter_context(patch('utils.rtc_pairs.load_pairs', side_effect=lambda *_a: copy.deepcopy(self.pairs)))
            stack.enter_context(patch('rtc_noisy_v2.cache.load_v2_cache', side_effect=self.load_cache))
            stack.enter_context(patch('rtc_noisy_v2.cache.check_suite'))
            stack.enter_context(patch('w2v_rebuild.data.sha256', return_value='fixture-sha'))
            stack.enter_context(patch('rtc_noisy.common.read_wave', side_effect=self.read_wave))
            stack.enter_context(patch('rtc_noisy.data.read_cached', side_effect=self.read_cached))
            stack.enter_context(patch('utils.rtc_data.RTCPairDataset._read',
                                      side_effect=lambda source: torch.full((64600,), self.markers[source])))
            stack.enter_context(patch.object(FeatureCollator, 'extract', synthetic_features))
            yield self

    def bundle(self, enabled):
        with redirect_stdout(io.StringIO()):
            bundle = DataBundle(self.args(enabled))
        for loader in [*bundle.train, *bundle.dev.values()]:
            loader.collate_fn.extractor = SimpleNamespace(sampling_rate=16000)
        return bundle


class LanguageDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_noisy_dev_unsorted_same_label_sources_remain_aligned(self):
        fixture = BundleFixture()
        with fixture.active():
            bundle = fixture.bundle(True)
            for condition in ('seen', 'heldout'):
                original = fixture.caches[str(fixture.root/condition)]
                expected = sorted(original, key=lambda row: (row['source'], row['band']))
                self.assertNotEqual([r['source'] for r in original], [r['source'] for r in expected])
                self.assertTrue(any(a['source'] != b['source'] and a['label'] == b['label']
                                    and language_id(a['source']) != language_id(b['source'])
                                    for a, b in zip(original, expected)))
                observed = []
                for batch in bundle.dev[condition]:
                    for index, source in enumerate(batch['source_ids']):
                        band = batch['bands'][index]
                        marker = fixture.cached_values[f'{condition}:{source}:{band}']
                        self.assertEqual(float(batch['features'][index, 0, 0]), marker)
                        self.assertEqual(int(batch['labels'][index]), fixture.labels[source])
                        self.assertEqual(int(batch['languages'][index]), language_id(source))
                        self.assertEqual(float(batch['language_weights'][index]), 1.)
                        observed.append((source, band))
                self.assertEqual(observed, [(row['source'], row['band']) for row in expected])

    def test_budget_uses_each_branch_pool_and_pair_view_layout(self):
        fixture = BundleFixture()
        with fixture.active():
            bundle = fixture.bundle(True)
            for name, counts in (('ordinary', fixture.ordinary_counts), ('rtc_pair', fixture.pair_counts),
                                 ('noisy_pair', fixture.noisy_counts)):
                report = bundle.language_budgets[name]
                self.assertEqual(report['counts'], counts)
                expected = [[q*sum(row)/row[0], (1-q)*sum(row)/row[1]]
                            for row, q in zip(counts, [.40, .35])]
                torch.testing.assert_close(torch.tensor(report['coefficients']), torch.tensor(expected))
            bundle.begin(1)
            for branch, loader in zip(('ordinary', 'rtc_pair', 'noisy_pair'), bundle.train):
                batch = next(iter(loader))
                lookup = torch.tensor(bundle.language_budgets[branch]['coefficients'])
                torch.testing.assert_close(batch['language_weights'], lookup[batch['labels'], batch['languages']])
                if branch != 'ordinary':
                    pairs = batch['pairs']
                    self.assertEqual(batch['source_ids'][:pairs], batch['source_ids'][pairs:])
                    self.assertTrue(torch.equal(batch['languages'][:pairs], batch['languages'][pairs:]))
                    self.assertTrue(torch.equal(batch['language_weights'][:pairs], batch['language_weights'][pairs:]))
            dataset = bundle.train[2].dataset
            for index in range(len(dataset)):
                for bank, band in ((0, 0), (1, 3)):
                    row = dataset[(index, bank, band)]
                    source = dataset.inner.sources[index]['offline']
                    self.assertEqual(row[-1]['source_id'], source)
                    self.assertEqual(row[-1]['language'], language_id(source))
                    self.assertEqual(float(row[0][0]), fixture.markers[source])
                    name = ('old', 'extra')[bank]
                    self.assertEqual(float(row[1][0]), fixture.cached_values[f'{name}:{source}:{band}'])

    def test_tagging_preserves_seeded_sampling_waveforms_and_defaults(self):
        fixture = BundleFixture()
        with fixture.active():
            plain, tagged = fixture.bundle(False), fixture.bundle(True)
            self.assertFalse(plain.language_budgets)
            self.assertTrue(all(not isinstance(loader.dataset, SourceTaggedDataset) for loader in plain.train))
            for epoch in (1, 2):
                plain.begin(epoch)
                tagged.begin(epoch)
                self.assertEqual(list(plain.real_sampler), list(tagged.real_sampler))
                self.assertEqual(list(plain.rotation), list(tagged.rotation))
                ordinary_ids = []
                left, right = list(zip(*plain.train)), list(zip(*tagged.train))
                self.assertEqual(len(left), plain.steps)
                self.assertEqual(len(left), len(right))
                for old_batches, new_batches in zip(left, right):
                    ordinary_ids.extend(new_batches[0]['ids'])
                    for old, new in zip(old_batches, new_batches):
                        self.assertNotIn('language_weights', old)
                        for key in ('features', 'mask', 'labels'):
                            torch.testing.assert_close(old[key], new[key], atol=0, rtol=0)
                        if 'ids' in old:
                            self.assertEqual(old['ids'], new['ids'])
                    old, old_layout = combine_batches(old_batches)
                    new, new_layout = combine_batches(new_batches)
                    self.assertEqual(old_layout, new_layout)
                    for key in ('features', 'mask', 'labels'):
                        torch.testing.assert_close(old[key], new[key], atol=0, rtol=0)
                    self.assertEqual(new['source_ids'], [source for batch in new_batches for source in batch['source_ids']])
                self.assertCountEqual(ordinary_ids, fixture.ids)
                self.assertEqual(len(ordinary_ids), len(set(ordinary_ids)))
                plain.end(plain.steps)
                tagged.end(tagged.steps)
                self.assertEqual(plain.sampler_state(), tagged.sampler_state())


if __name__ == '__main__':
    unittest.main()
