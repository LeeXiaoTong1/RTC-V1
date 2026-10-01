"""Tiny official-style fixtures: missing pairs, source budgets and audible masks."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from transformers import SeamlessM4TFeatureExtractor

from w2v_aasist.data import read_protocol, FeatureCollator
from .data import (canonical_sources, resolve_pair_manifest, PairedEpochPlan,
                   PairedDataset, PairedCollator, loader)

torch.set_num_threads(1)


def fixture(root, per_group=2, samples=8000):
    root = Path(root); audio = root / 'data' / 'wav' / 'train'
    labels, pairs = [], []
    for language in ('en', 'zh'):
        for label in (0, 1):
            for i in range(per_group):
                stem = f'{language}_{label}_{i}'
                a, b = f'offline/{language}/{stem}.wav', f'online/{language}/different_{stem}.wav'
                t = np.arange(samples) / 16000
                wave = (.08 * np.sin(2 * np.pi * (220 + 37 * i + label * 9) * t)).astype(np.float32)
                if i == 1: wave[len(wave) // 3:len(wave) // 2] = 0.
                for name in (a, b):
                    path = audio / name; path.parent.mkdir(parents=True, exist_ok=True)
                    if name == b and i == 0 and label == 1: continue
                    sf.write(path, wave, 16000, subtype='FLOAT')
                    labels.append(name + (' bonafide' if label else ' spoof'))
                pairs.append([f'train/{a}', f'train/{b}' if (audio / b).is_file() else ''])
    protocol = root / 'data' / 'train_label.txt'; protocol.write_text('\n'.join(labels) + '\n')
    pair_file = root / 'data' / 'meta' / 'train_offline_online_pairs.csv'; pair_file.parent.mkdir()
    with pair_file.open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['offline_id', 'online_id']); writer.writerows(pairs)
    processor = root / 'processor'; SeamlessM4TFeatureExtractor().save_pretrained(processor)
    cfg = dict(train_data_path=str(audio), train_protocol=str(protocol), train_pair_manifest=None,
               seed=57, source_batch=3, short_loss_weight=.3, short_min_seconds=1., short_max_seconds=1.,
               short_prefix_probability=.5, rawboost=0, raw_config={}, short_rawboost_probability=.25,
               processing_silence_probability=0., workers=0, device='cpu', ssl_path=str(processor),
               microbatch=4, frame_budget=1600, prefetch_factor=2)
    records = read_protocol(protocol, audio)
    sources = canonical_sources(records, pair_file)
    cache_rows = []
    for s in sources:
        for v, condition in enumerate(('noisy_a', 'noisy_b')):
            cache_rows.append(dict(s['conditions']['offline'], noisy=True, condition=condition,
                                   version=v, full_length=True, output_samples=samples, band=v,
                                   processing_family=('ffmpeg', 'webrtc')[v]))
    return cfg, records, sources, cache_rows


class PairedDataTests(unittest.TestCase):
    def test_official_empty_online_is_preserved_and_path_is_resolved(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, records, sources, rows = fixture(d)
            path = resolve_pair_manifest(cfg)
            self.assertEqual(path.name, 'train_offline_online_pairs.csv')
            self.assertEqual(cfg['train_pair_manifest'], str(path))
            self.assertEqual(len(sources), 8)
            self.assertEqual(sum(s['missing_online'] for s in sources), 2)
            plan = PairedEpochPlan(sources, rows, 3, cfg['seed'])
            self.assertEqual(plan.coverage()['condition_rows_per_epoch'],
                             {'offline': 8, 'online': 6, 'noisy_a': 8, 'noisy_b': 8})
            self.assertEqual(plan.steps, 3)
            for epoch in (1, 2):
                flat = [i for batch in plan.batches(epoch) for i in batch]
                self.assertEqual(sorted(flat), list(range(8)))
                self.assertEqual(len(flat), len(set(flat)))
                self.assertEqual(list(plan.tickets(epoch, 1)), plan.batches(epoch)[1:])
            self.assertEqual(plan.batches(1), plan.batches(1))
            self.assertNotEqual(plan.batches(1), plan.batches(2))

    def test_bad_pair_mapping_and_ambiguous_discovery_fail(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, records, sources, rows = fixture(d)
            path = resolve_pair_manifest(cfg)
            original = path.read_text()
            lines = original.splitlines()
            path.write_text(original + lines[1] + '\n')
            with self.assertRaisesRegex(ValueError, 'duplicate Offline'):
                canonical_sources(records, path)
            path.write_text('\n'.join(lines[:-1]) + '\n')
            with self.assertRaisesRegex(ValueError, 'cover every'):
                canonical_sources(records, path)
            path.write_text(original)
            duplicate = Path(d) / 'data' / 'train_offline_online_pairs.csv'; duplicate.write_text(original)
            cfg['train_pair_manifest'] = None
            with self.assertRaisesRegex(ValueError, 'one candidate'):
                resolve_pair_manifest(cfg)
            changed = [dict(r, label=1-r['label']) if r['domain'] == 'online' else r for r in records]
            with self.assertRaisesRegex(ValueError, 'language/label mismatch'):
                canonical_sources(changed, path)

    def test_source_class_weights_are_not_counted_by_view_count(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, _, sources, rows = fixture(d)
            plan = PairedEpochPlan(sources, rows, 4, cfg['seed'])
            torch.testing.assert_close(plan.weights, torch.ones(2), rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'both complete'):
                PairedEpochPlan(sources, rows[:-1], 4)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                PairedEpochPlan(sources, rows + [rows[0]], 4)

    def test_full_reference_never_rawboosted_short_has_own_budget(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, _, sources, rows = fixture(d, samples=64000)
            plan = PairedEpochPlan(sources, rows, 4)
            cfg.update(rawboost=1, short_rawboost_probability=1.)
            ds = PairedDataset(plan.records, cfg, 1)
            with patch('utils.data_utils.process_rawboost_feature', side_effect=lambda w, *_: w * 2) as augment:
                example = ds[0]
            self.assertEqual(augment.call_count, 2)
            for condition in ('offline', 'online', 'noisy_a', 'noisy_b'):
                views = [r for r in example if r['condition'] == condition]
                self.assertEqual([r['view'] for r in views], ['full', 'short'])
                self.assertAlmostEqual(sum(r['view_weight'] for r in views), 1.)
                full, short = views
                original, _ = sf.read(full['audio'], dtype='float32')
                np.testing.assert_array_equal(full['wave'], original)
                cropped = original[short['crop_start_sample']:short['crop_start_sample'] + short['crop_samples']]
                np.testing.assert_array_equal(short['wave'], cropped * (2 if condition in ('offline', 'online') else 1))
                self.assertEqual(full['source_id'], short['source_id'])
                self.assertEqual(full['source_group'], short['source_group'])

    def test_short_dedup_and_internal_silence_remain_valid_audio(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, _, sources, rows = fixture(d, samples=8000)
            plan = PairedEpochPlan(sources, rows, 4)
            ex = PairedDataset(plan.records, cfg, 1)[1]
            self.assertTrue(all(r['view'] == 'full' and r['view_weight'] == 1 for r in ex))
            batch = PairedCollator(cfg['ssl_path'])([ex])
            for row in batch:
                self.assertTrue(bool(row['mask'].all()))
                self.assertEqual(row['audibility_mask'].numel(), row['mask'].shape[1])
                self.assertFalse(bool(row['audibility_mask'].all()))

    def test_short_noisy_silence_is_deterministic_and_does_not_touch_full(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, _, sources, rows = fixture(d, samples=64000)
            cfg['processing_silence_probability'] = 1.
            plan = PairedEpochPlan(sources, rows, 4)
            ds = PairedDataset(plan.records, cfg, 3)
            first, second = ds[0], ds[0]
            silenced = 0
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a['wave'], b['wave'])
                if a['view'] == 'full': self.assertEqual(a['augmentation'], 'unchanged')
                if a['augmentation'] == 'short_local_silence':
                    silenced += 1
                    self.assertTrue(a['noisy'])
                    original = next(r['wave'] for r in first if r['condition'] == a['condition'] and r['view'] == 'full')
                    native = original[a['crop_start_sample']:a['crop_start_sample'] + a['crop_samples']]
                    self.assertLessEqual(np.count_nonzero(a['wave'] != native) / len(native), .05)
                    self.assertGreaterEqual(np.dot(a['wave'], a['wave']) / np.dot(native, native), .9)
            self.assertEqual(silenced, 2)

    def test_loader_keeps_source_batch_complete_and_feature_values_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, _, sources, rows = fixture(d, samples=24001)
            plan = PairedEpochPlan(sources, rows, 3)
            ticket = plan.batches(1)[0]
            ds = PairedDataset(plan.records, cfg, 1)
            reference = FeatureCollator(cfg['ssl_path'])([r for i in ticket for r in ds[i]])
            batch = next(iter(loader(plan.records, cfg, training=True, epoch=1, batches=[ticket])))
            self.assertEqual(len(set(e['source_id'] for e in batch)), 3)
            for got, wanted in zip(batch, reference):
                self.assertEqual(got['source_id'], wanted['source_id'])
                torch.testing.assert_close(got['features'], wanted['features'], rtol=0, atol=0)
                torch.testing.assert_close(got['mask'], wanted['mask'], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
