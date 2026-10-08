"""Source provenance, exact tickets and bounded worker-memory contracts."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from w2v_v315.augment import recipe
from . import data
from .data import SourcePlan, RawWaveCache, Triplets, TripletCollator, TrainingLoader, validate_batch


def rows_fixture(count=5):
    rows = []
    for language in ('en', 'zh'):
        for label in (0, 1):
            for number in range(count):
                source = f'offline/{language}/{label}_{number}.wav'
                sha = hashlib.sha256(source.encode()).hexdigest()
                for condition in ('offline', 'online', 'noisy_a', 'noisy_b'):
                    if condition == 'online' and number % 2:
                        continue
                    rows.append(dict(id=source.replace('offline/', 'online/') if condition=='online' else source,
                        source_id=source, group_id=sha, source_sha256=sha, audio_sha256=sha,
                        language=language, label=label, condition=condition, split='train', view='full', full_length=True))
    return rows


class Bank:
    def sample(self, rng):
        return np.sin(np.arange(1600, dtype=np.float32)*.13), 'noise'


class Engine:
    def __call__(self, wave, recipe):
        return np.tanh(wave*1.2).astype(np.float32)


def wave_row(path, wave):
    import soundfile as sf
    sf.write(path, wave, 16000, subtype='FLOAT')
    stat = path.stat()
    return dict(audio=str(path), audio_size=stat.st_size, audio_mtime_ns=stat.st_mtime_ns,
                audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), output_samples=len(wave))


class EchoTickets(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, ticket):
        return (os.getpid(), ticket.occurrence, ticket.recipe_json)


def passthrough(rows):
    return rows


def echo_loader(dataset, cfg, persistent=False, **kwargs):
    return DataLoader(EchoTickets(), collate_fn=passthrough, num_workers=1,
        multiprocessing_context='spawn', persistent_workers=persistent, prefetch_factor=2, **kwargs)


class DataTests(unittest.TestCase):
    def test_complete_tickets_deterministic_resume_balanced_source_ce(self):
        rows = rows_fixture(65)
        plan = SourcePlan(rows, 16, 315101)
        saw_warmup = False
        for epoch in (0, 1, 2):
            batches = plan.batches(epoch)
            self.assertEqual(batches[2:5], plan.batches(epoch, 2, 5))
            for batch in batches:
                examples = []
                for ticket in batch:
                    self.assertEqual(rows[ticket.online]['condition'], 'online')
                    self.assertEqual(json.loads(ticket.recipe_json), recipe(plan.seed, ticket.occurrence,
                        ticket.phase, rows[ticket.original]['group_id'], warm=ticket.warm))
                    saw_warmup |= ticket.noisy_mass < .4
                    for role, index, mass in (('online', ticket.online, .9-ticket.noisy_mass),
                            ('reference', ticket.original, .1), ('noisy', ticket.original, ticket.noisy_mass)):
                        examples.append(dict(rows[index], role=role, ce_weight=mass/16, pair_occurrence=ticket.occurrence))
                self.assertEqual(len(examples), 48)
                self.assertEqual(len(validate_batch(examples, 16)), 16)
                role_mass = {role: sum(row['ce_weight'] for row in examples if row['role']==role)
                             for role in ('online', 'reference', 'noisy')}
                self.assertAlmostEqual(role_mass['reference'], .1)
                self.assertGreaterEqual(role_mass['online'], .5-1e-7)
                self.assertLessEqual(role_mass['noisy'], .4+1e-7)
        self.assertTrue(saw_warmup)
        wrong = copy.deepcopy(examples)
        for row in wrong:
            if row['role']=='online':
                row['ce_weight'] -= .05/16
            elif row['role']=='reference':
                row['ce_weight'] += .05/16
        with self.assertRaisesRegex(ValueError, 'CE schedule'):
            validate_batch(wrong, 16)
        with self.assertRaises(ValueError):
            plan.batches(0, 2, 1)

    def test_missing_online_excluded_reported_and_malformed_pairs_fail(self):
        rows = rows_fixture(5)
        plan = SourcePlan(rows, 16, 17)
        coverage = plan.coverage()
        self.assertEqual(coverage['canonical_sources'], 20)
        self.assertEqual(coverage['missing_online_sources'], 8)
        self.assertEqual(coverage['available_sources'], 12)
        self.assertEqual(coverage['excluded_source_groups'], {'en/0': 2, 'en/1': 2, 'zh/0': 2, 'zh/1': 2})
        selected = {rows[t.original]['source_id'] for batch in plan.batches(0) for t in batch}
        self.assertFalse(selected.intersection(coverage['excluded_missing_online_ids']))
        self.assertEqual(coverage['new_audio_disk_cache_bytes'], 0)
        with self.assertRaisesRegex(ValueError, 'all four'):
            SourcePlan([row for row in rows if not (row['condition']=='online' and row['language']=='en' and row['label']==1)], 16, 17)
        malformed = copy.deepcopy(rows)
        next(row for row in malformed if row['condition']=='online')['source_sha256'] = 'f'*64
        with self.assertRaises(ValueError):
            SourcePlan(malformed, 16, 17)
        duplicate = copy.deepcopy(rows)
        online = [row for row in duplicate if row['condition']=='online']
        online[1]['id'] = online[0]['id']
        with self.assertRaisesRegex(ValueError, 'multiple sources'):
            SourcePlan(duplicate, 16, 17)

    def test_raw_cache_byte_limit_unchanged_values_mutation_and_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/f'{i}.wav' for i in range(3)]
            rows = [wave_row(path, np.arange(1000, dtype=np.float32)*.0001+i*.01) for i, path in enumerate(paths)]
            cached, uncached = RawWaveCache(8000), RawWaveCache(0)
            with patch.object(data, 'verified_wave', wraps=data.verified_wave) as reader:
                a = cached.get(rows[0]); a[:] = 7
                np.testing.assert_array_equal(cached.get(rows[0]), uncached.get(rows[0]))
                self.assertEqual(reader.call_count, 2)
                for row in rows:
                    np.testing.assert_array_equal(cached.get(row), uncached.get(row))
                    self.assertLessEqual(cached.used, 8000)
                self.assertEqual(uncached.used, 0)
                self.assertEqual(len(cached.cache), 2)
            with paths[-1].open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'waveform changed'):
                cached.get(rows[-1])

    def test_cached_and_uncached_triplets_equal_and_online_masks_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = rows_fixture(1)
            for index, row in enumerate(rows):
                if row['condition'] not in ('offline', 'online'):
                    continue
                wave = np.sin(np.arange(32000, dtype=np.float32)*(.1 if row['condition']=='offline' else .11)+index*.17)*.15
                row.update(wave_row(Path(directory)/f'{index}.wav', wave))
                if row['condition']=='offline':
                    for other in rows:
                        if other['source_id']==row['source_id']:
                            other.update(group_id=row['audio_sha256'], source_sha256=row['audio_sha256'])
            plan = SourcePlan(rows, 16, 315101)
            ticket = plan.batches(0)[0][0]
            from dataclasses import replace
            r = json.loads(ticket.recipe_json)
            r['echo'] = None
            r['dropout'] = dict(position=.5, seconds=.1, gain=0.)
            ticket = replace(ticket, recipe_json=json.dumps(r))
            cfg = dict(source_batch=16, rolling_cache_bytes=0)
            a, b = Triplets(rows, cfg, directory), Triplets(rows, cfg, directory)
            for dataset, cap in ((a, 1_000_000), (b, 0)):
                dataset.bank, dataset.engines, dataset.raw_cache = Bank(), Engine(), RawWaveCache(cap)
            first, second = a[ticket], b[ticket]
            self.assertEqual([row['role'] for row in first], ['online', 'reference', 'noisy'])
            for x, y in zip(first, second):
                np.testing.assert_array_equal(x['wave'], y['wave'])
                self.assertEqual(x['aux_spans'], y['aux_spans'])
                self.assertEqual(len(x['wave']), 32000)
                self.assertIs(x['online_pair_eligible'], True)
            self.assertEqual(first[0]['aux_spans'], [])
            self.assertTrue(first[1]['aux_spans'])
            self.assertEqual(first[1]['aux_spans'], first[2]['aux_spans'])
            self.assertFalse((Path(directory)/'rolling_pairs').exists())
            wrong = replace(ticket, online=ticket.original)
            with self.assertRaisesRegex(ValueError, 'official Train Offline/Online'):
                a[wrong]

    def test_missing_span_exclusion_preserves_full_feature_sequence(self):
        collator = TripletCollator('.')
        def features(rows):
            return [dict(row, features=torch.zeros(1, 20, 160), mask=torch.ones(1, 20, dtype=torch.long))
                    for row in rows]
        collator.features = features
        rows = [dict(wave=np.zeros(6400, np.float32), aux_spans=spans)
                for spans in ([], [[3000, 3200]], [[3000, 3200]])]
        output = collator([rows])
        self.assertTrue(output[0]['aux_valid'].all())
        self.assertFalse(output[1]['aux_valid'].all())
        np.testing.assert_array_equal(output[1]['aux_valid'], output[2]['aux_valid'])
        self.assertTrue(all(row['features'].shape==(1, 20, 160) for row in output))
        self.assertTrue(all(isinstance(row['features'], np.ndarray) for row in output))

    def test_persistent_worker_pid_and_recipe_survive_segment_and_epoch(self):
        plan = SourcePlan(rows_fixture(17), 16, 9)
        cfg = dict(source_batch=16, rolling_cache_bytes=0)
        with patch.object(data, '_loader', side_effect=echo_loader):
            session = TrainingLoader(plan, cfg, '.')
        try:
            a = list(session.segment(0, 0, 1))
            b = list(session.segment(0, 1, 2))
            c = list(session.segment(1, 0, 1))
            self.assertEqual({row[0] for batch in a+b+c for row in batch}, {a[0][0][0]})
            for actual, expected in ((a, plan.batches(0, 0, 1)), (b, plan.batches(0, 1, 2)), (c, plan.batches(1, 0, 1))):
                self.assertEqual([[(row[1], row[2]) for row in batch] for batch in actual],
                                 [[(ticket.occurrence, ticket.recipe_json) for ticket in batch] for batch in expected])
        finally:
            session.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            session.segment(1, 0, 1)


if __name__ == '__main__':
    unittest.main()
