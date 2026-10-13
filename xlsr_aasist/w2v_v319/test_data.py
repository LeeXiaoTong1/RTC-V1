"""Actual-waveform provenance, balanced quotas, sixfold view and resume checks."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch

from . import data
from .data import SourcePlan, Triplets, RawWaveCache, validate_batch, TripletCollator
from .views import FAMILIES, recipe, augment


def rows_fixture(count=11, directory=None):
    rows = []
    for language in ('en', 'zh'):
        for label in (0, 1):
            for number in range(count):
                source = f'offline/{language}/{label}_{number}.wav'
                sha = hashlib.sha256(source.encode()).hexdigest()
                off = None
                for condition in ('offline', 'online', 'noisy_a', 'noisy_b'):
                    if condition == 'online' and number % 2:
                        continue
                    row = dict(id=source.replace('offline/', 'online/') if condition == 'online' else source,
                        source_id=source, group_id=sha, source_sha256=sha, audio_sha256=sha,
                        language=language, label=label, condition=condition, split='train', view='full', full_length=True)
                    if directory and condition in ('offline', 'online'):
                        path = Path(directory)/f'{language}_{label}_{number}_{condition}.wav'
                        samples = np.arange(48000, dtype=np.float32)
                        phase = label*.27 + (0. if language == 'en' else .51)
                        wave = (.15*np.sin(samples*(.09+number*.001)+phase)+ .035*np.cos(samples*.12)).astype(np.float32)
                        if condition == 'online':
                            wave = np.tanh(wave*1.1)
                        sf.write(path, wave, 16000, subtype='FLOAT')
                        stat = path.stat()
                        row.update(audio=str(path), audio_size=stat.st_size, audio_mtime_ns=stat.st_mtime_ns,
                            audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), output_samples=len(wave))
                        if condition == 'offline':
                            sha = row['audio_sha256']
                            row.update(group_id=sha, source_sha256=sha)
                            off = row
                    if off:
                        row.update(group_id=sha, source_sha256=sha)
                    rows.append(row)
    return rows


class Bank:
    scenes = None
    def sample(self, rng):
        return np.sin(np.arange(16000, dtype=np.float32)*.071)*.05, 'noise'


class Engine:
    def __init__(self):
        self.calls = []
    def __call__(self, wave, recipe):
        self.calls.append((wave.copy(), copy.deepcopy(recipe)))
        return np.tanh(wave*1.2).astype(np.float32)


def dataset(rows, cfg, directory):
    value = Triplets(rows, cfg, directory)
    value.bank, value.engines, value.raw_cache = Bank(), Engine(), RawWaveCache(2*1024**2)
    return value


class DataTests(unittest.TestCase):
    def test_profile_quota_one_still_includes_longest_actual_source(self):
        rows = rows_fixture(13)
        for row in rows:
            number = int(row['source_id'].rsplit('_', 1)[1].split('.')[0])
            row['output_samples'] = 16000+number*100
        cfg = dict(epochs=5, source_batch=16, group_quotas=[1, 3, 8, 4], scene_probability=0., ssl_path='fixture')
        plan = SourcePlan(rows, 16, 31901, cfg)
        tickets = []
        class ProbeDataset:
            def __getitem__(self, ticket):
                tickets.append(ticket)
                return [dict(source_index=ticket.original, features=np.zeros((1, 12, 160), np.float32),
                             mask=np.ones((1, 12), np.int64)) for _ in range(3)]
        with patch.object(data, 'Triplets', lambda *a, **k: ProbeDataset()), \
             patch.object(data, 'TripletCollator', lambda *a: lambda units: [r for group in units for r in group]):
            values = data.probe_examples(plan, cfg, '.')
        self.assertEqual(len(values), 48)
        en_fake = [t for t in tickets if (rows[t.original]['language'], rows[t.original]['label']) == ('en', 0)]
        self.assertEqual(len(en_fake), 1)
        self.assertEqual(rows[en_fake[0].original]['output_samples'], 17200)
        for t in tickets:
            for value in (t.offline_view_json, t.online_view_json):
                self.assertEqual(json.loads(value)['recipe']['rate'], 970)

    def test_five_epochs_guarantee_all_six_english_versions_with_minimal_extra_steps(self):
        rows = rows_fixture(31)
        # Keep many EN-fake sources and much smaller other groups. The former
        # would miss part of its six-version set under the old epoch budget.
        rows = [r for r in rows if (r['language'], r['label']) == ('en', 0)
                or int(r['source_id'].rsplit('_', 1)[1].split('.')[0]) < 7]
        cfg = dict(epochs=5, source_batch=16, group_quotas=[4, 3, 5, 4], scene_probability=0.)
        plan = SourcePlan(rows, 16, 31901, cfg)
        self.assertEqual(plan.original_steps, 4)
        self.assertEqual(plan.coverage_steps, 10)
        self.assertEqual(plan.steps, 10)
        seen = {}
        for epoch in range(5):
            for batch in plan.batches(epoch):
                self.assertEqual(len(batch), 16)
                self.assertEqual(len({t.original for t in batch}), 16)
                for t in batch:
                    r = rows[t.original]
                    if r['language'] != 'en':
                        continue
                    for kind, value in (('offline', t.offline_view_json), ('online', t.online_view_json)):
                        if kind == 'online' and t.online is None:
                            continue
                        spec = json.loads(value)
                        seen.setdefault((r['source_id'], kind), set()).add(spec['view_id'])
        expected = sum(1+int('online' in item['indices']) for item in plan.inventory.values() if item['language'] == 'en')
        self.assertEqual(len(seen), expected)
        self.assertTrue(all(len(versions) == 6 for versions in seen.values()))
        final = plan.coverage(4)
        for key in ('en/0', 'en/1'):
            for values in final['source_groups'][key]['streams'].values():
                self.assertEqual(values['cumulative']['fixed'], values['available_fixed_views'])
        self.assertEqual(final['planned_full_wave_views'], 5*10*48)
        self.assertEqual(final['additional_steps_per_epoch'], 6)

    def test_fixed_six_versions_and_common_label_blind_probability(self):
        plan = SourcePlan(rows_fixture(), 16, 31901, dict(scene_probability=0.))
        for group, pool in plan.pools.items():
            for source in pool:
                for kind in ('offline', 'online'):
                    values = [plan.views.specification(source, kind, c) for c in range(12)]
                    self.assertEqual(sum(v['original'] for v in values), 2)
                    first = values[:6]
                    factor = plan.views.factors[group]
                    if factor == 6:
                        self.assertEqual({v['recipe']['family'] for v in first}, set(FAMILIES))
                        self.assertEqual(len({v['view_id'] for v in first}), 6)
                        self.assertEqual([v['view_id'] for v in first], [v['view_id'] for v in values[6:]])
                    elif factor == 3:
                        self.assertEqual(len({v['view_id'] for v in values}), 3)
                        self.assertEqual(len({v['recipe']['family'] for v in values}), 3)
                    else:
                        self.assertEqual(sum(v['dynamic'] for v in values), 10)
                        self.assertEqual(sum(not v['dynamic'] for v in values), 2)
                        self.assertEqual(len({v['view_id'] for v in values}), 11)
                    expected = dict(fixed=len({v['view_id'] for v in values if not v['dynamic']}),
                                    dynamic=sum(v['dynamic'] for v in values),
                                    realized=len({v['view_id'] for v in values}))
                    self.assertEqual(plan.views.cumulative(source, kind, 12), expected)
                for cycle in range(6):
                    a = plan.views.specification(source, 'offline', cycle)['recipe']
                    b = plan.views.specification(source, 'online', cycle)['recipe']
                    self.assertEqual(a['family'], b['family'])
                    if a['family'].startswith('speed_'):
                        self.assertEqual(a['rate'], b['rate'])
                    elif a['family'] == 'temporal_context':
                        self.assertEqual((a['keep'], a['start']), (b['keep'], b['start']))
        with self.assertRaises(ValueError):
            SourcePlan(rows_fixture(), 16, 1, dict(view_factors=[5, 5, 1, 3]))

    def test_resume_quotas_unique_sources_and_missing_online_preserved(self):
        rows = rows_fixture(13)
        cfg = dict(scene_probability=.8, source_batch=16, group_quotas=[4, 3, 5, 4])
        plan, replay = SourcePlan(rows, 16, 31901, cfg), SourcePlan(rows, 16, 31901, cfg)
        missing_seen = set()
        for epoch in range(7):
            batches = plan.batches(epoch)
            self.assertEqual(batches[1:], replay.batches(epoch, 1))
            for batch in batches:
                examples = []
                self.assertEqual(len({t.original for t in batch}), 16)
                for t in batch:
                    base = rows[t.original]
                    if t.online is None:
                        missing_seen.add(base['source_id'])
                    for role, mass in (('online', .3), ('reference', .1), ('noisy', .6)):
                        examples.append(dict(base, role=role, source_weight=t.source_weight,
                            ce_weight=t.source_weight*mass, pair_occurrence=t.occurrence,
                            condition=('online' if t.online is not None else 'rtc_online_fallback') if role == 'online' else 'rtc_'+role,
                            online_pair_eligible=t.online is not None))
                self.assertEqual(len(validate_batch(examples, 16, cfg)), 16)
        self.assertEqual(missing_seen, set(plan.missing_online))
        report = plan.coverage(0)
        recipes = [json.loads(t.recipe_json) for b in plan.batches(0) for t in b]
        from collections import Counter
        self.assertEqual(report['processing_families'], dict(Counter(r['family'] for r in recipes)))
        self.assertEqual(report['noise_types'], dict(Counter(r['noise_type'] for r in recipes)))
        self.assertFalse(report['excluded_missing_online_ids'])
        self.assertEqual(report['canonical_sources'], report['available_sources'])
        self.assertEqual(report['new_audio_disk_cache_bytes'], 0)
        self.assertEqual(report['source_groups']['en/0']['streams']['offline']['available_fixed_views'], 13*6)
        self.assertEqual(report['source_groups']['zh/0']['streams']['offline']['available_fixed_views'], 13)
        bad = copy.deepcopy(examples)
        bad[0]['condition'] = 'offline'
        with self.assertRaisesRegex(ValueError, 'Raw Offline'):
            validate_batch(bad, 16, cfg)
        bad = copy.deepcopy(examples)
        bad[0]['source_weight'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'auxiliary budget'):
            validate_batch(bad, 16, cfg)
        with self.assertRaises(ValueError):
            plan.batches(0, 2, 1)

    def test_actual_audio_triplets_shared_base_fallback_and_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = rows_fixture(7, folder)
            cfg = dict(source_batch=16, group_quotas=[4, 3, 5, 4], scene_probability=0.)
            plan = SourcePlan(rows, 16, 31901, cfg)
            loader = dataset(rows, cfg, folder)
            tickets = [t for batch in plan.batches(0) for t in batch]
            for missing in (False, True):
                t = next(t for t in tickets if (t.online is None) == missing)
                r = json.loads(t.recipe_json)
                r['dropout'] = dict(position=.5, seconds=.08, gain=0.)
                t = replace(t, recipe_json=json.dumps(r))
                loader.engines.calls.clear()
                with patch.object(data, 'augment', wraps=augment) as call:
                    values = loader[t]
                self.assertEqual(call.call_count, 2)
                self.assertEqual([r['role'] for r in values], ['online', 'reference', 'noisy'])
                self.assertTrue(all(r['condition'] != 'offline' for r in values))
                self.assertEqual(len(loader.engines.calls), 3 if missing else 2)
                self.assertEqual(loader.engines.calls[0][1], loader.engines.calls[1][1])
                self.assertEqual(values[1]['base_view_id'], values[2]['base_view_id'])
                self.assertEqual(values[1]['aux_spans'], values[2]['aux_spans'])
                self.assertTrue(values[1]['aux_spans'])
                self.assertEqual(values[0]['aux_spans'], [])
                self.assertAlmostEqual(sum(r['ce_weight'] for r in values), t.source_weight)
                if missing:
                    self.assertFalse(values[0]['online_pair_eligible'])
                    self.assertEqual(values[0]['bridge_unavailable_reason'], 'missing_official_online')
                    self.assertNotEqual(loader.engines.calls[0][1]['seed'], loader.engines.calls[-1][1]['seed'])
                repeated = loader[t]
                for a, b in zip(values, repeated):
                    np.testing.assert_array_equal(a['wave'], b['wave'])
            self.assertFalse((Path(folder)/'rolling_pairs').exists())

    def test_bridge_crops_skip_but_speed_is_eligible(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = rows_fixture(7, folder)
            cfg = dict(source_batch=16, scene_probability=0.)
            plan = SourcePlan(rows, 16, 31901, cfg)
            t = next(t for batch in plan.batches(0) for t in batch if t.online is not None)
            loader = dataset(rows, cfg, folder)
            for variant in (1, 2, 3, 4, 5):
                specs = []
                for kind in ('offline', 'online'):
                    r = recipe(31901, rows[t.original]['source_id'], kind, variant)
                    if variant == 4:
                        r.update(keep=.9, start=.5)
                    specs.append(json.dumps(dict(recipe=r, view_id=kind+str(variant))))
                values = loader[replace(t, offline_view_json=specs[0], online_view_json=specs[1])]
                self.assertEqual(values[0]['online_pair_eligible'], variant != 4)
                self.assertTrue(values[0]['noise_pair_eligible'])
            invalid = replace(t, online=t.original)
            with self.assertRaisesRegex(ValueError, 'Official Online'):
                loader[invalid]

    def test_feature_masks_keep_shared_erased_positions_and_all_frames(self):
        collator = TripletCollator('.')
        collator.features = lambda rows: [dict(r, features=torch.zeros(1, 30, 160), mask=torch.ones(1, 30, dtype=torch.long)) for r in rows]
        output = collator([[dict(wave=np.zeros(9600, np.float32), aux_spans=spans)
                           for spans in ([], [[3200, 3840]], [[3200, 3840]])]])
        self.assertTrue(output[0]['aux_valid'].all())
        np.testing.assert_array_equal(output[1]['aux_valid'], output[2]['aux_valid'])
        self.assertFalse(output[1]['aux_valid'].all())
        self.assertTrue(all(r['features'].shape == (1, 30, 160) for r in output))


if __name__ == '__main__':
    unittest.main()
