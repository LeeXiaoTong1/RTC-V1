"""CPU tests for timing, ambiguity, integrity and resumable preparation."""
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .phonemes import (SCHEMA, MODEL_ID, MODEL_REVISION, ALGORITHM, atomic_json,
    digest_file, lineage, map_segments, match_segments, greedy_tokens, ctc_align,
    segment_support, bind_manifest, PhonemeCache)
from .prepare_phonemes import prepare


def segment(token, start=0, end=800, instance=0):
    return dict(token=token, start_sample=start, end_sample=end, instance=instance, posterior=.9)


def as_segments(tokens): return [segment(t, i * 1000, i * 1000 + 800, i) for i, t in enumerate(tokens)]


def all_optimal(a, b):
    """Tiny brute-force oracle, independent of the production LCS tables."""
    possible = []
    for size in range(min(len(a), len(b)) + 1):
        found = []
        for ai in itertools.combinations(range(len(a)), size):
            for bi in itertools.combinations(range(len(b)), size):
                if [a[i] for i in ai] == [b[j] for j in bi]: found.append(set(zip(ai, bi)))
        if found: possible = found
    return set.intersection(*possible) if possible else set()


class TimingTests(unittest.TestCase):
    def test_speed_crop_containment_and_instance(self):
        spans = as_segments(['a', 'b', 'a', 'c'])
        mapped = map_segments(spans, dict(scale=.5, offset=-450, input_samples=4000, output_samples=1100))
        self.assertEqual([s['instance'] for s in mapped], [1, 2])
        self.assertEqual([(s['start_sample'], s['end_sample']) for s in mapped], [(50, 450), (550, 950)])

    def test_delay_drops_clipped_not_stretched(self):
        value = map_segments(as_segments(['a', 'b']), dict(scale=1, offset=300, output_samples=1800))
        self.assertEqual(len(value), 1)
        self.assertEqual(value[0]['start_sample'], 300)

    def test_invalid_map(self):
        for bad in (0, -1, float('nan')):
            with self.assertRaises(ValueError): map_segments([], dict(scale=bad, offset=0, output_samples=1000))


class MatchingTests(unittest.TestCase):
    def test_repeated_instances_and_deletions(self):
        self.assertEqual(match_segments(as_segments('aba'), as_segments('aba')), [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(match_segments(as_segments('aa'), as_segments('a')), [])
        self.assertEqual(match_segments(as_segments('ab'), as_segments('ba')), [])
        self.assertEqual(match_segments(as_segments('abcd'), as_segments('acd')), [(0, 0), (2, 1), (3, 2)])

    def test_all_small_binary_sequences_against_oracle(self):
        seq = [x for n in range(5) for x in itertools.product('ab', repeat=n)]
        for a in seq:
            for b in seq:
                self.assertEqual(set(match_segments(as_segments(a), as_segments(b))), all_optimal(a, b), (a, b))


class CTCTests(unittest.TestCase):
    def test_repeated_phone_has_distinct_support(self):
        probs = np.full((7, 3), .005)
        for i, token in enumerate([0, 1, 0, 1, 0, 2, 0]): probs[i, token] = .99
        log_probs = np.log(probs)
        tokens = greedy_tokens(log_probs, 0)
        self.assertEqual(tokens, [1, 1, 2])
        aligned = ctc_align(log_probs, tokens, 0)
        self.assertEqual([a['start_frame'] for a in aligned], [1, 3, 5])
        spans = segment_support(aligned, {1: 'a', 2: 'b'}, 2320)
        self.assertEqual([s['token'] for s in spans], ['a', 'a', 'b'])
        self.assertLessEqual(spans[0]['end_sample'], spans[1]['start_sample'])
        self.assertIn('approximate', spans[0]['alignment'])

    def test_long_blank_gap_not_assigned_to_phone(self):
        support = [dict(token_id=1, instance=0, start_frame=0, end_frame=1, posterior=.9),
                   dict(token_id=2, instance=1, start_frame=100, end_frame=101, posterior=.9)]
        spans = segment_support(support, {1: 'a', 2: 'b'}, 32400)
        self.assertGreater(spans[1]['start_sample'] - spans[0]['end_sample'], 20000)

    def test_impossible_repeated_tokens_fail(self):
        with self.assertRaises(ValueError): ctc_align(np.log(np.full((2, 3), 1 / 3)), [1, 1], 0)


class CacheTests(unittest.TestCase):
    def setUp(self):
        keys = ('HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE')
        saved = {k: os.environ.get(k) for k in keys}
        def restore():
            for key, value in saved.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value
        self.addCleanup(restore)

    def fixture(self, root):
        import soundfile as sf
        parent = root / 'parent'; parent.mkdir()
        atomic_json(parent / 'config.json', {'version': '3.18.5'})
        rows = []
        for domain, multiplier in [('offline', 1), ('online', .8)]:
            audio = root / (domain + '.wav')
            wave = (np.sin(np.arange(16000) * .02) * .1 * multiplier).astype(np.float32)
            sf.write(audio, wave, 16000, subtype='FLOAT')
            sha = digest_file(audio)
            if domain == 'offline': group = sha
            rows.append(dict(id=domain + '.wav', source_id='offline.wav', group_id=group,
                audio_sha256=sha, condition=domain, domain=domain, language='en', label=0,
                output_samples=16000, split='train', full_length=True, audio=str(audio)))
        atomic_json(parent / 'train_rows.json', rows)
        return parent, rows

    def test_prepare_resume_identity_hash_and_no_model_needed_on_reuse(self):
        calls = []
        class FakeRecognizer:
            def __init__(self, *args): calls.append('load')
            def __call__(self, wave): return [segment('a', 400, 1200)], {'fixture': True}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent, rows = self.fixture(root)
            cache = root / 'phones'; model_cache = root / 'models'
            binding = prepare(parent, cache, model_cache, 'cpu', True, FakeRecognizer)
            original_manifest_hash = binding['sha256']
            loaded = PhonemeCache(binding['path'], rows)
            self.assertEqual(loaded.get(rows[0])['segments'][0]['token'], 'a')
            with patch.object(Path, 'is_file', side_effect=AssertionError('Worker repeated inventory filesystem stats')):
                fast = PhonemeCache(binding['path'], rows, binding=binding)
                self.assertEqual(fast.get(rows[0])['segments'][0]['token'], 'a')
            with self.assertRaises(ValueError):
                PhonemeCache(binding['path'], rows, binding=dict(binding, sha256='0' * 64))
            def fail(*args): self.fail('Idempotent preparation loaded the model')
            again = prepare(parent, cache, model_cache, 'cpu', True, fail)
            self.assertEqual(original_manifest_hash, again['sha256'])
            self.assertEqual(calls, ['load'])
            changed = [dict(r) for r in rows]; changed[0]['label'] = 1
            with self.assertRaises(ValueError): bind_manifest(binding['path'], changed)
            record_path = cache / json.loads(Path(binding['path']).read_text())['records'][rows[0]['id']]['path']
            record_path.write_text(record_path.read_text() + ' ', encoding='utf-8')
            with self.assertRaises(ValueError): PhonemeCache(binding['path'], rows).get(rows[0])
            with self.assertRaises(ValueError): prepare(parent, cache, model_cache, 'cpu', True, fail)

    def test_reject_dev_and_audio_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent, rows = self.fixture(root)
            with self.assertRaises(ValueError): lineage(dict(rows[0], split='dev'))
            with Path(rows[0]['audio']).open('ab') as f: f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                prepare(parent, root / 'cache', root / 'models', 'cpu', True, lambda *a: None)

    def test_interrupted_preparation_is_resumable_but_not_trainable(self):
        class Interrupted:
            def __init__(self, *args): self.calls = 0
            def __call__(self, wave):
                self.calls += 1
                if self.calls == 2: raise RuntimeError('Interrupted fixture')
                return [segment('a', 400, 1200)], {}
        class Finished:
            def __init__(self, *args): pass
            def __call__(self, wave): return [segment('b', 400, 1200)], {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent, rows = self.fixture(root); cache = root / 'cache'
            with self.assertRaisesRegex(RuntimeError, 'Interrupted'):
                prepare(parent, cache, root / 'models', 'cpu', True, Interrupted)
            self.assertFalse((cache / 'manifest.json').exists())
            with self.assertRaises(FileNotFoundError): bind_manifest(cache / 'manifest.json', rows)
            result = prepare(parent, cache, root / 'models', 'cpu', True, Finished)
            self.assertEqual(result['records'], 2)
            summary = json.loads((cache / 'manifest.json').read_text())['summary']
            self.assertEqual(summary['reused'], 1)
            self.assertEqual(summary['prepared'], 1)


if __name__ == '__main__': unittest.main()
