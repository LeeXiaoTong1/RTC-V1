"""Source metadata wiring and a real one-epoch CPU engine/export smoke test."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from . import train as engine
from .core import sha256, load_checkpoint
from .data import SourceTaggedDataset, FeatureCollator, combine_batches
from .language import LanguageBudget, language_id
from .group_metrics import export_language_report
from .refinement_engine_tests import FixtureBundle, build_fixture_detector
from . import refinement_engine_tests as fixtures


class WaveRows(Dataset):
    def __init__(self, ids, labels, kind):
        self.ids, self.labels, self.kind = ids, labels, kind

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ticket):
        i = ticket[0] if isinstance(ticket, tuple) else ticket
        wave = torch.full((64600,), float(i+1))
        if self.kind == 'ordinary':
            return wave, self.labels[i], self.ids[i]
        return wave, wave+10, self.labels[i], 3, 1


def collator(kind):
    result = FeatureCollator('unused', kind, source_metadata=True)
    result.extractor = object()
    result.extract = lambda waves: (torch.stack([w[0].expand(12, 160).clone() for w in waves]),
                                    torch.ones(len(waves), 12, dtype=torch.long))
    return result


class LanguageFixtureBundle(FixtureBundle):
    def __init__(self, args):
        super().__init__(args)
        self.steps = 1
        self.train = [stream[:1] for stream in self.train]
        self.language_budgets = {}
        for kind, stream in zip(('ordinary', 'rtc_pair', 'noisy_pair'), self.train):
            for batch in stream:
                n = batch.get('pairs', len(batch['labels']))
                languages = [0, 0, 1, 1] * (n//4)
                ids = [f'offline/{("en", "zh")[lang]}/{kind}_{i}.wav' for i, lang in enumerate(languages)]
                budget = LanguageBudget(ids, batch['labels'][:n], args.en_real_budget, args.en_fake_budget)
                self.language_budgets[kind] = budget.report
                languages = torch.tensor(languages * (2 if 'pairs' in batch else 1))
                batch.update(languages=languages, source_ids=ids * (2 if 'pairs' in batch else 1),
                             language_weights=budget.coefficients(batch['labels'], languages))
        generator = torch.Generator().manual_seed(12)

        def dev_batch(ids, labels, bands=None):
            item = {'features': torch.randn(len(ids), 24, 160, generator=generator),
                    'mask': torch.ones(len(ids), 24, dtype=torch.long),
                    'labels': torch.tensor(labels), 'source_ids': ids}
            item.update({'ids': ids} if bands is None else {'bands': bands})
            return item

        clean_ids = [f'{domain}/{lang}/{label}.wav' for domain in ('online', 'offline')
                     for lang in ('en', 'zh') for label in (0, 1)]
        noisy_ids = [f'offline/{lang}/{label}.wav' for band in range(4)
                     for lang in ('en', 'zh') for label in (0, 1)]
        self.dev = {'clean': [dev_batch(clean_ids, [0, 1]*4)],
                    'seen': [dev_batch(noisy_ids, [0, 1]*8, [b for b in range(4) for _ in range(4)])],
                    'heldout': [dev_batch(noisy_ids, [0, 1]*8, [b for b in range(4) for _ in range(4)])]}


class LanguageEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tags_preserve_waveforms_rng_ticket_and_pair_layout(self):
        ids = ['offline/en/f.wav', 'offline/en/r.wav', 'offline/zh/f.wav', 'offline/zh/r.wav']
        labels = [0, 1, 0, 1]
        budget = LanguageBudget(ids, labels)
        batches = []
        for kind in ('ordinary', 'pair', 'noisy_pair'):
            inner = WaveRows(ids, labels, kind)
            tagged = SourceTaggedDataset(inner, ids, labels, budget)
            state = torch.get_rng_state()
            rows = [tagged[(i, 2, 3)] for i in range(4)]
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            for i, row in enumerate(rows):
                expected = inner[i]
                for actual, original in zip(row[:-1], expected):
                    if isinstance(actual, torch.Tensor):
                        torch.testing.assert_close(actual, original, rtol=0, atol=0)
                    else:
                        self.assertEqual(actual, original)
            b = collator(kind)(rows)
            repeats = 1 if kind == 'ordinary' else 2
            self.assertEqual(b['source_ids'], ids*repeats)
            self.assertEqual(b['languages'].tolist(), [0, 0, 1, 1]*repeats)
            self.assertEqual(b['labels'].tolist(), labels*repeats)
            torch.testing.assert_close(b['language_weights'], budget.coefficients(b['labels'], b['languages']))
            batches.append(b)
        merged, layout = combine_batches(batches)
        self.assertEqual(layout, (4, 4, 4))
        self.assertEqual(merged['source_ids'], ids*5)
        self.assertEqual(merged['features'].shape, (20, 12, 160))
        broken = copy.deepcopy(batches)
        del broken[1]['languages']
        with self.assertRaises(ValueError):
            combine_batches(broken)
        malformed = rows.copy()
        malformed[0] = (*rows[0][:-1], {**rows[0][-1], 'label': 1})
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            collator('pair')(malformed)

    def test_interrupted_scores_are_preserved_before_replaying_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for name in ('epoch_001_scores.jsonl', 'epoch_002_scores.jsonl', 'epoch_002_evaluation.json', 'last.pt'):
                (out/name).write_text(name)
            engine.archive_uncommitted_diagnostics(out, 2)
            self.assertTrue((out/'epoch_001_scores.jsonl').is_file())
            self.assertFalse((out/'epoch_002_scores.jsonl').exists())
            copies = list((out/'interrupted_diagnostics').glob('*/epoch_002_scores.jsonl'))
            self.assertEqual(len(copies), 1)
            self.assertEqual(copies[0].read_text(), 'epoch_002_scores.jsonl')
            self.assertEqual((out/'last.pt').read_text(), 'last.pt')

    def test_one_epoch_actual_updates_scores_report_and_resume_guards(self):
        helper = fixtures.RefinementEngineTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, original = helper.make_baseline(root)
            digest = sha256(baseline)
            out = root/'new'/'stage3'
            args = helper.arguments(out, 1) + ['--finetune_from', str(baseline),
                                              '--language_weighting', '--adaptation_control']
            def run(argv):
                with patch.object(engine, 'DataBundle', LanguageFixtureBundle), \
                     patch.object(engine.Detector, 'load', side_effect=build_fixture_detector), \
                     patch.object(engine, 'source_hashes', return_value={}), \
                     patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                    engine.main()
            run(args)
            self.assertEqual(sha256(baseline), digest)
            last = load_checkpoint(out/'last.pt')
            self.assertEqual((last['epoch'], last['global_step']), (1, 1))
            self.assertTrue(last['config']['language_weighting'])
            self.assertEqual(last['config']['en_real_budget'], .35)
            self.assertEqual(last['config']['en_fake_budget'], .4)
            for name, value in original.items():
                if name.startswith('backbone.encoder.layers.'):
                    layer = int(name.split('.')[3])
                    if layer < 20:
                        torch.testing.assert_close(last['model'][name], value, rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(last['model'][k], v) for k, v in original.items() if k.startswith('head.')))
            metrics = json.loads((out/'metrics.jsonl').read_text())
            self.assertEqual(sum(x['count'] for x in metrics['train_groups']['all'].values()), 40)
            self.assertEqual(sum(x['count'] for x in metrics['train_groups']['ordinary'].values()), 24)
            self.assertEqual(sum(x['count'] for x in metrics['train_groups']['noisy_processed'].values()), 4)
            for name in ('baseline_scores.jsonl', 'epoch_001_scores.jsonl'):
                scores = [json.loads(line) for line in (out/name).read_text().splitlines()]
                self.assertEqual(len(scores), 40)
                self.assertEqual({s['language'] for s in scores}, {'en', 'zh'})
                self.assertTrue(all(language_id(s['source_id']) == ('en', 'zh').index(s['language']) for s in scores))
            self.assertIn('language_groups', metrics['dev'])
            export = export_language_report(out, download_dir=root/'download')
            self.assertTrue(Path(export['download_archive']).is_file())
            summary = json.loads(Path(export['summary']).read_text(encoding='utf-8'))
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['epochs'][0]['train_groups'], metrics['train_groups'])
            resumed = helper.arguments(out, 1) + ['--resume', str(out/'last.pt'),
                                                  '--language_weighting', '--adaptation_control']
            before = {name: sha256(out/name) for name in ('last.pt', 'best_model.pt', 'metrics.jsonl', 'epoch_001_scores.jsonl')}
            run(resumed)
            self.assertEqual(before, {name: sha256(out/name) for name in before})
            with self.assertRaisesRegex(ValueError, 'en_real_budget'):
                run(resumed + ['--en_real_budget', '.45'])
            self.assertEqual(sha256(baseline), digest)


if __name__ == '__main__':
    unittest.main()
