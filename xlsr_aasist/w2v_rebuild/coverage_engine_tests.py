"""Exercise real DataBundle/cropping/sampling/engine/export with a tiny CPU model."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from utils.data_utils import SpoofAudioDataset
from . import train as engine
from .core import atomic_save, load_checkpoint, sha256
from .data import DataBundle, FeatureCollator
from .group_metrics import export_language_report
from .language_data_tests import BundleFixture
from .refinement_engine_tests import RefinementEngineTests, build_fixture_detector


def training_features(_collator, waves):
    features = torch.stack([wave[:3840] for wave in waves]).reshape(len(waves), 24, 160)
    return features, torch.ones((len(waves), 24), dtype=torch.long)


class CoverageEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_one_epoch_actual_crops_updates_freezing_reports_and_exact_resume(self):
        fixture = BundleFixture()
        for row in fixture.caches[str(fixture.root/'extra')]:
            row['processing'] = {'family': 'webrtc'}
        def build_dataset(_protocol, root, mode, args, **kwargs):
            inner = SpoofAudioDataset(fixture.ids, root, fixture.labels, args=args,
                                      use_rawboost=mode == 'train', algo=0)
            return inner, fixture.ids, fixture.labels
        fixture.build_dataset = build_dataset
        def load_wave(path, **kwargs):
            parts = Path(path).parts
            pos = next(i for i, value in enumerate(parts) if value in ('offline', 'online'))
            source = '/'.join(parts[pos:])
            wave = np.sin(np.arange(150000, dtype=np.float32)*.03)*.1
            wave += np.float32(fixture.markers[source]*.0001)
            return wave, 16000
        helper = RefinementEngineTests()
        with tempfile.TemporaryDirectory() as temporary, fixture.active(), ExitStack() as patches:
            patches.enter_context(patch('utils.data_utils.librosa.load', side_effect=load_wave))
            patches.enter_context(patch('utils.data_utils.NoiseAugment.from_env', return_value=lambda x, sr: x))
            patches.enter_context(patch('utils.data_utils.process_rawboost_feature', side_effect=lambda x, *a: x))
            patches.enter_context(patch.object(FeatureCollator, 'extract', training_features))
            # Use real source metadata and the real sampler; only acoustic I/O and
            # the expensive nonlearned feature extraction are replaced in this test.
            original_init = FeatureCollator.__init__
            def collator_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.extractor = object()
            patches.enter_context(patch.object(FeatureCollator, '__init__', collator_init))
            root = Path(temporary)
            baseline, initial_weights = helper.make_baseline(root)
            fixture_args = fixture.args(True)
            fixture_args.coverage_training = True
            fixture_args.coverage_prefix_probability = .5
            fixture_args.noisy_bank_policy = 'cycle'
            with redirect_stdout(io.StringIO()):
                fingerprints = DataBundle(fixture_args).base_fingerprints
            saved = load_checkpoint(baseline)
            saved['data_fingerprints'] = fingerprints
            atomic_save(saved, baseline)
            digest = sha256(baseline)
            out = root/'new'/'stage3'
            arguments = helper.arguments(out, 1) + ['--finetune_from', str(baseline), '--language_weighting',
                '--coverage_training', '--adaptation_control', '--noisy_extra_fraction', '.2']
            for key in ('train_data_path', 'dev_data_path', 'train_protocol', 'dev_protocol', 'rtc_pairs',
                        'train_noise_manifest', 'train_noisy_cache', 'dev_noisy_cache', 'dev_heldout_cache', 'ssl_path'):
                arguments += ['--'+key, getattr(fixture_args, key)]
            arguments += ['--extra_train_noisy_cache', fixture_args.extra_train_noisy_cache[0]]
            def run(argv):
                with patch.object(engine.Detector, 'load', side_effect=build_fixture_detector), \
                     patch.object(engine, 'source_hashes', return_value={}), \
                     patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                    engine.main()
            run(arguments)
            self.assertEqual(sha256(baseline), digest)
            last = load_checkpoint(out/'last.pt')
            self.assertEqual(last['global_step'], 2)
            self.assertEqual(last['sampler']['format'], 'rtc_coverage_rotation_v1')
            for name, value in initial_weights.items():
                if name.startswith('backbone.encoder.layers.') and int(name.split('.')[3]) < 20:
                    torch.testing.assert_close(last['model'][name], value, rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(last['model'][k], v) for k, v in initial_weights.items()
                                if k.startswith('head.')))
            actual = json.loads((out/'coverage_actual_epoch_001.json').read_text())
            self.assertEqual(actual['ordinary_unique_files'], len(fixture.ids))
            self.assertEqual(actual['ordinary_views'], len(fixture.ids))
            self.assertEqual(actual['noisy_processed_views'], 8)
            self.assertTrue(any(k.endswith('|random') for k in actual['crop_counts']))
            self.assertTrue(any(k.endswith('|prefix') for k in actual['crop_counts']))
            metrics = json.loads((out/'metrics.jsonl').read_text())
            self.assertEqual(metrics['coverage'], actual)
            self.assertEqual(metrics['train_groups']['ordinary']['en-real']['count'], 6)
            self.assertEqual(last['config']['language_budgets']['noisy_pair']['coefficients'], [[1., 1.], [1., 1.]])
            for name in ('baseline_scores.jsonl', 'epoch_001_scores.jsonl'):
                self.assertTrue((out/name).is_file())
            report = export_language_report(out, download_dir=root/'download')
            with zipfile.ZipFile(report['download_archive']) as package:
                self.assertIn('stage3/coverage_actual_epoch_001.json', package.namelist())
                self.assertIn('stage3/coverage_plan_epoch_001.json', package.namelist())
                self.assertEqual(json.loads(package.read('summary.json'))['status'], 'complete')
                self.assertFalse(any(x.endswith('.pt') for x in package.namelist()))
            resumed = arguments.copy()
            index = resumed.index('--finetune_from')
            resumed[index:index+2] = ['--resume', str(out/'last.pt')]
            before = {name: sha256(out/name) for name in ('last.pt', 'best_model.pt', 'metrics.jsonl')}
            run(resumed)
            self.assertEqual(before, {name: sha256(out/name) for name in before})
            with self.assertRaisesRegex(ValueError, 'coverage_prefix_probability'):
                run(resumed + ['--coverage_prefix_probability', '.8'])
            self.assertEqual(sha256(baseline), digest)


if __name__ == '__main__':
    unittest.main()
