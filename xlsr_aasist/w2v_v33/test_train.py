"""Real tiny w2v-BERT: protected warm start, full/short training and exact resume."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from w2v_aasist.runtime import atomic_save, sha256
from .data import PairedEpochPlan
from w2v_v3.model import Detector, HeadConfig
from . import train as training
from .control import Controller
from w2v_v31.test_control import dev


torch.set_num_threads(1)


def fixture(root):
    import soundfile as sf
    from transformers import SeamlessM4TFeatureExtractor
    from w2v_aasist.tests import tiny_detector
    ssl = root / 'ssl'
    SeamlessM4TFeatureExtractor().save_pretrained(ssl)
    original = tiny_detector()
    baseline = root / 'original.pt'
    fingerprints = {str(ssl / 'preprocessor_config.json'): sha256(ssl / 'preprocessor_config.json')}
    original_state = {'schema': 'rtc_w2v_rebuild_v1', 'model': original.state_dict(),
                      'model_config': original.backbone.config.to_dict(),
                      'data_fingerprints': fingerprints, 'dev': {}}
    atomic_save(baseline, original_state)
    head = HeadConfig(input_dim=16, projection=8, expansion=32, blocks=4,
                      kernels=(3, 7), merge_kernel=3, dropout=.1)
    warm_model = Detector.from_original(original_state, head_config=asdict(head), checkpointing=False)
    warm = root / 'v3_best.pt'
    atomic_save(warm, {'schema': 'rtc_w2v_multiconv_v3', 'kind': 'weights', 'tag': 'epoch_2_step_2417',
                       'model': warm_model.state_dict(), **warm_model.architecture(),
                       'data_fingerprints': fingerprints})
    ordinary, noisy, dev_noisy = [], [], []
    for i in range(8):
        wave = np.random.default_rng(i).normal(0, .03, 9200 + i*80).astype(np.float32)
        path = root / f'{i}.wav'
        sf.write(path, wave, 16000, subtype='FLOAT')
        row = dict(id=f'{i}.wav', audio=str(path), label=i % 2, language='en',
                   domain='online' if i < 4 else 'offline', noisy=False, band=-1)
        ordinary.append(row)
        if i >= 4:
            for version in (0, 1):
                noisy.append(dict(row, noisy=True, band=version*2, version=version,
                                  full_length=True, output_samples=len(wave),
                                  source_audio=str(path), source_sha256=sha256(path)))
            for band in range(4):
                dev_noisy.append(dict(row, noisy=True, band=band, full_length=True,
                                      output_samples=len(wave)))
    sources = []
    for i in range(4):
        off, on = ordinary[i + 4], ordinary[i]
        sources.append(dict(off, source_id=off['id'], missing_online=False,
                            conditions={'offline': dict(off, condition='offline'),
                                        'online': dict(on, condition='online')}))
    paired_cache = [dict(row, source_id=row['id'], condition='noisy_a' if row['version'] == 0 else 'noisy_b') for row in noisy]
    plan = PairedEpochPlan(sources, paired_cache, source_batch=2, seed=7)
    validation = {'clean': ordinary, 'seen': dev_noisy, 'heldout': dev_noisy}
    cfg = dict(device='cpu', amp='none', eval_amp='none', seed=71,
               baseline=str(baseline), baseline_sha256=sha256(baseline),
               warm_checkpoint=str(warm), warm_checkpoint_sha256=sha256(warm),
               source_batch=2, arm='candidate', pair_weight=.02, pair_ramp_fraction=.1,
               aux_max_tokens=16, head_epochs=0, joint_epochs=2,
               evals_per_epoch=2, trainable_layers=1, head_lr=.0001, joint_head_lr=.0001,
               encoder_lr=.00001, weight_decay=.0001, workers=0, ssl_path=str(ssl),
               max_seconds=0., rawboost=0, raw_config={}, joint_warmup_steps=1,
               noisy_weight_start=.5, noisy_weight=.5, noisy_ramp_epochs=0.,
               cka_weight=.01, cka_warmup_steps=0, grad_clip=1.,
               checkpointing=False, microbatch=2, frame_budget=100,
               production_layout=False, eval_batch=8, offload_activations=False,
               full_noisy=True, short_loss_weight=.3, short_min_seconds=.42,
               short_max_seconds=.47, short_prefix_probability=.5,
               # Guarantee a protected fallback while still running actual Dev
               # inference and all training/gradient/optimizer/control paths.
               processing_enabled=True, processing_identity_probability=.5,
               processing_single_probability=.35, processing_silence_probability=.05,
               fusion_chunk_layers=3, selection_min_delta=.99, en_real_tolerance=.005,
               noisy_fake_tolerance=.003, clean_tolerance=.001)
    discovery = lambda _: (plan, validation, torch.ones(2), torch.tensor([2, 2]), fingerprints)
    return cfg, discovery, fingerprints, warm_model


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Tests emulate one deployed, immutable source version even when other
        # agents are editing independent files in the shared development tree.
        fingerprints = training.source_fingerprints()
        stable_sources = patch.object(training, 'source_fingerprints', return_value=fingerprints)
        stable_sources.start()
        self.addCleanup(stable_sources.stop)

    def assert_same_training(self, actual, expected):
        self.assertEqual(actual['source_exposures'], expected['source_exposures'])
        self.assertEqual(actual['epoch_sources'], expected['epoch_sources'])
        self.assertEqual(actual['controller'], expected['controller'])
        self.assertEqual(actual['history'], expected['history'])
        self.assertEqual(actual['global_steps'], expected['global_steps'])
        for key, value in expected['model'].items():
            torch.testing.assert_close(actual['model'][key], value, rtol=0, atol=0)
        self.assertEqual(actual['optimizer']['param_groups'], expected['optimizer']['param_groups'])
        self.assertEqual(set(actual['optimizer']['state']), set(expected['optimizer']['state']))
        for index, state in expected['optimizer']['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(actual['optimizer']['state'][index][key], value, rtol=0, atol=0)

    def test_real_training_bootstrap_fallback_three_resume_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg, discovery, fingerprints, warm_model = fixture(root)
            first = root / 'first'
            actual_validate = training.validate
            with patch.object(training, 'build_data', discovery):
                training.train(cfg, first)
                final = training.read_state(first / 'last.pt')
                self.assertTrue(final['complete'])
                self.assertEqual(final['global_steps'], 4)
                self.assertEqual(len(final['history']), 5)
                self.assertEqual(final['history'][0]['tag'], 'baseline')
                self.assertEqual(final['history'][0]['global_steps'], 0)
                self.assertEqual(final['history'][0]['decision']['remaining_evaluations'], 4)
                self.assertEqual(final['controller']['best_safe']['tag'], 'baseline')
                selected = training.read_state(first / 'best_model.pt')
                self.assertEqual(selected['tag'], 'baseline')
                for key, value in warm_model.state_dict().items():
                    torch.testing.assert_close(selected['model'][key], value, rtol=0, atol=0)
                # Actual expanded views reached the model, not just metadata.
                for record in final['history'][1:]:
                    self.assertGreater(record['train']['short_views'], 0)
                    self.assertGreater(record['train']['short_ce_contribution'], 0)
                    self.assertEqual(record['train']['source_count'], 2)
                    self.assertTrue(any(key.endswith('/short') for key in record['train_groups']))
                self.assertTrue(any(not torch.equal(value, final['model'][key])
                                    for key, value in warm_model.state_dict().items()
                                    if key.startswith('head.')))
                self.assertTrue(any(not torch.equal(value, final['model'][key])
                                    for key, value in warm_model.state_dict().items()
                                    if key.startswith('backbone.encoder.layers.1.')))
                for key, value in warm_model.state_dict().items():
                    if key.startswith('backbone.encoder.layers.0.'):
                        torch.testing.assert_close(final['model'][key], value, rtol=0, atol=0)

                # Crash after the initial validation checkpoint, before the first
                # update. Restart must reproduce the uninterrupted epoch seed.
                initial_resume = root / 'initial_resume'
                def before_first_update(model, examples, optimizer, *args, **kwargs):
                    for key, value in warm_model.state_dict().items():
                        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                    self.assertEqual(len(optimizer.state), 0)
                    raise RuntimeError('interrupt before first optimizer update')
                with patch.object(training, 'supervised_step', side_effect=before_first_update):
                    with self.assertRaisesRegex(RuntimeError, 'before first optimizer update'):
                        training.train(cfg, initial_resume)
                initial = training.read_state(initial_resume / 'last.pt')
                self.assertEqual((initial['epoch'], initial['cursor'], initial['global_steps']), (1, 0, 0))
                self.assertEqual(initial['controller']['phase_evals'], 0)
                self.assertFalse(initial['optimizer']['state'])
                self.assertEqual(initial['tag'], 'baseline')
                training.train(cfg, initial_resume, initial_resume / 'last.pt')
                self.assert_same_training(training.read_state(initial_resume / 'last.pt'), final)

                for name, interruption, epoch, cursor in (
                        ('half_resume', 'epoch_1_scores.jsonl', 1, 1),
                        ('epoch_resume', 'epoch_2_step_1_scores.jsonl', 1, 2)):
                    destination = root / name
                    def interrupt(model, validation, cfg, device, path):
                        if path.name == interruption:
                            raise RuntimeError('interrupt next validation')
                        return actual_validate(model, validation, cfg, device, path)
                    with patch.object(training, 'validate', side_effect=interrupt):
                        with self.assertRaisesRegex(RuntimeError, 'interrupt next validation'):
                            training.train(cfg, destination)
                    boundary = training.read_state(destination / 'last.pt')
                    self.assertEqual((boundary['epoch'], boundary['cursor']), (epoch, cursor))
                    self.assertEqual(boundary['controller']['phase'], 'joint')
                    training.train(cfg, destination, destination / 'last.pt')
                    self.assert_same_training(training.read_state(destination / 'last.pt'), final)

            self.assertEqual(sha256(cfg['baseline']), cfg['baseline_sha256'])
            self.assertEqual(sha256(cfg['warm_checkpoint']), cfg['warm_checkpoint_sha256'])
            completed = json.loads((first / 'completed.json').read_text())
            self.assertTrue(completed['starting_v3_best_preserved'])
            self.assertFalse(completed['eligible_improvement'])
            for name in ('baseline.json', 'epoch_1_step_1.json', 'epoch_1.json', 'epoch_2.json',
                         'best_model.pt', 'best_weighted.pt', 'best_noisy.pt', 'control_best.pt'):
                self.assertTrue((first / name).is_file(), name)
            self.assertEqual(json.loads((first / 'planned_coverage.json').read_text())['sources_per_epoch'], 4)
            self.assertEqual(final['source_exposures'], 8)
            self.assertEqual(final['epoch_sources'], 4)
            self.assertTrue((first / 'acceptance.json').is_file())
            self.assertTrue((first / 'baseline_fingerprint.json').is_file())
            # A crash after last.pt commit may leave report.md behind. Completed
            # exact resume must repair it without performing any extra updates.
            (first / 'report.md').write_text('stale report', encoding='utf-8')
            with patch.object(training, 'build_data', discovery), patch.object(
                    training, 'supervised_step', side_effect=AssertionError('completed run must not update')):
                training.train(cfg, first, first / 'last.pt')
            self.assertIn('epoch_2', (first / 'report.md').read_text(encoding='utf-8'))

            # Exact resume must reject changed metadata and changed source code.
            changed = dict(fingerprints, unexpected_input='changed')
            with patch.object(training, 'build_data', lambda _: (*discovery(_)[:4], changed)):
                with self.assertRaisesRegex(ValueError, 'identical config, metadata, code, versions'):
                    training.train(cfg, first, first / 'last.pt')
            with patch.object(training, 'build_data', discovery), patch.object(
                    training, 'source_fingerprints', return_value={'changed_source': 'yes'}):
                with self.assertRaisesRegex(ValueError, 'identical config, metadata, code, versions'):
                    training.train(cfg, first, first / 'last.pt')

    def test_warm_checkpoint_identity_and_weight_kind_are_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            cfg, _, _, _ = fixture(Path(td))
            with self.assertRaisesRegex(ValueError, 'unchanged MultiConv best'):
                training.initialize({**cfg, 'warm_checkpoint_sha256': 'invalid'})
            warm = training.read_state(cfg['warm_checkpoint'])
            warm['kind'] = 'training'
            wrong = Path(td) / 'last_not_weights.pt'
            atomic_save(wrong, warm)
            with self.assertRaisesRegex(ValueError, 'weight checkpoint, not last.pt'):
                training.initialize({**cfg, 'warm_checkpoint': str(wrong), 'warm_checkpoint_sha256': sha256(wrong)})
            warm['kind'], warm['tag'] = 'weights', 'epoch_1'
            atomic_save(wrong, warm)
            with self.assertRaisesRegex(ValueError, 'epoch_2_step_2417'):
                training.initialize({**cfg, 'warm_checkpoint': str(wrong), 'warm_checkpoint_sha256': sha256(wrong)})

    def test_control_and_zero_weight_candidate_share_identical_forward_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg, discovery, _, _ = fixture(root)
            cfg.update(joint_epochs=1, pair_weight=0.)
            states = []
            with patch.object(training, 'build_data', discovery):
                for arm in ('control', 'candidate'):
                    destination = root / arm
                    training.train({**cfg, 'arm': arm}, destination)
                    states.append(training.read_state(destination / 'last.pt'))
                    for record in states[-1]['history']:
                        self.assertEqual(record.pop('arm'), arm)
            self.assert_same_training(*states)

    def test_non_fp32_dev_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as td:
            cfg, _, _, _ = fixture(Path(td))
            with self.assertRaisesRegex(ValueError, 'fixed FP32'):
                training.train({**cfg, 'eval_amp': 'bf16'}, Path(td) / 'wrong_precision')

    def test_first_full_pair_weight_is_diagnosed_once_across_committed_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg, discovery, _, _ = fixture(root)
            # Two sources/update: the 3-source ramp ends on update 2. Its
            # validation is committed before interrupting update 3.
            cfg.update(pair_warmup_fraction=.75, diagnose_aux_steps=[])
            run = root / 'ramp_resume'
            actual_step = training.supervised_step
            calls = []
            def observed_step(*args, **kwargs):
                if len(calls) == 2:
                    raise RuntimeError('interrupt after committed full-weight diagnostic')
                calls.append((kwargs['pair_weight'], kwargs['diagnose_aux_grad']))
                return actual_step(*args, **kwargs)
            with patch.object(training, 'build_data', discovery):
                with patch.object(training, 'supervised_step', side_effect=observed_step):
                    with self.assertRaisesRegex(RuntimeError, 'committed full-weight'):
                        training.train(cfg, run)
                boundary = training.read_state(run / 'last.pt')
                self.assertEqual(boundary['source_exposures'], 4)
                self.assertEqual(calls[0][1], False)
                self.assertLess(calls[0][0], cfg['pair_weight'])
                self.assertEqual(calls[1], (cfg['pair_weight'], True))
                resumed_calls = []
                def resumed_step(*args, **kwargs):
                    resumed_calls.append(kwargs['diagnose_aux_grad'])
                    return actual_step(*args, **kwargs)
                with patch.object(training, 'supervised_step', side_effect=resumed_step):
                    training.train(cfg, run, run / 'last.pt')
                self.assertEqual(resumed_calls, [False, False])
            rows = [json.loads(line) for line in (run / 'pair_gradient_diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['global_step'], 2)
            self.assertEqual(rows[0]['pair_weight'], cfg['pair_weight'])
            self.assertGreater(rows[0]['weighted_pair_shared_feature_grad_norm'], 0)

    def test_interrupted_promotion_recovers_last_committed_winners(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            controller = Controller({})
            controller.initialize(dev(), 'baseline')
            names = [*training.WINNERS.values(), 'control_best.pt']
            for name in names:
                atomic_save(run / name, {'schema': training.SCHEMA, 'tag': 'baseline', 'model': {'x': torch.ones(1)}})
            training.backup_winners(run, names)
            for name in names[:3]:
                atomic_save(run / name, {'schema': training.SCHEMA, 'tag': 'uncommitted', 'model': {'x': torch.zeros(1)}})
            training.recover_winners(run, controller)
            for name in names:
                self.assertEqual(training._checkpoint_tag(run / name), 'baseline')
                self.assertFalse((run / (name + '.previous')).exists())
            training.backup_winners(run, names)
            for name in names:
                atomic_save(run / name, {'schema': training.SCHEMA, 'tag': 'accepted', 'model': {'x': torch.zeros(1)}})
            controller.observe(dev(.96, .95), 'accepted')
            training.recover_winners(run, controller)
            for name in names:
                self.assertEqual(training._checkpoint_tag(run / name), 'accepted')
                self.assertFalse((run / (name + '.previous')).exists())


if __name__ == '__main__':
    unittest.main()
