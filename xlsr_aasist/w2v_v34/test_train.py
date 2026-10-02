"""Real tiny w2v-BERT: complete encoder updates, submission identity and exact resume."""
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
from w2v_v33.data import PairedEpochPlan
from w2v_v3.model import Detector, HeadConfig
from . import train as training
from w2v_v33.control import Controller
from w2v_v31.test_control import dev


torch.set_num_threads(1)


def fixture(root):
    from w2v_v33.test_train import fixture as paired_fixture
    cfg, discovery, fingerprints, warm_model = paired_fixture(root)
    source = training.read_state(cfg['warm_checkpoint'])
    source.update(schema='rtc_w2v_multiconv_v33', tag='baseline')
    warm = root/'submitted_v33.pt'
    atomic_save(warm, source)
    cfg.update(warm_checkpoint=str(warm), warm_checkpoint_sha256=sha256(warm),
               expected_warm_tag='baseline', trainable_layers=2, layer_decay=.9,
               source_provenance={'selected_arm': 'candidate', 'checkpoint_tag': 'baseline',
                                  'checkpoint_sha256': sha256(warm), 'baseline_fallback': True},
               platform_reference={'weighted': 93.3941, 'status': 'user_reported'},
               source_data_fingerprints=fingerprints)
    cfg['source_provenance']['file_fingerprints'] = {str(warm): sha256(warm)}
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
        self.assertEqual(actual['update_diagnostics'].keys(), expected['update_diagnostics'].keys())
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
                self.assertTrue(any(not torch.equal(value, final['model'][key])
                                    for key, value in warm_model.state_dict().items()
                                    if key.startswith('backbone.encoder.layers.0.')))
                for key, value in warm_model.state_dict().items():
                    if key.startswith('backbone.') and not key.startswith('backbone.encoder.layers.'):
                        torch.testing.assert_close(final['model'][key], value, rtol=0, atol=0)
                self.assertEqual(final['schema'], 'rtc_w2v_multiconv_v34')
                self.assertEqual(selected['schema'], 'rtc_w2v_multiconv_v34')
                self.assertEqual(selected['source_provenance']['warm_checkpoint_tag'], 'baseline')
                self.assertEqual(selected['source_provenance']['source_provenance']['checkpoint_sha256'], cfg['warm_checkpoint_sha256'])
                exported = Detector.from_checkpoint(selected, checkpointing=False)
                for key, value in warm_model.state_dict().items():
                    torch.testing.assert_close(exported.state_dict()[key], value, rtol=0, atol=0)
                self.assertTrue((first/'optimizer_groups.json').is_file())
                self.assertTrue((first/'epoch_1_step_1_parameter_updates.json').is_file())

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
            self.assertTrue(completed['starting_v33_best_preserved'])
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
                with self.assertRaisesRegex(ValueError, 'identical config, metadata, code, versions|data fingerprints differ'):
                    training.train(cfg, first, first / 'last.pt')
            with patch.object(training, 'build_data', discovery), patch.object(
                    training, 'source_fingerprints', return_value={'changed_source': 'yes'}):
                with self.assertRaisesRegex(ValueError, 'identical config, metadata, code, versions|data fingerprints differ'):
                    training.train(cfg, first, first / 'last.pt')

    def test_warm_checkpoint_identity_and_weight_kind_are_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            cfg, _, _, _ = fixture(Path(td))
            with self.assertRaisesRegex(ValueError, 'unchanged V3.3 weight'):
                training.initialize({**cfg, 'warm_checkpoint_sha256': 'invalid'})
            warm = training.read_state(cfg['warm_checkpoint'])
            warm['kind'] = 'training'
            wrong = Path(td) / 'last_not_weights.pt'
            atomic_save(wrong, warm)
            with self.assertRaisesRegex(ValueError, 'weight checkpoint, not last.pt'):
                training.initialize({**cfg, 'warm_checkpoint': str(wrong), 'warm_checkpoint_sha256': sha256(wrong)})
            warm['kind'], warm['tag'] = 'weights', 'epoch_1'
            atomic_save(wrong, warm)
            with self.assertRaisesRegex(ValueError, 'expected_warm_tag'):
                training.initialize({**cfg, 'warm_checkpoint': str(wrong), 'warm_checkpoint_sha256': sha256(wrong)})

    def test_rejects_old_schema_wrong_preprocessor_and_config_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg, discovery, _, _ = fixture(root)
            source = training.read_state(cfg['warm_checkpoint'])
            old = root/'old_v3.pt'
            source['schema'] = 'rtc_w2v_multiconv_v3'
            atomic_save(old, source)
            with self.assertRaisesRegex(ValueError, 'V3.3 weight checkpoint'):
                training.initialize({**cfg, 'warm_checkpoint': str(old), 'warm_checkpoint_sha256': sha256(old)})
            source['schema'] = 'rtc_w2v_multiconv_v33'
            source['data_fingerprints'] = {'preprocessor_config.json': 'bad'}
            atomic_save(old, source)
            with self.assertRaisesRegex(ValueError, 'Feature extractor differs'):
                training.initialize({**cfg, 'warm_checkpoint': str(old), 'warm_checkpoint_sha256': sha256(old)})
            run = root/'run'
            with patch.object(training, 'build_data', discovery):
                training.train(cfg, run)
                with self.assertRaisesRegex(ValueError, 'Exact V3.4 resume'):
                    training.train({**cfg, 'layer_decay': .85}, run, run/'last.pt')

    def test_source_identity_changes_are_rejected_before_data_or_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg, discovery, fingerprints, _ = fixture(root)
            record = root/'submission_meta.json'
            record.write_text('{}', encoding='utf-8')
            cfg['source_provenance']['file_fingerprints'][str(record)] = sha256(record)
            record.write_text('{"changed":true}', encoding='utf-8')
            with patch.object(training, 'build_data', side_effect=AssertionError('must fail before data')):
                with self.assertRaisesRegex(ValueError, 'source provenance changed'):
                    training.train(cfg, root/'refused')

    def test_nonreentrant_checkpointing_updates_early_and_late_encoder_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg, discovery, _, warm = fixture(root)
            cfg.update(checkpointing=True, joint_epochs=1)
            run = root/'checkpointed'
            with patch.object(training, 'build_data', discovery):
                training.train(cfg, run)
            state = training.read_state(run/'last.pt')
            for layer in (0, 1):
                prefix = f'backbone.encoder.layers.{layer}.'
                self.assertTrue(any(not torch.equal(value, state['model'][key])
                                    for key, value in warm.state_dict().items() if key.startswith(prefix)))
            for record in state['history'][1:]:
                for layer in (0, 1):
                    diag = record['parameter_updates']['groups'][f'encoder.layer_{layer:02d}']
                    self.assertGreater(diag['sampled_gradient_l2'], 0.)
                    self.assertGreater(diag['sampled_changed_coordinates'], 0)

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
                        self.assertEqual(record.pop('arm'), 'V3.4')
                        self.assertEqual(record.pop('source_arm'), arm)
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

    def test_restore_reduce_reinstates_adam_and_layer_ratios_across_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg, discovery, _, _ = fixture(root)
            cfg.update(selection_min_delta=.0001, lr_factor=.5)
            metrics = {
                'baseline_scores.jsonl': dev(.950, .940, .970, .860),
                'epoch_1_step_1_scores.jsonl': dev(.952, .943, .971, .870),
                # Severe deterioration after a real Adam update: restore the
                # accepted half-epoch weights AND their nonempty moments.
                'epoch_1_scores.jsonl': dev(.944, .932, .967, .820),
                'epoch_2_step_1_scores.jsonl': dev(.951, .942, .971, .865),
                'epoch_2_scores.jsonl': dev(.953, .944, .972, .880),
            }
            def measured(model, rows, recipe, device, path):
                value = dict(metrics[path.name])
                value.update(seen_f1=value['noisy_f1'], heldout_f1=value['noisy_f1'])
                return value

            actual_step = training.supervised_step
            checked = []
            def assert_restored_before_update(model, optimizer, run):
                selected = training.read_state(run/'control_best.pt')
                self.assertEqual(selected['tag'], 'epoch_1_step_1')
                self.assertTrue(selected['optimizer']['state'])
                for key, value in selected['model'].items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                current = optimizer.state_dict()
                self.assertEqual(set(current['state']), set(selected['optimizer']['state']))
                for index, slots in selected['optimizer']['state'].items():
                    for name, value in slots.items():
                        torch.testing.assert_close(current['state'][index][name], value, rtol=0, atol=0)
                scale = training.lr_scale({**cfg, 'lr_warmup_steps': cfg['joint_warmup_steps']},
                                          phase_step=2, phase_steps=4, controller_scale=.5)
                ratios = []
                for group, saved in zip(current['param_groups'], selected['optimizer']['param_groups']):
                    self.assertEqual(group['name'], saved['name'])
                    self.assertEqual(group['base_lr'], saved['base_lr'])
                    self.assertEqual(group['params'], saved['params'])
                    self.assertAlmostEqual(group['lr'], saved['base_lr']*scale, places=14)
                    ratios.append(group['lr']/group['base_lr'])
                self.assertTrue(all(abs(ratio-scale) < 1e-12 for ratio in ratios))
                checked.append(str(run))

            with patch.object(training, 'build_data', discovery), patch.object(training, 'validate', measured):
                uninterrupted = root/'uninterrupted_recovery'
                calls = [0]
                def observed(model, examples, optimizer, *args, **kwargs):
                    if calls[0] == 2:
                        assert_restored_before_update(model, optimizer, uninterrupted)
                    calls[0] += 1
                    return actual_step(model, examples, optimizer, *args, **kwargs)
                with patch.object(training, 'supervised_step', side_effect=observed):
                    training.train(cfg, uninterrupted)
                final = training.read_state(uninterrupted/'last.pt')
                self.assertEqual(calls[0], 4)
                self.assertEqual(final['history'][2]['decision']['action'], 'restore_reduce')
                self.assertEqual(final['controller']['reductions'], 1)

                resumed = root/'resumed_recovery'
                calls[0] = 0
                def interrupt(model, examples, optimizer, *args, **kwargs):
                    if calls[0] == 2:
                        assert_restored_before_update(model, optimizer, resumed)
                        raise RuntimeError('interrupt after committed restore_reduce')
                    calls[0] += 1
                    return actual_step(model, examples, optimizer, *args, **kwargs)
                with patch.object(training, 'supervised_step', side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, 'committed restore_reduce'):
                        training.train(cfg, resumed)
                boundary = training.read_state(resumed/'last.pt')
                self.assertEqual((boundary['epoch'], boundary['cursor'], boundary['global_steps']), (1, 2, 2))
                self.assertEqual(boundary['tag'], 'epoch_1_step_1')
                self.assertEqual(boundary['controller']['lr_scale'], .5)
                resumed_calls = [0]
                def after_resume(model, examples, optimizer, *args, **kwargs):
                    if resumed_calls[0] == 0:
                        assert_restored_before_update(model, optimizer, resumed)
                    resumed_calls[0] += 1
                    return actual_step(model, examples, optimizer, *args, **kwargs)
                with patch.object(training, 'supervised_step', side_effect=after_resume):
                    training.train(cfg, resumed, resumed/'last.pt')
                self.assertEqual(resumed_calls[0], 2)
                self.assert_same_training(training.read_state(resumed/'last.pt'), final)
            self.assertEqual(len(checked), 3)
            self.assertEqual(sha256(cfg['warm_checkpoint']), cfg['warm_checkpoint_sha256'])
            self.assertEqual(sha256(cfg['baseline']), cfg['baseline_sha256'])

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
