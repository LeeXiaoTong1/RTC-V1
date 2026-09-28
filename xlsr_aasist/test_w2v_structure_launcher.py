"""Read-only default, reviewed diagnostic binding, and checkpoint protection."""
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import start_w2v_structure as launch
from start_w2v_coverage import coverage_config
from w2v_rebuild.core import sha256
from w2v_rebuild.structure_recipe import structure_config, structure_recipe_contract


class StructureLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root/'previous run;$(literal)'
        (self.source/'stage3').mkdir(parents=True)
        self.baseline = self.root/'original'/'stage3'/'best_model.pt'
        self.baseline.parent.mkdir(parents=True)
        self.baseline.write_bytes(b'original weights')
        self.cache = self.root/'existing diverse cache'
        self.cache.mkdir()
        (self.cache/'config.json').write_text(json.dumps({'role': 'train', 'generation': 1,
                                                        'processing': {'profile': 'diverse'}}))
        (self.cache/'manifest.jsonl').write_text('existing cache')
        self.config = {'stage': 3, 'adaptation': True, 'baseline_path': str(self.baseline),
                       'init_sha256': sha256(self.baseline), 'extra_train_noisy_cache': [str(self.cache)],
                       'feature_cache': str(self.root/'features'), 'pair_warmup_epochs': 2.,
                       'algo': 5, 'eval_batch': 16, 'eval_microbatch': 4,
                       'train_data_path': 'train audio', 'dev_data_path': 'dev audio',
                       'train_protocol': 'train.txt', 'dev_protocol': 'dev.txt', 'rtc_pairs': 'pairs.jsonl',
                       'train_noise_manifest': 'noise.jsonl', 'train_noisy_cache': 'old train cache',
                       'dev_noisy_cache': 'fixed seen', 'dev_heldout_cache': 'fixed held', 'ssl_path': 'pretrained',
                       'noise_environment': {'RTC_B_NOISE_PROB': '.5'}}
        self.source_config = self.source/'stage3'/'config.json'
        self.source_config.write_text(json.dumps(self.config))
        self.audit = self.root/'reviewed audit'
        self.audit.mkdir()
        (self.audit/'summary.json').write_text(json.dumps({'status': 'complete', 'recommendation': 'ready_for_review'}))
        self.manifest = {'format': 'w2v_structure_audit_v1', 'baseline_path': str(self.baseline),
                         'baseline_sha256': sha256(self.baseline),
                         'input_sha256': {str(self.source_config.resolve()): sha256(self.source_config)},
                         'training_recipe_contract': structure_recipe_contract(structure_config(self.config))}

    def invoke(self, extra=(), train=None, audit_result=0, audit_error=None, upload=False):
        args = ['--from-run', str(self.source), '--download-dir', str(self.root/'download'), *extra]
        if upload:
            args.append('--upload-temp')
        validator = SimpleNamespace(validate_reviewed_audit=lambda *_:
                                    self._validate_stub(audit_error))
        with patch.object(launch, 'ROOT', self.root), \
             patch.object(launch, 'recovery_lock', return_value=nullcontext()), \
             patch.object(launch, 'run_training', side_effect=train) as training, \
             patch.object(launch.subprocess, 'run', return_value=SimpleNamespace(returncode=audit_result)) as auditing, \
             patch.dict('sys.modules', {'audit_w2v_structure': validator}), \
             redirect_stdout(io.StringIO()) as output:
            result = launch.main(args)
        return result, training, auditing, output.getvalue()

    def _validate_stub(self, error):
        if error:
            raise error
        return self.manifest

    def training_args(self):
        return ['--mode', 'train', '--reviewed-audit', str(self.audit)]

    def test_recipe_changes_only_explicit_structure_controls(self):
        prior = coverage_config(self.config, self.cache)
        actual = structure_config(self.config)
        self.assertEqual({k: v for k, v in actual.items() if k not in prior},
                         {'local_structure_weight': .02, 'local_structure_warmup_steps': 100,
                          'noisy_selection': True})
        for key in prior:
            self.assertEqual(actual[key], prior[key])
        command = launch.structure_command(actual, self.root/'new run'/'stage3', self.baseline)
        self.assertEqual(command[command.index('--finetune_from')+1], str(self.baseline))
        self.assertEqual(command[command.index('--local_structure_weight')+1], '0.02')
        self.assertEqual(command[command.index('--epochs')+1], '1')
        self.assertIn('--noisy_selection', command)
        self.assertNotIn('--resume', command)

    def test_default_launches_only_audit_and_preserves_literal_paths(self):
        before = sha256(self.baseline)
        code, training, auditing, output = self.invoke(upload=True)
        self.assertEqual(code, 0)
        training.assert_not_called()
        command = auditing.call_args.args[0]
        self.assertEqual(command[command.index('--from-run')+1], str(self.source))
        self.assertIn('--upload-temp', command)
        self.assertEqual(command[command.index('--train-per-group')+1], '64')
        self.assertEqual(command[command.index('--dev-per-group')+1], '32')
        self.assertIn('MODE=read_only_audit', output)
        self.assertEqual(sha256(self.baseline), before)
        self.assertFalse(list((self.root/'exp').glob('w2v_structure_train*')))

    def test_audit_failure_cannot_fall_through_to_training(self):
        code, training, auditing, _ = self.invoke(audit_result=8)
        self.assertEqual(code, 8)
        training.assert_not_called()
        self.assertEqual(auditing.call_count, 1)

    def test_preview_does_not_start_gpu_or_create_outputs(self):
        code, training, auditing, output = self.invoke(['--preview'])
        self.assertEqual(code, 0)
        training.assert_not_called()
        auditing.assert_not_called()
        self.assertFalse((self.root/'exp').exists())
        self.assertIn('Preview only', output)

    def test_train_requires_explicit_reviewed_directory(self):
        with self.assertRaises(SystemExit):
            self.invoke(['--mode', 'train'])
        self.assertFalse((self.root/'exp').exists())

    def test_rejected_or_inconclusive_audit_stops_before_training(self):
        for recommendation in ('reject', 'inconclusive'):
            with self.subTest(recommendation=recommendation), self.assertRaisesRegex(ValueError, recommendation):
                self.invoke(self.training_args(), audit_error=ValueError(recommendation))
        self.assertFalse((self.root/'exp').exists())

    def test_changed_original_best_is_rejected(self):
        self.baseline.write_bytes(b'changed weights')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.invoke(self.training_args())
        self.assertFalse((self.root/'exp').exists())

    def test_source_config_and_recipe_cannot_drift_after_diagnostic(self):
        self.config['head_lr'] = 9e-5
        self.source_config.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, 'source config'):
            self.invoke(self.training_args())
        self.manifest['input_sha256'][str(self.source_config.resolve())] = sha256(self.source_config)
        self.manifest['training_recipe_contract']['settings']['local_structure_weight'] = .9
        with self.assertRaisesRegex(ValueError, 'recipe differs'):
            self.invoke(self.training_args())
        self.assertFalse((self.root/'exp').exists())

    def test_reviewed_preview_performs_no_training_and_no_output_write(self):
        code, training, auditing, output = self.invoke(self.training_args()+['--preview'])
        self.assertEqual(code, 0)
        training.assert_not_called()
        auditing.assert_not_called()
        self.assertFalse((self.root/'exp').exists())
        self.assertIn('Verified the reviewed diagnostic', output)

    def test_failure_keeps_partial_diagnostics_and_original_best(self):
        def failure(command, env, log_path):
            Path(log_path).write_text('deliberate failure before first epoch')
            self.assertEqual(env['RTC_B_NOISE_PROB'], '.5')
            return 7
        code, training, auditing, output = self.invoke(self.training_args(), train=failure)
        self.assertEqual(code, 7)
        self.assertEqual(training.call_count, 1)
        auditing.assert_not_called()
        self.assertIn('ORIGINAL_BEST_PRESERVED=True', output)
        self.assertIn('STRUCTURE_TRAINING_RESULT=failed', output)
        archive = next((self.root/'download').glob('*.zip'))
        with zipfile.ZipFile(archive) as package:
            self.assertEqual(json.loads(package.read('summary.json'))['status'], 'failed')
            self.assertFalse(any(name.endswith(('.pt', '.wav', '.flac')) for name in package.namelist()))

    def test_success_and_upload_failure_keep_local_report(self):
        def success(command, env, log_path):
            stage = Path(command[command.index('--out')+1])
            stage.mkdir()
            Path(log_path).write_text('one epoch complete')
            (stage/'config.json').write_text(json.dumps(structure_config(self.config)))
            (stage/'completed.json').write_text(json.dumps({'status': 'no_eligible_improvement',
                    'best_model': str(stage/'best_model.pt'), 'candidate_model': None}))
            (stage/'best_model.pt').write_bytes(b'never upload')
            return 0
        with patch('audit_w2v_train.upload_report', side_effect=OSError('network unavailable')):
            code, training, auditing, output = self.invoke(self.training_args(), train=success, upload=True)
        self.assertEqual(code, 0)
        auditing.assert_not_called()
        self.assertIn('UPLOAD_COMPLETE=False', output)
        self.assertIn('STRUCTURE_TRAINING_RESULT=no_eligible_improvement', output)
        self.assertTrue(list((self.root/'download').glob('*.zip')))
        plan = json.loads(next((self.root/'exp').glob('w2v_structure_train_*/structure_plan.json')).read_text())
        self.assertEqual(plan['review_action'], 'explicit --mode train --reviewed-audit')
        self.assertEqual(plan['audit_manifest'], self.manifest)
        self.assertEqual(plan['audit_summary']['recommendation'], 'ready_for_review')


if __name__ == '__main__':
    unittest.main()
