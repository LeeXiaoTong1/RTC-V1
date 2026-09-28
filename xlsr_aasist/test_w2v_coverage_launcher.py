"""Protect initialization, fixed recipe, exports and failure status without a GPU."""
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import start_w2v_coverage as coverage
from start_w2v_en import english_config
from w2v_rebuild.core import sha256


class CoverageLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'previous run;$(literal)'
        (self.source/'stage3').mkdir(parents=True)
        self.baseline = self.root/'original'/'stage3'/'best_model.pt'
        self.baseline.parent.mkdir(parents=True)
        self.baseline.write_bytes(b'original weights')
        self.cache = self.root/'existing cache'
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
        self.write_config()

    def write_config(self):
        (self.source/'stage3'/'config.json').write_text(json.dumps(self.config))

    def test_only_coverage_recipe_changes_and_arguments_remain_literal(self):
        original = english_config(self.config, self.cache)
        new = coverage.coverage_config(self.config, self.cache)
        self.assertEqual({k: v for k, v in new.items() if k not in original},
                         {'coverage_training': True, 'coverage_prefix_probability': .5})
        for key in original:
            self.assertEqual(new[key], original[key])
        command = coverage.coverage_command(new, self.root/'new run'/'stage3', self.baseline)
        self.assertIn('--coverage_training', command)
        self.assertEqual(command[command.index('--epochs')+1], '1')
        self.assertEqual(command[command.index('--finetune_from')+1], str(self.baseline))
        self.assertNotIn('--resume', command)
        self.assertNotIn('--preflight', command)
        self.assertNotIn('--check_data', command)

    def invoke(self, run=False, training=None, upload=False):
        args = ['--from-run', str(self.source), '--download-dir', str(self.root/'download')]
        if run:
            args.append('--run')
        if upload:
            args.append('--upload-temp')
        with patch.object(coverage, 'ROOT', self.root), \
             patch.object(coverage, 'recovery_lock', return_value=nullcontext()), \
             patch.object(coverage, 'run_training', side_effect=training) as child, \
             redirect_stdout(io.StringIO()) as output:
            result = coverage.main(args)
        return result, child, output.getvalue()

    def test_preview_does_not_train_write_outputs_or_modify_baseline(self):
        before = sha256(self.baseline)
        result, child, output = self.invoke()
        self.assertEqual(result, 0)
        child.assert_not_called()
        self.assertFalse((self.root/'exp').exists())
        self.assertIn('Preview only', output)
        self.assertEqual(sha256(self.baseline), before)

    def test_hash_mismatch_rejected_before_training(self):
        self.baseline.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.invoke(True)
        self.assertFalse((self.root/'exp').exists())

    def test_failure_stays_failure_and_partial_diagnostics_are_downloadable(self):
        def failed(command, env, log_path):
            Path(log_path).write_text('failure before first epoch')
            self.assertEqual(env['RTC_B_NOISE_PROB'], '.5')
            return 7
        result, child, output = self.invoke(True, failed)
        self.assertEqual(result, 7)
        self.assertEqual(child.call_count, 1)
        self.assertIn('COVERAGE_TRAINING_RESULT=failed', output)
        self.assertIn('ORIGINAL_BEST_PRESERVED=True', output)
        archive = next((self.root/'download').glob('*.zip'))
        with zipfile.ZipFile(archive) as package:
            self.assertIn('coverage_execution.log', package.namelist())
            self.assertIn('coverage_plan.json', package.namelist())
            summary = json.loads(package.read('summary.json'))
            self.assertEqual(summary['status'], 'failed')
            self.assertIn(b'Coverage-repair', package.read('report.md'))
            self.assertFalse(any(name.endswith(('.pt', '.wav', '.flac')) for name in package.namelist()))

    def test_success_and_upload_failure_preserve_local_report_and_baseline(self):
        def success(command, env, log_path):
            stage = Path(command[command.index('--out')+1])
            stage.mkdir()
            Path(log_path).write_text('one epoch complete')
            (stage/'config.json').write_text(json.dumps(coverage.coverage_config(self.config, self.cache)))
            (stage/'completed.json').write_text(json.dumps({'status': 'no_eligible_improvement',
                    'best_model': str(stage/'best_model.pt'), 'candidate_model': None}))
            (stage/'best_model.pt').write_bytes(b'never upload')
            return 0
        before = sha256(self.baseline)
        with patch('audit_w2v_train.upload_report', side_effect=OSError('network unavailable')):
            result, child, output = self.invoke(True, success, True)
        self.assertEqual(result, 0)
        self.assertIn('UPLOAD_COMPLETE=False', output)
        self.assertIn('COVERAGE_TRAINING_RESULT=no_eligible_improvement', output)
        self.assertTrue(list((self.root/'download').glob('*.zip')))
        self.assertEqual(sha256(self.baseline), before)


if __name__ == '__main__':
    unittest.main()
