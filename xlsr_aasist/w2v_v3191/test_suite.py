"""Orchestration tests with a fake executor: no GPU or actual training."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from .suite import (ARMS, StopState, prepare_manifest, read, write, run_suite,
                    validate_manifest, request_stop, main, package_report)


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rtc_v3191_suite_test_',
                                                dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name).resolve()
        self.parent = self.root/'parent'
        self.parent.mkdir()
        (self.parent/'last.pt').write_bytes(b'parent checkpoint bytes')
        for name in ('config.json', 'train_rows.json', 'dev_rows.json', 'dev_partition.json'):
            write(self.parent/name, {'seed': 77})
        self.path = self.root/'suite'
        self.manifest = prepare_manifest(self.path, self.parent,
            ['--parent-run', str(self.parent), '--epochs', '2', '--seed', '77'], 2, 77)
        self.calls = []
        self.comparisons = []
        self.outcomes = {}

    def tearDown(self):
        self.temp.cleanup()

    def fake_executor(self, command, env, log, stopped):
        self.calls.append(command)
        record = Path(env['V3191_SUITE_RUN_RECORD'])
        arm = record.name.split('.')[0]
        if '--resume' in command:
            run = Path(command[command.index('--resume')+1])
        else:
            run = self.root/('run_'+arm)
            run.mkdir(exist_ok=False)
            write(run/'config.json', dict(version='3.19.1', robust_arm=arm, epochs=2,
                sampling_seed=77, parent_init=dict(run=str(self.parent),
                    checkpoint_sha256=self.manifest['immutable']['parent_files'][str(self.parent/'last.pt')])))
        write(record, {'run': str(run)})
        (run/'paused.json').unlink(missing_ok=True)
        outcome = self.outcomes.get(arm, 'complete')
        if outcome in ('request_stop', 'finish_and_stop'):
            write(self.path/'stop_requested.json', {'reason': 'test'})
            self.assertTrue(stopped())
            self.assertTrue((run/'stop_requested.json').exists())
        if outcome in ('complete', 'finish_and_stop'):
            write(run/'completed.json', dict(status='complete', completed_epochs=2))
        elif outcome in ('paused', 'request_stop'):
            write(run/'paused.json', {'status': 'paused'})
        elif outcome == 'wrong_arm':
            config = read(run/'config.json')
            config['robust_arm'] = 'wrong'
            write(run/'config.json', config)
        return 7 if outcome == 'fail' else 0

    def compare(self, paths, **kwargs):
        self.comparisons.append((paths, kwargs))
        return dict(comparisons=[], checked_runs=paths)

    def run_now(self):
        return run_suite(self.path, executor=self.fake_executor, comparer=self.compare)

    def test_three_arms_share_parent_seed_epochs_and_compare_once(self):
        result = self.run_now()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(self.calls), 3)
        self.assertEqual([c[c.index('--arm')+1] for c in self.calls], list(ARMS))
        for command in self.calls:
            self.assertEqual(command[command.index('--parent-run')+1], str(self.parent))
            self.assertEqual(command[command.index('--epochs')+1], '2')
            self.assertEqual(command[command.index('--seed')+1], '77')
            self.assertNotIn('--submit-best', command)
            self.assertNotIn('--upload-submission', command)
        self.assertEqual(len(self.comparisons), 1)
        self.assertTrue((self.path/'comparison.json').exists())

    def test_failed_arm_resumes_existing_run_and_never_restarts_from_parent(self):
        self.outcomes['expanded'] = 'fail'
        result = self.run_now()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(result['arms']['hard']['status'], 'pending')
        self.assertFalse(self.comparisons)
        self.outcomes.clear()
        result = self.run_now()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(self.calls), 4)
        resumed = self.calls[2]
        self.assertIn('--resume', resumed)
        self.assertEqual(resumed[resumed.index('--resume')+1], str(self.root/'run_expanded'))
        self.assertNotIn('--parent-run', resumed)

    def test_graceful_pause_never_advances_even_with_zero_child_exit(self):
        self.outcomes['baseline'] = 'request_stop'
        result = self.run_now()
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result['arms']['expanded']['status'], 'pending')
        self.assertFalse(self.comparisons)
        self.outcomes.clear()
        (self.path/'stop_requested.json').unlink()
        result = self.run_now()
        self.assertEqual(result['status'], 'complete')
        self.assertIn('--resume', self.calls[1])

    def test_stop_racing_final_epoch_does_not_start_next_arm(self):
        self.outcomes['baseline'] = 'finish_and_stop'
        result = self.run_now()
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(result['arms']['baseline']['status'], 'complete')
        self.assertEqual(len(self.calls), 1)
        self.outcomes.clear()
        (self.path/'stop_requested.json').unlink()
        result = self.run_now()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(self.calls), 3)  # completed baseline was skipped

    def test_run_record_recovers_crash_before_suite_manifest_update(self):
        # Model run was initialized, but suite.json was never told its path.
        command = ['python', '--arm', 'baseline']
        self.outcomes['baseline'] = 'fail'
        self.fake_executor(command, {'V3191_SUITE_RUN_RECORD': str(self.path/'baseline.run.json')},
                           self.path/'baseline.log', lambda: False)
        self.outcomes.clear()
        self.calls.clear()
        result = self.run_now()
        self.assertEqual(result['status'], 'complete')
        self.assertIn('--resume', self.calls[0])

    def test_parent_mutation_prevents_any_child_launch(self):
        (self.parent/'last.pt').write_bytes(b'different checkpoint')
        with self.assertRaisesRegex(ValueError, 'parent file changed'):
            self.run_now()
        self.assertFalse(self.calls)

    def test_wrong_child_arm_stops_suite(self):
        self.outcomes['baseline'] = 'wrong_arm'
        with self.assertRaisesRegex(ValueError, 'Child arm'):
            self.run_now()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(read(self.path/'suite.json')['status'], 'failed')

    def test_immutable_common_args_tampering_is_rejected(self):
        changed = copy.deepcopy(self.manifest)
        changed['immutable']['epochs'] = 3
        with self.assertRaisesRegex(ValueError, 'Immutable'):
            validate_manifest(changed)

    def test_explicit_resume_rejects_training_overrides(self):
        with self.assertRaisesRegex(ValueError, 'no training overrides'):
            main(['--resume-suite', str(self.path), '--epochs', '5'])

    def test_signal_stop_is_durable_and_stop_command_uses_no_training(self):
        state = StopState(self.path)
        state.signal()
        self.assertTrue(state.requested())
        self.assertTrue((self.path/'stop_requested.json').exists())
        request_stop(self.path)
        self.assertEqual(read(self.path/'suite.json')['status'], 'pending')
        self.assertFalse(self.calls)

    def test_suite_archive_excludes_audio_checkpoints_and_unlisted_files(self):
        result = self.run_now()
        for arm in ARMS:
            run = Path(result['arms'][arm]['run'])
            (run/'last.pt').write_bytes(b'private model')
            (run/'example.wav').write_bytes(b'private audio')
            (run/'unlisted.txt').write_text('private extra', encoding='utf-8')
            write(run/'robust_final.json', {'diagnostic': arm})
        delivery = package_report(self.path, result)
        with zipfile.ZipFile(delivery['archive']) as archive:
            names = set(archive.namelist())
            self.assertTrue({'suite.json', 'comparison.json', 'REPORT_MANIFEST.json'} <= names)
            self.assertTrue(all(arm+'/robust_final.json' in names for arm in ARMS))
            self.assertFalse(any(n.endswith(('.pt', '.wav', 'unlisted.txt')) for n in names))

    def test_opt_in_upload_is_reused_without_duplicate_network_call(self):
        result = self.run_now()
        result['immutable']['upload_reports'] = True
        calls = []
        def uploader(path):
            calls.append(path)
            return 'https://temp.sh/example/report.zip'
        first = package_report(self.path, result, uploader=uploader)
        second = package_report(self.path, result, uploader=uploader)
        self.assertEqual(first['url'], second['url'])
        self.assertEqual(len(calls), 1)
        self.assertTrue(first['uploaded'])

    def test_delivery_failure_retains_completed_arm_state(self):
        result = self.run_now()
        result['immutable']['upload_reports'] = True
        def uploader(_):
            raise RuntimeError('simulated network failure')
        with self.assertRaisesRegex(RuntimeError, 'completed arms are retained'):
            package_report(self.path, result, uploader=uploader)
        self.assertEqual(read(self.path/'report_status.json')['status'], 'failed')
        self.assertTrue((self.path/'report.zip').exists())
        self.assertTrue(all(s['status'] == 'complete' for s in read(self.path/'suite.json')['arms'].values()))


if __name__ == '__main__':
    unittest.main()
