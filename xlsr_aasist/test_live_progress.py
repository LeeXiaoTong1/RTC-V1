import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import PurePosixPath

import live_progress as live
from w2v_aasist.progress import progress, training_bar


class LiveProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'progress.json'
        self.env = patch.dict(os.environ, {'AASIST_PROGRESS_FILE': str(self.path),
                                          'AASIST_PROGRESS_OWNER': str(os.getpid())})
        self.env.start()
        self.addCleanup(self.env.stop)
        live._last_write = 0.

    def test_completed_steps_and_current_loss_reach_separate_state(self):
        output = io.StringIO()
        with patch('sys.stderr.isatty', return_value=False), patch('sys.stdout', output):
            bar = training_bar(range(3), 3, 'Train epoch 1/2')
            for step in bar:
                # This step is not completed yet and must not be displayed as done.
                self.assertLessEqual(live.read_state(self.path)['current'], step)
                bar.set_postfix(loss=.03/(step+1))
        value = live.read_state(self.path)
        self.assertEqual((value['current'], value['total'], value['loss']), (3, 3, .01))
        self.assertEqual(output.getvalue(), '')
        with patch('sys.stderr.isatty', return_value=False), patch('sys.stdout', output):
            list(progress(range(2), total=2, label='epoch_1 Dev seen'))
        value = live.read_state(self.path)
        self.assertEqual(value['label'], 'epoch_1 Dev seen')
        self.assertIsNone(value['loss'])
        self.assertNotIn('\r', output.getvalue())
        self.assertFalse(list(Path(self.tmp.name).glob('*.tmp')))

    def test_failed_step_is_not_counted(self):
        with patch('sys.stderr.isatty', return_value=False):
            iterator = iter(training_bar(range(2), 2, 'Train'))
            next(iterator)
            iterator.close()
        self.assertEqual(live.read_state(self.path)['current'], 0)

    def test_throttled_state_and_writer_failure_cannot_abort_training(self):
        with patch('live_progress.time.monotonic', return_value=100.):
            live.publish('A', force=True)
            live.publish('B')
            self.assertEqual(live.read_state(self.path)['label'], 'A')
        with patch('live_progress.os.replace', side_effect=OSError('disk full')), patch('sys.stderr', io.StringIO()):
            live.publish('C', force=True)
        self.assertEqual(live.read_state(self.path)['label'], 'A')

    def test_single_line_respects_terminal_width_and_shows_staleness(self):
        value = dict(label='Train epoch 1/2\n', current=50, total=100, loss=.2,
                     elapsed=500., updated_at=1000.)
        line = live.render(value, 160, now=1040.)
        self.assertIn('50.00%', line)
        self.assertIn('loss=0.20000', line)
        self.assertIn('no update 40s', line)
        self.assertNotIn('\n', line)
        self.assertLess(len(live.render(value, 60)), 60)
        normal = live.render(dict(value, current=1000, total=3158), 80, now=1000.)
        self.assertLess(len(normal), 80)
        self.assertIn('loss=', normal)
        self.assertIn('ETA=', normal)
        self.assertNotIn('100.00%', live.render(dict(label='Saving', updated_at=1000.), 120, now=1000.))

    def test_viewer_interrupt_is_read_only_and_sends_no_signal(self):
        live.publish('Train', total=10, force=True)
        before = self.path.read_bytes()
        output = io.StringIO()
        output.isatty = lambda: True
        with patch('sys.stdout', output), patch('live_progress.time.sleep', side_effect=KeyboardInterrupt), \
                patch('live_progress.os.kill') as kill, patch('live_progress.subprocess.call') as call:
            live.watch(self.path)
        kill.assert_not_called()
        call.assert_not_called()
        self.assertEqual(before, self.path.read_bytes())
        self.assertIn('\r', output.getvalue())

    def test_runner_forwards_flags_and_reports_real_exit_status(self):
        with patch('live_progress.subprocess.call', return_value=7) as child:
            self.assertEqual(live.run_job('full', ['--', '--prepare-only', '--workers', '2']), 7)
        self.assertEqual(child.call_args.args[0][-4:], ['w2v_aasist.full_workflow', '--prepare-only', '--workers', '2'])
        self.assertEqual(live.read_state(self.path)['status'], 'failed')
        with patch('live_progress.subprocess.call', return_value=0):
            self.assertEqual(live.run_job('train', []), 0)
        self.assertEqual(live.read_state(self.path)['status'], 'complete')

    def test_legacy_log_uses_reported_counts_without_extrapolation(self):
        log = Path(self.tmp.name) / 'train.log'
        log.write_text('STEP 100/3158 CE=0.123 grad=1 seconds/step=2.0 ETA_min=101.9\n')
        value = live.log_state(log)
        self.assertEqual((value['current'], value['total']), (100, 3158))
        self.assertTrue(value['legacy'])
        log.write_text('Dev seen: 200/400 elapsed=3.0m ETA=3.0m\n')
        self.assertEqual(live.log_state(log)['current'], 200)

    def test_cache_guard_ignores_only_own_supervisor_and_keeps_other_job_guard(self):
        from w2v_aasist import full_workflow as workflow

        class Entry:
            def __init__(self, pid, command):
                self.name, self.command = str(pid), command
            def stat(self):
                return SimpleNamespace(st_uid=7)
            def __truediv__(self, name):
                return SimpleNamespace(read_bytes=lambda: self.command)

        entries = [Entry(101, b'python\0-u\0live_progress.py\0--run\0full\0--\0--source-config\0exp/w2v_old/config.json\0')]
        proc = SimpleNamespace(is_dir=lambda: True, glob=lambda _: entries)
        with patch.dict(os.environ, {'AASIST_PROGRESS_OWNER': '101'}), \
                patch.object(workflow, 'Path', side_effect=lambda p: proc if p == '/proc' else PurePosixPath(p)), \
                patch.object(workflow.os, 'name', 'posix'), \
                patch.object(workflow.os, 'getuid', return_value=7, create=True), \
                patch.object(workflow.os, 'getpid', return_value=303), \
                patch.object(workflow.os, 'getppid', return_value=101):
            workflow.ensure_idle()
            entries.append(Entry(202, b'python\0-m\0w2v_aasist.train\0'))
            with self.assertRaisesRegex(RuntimeError, '202'):
                workflow.ensure_idle()


if __name__ == '__main__':
    unittest.main()
