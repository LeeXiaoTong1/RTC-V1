"""Offline deployment checks: persistent metrics and a read-only live viewer."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import v37_console


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ConsoleTests(unittest.TestCase):
    def fixture(self, folder, status='running'):
        log = Path(folder)/'v37.log'
        log.write_text('baseline Clean=98 Noisy=91 Weighted=93.1\n'
                       'head_only Clean=98 Noisy=92 Weighted=93.8\n'
                       'language_debias Clean=98 Noisy=93 Weighted=94.5\n', encoding='utf-8')
        state = Path(str(log)+'.progress.json')
        state.write_text(json.dumps(dict(label='Full Train refit', current=2, total=10,
                                        status=status, owner_pid=123)), encoding='utf-8')
        return log, state

    def test_once_preserves_every_result_before_progress_without_mutating_files(self):
        with tempfile.TemporaryDirectory() as folder:
            log, state = self.fixture(folder)
            before = {p: p.read_bytes() for p in (log, state)}
            output = io.StringIO()
            with patch.object(v37_console, 'alive', return_value=True), \
                    patch.object(v37_console.time, 'sleep', side_effect=AssertionError('read-only once')):
                v37_console.watch(log, once=True, stream=output)
            self.assertEqual(before, {p: p.read_bytes() for p in (log, state)})
            text = output.getvalue()
            for metric in ('Weighted=93.1', 'Weighted=93.8', 'Weighted=94.5'):
                self.assertEqual(text.count(metric), 1)
                self.assertLess(text.index(metric), text.index('2/10'))
            self.assertNotIn('\x1b[2J', text)

    def test_completed_large_log_is_drained_through_last_submission_line(self):
        with tempfile.TemporaryDirectory() as folder:
            log, _ = self.fixture(folder, status='complete')
            with log.open('a', encoding='utf-8') as handle:
                handle.write(('retained diagnostic row\n'*100000)+'SUBMISSION_BASELINE_FALLBACK=True\n')
            output = io.StringIO()
            v37_console.watch(log, once=True, stream=output)
            self.assertEqual(output.getvalue().count('retained diagnostic row'), 100000)
            self.assertEqual(output.getvalue().count('SUBMISSION_BASELINE_FALLBACK=True'), 1)

    def test_interrupt_closes_only_viewer_without_sending_any_process_signal(self):
        with tempfile.TemporaryDirectory() as folder:
            log, _ = self.fixture(folder)
            output = Terminal()
            with patch.object(v37_console, 'alive', return_value=True), \
                    patch.object(v37_console.time, 'sleep', side_effect=KeyboardInterrupt), \
                    patch('os.kill', side_effect=AssertionError('viewer must never signal')):
                v37_console.watch(log, stream=output)
            text = output.getvalue()
            self.assertIn('Weighted=94.5', text)
            self.assertIn('no signal was sent', text)
            self.assertNotIn('\x1b[2J', text)

    def test_dead_owner_is_reported_as_stopped(self):
        with tempfile.TemporaryDirectory() as folder:
            log, _ = self.fixture(folder)
            output = io.StringIO()
            with patch.object(v37_console, 'alive', return_value=False):
                v37_console.watch(log, once=True, stream=output)
            self.assertIn('Background job stopped', output.getvalue())


@unittest.skipUnless(shutil.which('bash'), 'Bash is needed for wrapper invocation checks')
class CleanupTests(unittest.TestCase):
    def invoke(self, args):
        script = Path(__file__).resolve().parent.parent/'cleanup_w2v_v37.sh'
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/'calls.txt'
            environment = dict(os.environ, V37_TEST_CALLS=log.as_posix())
            # A Bash function shadows Python for the entire child script. No
            # training process or real cleanup implementation is ever launched.
            command = ('python() { printf "%s\\n" "$*" >> "$V37_TEST_CALLS"; }; '
                       'export -f python; bash "$@"')
            result = subprocess.run([shutil.which('bash'), '-c', command, 'cleanup-test',
                                     script.as_posix(), *args], env=environment,
                                    capture_output=True, text=True, timeout=15)
            calls = log.read_text().splitlines() if log.exists() else []
            return result.returncode, calls

    def test_cleanup_is_preview_unless_apply_is_explicit(self):
        status, calls = self.invoke([])
        self.assertEqual(status, 0)
        self.assertEqual(calls, ['-m w2v_v36.stop', '-m w2v_v35.maintenance'])
        status, calls = self.invoke(['--apply'])
        self.assertEqual(status, 0)
        self.assertEqual(calls, ['-m w2v_v36.stop --apply', '-m w2v_v35.maintenance --apply'])

    def test_bad_cleanup_arguments_never_invoke_stop_or_maintenance(self):
        for args in (['--force'], ['--apply', '--apply'], ['--apply', '/tmp/other']):
            with self.subTest(args=args):
                status, calls = self.invoke(args)
                self.assertEqual(status, 2)
                self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
