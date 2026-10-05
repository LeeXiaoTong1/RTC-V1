"""Native exit reporting, persistent terminal metrics, and shell handoff."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from .common import atomic_json, read_json
import v39_console
import v39_status


class Terminal(io.StringIO):
    def isatty(self):
        return True


class RuntimeTests(unittest.TestCase):
    def test_segfault_is_recorded_and_previous_metrics_remain_visible(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            log = Path(folder) / 'run.log'
            log.write_text('[Dev] baseline Weighted=96.3\n[Dev] candidate Weighted=96.4\n')
            value = v39_status.record(log, 139)
            self.assertEqual(value['signal'], 'SIGSEGV')
            output = io.StringIO()
            v39_console.watch(log, once=True, stream=output)
            for text in ('Weighted=96.3', 'Weighted=96.4', 'exit_code=139', 'SIGSEGV'):
                self.assertIn(text, output.getvalue())
            self.assertNotIn('\x1b[2J', output.getvalue())

    def test_interrupt_does_not_signal_training_and_missing_exit_record_is_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'run.log'
            log.write_text('prior result\n')
            atomic_json(str(log) + '.progress.json', dict(label='fitting', owner_pid=123, status='running', updated_at=0))
            output = Terminal()
            with patch.object(v39_console, 'alive', return_value=True), \
                    patch.object(v39_console.time, 'sleep', side_effect=KeyboardInterrupt), \
                    patch('os.kill', side_effect=AssertionError('Viewer cannot signal')):
                v39_console.watch(log, stream=output)
            self.assertIn('background job was not signalled', output.getvalue())
            output = io.StringIO()
            with patch.object(v39_console, 'alive', return_value=False):
                v39_console.watch(log, once=True, stream=output)
            self.assertIn('without a final status', output.getvalue())

    def test_supervisor_records_workflow_exit_and_forwards_arguments(self):
        bash = shutil.which('bash')
        if not bash:
            self.skipTest('Bash unavailable')
        script = Path(__file__).resolve().parent.parent / 'v39_supervise.sh'
        with tempfile.TemporaryDirectory() as folder:
            call_log = Path(folder) / 'calls.txt'
            # Mock only the invoked executable; execute the real shell wrapper.
            code = ('python() { printf "%s\\n" "$*" >> "$V39_TEST_CALLS"; '
                    'if [[ "$1" == -X ]]; then return 139; fi; return 0; }; '
                    'export -f python; bash "$@"')
            result = subprocess.run([bash, '-c', code, 'supervisor-test', script.as_posix(),
                str(Path(folder) / 'log'), '--source-run', 'path with spaces', '--upload-temp'],
                env=dict(os.environ, V39_TEST_CALLS=call_log.as_posix()), capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 139, result.stderr)
            calls = call_log.read_text().splitlines()
            self.assertIn('-m w2v_v39.workflow --source-run path with spaces --upload-temp', calls[0])
            self.assertIn('--exit-code 139', calls[-1])


if __name__ == '__main__':
    unittest.main()
