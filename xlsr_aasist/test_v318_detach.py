import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import v318_detach


class Tests(unittest.TestCase):
    @unittest.skipUnless(os.name=='posix','Linux/POSIX session integration')
    def test_viewer_group_interrupt_does_not_reach_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'v318_supervise.sh').write_text('#!/bin/bash\nshift\nexec "$@"\n')
            (root/'child.py').write_text('''import os, signal, time
from pathlib import Path
signal.signal(signal.SIGINT,signal.SIG_DFL)
Path('ready').write_text(str(os.getsid(0)))
deadline=time.monotonic()+10
while not Path('release').exists() and time.monotonic()<deadline:time.sleep(.02)
assert Path('release').exists()
Path('done').write_text('ok')
''')
            harness='''import os, signal, sys, time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from v318_detach import launch
signal.signal(signal.SIGINT,signal.SIG_IGN)
root=Path(sys.argv[2])
pid=launch(root/'log',[sys.executable,str(root/'child.py')],root)
deadline=time.monotonic()+10
while not (root/'ready').exists() and time.monotonic()<deadline:time.sleep(.02)
assert int((root/'ready').read_text())==pid
assert pid!=os.getsid(0)
os.killpg(os.getpgrp(),signal.SIGINT)
(root/'release').touch()
while not (root/'done').exists() and time.monotonic()<deadline:time.sleep(.02)
assert (root/'done').read_text()=='ok'
os.waitpid(pid,0)
'''
            result=subprocess.run([sys.executable,'-c',harness,str(Path(__file__).resolve().parent),str(root)],
                                  start_new_session=True,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_new_session_and_file_redirection_are_required(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            log = root/'training log.txt'
            calls = []
            def start(command, **kw):
                calls.append((command, kw))
                kw['stdout'].write(b'fixture output\n')
                return SimpleNamespace(pid=1234)
            with patch.object(v318_detach.os, 'name', 'posix'):
                # Path construction must remain native on Windows test hosts.
                with patch.object(v318_detach, 'Path', type(root)):
                    pid = v318_detach.launch(log, ['--data-run', 'path with spaces', '--workers', '2'], root, start)
            command, kw = calls[0]
            self.assertEqual(pid, 1234)
            self.assertEqual(command[-4:], ['--data-run', 'path with spaces', '--workers', '2'])
            self.assertTrue(kw['start_new_session'] and kw['close_fds'])
            self.assertEqual(kw['stdin'], v318_detach.subprocess.DEVNULL)
            self.assertEqual(kw['stderr'], v318_detach.subprocess.STDOUT)
            self.assertEqual(log.read_bytes(), b'fixture output\n')


if __name__ == '__main__': unittest.main()
