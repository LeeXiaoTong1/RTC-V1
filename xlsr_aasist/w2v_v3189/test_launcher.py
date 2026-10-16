"""Detached launcher binds only this version's supervisor and preserves argv."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from .launcher import launch


class LauncherTests(unittest.TestCase):
    def test_launch_uses_isolated_session_and_exact_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();(root/'exp').mkdir();log=root/'exp'/'v3189_test.log'
            popen=Mock(return_value=SimpleNamespace(pid=123))
            # POSIX check must not change pathlib's Windows path implementation.
            import w2v_v3189.launcher as module
            with patch.object(module,'os',SimpleNamespace(name='posix')):
                self.assertEqual(launch(log,['--resume','a path with spaces'],root,popen),123)
            args,kwargs=popen.call_args
            self.assertEqual(args[0],['bash',str(root/'v3189_supervise.sh'),str(log),'--resume','a path with spaces'])
            self.assertTrue(kwargs['start_new_session']);self.assertTrue(kwargs['close_fds'])
            self.assertEqual(kwargs['cwd'],root)


if __name__=='__main__':unittest.main()
