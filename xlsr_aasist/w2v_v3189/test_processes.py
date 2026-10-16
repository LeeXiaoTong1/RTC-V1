"""The automatic dashboard must not block its own worker or best export."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .processes import ensure_idle, is_read_only_viewer


class ProcessTests(unittest.TestCase):
    def test_exact_read_only_module_with_python_options_is_allowed(self):
        for argv in (['python','-u','-m','w2v_v3189.monitor'],
                     ['/env/bin/python3.11','-X','faulthandler','-u','-m','w2v_v3189.monitor','--run','exp/w2v_fixture'],
                     ['python','-Wignore','-B','-m','w2v_v3189.monitor','--once']):
            with self.subTest(argv=argv):
                self.assertTrue(is_read_only_viewer(argv))

    def test_training_and_argument_mentions_are_never_exempted(self):
        for argv in (['python','-m','w2v_v3189.workflow','--run','w2v_v3189.monitor'],
                     ['python','-m','w2v_v3189.evaluate','--note','-m','w2v_v3189.monitor'],
                     ['python','-c','run_training()','-m','w2v_v3189.monitor'],
                     ['python','main_train.py','-m','w2v_v3189.monitor'],
                     ['python','-m','w2v_v3189.monitor_extra'],
                     ['python','-X','-m','w2v_v3189.monitor'],
                     ['python','-m'], []):
            with self.subTest(argv=argv):
                self.assertFalse(is_read_only_viewer(argv))

    def table(self, root, rows):
        for pid, argv in rows.items():
            entry=root/str(pid);entry.mkdir()
            (entry/'cmdline').write_bytes(('\0'.join(argv)+'\0').encode())
        stack=ExitStack()
        stack.enter_context(patch('os.getuid',return_value=root.stat().st_uid,create=True))
        stack.enter_context(patch('os.getpid',return_value=9999))
        stack.enter_context(patch('os.getppid',return_value=9998))
        return stack

    def test_auto_viewer_can_exist_during_both_worker_startup_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.table(root,{930406:['python','-u','-m','w2v_v3189.monitor'],
                                  9999:['python','-X','faulthandler','-u','-m','w2v_v3189.workflow']}):
                ensure_idle(proc_root=root)
                ensure_idle(proc_root=root)

    def test_viewer_does_not_hide_another_training_evaluation_or_cache_job(self):
        for module in ('w2v_v3189.workflow','w2v_v3189.evaluate','w2v_v3186.workflow','rtc_noisy.prepare'):
            with self.subTest(module=module),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                with self.table(root,{930406:['python','-u','-m','w2v_v3189.monitor'],
                                      930407:['python','-m',module]}):
                    with self.assertRaises(RuntimeError) as error:
                        ensure_idle(proc_root=root)
                    self.assertIn(module,str(error.exception))
                    self.assertNotIn('930406',str(error.exception))

    def test_other_user_and_disappeared_process_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.table(root,{101:['python','-m','w2v_v3189.workflow']}):
                with patch('os.getuid',return_value=root.stat().st_uid+1,create=True):
                    ensure_idle(proc_root=root)
                (root/'101'/'cmdline').unlink()
                ensure_idle(proc_root=root)

    def test_launcher_worker_and_export_use_the_versioned_guard(self):
        root=Path(__file__).resolve().parents[1]
        self.assertIn('python -m w2v_v3189.processes',(root/'run_w2v_v3189.sh').read_text())
        for name in ('workflow.py','evaluate.py'):
            content=(root/'w2v_v3189'/name).read_text()
            self.assertIn('from .processes import ensure_idle',content)
            self.assertNotIn('from w2v_aasist.full_workflow import ensure_idle',content)


if __name__=='__main__':
    unittest.main()
