"""Process identity and signal boundaries; never signal real processes."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock,patch
from . import stop as module


def fixture(root):
    (root/'exp').mkdir();log=root/'exp'/'v3186_fixture.log';log.write_text('')
    (root/'exp'/'.latest_v3186_pid').write_text('101')
    (root/'exp'/'.latest_v3186_log').write_text(str(log))
    leader=module.Process(101,1,101,101,500,1000,'S',('bash',str(root/'v3186_supervise.sh'),str(log)),str(root))
    child=module.Process(102,101,101,101,501,1000,'R',('python','-m','w2v_v3186.workflow'),str(root))
    worker=replace(child,pid=103,ppid=102,start=502)
    return log,{p.pid:p for p in (leader,child,worker)}


class StopTests(unittest.TestCase):
    def test_only_verified_detached_session_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();log,records=fixture(root)
            leader,members=module.verify_session(records,101,root,log,1000,999)
            self.assertEqual({p.pid for p in members},{101,102,103})
            for change in (dict(uid=2000),dict(session=99),dict(group=999),dict(cwd=str(root.parent)),
                           dict(command=('bash','another_script.sh',str(log)))):
                with self.subTest(change=change),self.assertRaises(ValueError):
                    module.verify_session({**records,101:replace(leader,**change)},101,root,log,1000,999)

    def test_rejects_unrelated_session_member_and_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();log,records=fixture(root)
            unrelated=replace(records[103],pid=104,ppid=1)
            with self.assertRaisesRegex(ValueError,'unverified'):
                module.verify_session({**records,104:unrelated},101,root,log,1000,999)
            outside=replace(unrelated,session=500)
            with self.assertRaisesRegex(ValueError,'unrelated'):
                module.verify_session({**records,104:outside},101,root,log,1000,999)

    def test_explicit_run_must_match_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();log,_=fixture(root);run=root/'exp'/'run';run.mkdir()
            (run/'config.json').write_text(json.dumps(dict(version='3.18.6',recipe='old_recipe')))
            log.write_text('V3186_RUN='+str(run)+'\n')
            self.assertEqual(module.verify_run(root,log,run),run)
            log.write_text('V3186_RUN='+str(root/'other')+'\n')
            with self.assertRaisesRegex(ValueError,'differs'):module.verify_run(root,log,run)

    def test_stop_signals_verified_group_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory).resolve();log,records=fixture(root)
            saved=root/'exp'/'last.pt';saved.write_bytes(b'committed epoch')
            kill=Mock();snapshot=Mock(side_effect=[records,records,{}])
            with patch.object(module,'LINUX',True),patch.object(module.os,'getuid',return_value=1000,create=True), \
                 patch.object(module.os,'getpgrp',return_value=999,create=True):
                self.assertTrue(module.stop(root,snapshot=snapshot,signal_group=kill))
            kill.assert_called_once_with(101,signal.SIGTERM)
            self.assertEqual(saved.read_bytes(),b'committed epoch')

    def test_dry_run_sends_no_signal(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory).resolve();_,records=fixture(root);kill=Mock()
            with patch.object(module,'LINUX',True),patch.object(module.os,'getuid',return_value=1000,create=True), \
                 patch.object(module.os,'getpgrp',return_value=999,create=True):
                self.assertTrue(module.stop(root,dry_run=True,snapshot=lambda:records,signal_group=kill))
            kill.assert_not_called()

    def test_pid_reuse_and_timeout_never_force_kill(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory).resolve();_,records=fixture(root);kill=Mock()
            with patch.object(module,'LINUX',True),patch.object(module.os,'getuid',return_value=1000,create=True), \
                 patch.object(module.os,'getpgrp',return_value=999,create=True):
                with self.assertRaisesRegex(RuntimeError,'reused'):
                    module.stop(root,snapshot=Mock(side_effect=[records,{**records,101:replace(records[101],start=999)}]),signal_group=kill)
                kill.assert_not_called()
                self.assertFalse(module.stop(root,timeout=1,snapshot=lambda:records,signal_group=kill,
                                             monotonic=Mock(side_effect=[0,2]),sleeper=Mock()))
                kill.assert_called_once_with(101,signal.SIGTERM)


if __name__=='__main__':unittest.main()
