"""A graceful request cannot target an unrelated process or destroy checkpoints."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from . import stop as module


def fixture(root):
    (root/'exp').mkdir();log=root/'exp'/'v3187_fixture.log'
    run=root/'exp'/'run';run.mkdir()
    (run/'config.json').write_text(json.dumps({'version':'3.18.7'}))
    log.write_text('V3187_RUN='+str(run)+'\n')
    (root/'exp'/'.latest_v3187_pid').write_text('101')
    (root/'exp'/'.latest_v3187_log').write_text(str(log))
    leader=module.Process(101,1,101,101,500,1000,'S',('bash',str(root/'v3187_supervise.sh'),str(log)),str(root))
    child=module.Process(102,101,101,101,501,1000,'R',('python','-m','w2v_v3187.workflow'),str(root))
    return log,run,{p.pid:p for p in (leader,child)}


class StopTests(unittest.TestCase):
    def test_only_verified_session_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();log,run,records=fixture(root)
            leader,members=module.verify_session(records,101,root,log,1000,999)
            self.assertEqual(len(members),2)
            for change in (dict(uid=2000),dict(session=999),dict(group=999),dict(cwd=str(root.parent))):
                with self.subTest(change=change),self.assertRaises(ValueError):
                    module.verify_session({**records,101:replace(leader,**change)},101,root,log,1000,999)

    def test_request_waits_for_checkpoint_acknowledgement_without_signals(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            root=Path(folder).resolve();log,run,records=fixture(root)
            checkpoint=run/'last.pt';checkpoint.write_bytes(b'optimizer and model')
            calls=[]
            def snapshot():
                calls.append(1)
                if len(calls)>=3:
                    (run/'paused.json').write_text(json.dumps({'status':'paused'}))
                    return {}
                return records
            with patch.object(module,'LINUX',True),patch.object(module.os,'getuid',return_value=1000,create=True), \
                 patch.object(module.os,'getpgrp',return_value=999,create=True),patch.object(module.os,'killpg',create=True) as kill:
                self.assertTrue(module.stop(root,snapshot=snapshot))
                kill.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(),b'optimizer and model')
            self.assertEqual(json.loads((run/'stop_requested.json').read_text())['action'],'pause')

    def test_timeout_leaves_request_and_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            root=Path(folder).resolve();log,run,records=fixture(root)
            with patch.object(module,'LINUX',True),patch.object(module.os,'getuid',return_value=1000,create=True), \
                 patch.object(module.os,'getpgrp',return_value=999,create=True):
                self.assertTrue(module.stop(root,dry_run=True,snapshot=lambda:records))
                self.assertFalse((run/'stop_requested.json').exists())
                self.assertFalse(module.stop(root,snapshot=lambda:records,monotonic=Mock(side_effect=[0,2]),timeout=1))
                self.assertTrue((run/'stop_requested.json').is_file())

    def test_reused_pid_cannot_write_request(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            root=Path(folder).resolve();log,run,records=fixture(root)
            with patch.object(module,'LINUX',True),patch.object(module.os,'getuid',return_value=1000,create=True), \
                 patch.object(module.os,'getpgrp',return_value=999,create=True):
                with self.assertRaisesRegex(RuntimeError,'reused'):
                    module.stop(root,snapshot=Mock(side_effect=[records,{**records,101:replace(records[101],start=999)}]))
                self.assertFalse((run/'stop_requested.json').exists())


if __name__=='__main__':unittest.main()
