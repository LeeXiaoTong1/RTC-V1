"""Pause exits without network delivery; final delivery errors remain visible."""
from contextlib import ExitStack,redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .common import RECIPE,atomic_json,read_json


class WorkflowDeliveryTests(unittest.TestCase):
    def fixture(self,root):
        run=root/'exp'/'run';run.mkdir(parents=True)
        atomic_json(run/'config.json',dict(version='3.18.9',recipe=RECIPE,saved_manifests={}))
        atomic_json(run/'train_rows.json',[]);atomic_json(run/'dev_rows.json',[])
        return run

    def patched(self,root,run,outcome,report_error=None):
        from . import workflow
        stack=ExitStack()
        stack.enter_context(patch.object(workflow,'ROOT',root))
        stack.enter_context(patch.object(workflow,'ensure_idle'))
        stack.enter_context(patch.object(workflow,'verify_inputs'))
        stack.enter_context(patch.object(workflow,'prepare_request',return_value={'out':'fixture'}))
        stack.enter_context(patch.object(workflow,'prepare_report_request',return_value=dict(destination=str(root/'reports'),upload=True)))
        train=stack.enter_context(patch.object(workflow,'run_experiment',side_effect=outcome if isinstance(outcome,Exception) else None,
                                             return_value=outcome if not isinstance(outcome,Exception) else None))
        finish=stack.enter_context(patch.object(workflow,'finish_request'))
        report=stack.enter_context(patch.object(workflow,'report',side_effect=report_error))
        stack.enter_context(patch('sys.argv',['workflow','--resume',str(run)]))
        return stack,workflow,train,finish,report

    def test_pause_and_preflight_do_not_export_or_upload(self):
        for status in ('paused','initialization_verified'):
            with self.subTest(status=status),tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
                root=Path(folder);run=self.fixture(root);(run/'failure.log').write_text('old failure')
                stack,workflow,train,finish,report=self.patched(root,run,dict(status=status))
                with stack:workflow.main()
                train.assert_called_once();finish.assert_not_called();report.assert_not_called()
                self.assertFalse((run/'failure.log').exists())

    def test_completed_run_delivers_both_artifacts(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            root=Path(folder);run=self.fixture(root)
            stack,workflow,train,finish,report=self.patched(root,run,dict(status='complete'))
            with stack:workflow.main()
            finish.assert_called_once();report.assert_called_once()

    def test_training_error_is_preserved_when_report_also_fails(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            root=Path(folder);run=self.fixture(root)
            stack,workflow,train,finish,report=self.patched(root,run,RuntimeError('training fixture'),RuntimeError('report fixture'))
            with stack,self.assertRaisesRegex(RuntimeError,'training fixture'):workflow.main()
            finish.assert_not_called();report.assert_called_once()
            self.assertIn('training fixture',(run/'failure.log').read_text())

    def test_report_error_after_success_is_not_silently_swallowed(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            root=Path(folder);run=self.fixture(root)
            stack,workflow,train,finish,report=self.patched(root,run,dict(status='complete'),RuntimeError('report fixture'))
            with stack,self.assertRaisesRegex(RuntimeError,'report fixture'):workflow.main()
            finish.assert_called_once();report.assert_called_once()


if __name__=='__main__':unittest.main()
