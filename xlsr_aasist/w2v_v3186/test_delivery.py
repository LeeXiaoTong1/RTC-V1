"""Standalone exports deliver both artifacts and retry without GPU inference."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .common import read_json,digest
from .test_reports import fixture,submission_fixture
from .delivery import deliver_export,main


class ManualDeliveryTests(unittest.TestCase):
    def test_manual_export_uploads_both_and_retry_reuses_outputs(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);run,destination=fixture(root);out=submission_fixture(root,run)
            # A standalone export need not have a training-time submission plan.
            (run/'submission_request.json').unlink()
            checkpoint=digest(run/'last.pt')
            with patch('w2v_v3186.delivery.upload_archive',return_value='https://temp.sh/s/submission.zip') as submit, \
                 patch('w2v_v3186.reports.upload_archive',return_value='https://temp.sh/r/report.zip') as report, \
                 patch('torch.load',side_effect=AssertionError('No checkpoint loading')):
                links=deliver_export(run,out)
                with patch('sys.argv',['upload','--run',str(run)]):main()
                self.assertEqual(submit.call_count,1);self.assertEqual(report.call_count,1)
            self.assertEqual(links['submission_url'],'https://temp.sh/s/submission.zip')
            self.assertEqual(links['report_url'],'https://temp.sh/r/report.zip')
            self.assertEqual(digest(run/'last.pt'),checkpoint)

    def test_submission_failure_still_generates_report_and_is_retryable(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);run,destination=fixture(root);out=submission_fixture(root,run)
            with patch('w2v_v3186.delivery.upload_archive',side_effect=RuntimeError('network')), \
                 patch('w2v_v3186.reports.upload_archive',return_value='https://temp.sh/r/report.zip'):
                with self.assertRaisesRegex(RuntimeError,'Retry without inference'):deliver_export(run,out)
            self.assertEqual(read_json(run/'artifact_links.json')['report_url'],'https://temp.sh/r/report.zip')
            self.assertTrue((out/'submission.zip').exists())

    def test_different_run_refused_before_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,destination=fixture(root);out=submission_fixture(root,run)
            from .common import atomic_json
            meta=read_json(out/'submission_meta.json');meta['run']=str(root/'other');atomic_json(out/'submission_meta.json',meta)
            with patch('w2v_v3186.delivery.upload_archive') as upload:
                with self.assertRaisesRegex(ValueError,'does not match'):deliver_export(run,out)
                upload.assert_not_called()


if __name__=='__main__':unittest.main()
