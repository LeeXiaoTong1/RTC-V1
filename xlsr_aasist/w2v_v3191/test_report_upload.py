"""Report-host outages preserve local artifacts and cannot stop later arms."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from .common import RECIPE, atomic_json, read_json, digest
from .reports import report, automatic_report
from .report_upload import ReportUploadError, upload_report


class ReportUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root/'run'
        self.run.mkdir()
        atomic_json(self.run/'config.json', dict(version='3.19.1', recipe=RECIPE))
        atomic_json(self.run/'completed.json', dict(status='complete', completed_epochs=2))
        (self.run/'last.pt').write_bytes(b'preserved checkpoint')
        self.output = self.root/'reports'

    def tearDown(self):
        self.temp.cleanup()

    def test_curl_22_is_nonfatal_only_for_automatic_delivery(self):
        before = digest(self.run/'last.pt')
        with patch('w2v_v3191.reports.upload_archive',
                   side_effect=subprocess.CalledProcessError(22, ['curl'])), redirect_stdout(io.StringIO()):
            self.assertIsNone(automatic_report(self.run, self.output, True))
            with self.assertRaises(ReportUploadError):
                report(self.run, self.output, True)
        status = read_json(self.run/'report_status.json')
        self.assertEqual(status['status'], 'failed')
        self.assertFalse(status['uploaded'])
        self.assertIsNone(status['url'])
        self.assertEqual(digest(status['archive']), status['zip_sha256'])
        self.assertEqual(digest(self.run/'last.pt'), before)
        self.assertEqual(read_json(self.run/'completed.json')['status'], 'complete')

    def test_local_packaging_failure_still_raises(self):
        with patch('w2v_v3191.reports.report_files', side_effect=OSError('disk error')):
            with self.assertRaisesRegex(RuntimeError, 'disk error'):
                automatic_report(self.run, self.output, True)

    def test_archive_mutation_during_upload_is_not_swallowed(self):
        def corrupt(archive):
            Path(archive).write_bytes(b'corrupted')
            return 'https://temp.sh/example/report.zip'
        with patch('w2v_v3191.reports.upload_archive', side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, 'changed during upload') as caught:
                automatic_report(self.run, self.output, True)
        self.assertNotIsInstance(caught.exception, ReportUploadError)

    def test_successful_retry_uses_existing_archive(self):
        with patch('w2v_v3191.reports.upload_archive', side_effect=OSError('offline')):
            automatic_report(self.run, self.output, True)
        before = read_json(self.run/'report_status.json')['zip_sha256']
        with patch('w2v_v3191.reports.upload_archive', return_value='https://temp.sh/example/report.zip') as upload:
            report(self.run, self.output, True)
            report(self.run, self.output, True)
        after = read_json(self.run/'report_status.json')
        self.assertEqual(after['zip_sha256'], before)
        self.assertEqual(after['status'], 'complete')
        self.assertEqual(upload.call_count, 1)

    def test_bad_remote_url_is_delivery_error(self):
        with self.assertRaises(ReportUploadError):
            upload_report(Path('report.zip'), lambda _: 'https://temp.sh/upload')


if __name__ == '__main__':
    unittest.main()
