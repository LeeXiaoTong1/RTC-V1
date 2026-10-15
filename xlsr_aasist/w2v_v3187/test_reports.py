"""Report contents, independent delivery, receipts and failure recovery without GPU."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
from .common import RECIPE,atomic_json,read_json,digest,fingerprint
from .reports import report,prepare_request,retry_outputs


def fixture(root):
    run=root/'run';run.mkdir();destination=root/'downloads';destination.mkdir()
    atomic_json(run/'config.json',dict(version='3.18.7',recipe=RECIPE,saved_manifests={}))
    atomic_json(run/'training_history.json',[])
    (run/'last.pt').write_bytes(b'retained checkpoint')
    (run/'audio.wav').write_bytes(b'excluded recording')
    (run/'weights.npz').write_bytes(b'excluded unrecognized tensor file')
    (run/'dev_scores_epoch_1_step_2.npz').write_bytes(b'prediction fixture')
    (run/'parent_baseline_scores.npz').write_bytes(b'actual zero-update predictions')
    (run/'parent_reference_scores.npz').write_bytes(b'cached parent predictions')
    (run/'diagnostics').mkdir();(run/'diagnostics'/'curves.html').write_text('<p>metrics</p>')
    atomic_json(run/'report_request.json',dict(destination=str(destination),upload=True))
    return run,destination


def submission_fixture(root,run):
    out=root/'submission';out.mkdir()
    with zipfile.ZipFile(out/'submission.zip','w') as z:z.writestr('scores.txt','item_1 0.25\n')
    meta=dict(version='3.18.7',recipe=RECIPE,score='P(fake=0)',kind='best',run=str(run),
        selected='epoch_1_step_2',checkpoint_sha256=digest(run/'last.pt'),protocol_sha256='protocol-fixture',
        count=1,id_order_sha256=fingerprint(['item_1']),calibration={'bias':0},zip_sha256=digest(out/'submission.zip'))
    atomic_json(out/'submission_meta.json',meta)
    atomic_json(run/'completed.json',dict(status='complete',best_tag=meta['selected'],checkpoint_sha256=meta['checkpoint_sha256']))
    atomic_json(run/'submission_request.json',dict(out=str(out),protocol_sha256=meta['protocol_sha256'],
        count=1,id_order_sha256=meta['id_order_sha256'],upload=True))
    return out


class ReportsTests(unittest.TestCase):
    def test_report_allowlist_distinct_links_and_idempotent_upload(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            run,destination=fixture(Path(directory))
            atomic_json(run/'submission_status.json',dict(status='complete',url='https://temp.sh/a/submission.zip'))
            before=(run/'last.pt').read_bytes()
            with patch('w2v_v3187.reports.upload_archive',return_value='https://temp.sh/b/report.zip') as upload:
                archive=report(run,destination,True)
                first_hash=digest(archive);report(run,destination,True)
                self.assertEqual(upload.call_count,1);self.assertEqual(digest(archive),first_hash)
                report(run,destination,True,force=True);self.assertEqual(upload.call_count,2)
            with zipfile.ZipFile(archive) as z:
                names=set(z.namelist());manifest=json.loads(z.read('REPORT_MANIFEST.json'))
                self.assertTrue({'config.json','diagnostics/curves.html','dev_scores_epoch_1_step_2.npz',
                    'parent_baseline_scores.npz','parent_reference_scores.npz'}<=names)
                self.assertFalse({'last.pt','audio.wav','weights.npz','report_status.json','download_links.txt'}&names)
                self.assertEqual({row['file'] for row in manifest['files']},names-{'REPORT_MANIFEST.json'})
            links=(run/'download_links.txt').read_text()
            self.assertIn('SUBMISSION_URL=https://temp.sh/a/submission.zip',links)
            self.assertIn('REPORT_URL=https://temp.sh/b/report.zip',links)
            self.assertEqual((run/'last.pt').read_bytes(),before)

    def test_report_request_persists_delivery_and_destination_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir()
            first=prepare_request(SimpleNamespace(download_dir=str(root/'custom'),upload_temp=True),run,[])
            resumed=prepare_request(SimpleNamespace(download_dir=str(root/'default'),upload_temp=False),run,['--resume',str(run)])
            self.assertEqual(first,resumed)
            with self.assertRaisesRegex(ValueError,'outside'):
                prepare_request(SimpleNamespace(download_dir=str(run/'bad'),upload_temp=False),run,['--download-dir=x'])

    def test_network_failure_keeps_report_and_training_for_retry(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            run,destination=fixture(Path(directory));checkpoint=digest(run/'last.pt')
            with patch('w2v_v3187.reports.upload_archive',side_effect=RuntimeError('network fixture')):
                with self.assertRaisesRegex(RuntimeError,'Retry without training or inference'):report(run,destination,True)
            status=read_json(run/'report_status.json');archive=Path(status['archive'])
            self.assertTrue(archive.is_file());self.assertEqual(status['status'],'failed')
            original=digest(archive)
            with patch('w2v_v3187.reports.upload_archive',return_value='https://temp.sh/b/report.zip'):
                report(run,destination,True)
            self.assertEqual(digest(archive),original);self.assertEqual(digest(run/'last.pt'),checkpoint)
            self.assertEqual(read_json(run/'report_status.json')['status'],'complete')

    def test_submission_failure_still_uploads_report_and_retry_reuses_scores(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);run,destination=fixture(root);out=submission_fixture(root,run)
            scores=digest(out/'submission.zip');checkpoint=digest(run/'last.pt')
            with patch('w2v_v3187.delivery.upload_archive',side_effect=RuntimeError('submission network fixture')), \
                 patch('w2v_v3187.reports.upload_archive',return_value='https://temp.sh/r/report.zip') as report_upload, \
                 patch('torch.load',side_effect=AssertionError('No checkpoint loading for retries')):
                with self.assertRaisesRegex(RuntimeError,'submission:'):retry_outputs(run)
                self.assertEqual(report_upload.call_count,1)
                self.assertEqual(read_json(run/'artifact_links.json')['report_url'],'https://temp.sh/r/report.zip')
            with patch('w2v_v3187.delivery.upload_archive',return_value='https://temp.sh/s/submission.zip') as submit_upload, \
                 patch('w2v_v3187.reports.upload_archive',return_value='https://temp.sh/r2/report.zip') as report_upload, \
                 patch('torch.load',side_effect=AssertionError('No checkpoint loading for retries')):
                links=retry_outputs(run);again=retry_outputs(run)
                self.assertEqual(links,again);self.assertEqual(submit_upload.call_count,1)
                self.assertEqual(report_upload.call_count,1)
            self.assertEqual(links['submission_url'],'https://temp.sh/s/submission.zip')
            self.assertEqual(links['report_url'],'https://temp.sh/r2/report.zip')
            self.assertEqual(digest(out/'submission.zip'),scores);self.assertEqual(digest(run/'last.pt'),checkpoint)

    def test_report_failure_keeps_successful_submission_link(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);run,destination=fixture(root);submission_fixture(root,run)
            with patch('w2v_v3187.delivery.upload_archive',return_value='https://temp.sh/s/submission.zip'), \
                 patch('w2v_v3187.reports.upload_archive',side_effect=RuntimeError('report network fixture')):
                with self.assertRaisesRegex(RuntimeError,'report:'):retry_outputs(run)
            links=read_json(run/'artifact_links.json')
            self.assertEqual(links['submission_url'],'https://temp.sh/s/submission.zip')
            self.assertIsNone(links['report_url']);self.assertEqual(links['report_status'],'failed')

if __name__=='__main__':unittest.main()
