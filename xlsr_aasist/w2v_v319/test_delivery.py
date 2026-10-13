"""Best-selector provenance, retry without inference, and separate safe ZIP links."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from w2v_v39.common import atomic_json, read_json, digest
from w2v_v313.state import identity
from . import delivery, reports


class DeliveryTests(unittest.TestCase):
    def fixture(self, root, kind='best_weighted'):
        run=root/'run';run.mkdir();protocol=root/'progress.txt';protocol.write_text('a\nb\n')
        cfg=dict(version='3.19',device='cpu',workers=0)
        done=dict(version='3.19',status='complete',checkpoint_sha256='a'*64,
                  selections=dict(best_weighted='epoch_4_step_100',best_guarded='starting_parent'))
        atomic_json(run/'config.json',cfg);atomic_json(run/'completed.json',done)
        args=SimpleNamespace(submit_best=True,upload_submission=True,submission_checkpoint=kind,
                             submission_protocol=str(protocol),submission_audio_root=str(root),
                             submission_out=str(root/'submission'),download_dir=str(root/'downloads'),
                             upload_temp=True)
        with patch.object(delivery,'read_protocol',return_value=[dict(id='a'),dict(id='b')]):
            request=delivery.prepare_request(args,run,cfg)
        reports.prepare_request(args,run,[])
        return run,cfg,done,request

    def exporter(self, done, calls):
        def export(run,protocol,audio_root,out,**kwargs):
            calls.append(kwargs['checkpoint_kind'])
            out=Path(out);out.mkdir()
            with zipfile.ZipFile(out/'submission.zip','w') as archive:
                archive.writestr('scores.txt','a 0.80\nb 0.10\n')
            atomic_json(out/'submission_meta.json',dict(version='3.19',run=str(Path(run).resolve()),
                score='P(fake)',checkpoint_kind=kwargs['checkpoint_kind'],
                selected=done['selections'][kwargs['checkpoint_kind']],
                checkpoint_sha256=done['checkpoint_sha256'],protocol_sha256=digest(protocol),
                zip_sha256=digest(out/'submission.zip'),count=2,id_order_sha256=identity(['a','b'])))
            return out/'submission.zip'
        return export

    def test_best_export_upload_failure_retry_never_repeats_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,cfg,done,request=self.fixture(root);calls=[]
            with patch.object(delivery,'ROOT',root), patch.object(delivery,'upload_archive',side_effect=OSError('offline')):
                with self.assertRaisesRegex(RuntimeError,'Retry without GPU inference'):
                    delivery.finish_request(run,cfg,request,exporter=self.exporter(done,calls))
            self.assertTrue((Path(request['out'])/'submission.zip').is_file())
            self.assertEqual(calls,['best_weighted'])
            metadata=dict(done,selected=done['selections']['best_weighted'])
            with patch.object(delivery,'ROOT',root), patch('w2v_v319.state.load_selected',return_value=({},metadata)), \
                 patch.object(delivery,'upload_archive',return_value='https://temp.sh/one/submission.zip') as upload:
                result=delivery.finish_request(run,cfg,request,exporter=self.exporter(done,calls))
                self.assertEqual(result['url'],'https://temp.sh/one/submission.zip')
                self.assertEqual(upload.call_count,1)
                delivery.finish_request(run,cfg,request,exporter=self.exporter(done,calls))
                self.assertEqual(upload.call_count,1)
            self.assertEqual(calls,['best_weighted'])
            self.assertIn('SUBMISSION_URL=https://temp.sh/one/submission.zip',(run/'download_links.txt').read_text())

    def test_guarded_is_explicit_and_wrong_best_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,cfg,done,request=self.fixture(root,'best_guarded');calls=[]
            request['upload']=False
            with patch.object(delivery,'ROOT',root):
                delivery.finish_request(run,cfg,request,exporter=self.exporter(done,calls))
            self.assertEqual(calls,['best_guarded'])
            meta=read_json(Path(request['out'])/'submission_meta.json');meta['selected']='epoch_4_step_100'
            atomic_json(Path(request['out'])/'submission_meta.json',meta)
            with patch('w2v_v319.state.load_selected',return_value=({},dict(done,selected='starting_parent'))):
                with self.assertRaisesRegex(ValueError,'different best'):
                    delivery.finish_request(run,cfg,request,exporter=self.exporter(done,calls))

    def test_report_excludes_weights_audio_and_unknown_arrays_and_delivers_two_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,cfg,done,request=self.fixture(root);calls=[]
            (run/'last.pt').write_bytes(b'private model')
            (run/'input.wav').write_bytes(b'private audio')
            (run/'random_features.npz').write_bytes(b'large features')
            diagnostic_arrays=('panel_scores_baseline_tune.npz','panel_scores_baseline_audit.npz',
                               'panel_scores_selected_audit.npz','panel_scores_epoch_1_step_100.npz')
            for name in diagnostic_arrays:(run/name).write_bytes(b'small verified prediction array')
            (run/'panel_scores_features.npz').write_bytes(b'unrecognized array')
            atomic_json(run/'training_history.json',[dict(epoch=1,metrics=dict(weighted_f1=.95))])
            with patch.object(delivery,'ROOT',root),patch.object(delivery,'upload_archive',return_value='https://temp.sh/one/submission.zip'):
                delivery.finish_request(run,cfg,request,exporter=self.exporter(done,calls))
            with patch.object(reports,'upload_archive',return_value='https://temp.sh/two/report.zip') as uploader:
                archive=reports.report(run,root/'downloads',upload=True)
                reports.report(run,root/'downloads',upload=True)
                self.assertEqual(uploader.call_count,1)
            with zipfile.ZipFile(archive) as z:
                self.assertIn('training_history.json',z.namelist())
                self.assertIn('REPORT_MANIFEST.json',z.namelist())
                for name in diagnostic_arrays:self.assertIn(name,z.namelist())
                for name in ('last.pt','input.wav','random_features.npz','panel_scores_features.npz','artifact_links.json'):
                    self.assertNotIn(name,z.namelist())
            links=(run/'download_links.txt').read_text()
            self.assertIn('SUBMISSION_URL=https://temp.sh/one/submission.zip',links)
            self.assertIn('REPORT_URL=https://temp.sh/two/report.zip',links)
            with patch.object(delivery,'upload_archive',side_effect=AssertionError('Should reuse scores link')), \
                 patch.object(reports,'upload_archive',side_effect=AssertionError('Should reuse report link')):
                reports.retry_outputs(run)
            self.assertEqual(calls,['best_weighted'])

    def test_submission_id_order_tamper_and_extra_archive_files_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,cfg,done,request=self.fixture(root);out=root/'fresh'
            self.exporter(done,[])(run,request['protocol'],request['audio_root'],out,checkpoint_kind='best_weighted')
            meta=read_json(out/'submission_meta.json');meta['id_order_sha256']=identity(['b','a'])
            atomic_json(out/'submission_meta.json',meta)
            with self.assertRaisesRegex(ValueError,'count/order'):delivery.check_archive(out)
            with zipfile.ZipFile(out/'submission.zip','a') as z:z.writestr('last.pt','not allowed')
            meta['zip_sha256']=digest(out/'submission.zip');atomic_json(out/'submission_meta.json',meta)
            with self.assertRaisesRegex(ValueError,'only scores'):delivery.check_archive(out)

    def test_resume_preserves_saved_selector_and_report_retries_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,cfg,done,request=self.fixture(root,'best_guarded')
            args=SimpleNamespace(submit_best=False)
            with patch.object(delivery,'read_protocol',return_value=[dict(id='a'),dict(id='b')]):
                self.assertEqual(delivery.prepare_request(args,run,cfg),request)
            with patch.object(reports,'upload_archive',return_value='https://temp.sh/two/report.zip'):
                with self.assertRaisesRegex(RuntimeError,'submission:'):
                    reports.retry_outputs(run)
            self.assertEqual(read_json(run/'report_status.json')['status'],'complete')


if __name__=='__main__':unittest.main()
