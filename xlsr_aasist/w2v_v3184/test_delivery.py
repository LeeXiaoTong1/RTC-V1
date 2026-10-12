"""Failure-path regressions; HTTP uploads are mocked, never sent in tests."""
from contextlib import redirect_stdout,nullcontext
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import zipfile
import torch
from w2v_aasist.evaluate import package_scores
from .common import atomic_json,read_json,digest,fingerprint,schema_for
from .config import parser,validate,verify_unique
from .delivery import check_archive,prepare_request,finish_request,upload_submission


def make_export(out,**overrides):
    ids=['online/en/a.wav','online/zh/b.wav']
    path=package_scores(ids,[.9,.1],out)
    meta=dict(version='3.18.4',kind='best',score='P(fake=0)',selected='epoch_1_step_3',
        checkpoint_sha256='fixture-checkpoint',count=2,id_order_sha256=fingerprint(ids),zip_sha256=digest(path),
        run='fixture-run',calibration={'scale':1.,'bias':0.},protocol_sha256='fixture-protocol')
    meta.update(overrides);atomic_json(Path(out)/'submission_meta.json',meta)
    return path,meta


class DeliveryTests(unittest.TestCase):
    def test_shared_idle_guard_recognizes_old_and_r2_jobs_but_not_viewer(self):
        # Exercise the protected shared guard directly: its substring rule
        # intentionally catches the new namespace without editing old code.
        from w2v_aasist import full_workflow
        with tempfile.TemporaryDirectory() as d:
            proc=Path(d);own=proc/'100';own.mkdir()
            (own/'cmdline').write_bytes(b'/usr/bin/python\0-m\0w2v_v3184.workflow\0')
            other=proc/'200';other.mkdir()
            fixture_os=SimpleNamespace(name='posix',getpid=lambda:100,getuid=lambda:own.stat().st_uid,
                getppid=lambda:99,environ={})
            def fixture_path(value):return proc if str(value)=='/proc' else Path(value)
            with patch.object(full_workflow,'os',fixture_os),patch.object(full_workflow,'Path',fixture_path):
                for module in ('w2v_v3183.workflow','w2v_v3183.evaluate','w2v_v3183r2.workflow','w2v_v3183r2.evaluate',
                               'w2v_v3184.workflow','w2v_v3184.evaluate'):
                    (other/'cmdline').write_bytes(('/usr/bin/python\0-m\0'+module+'\0').encode())
                    with self.subTest(module=module),self.assertRaisesRegex(RuntimeError,'existing job'):
                        full_workflow.ensure_idle()
                (other/'cmdline').write_bytes(b'/usr/bin/python\0v3184_progress.py\0/exp/v3184.log\0')
                full_workflow.ensure_idle()

    def test_progress_recognizes_r2_training_and_does_not_confuse_parent(self):
        from v3184_progress import TRAIN_PATTERN,Steps,progress
        line='[Train] V3.18.4 epoch 2/6 phase=joint updates=2417; C3'
        self.assertEqual(TRAIN_PATTERN.search(line).groups(),('2','6','2417'))
        self.assertIsNone(TRAIN_PATTERN.search('[Train] V3.18.3 epoch 2/6 updates=2417'))
        reader=Steps();reader.latest=dict(epoch=2,step=17,loss=.35);reader.count=17;reader.seconds=34.
        self.assertIn('17/2417',progress(reader,2,6,2417))
        self.assertIn('waiting for first',progress(reader,3,6,2417))

    def test_old_archive_cannot_be_uploaded_as_r2(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);make_export(out,version='3.18.3')
            with patch('w2v_v3184.delivery.upload_archive') as send:
                with self.assertRaisesRegex(ValueError,'V3.18.4'):upload_submission(out)
                send.assert_not_called()

    def test_export_invalid_inputs_fail_before_loading_gpu_model(self):
        from .evaluate import export
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';run.mkdir();proto=root/'protocol.txt';proto.write_text('a.wav\n')
            meta=dict(checkpoint_sha256='expected',selected='epoch_1',kind='best')
            factory=Mock(side_effect=AssertionError('Model must not load'))
            with patch('w2v_v3184.evaluate.selected',return_value=({'seed':1},{},meta)):
                with self.assertRaises(FileNotFoundError):
                    export(run,root/'out',protocol=root/'missing.txt',audio_root=root,device='cpu',verify=False,model_factory=factory)
                atomic_json(run/'calibration_best.json',dict(checkpoint_sha256='other',selected='epoch_1'))
                with patch('w2v_v3184.evaluate.read_protocol',return_value=[{'id':'a.wav'}]):
                    with self.assertRaisesRegex(ValueError,'Calibration belongs'):
                        export(run,root/'out',protocol=proto,audio_root=root,device='cpu',verify=False,model_factory=factory)
                factory.assert_not_called()

    def test_unique_verification_preserves_conflict_detection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'asset';p.write_text('asset');sha=digest(p)
            with patch('w2v_v3184.config.verify_files') as verify:
                verify_unique({str(p):sha},{str(p.parent/'.'/p.name):sha})
            self.assertEqual(len(verify.call_args.args[0]),1)
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                verify_unique({str(p):sha},{str(p):'changed'})
            p.write_text('different')
            with self.assertRaises(ValueError):verify_unique({str(p):sha})

    def test_upload_failure_retry_and_saved_link(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            out=Path(d);archive,meta=make_export(out)
            with patch('w2v_v3184.delivery.upload_archive',side_effect=RuntimeError('offline')):
                with self.assertRaisesRegex(RuntimeError,'without GPU'):upload_submission(out)
            self.assertEqual(read_json(out/'upload_status.json')['status'],'failed')
            self.assertEqual(digest(archive),meta['zip_sha256']);self.assertFalse((out/'temp_download_url.txt').exists())
            with patch('w2v_v3184.delivery.upload_archive',return_value='https://temp.sh/abc/submission.zip') as send:
                self.assertEqual(upload_submission(out),'https://temp.sh/abc/submission.zip')
                send.assert_called_once_with(archive)
                upload_submission(out);self.assertEqual(send.call_count,1)
                upload_submission(out,force=True);self.assertEqual(send.call_count,2)
            self.assertEqual(read_json(out/'upload_receipt.json')['selected'],'epoch_1_step_3')
            self.assertEqual((out/'temp_download_url.txt').read_text().strip(),'https://temp.sh/abc/submission.zip')

    def test_bad_archive_score_order_and_service_response_rejected(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            out=Path(d);archive,meta=make_export(out)
            with patch('w2v_v3184.delivery.upload_archive',return_value='https://example.com/wrong'):
                with self.assertRaisesRegex(RuntimeError,'Unexpected'):upload_submission(out)
            with zipfile.ZipFile(archive,'a') as z:z.writestr('model.pt','forbidden upload fixture')
            meta['zip_sha256']=digest(archive);atomic_json(out/'submission_meta.json',meta)
            with self.assertRaisesRegex(ValueError,'only scores'):check_archive(out)
            archive,meta=make_export(out,id_order_sha256='wrong-order')
            with self.assertRaisesRegex(ValueError,'count/order'):check_archive(out)
            archive.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash changed'):check_archive(out)

    def test_request_preflight_resume_and_set_lock(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);run=root/'run';run.mkdir();audio=root/'audio';audio.mkdir()
            (audio/'a.wav').write_bytes(b'preflight checks existence only')
            protocol=root/'eval.txt';protocol.write_text('a.wav\n')
            argv=['--submit-best','--upload-submission','--submission-protocol',str(protocol),
                  '--submission-audio-root',str(audio),'--submission-out',str(root/'out')]
            args=parser().parse_args(argv);validate(args)
            with patch('w2v_v3184.delivery.read_protocol',wraps=__import__('w2v_aasist.data',fromlist=['read_protocol']).read_protocol) as reader:
                request=prepare_request(args,run,{})
                self.assertEqual(reader.call_count,1)
            resumed=prepare_request(parser().parse_args(['--resume',str(run)]),run,{})
            self.assertEqual(request,resumed);self.assertEqual(request['kind'],'best')
            alternate=root/'other.txt';alternate.write_text('a.wav\n')
            changed=parser().parse_args(['--submit-best','--submission-protocol',str(alternate)])
            with self.assertRaisesRegex(ValueError,'differs'):prepare_request(changed,run,{})
            protocol.write_text('missing.wav\n')
            with self.assertRaisesRegex(ValueError,'Pinned'):prepare_request(parser().parse_args([]),run,{})
            with self.assertRaisesRegex(ValueError,'requires'):validate(parser().parse_args(['--upload-submission']))

    def test_failed_export_leaves_no_final_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';run.mkdir();protocol=root/'eval.txt';protocol.write_text('a.wav\n')
            out=root/'submission'
            atomic_json(run/'completed.json',{'status':'complete','best_tag':'epoch_1_step_3','checkpoint_sha256':'fixture'})
            request=dict(out=str(out),protocol=str(protocol),protocol_sha256=digest(protocol),audio_root=str(root),
                count=1,id_order_sha256=fingerprint(['a.wav']),upload=True)
            def broken(run,stage,**kwargs):
                stage.mkdir();(stage/'partial.txt').write_text('incomplete');raise RuntimeError('inference failed')
            with self.assertRaisesRegex(RuntimeError,'inference failed'):
                finish_request(run,{'device':'cpu','workers':0},request,broken)
            self.assertFalse(out.exists());self.assertEqual(list(root.glob('submission.pending-*')),[])
            self.assertEqual(read_json(run/'submission_status.json')['status'],'failed')
            self.assertEqual(read_json(run/'completed.json')['status'],'complete')

    def test_best_selector_does_not_use_last_weight_block(self):
        from .evaluate import selected
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);cfg={'version':'3.18.4','omni_arch':'3b'}
            atomic_json(run/'config.json',cfg)
            state=dict(schema=schema_for(cfg),identity=fingerprint(cfg),best_tag='epoch_1',last_tag='epoch_2',
                best_model={'head.fixture':torch.tensor([1.])},model={'head.fixture':torch.tensor([9.])},
                history=[{'tag':'epoch_1'},{'tag':'epoch_2'}])
            torch.save(state,run/'last.pt')
            atomic_json(run/'completed.json',dict(version='3.18.4',status='complete',best_tag='epoch_1',last_tag='epoch_2',checkpoint_sha256=digest(run/'last.pt')))
            _,best,meta=selected(run,'best');_,last,_=selected(run,'last')
            self.assertEqual(float(best['head.fixture']),1.);self.assertEqual(float(last['head.fixture']),9.)
            self.assertEqual(meta['selected'],'epoch_1')

    def test_workflow_never_submits_after_training_failure(self):
        from .workflow import main
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            cfg=dict(seed=1,stream_sources=16,group_quotas=[4,2,7,3])
            with patch('sys.argv',['workflow','--submit-best']),patch('w2v_v3184.workflow.ROOT',Path(d)), \
                 patch('w2v_v3184.workflow.ensure_idle'),patch('w2v_v3184.workflow.run_lock',return_value=nullcontext()), \
                 patch('w2v_v3184.workflow.configuration',return_value=(cfg,[],[])),patch('w2v_v3184.workflow.describe',return_value={}), \
                 patch('w2v_v3184.workflow.prepare_request',return_value={'upload':True}),patch('w2v_v3184.workflow.report'), \
                 patch('w2v_v3184.workflow.run_experiment',side_effect=RuntimeError('training failed')), \
                 patch('w2v_v3184.workflow.finish_request') as submit:
                with self.assertRaisesRegex(RuntimeError,'training failed'):main()
                submit.assert_not_called()

    def test_workflow_rejects_old_run_resume_before_input_checks(self):
        from .workflow import main
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            run=Path(d)/'old';run.mkdir();atomic_json(run/'config.json',{'version':'3.18.3'})
            with patch('sys.argv',['workflow','--resume',str(run)]),patch('w2v_v3184.workflow.ROOT',Path(d)), \
                 patch('w2v_v3184.workflow.ensure_idle'),patch('w2v_v3184.workflow.run_lock',return_value=nullcontext()), \
                 patch('w2v_v3184.workflow.verify_inputs') as verify:
                with self.assertRaisesRegex(ValueError,'--parent-run'):main()
                verify.assert_not_called()


if __name__=='__main__':unittest.main()
