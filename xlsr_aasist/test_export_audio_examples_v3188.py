"""Quartet identity, pre-RTC capture, live-run isolation and retry tests."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import soundfile as sf
import export_audio_examples_v3188 as export
from w2v_v3184.augment import NoiseBank, Engines, recipe
from w2v_v3184.noise import RECORDED


class FakeRTC:
    def __init__(self, _): self.calls = 0
    def __call__(self, wave, rec):
        self.calls += 1
        return (.5*wave).astype(np.float32)


def fixture(root):
    root=Path(root); run=root/'live';run.mkdir();audio=root/'audio';audio.mkdir()
    rows=[]; rng=np.random.default_rng(8)
    for language,label in export.GROUPS:
        key=f'{language}-{label}';x=rng.normal(0,.08,16000).astype(np.float32)
        off=audio/(key+'-source.wav');sf.write(off,x,16000,subtype='FLOAT')
        for kind,wave in (('offline',x),('online',np.pad(x*.8,(40,80)))):
            path=audio/(key+'-'+kind+'.wav');sf.write(path,wave,16000,subtype='FLOAT')
            stat=path.stat()
            rows.append(dict(id=kind+':'+key,source_id=key,group_id=key,source_sha256=export.digest(off),
                language=language,label=label,condition=kind,split='train',full_length=True,
                audio=str(path),audio_size=stat.st_size,audio_mtime_ns=stat.st_mtime_ns))
    noises=[]
    for category in RECORDED:
        path=audio/('noise-'+category+'.wav');sf.write(path,rng.normal(0,.1,32000).astype(np.float32),16000,subtype='FLOAT')
        stat=path.stat()
        noises.append(dict(path=str(path),sha256=export.digest(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
            category=category,source_group=category,original_sources=[category],original_recording=category,split='train'))
    export.write_json(run/'train_rows.json',rows)
    cfg=dict(version='3.18.8',recipe='v3188_single_noisy_memory_bounded_v1',seed=31801,warm_epochs=0,
        source_epoch_offset=5,ffmpeg='fixture',noise_records={'train':noises},scene_noise_records={'train':noises},
        scene_probability=1.,rtc_policy='three-profile',rtc_stress_probability=.35,rtc_stress_ramp_epochs=2.,
        saved_manifests={'train_rows.json':export.digest(run/'train_rows.json')})
    export.write_json(run/'config.json',cfg)
    export.write_json(run/'progress.json',dict(epoch=2,step=55,phase='joint',status='training'))
    (run/'last.pt').write_bytes(b'DO NOT LOAD OR CHANGE THE LIVE CHECKPOINT')
    args=export.parser().parse_args(['--run',str(run),'--output-root',str(root/'exports'),
        '--min-seconds','0.2','--max-seconds','2','--scene','rain'])
    return run,cfg,rows,args


class ExportTests(unittest.TestCase):
    def test_exact_pre_rtc_capture_all_six_scenes(self):
        with tempfile.TemporaryDirectory() as folder:
            _,cfg,rows,_=fixture(folder)
            source=sf.read(rows[0]['audio'],dtype='float32')[0]
            bank=NoiseBank(cfg['noise_records']['train'],1024**2,cfg['scene_noise_records']['train'])
            for scene in (*RECORDED,'echo'):
                with self.subTest(scene=scene):
                    rec=recipe(1,'test','source',0,5,0,scene=scene,rtc_policy='three-profile')
                    engine=FakeRTC(None)
                    mixed,received,meta=export.generate_stages(source,rec,bank,engine)
                    self.assertEqual(engine.calls,1)
                    np.testing.assert_array_equal(received,mixed*.5)
                    self.assertFalse(np.array_equal(source,mixed))
                    self.assertEqual(meta['recipe']['category'],scene)

    def test_complete_quartets_preserve_live_run_and_reproduce(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            run,cfg,rows,args=fixture(folder)
            before={p.name:export.digest(p) for p in run.iterdir()}
            with patch('torch.load',side_effect=AssertionError('Never load a checkpoint')):
                archive=export.export_examples(args,FakeRTC)
                again=export.export_examples(args,FakeRTC)
            self.assertEqual(before,{p.name:export.digest(p) for p in run.iterdir()})
            manifest=export.validate_archive(archive)
            self.assertEqual(manifest['groups'],4)
            self.assertEqual(manifest['recipe_epoch'],6)
            self.assertEqual({(c['language'],c['label']) for c in manifest['cases']},set(export.GROUPS))
            with zipfile.ZipFile(archive) as z,zipfile.ZipFile(again) as other:
                self.assertEqual(len(z.namelist()),19)
                self.assertNotIn('last.pt',z.namelist())
                self.assertIn('index.html',z.namelist())
                for case in manifest['cases']:
                    for filename,_ in export.STAGES:
                        data=z.read(case['name']+'/'+filename)
                        # libsndfile may timestamp the WAV PEAK chunk. The
                        # decoded waveform, not its header timestamp, repeats.
                        first,sr=sf.read(io.BytesIO(data),dtype='float32')
                        second,other_sr=sf.read(io.BytesIO(other.read(case['name']+'/'+filename)),dtype='float32')
                        self.assertEqual((sr,other_sr),(16000,16000))
                        np.testing.assert_array_equal(first,second)
                    a=sf.read(io.BytesIO(z.read(case['name']+'/03_noisy_before_rtc.wav')),dtype='float32')[0]
                    b=sf.read(io.BytesIO(z.read(case['name']+'/04_noisy_online.wav')),dtype='float32')[0]
                    np.testing.assert_array_equal(b,a*.5)

    def test_canonical_pair_mismatch_and_manifest_tamper_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            run,_,rows,args=fixture(folder)
            rows[1]['label']=1-rows[0]['label']
            with self.assertRaisesRegex(ValueError,'metadata disagrees'):
                export.select_pairs(rows,1,1,.1,2)
            (run/'train_rows.json').write_text('[]',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'fingerprint'):export.load_inputs(str(run))

    def test_cannot_export_into_live_run(self):
        with tempfile.TemporaryDirectory() as folder:
            run,_,_,args=fixture(folder);args.output_root=str(run/'examples')
            with self.assertRaisesRegex(ValueError,'outside the live'):
                export.export_examples(args,FakeRTC)

    def test_upload_retry_retains_audio_and_validates_contents(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            _,_,_,args=fixture(folder)
            archive=export.export_examples(args,FakeRTC);checksum=export.digest(archive)
            def failed(*_,**__):raise subprocess.CalledProcessError(22,'curl')
            with self.assertRaises(subprocess.CalledProcessError):export.upload(archive,failed)
            self.assertEqual(export.digest(archive),checksum)
            calls=[]
            def success(cmd,**kwargs):
                calls.append(cmd);return SimpleNamespace(stdout='https://temp.sh/test-fixture/audio.zip\n')
            self.assertEqual(export.upload(archive,success),'https://temp.sh/test-fixture/audio.zip')
            self.assertEqual(calls[0][-1],'https://temp.sh/upload')
            self.assertEqual(archive.with_suffix('.url.txt').read_text().strip(),'https://temp.sh/test-fixture/audio.zip')
            with zipfile.ZipFile(archive,'a') as z:z.writestr('unrequested_checkpoint.pt',b'bad')
            with self.assertRaisesRegex(ValueError,'Unexpected files'):export.upload(archive,success)
            self.assertEqual(len(calls),1)

    def test_native_ffmpeg_and_opus_endpoints(self):
        from w2v_v318.runtime import bundled_ffmpeg
        try:ffmpeg=bundled_ffmpeg()
        except (ImportError,FileNotFoundError,RuntimeError) as error:self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as folder:
            _,cfg,rows,_=fixture(folder)
            source=sf.read(rows[0]['audio'],dtype='float32')[0]
            bank=NoiseBank(cfg['noise_records']['train'],1024**2,cfg['scene_noise_records']['train'])
            for family in ('light','ffmpeg'):
                with self.subTest(family=family):
                    rec=recipe(2,'native','source',0,5,0,scene='rain',rtc_policy='three-profile')
                    rec.update(family=family,codec='opus')
                    mixed,received,meta=export.generate_stages(source,rec,bank,Engines(ffmpeg))
                    self.assertEqual((len(mixed),len(received)),(len(source),len(source)))
                    self.assertTrue(np.isfinite(received).all())
                    self.assertFalse(np.array_equal(mixed,received))


if __name__=='__main__':unittest.main()
