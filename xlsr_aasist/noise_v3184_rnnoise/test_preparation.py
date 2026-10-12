"""Network recovery, archive containment, speech removal and split integrity."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from .download import fetch, sha256
from .prepare import index_archive, screen_recording, write_outputs, SOURCES, HERE
from .screen import infer_rate, decode_raw, safe_intervals, assign_splits, SpeechScreen, FRAME
from w2v_v3184.noise import bind_manifest, SceneBank, scene_wave
from w2v_v3184.augment import recipe, generate
from w2v_v3184.test_support import scene_fixture


class Response(io.BytesIO):
    def __init__(self, data, status=206, content_range=''):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': content_range}


def archive_at(path, entries):
    with tarfile.open(path, 'w:gz') as archive:
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.type = kind
            archive.addfile(info, io.BytesIO(data))


class PreparationTests(unittest.TestCase):
    def test_range_download_resume_and_complete_reuse(self):
        data = b'abcdefghij'*9
        row = dict(url='https://example.com/noise',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        requests = []
        def opener(request, timeout):
            lo, hi = map(int,request.get_header('Range')[6:].split('-'))
            requests.append(lo)
            return Response(data[lo:hi+1],content_range=f'bytes {lo}-{hi}/{len(data)}')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'a.gz'
            parts = p.with_name('a.gz.parts')
            parts.mkdir()
            (parts/'0').write_bytes(data[:20])
            (parts/'0.sha256').write_text(hashlib.sha256(data[:20]).hexdigest())
            fetch(row,p,jobs=2,opener=opener,chunk_bytes=20)
            self.assertNotIn(0,requests)
            self.assertEqual(p.read_bytes(),data)
            fetch(row,p,opener=lambda *a,**k:self.fail('network for completed cache'))
            p.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Corrupt completed'):
                fetch(row,p,opener=opener)

    def test_bad_range_and_final_checksum_never_publish(self):
        row = dict(url='https://example.com/noise',bytes=4,sha256=hashlib.sha256(b'good').hexdigest())
        for response in (lambda:Response(b'good',content_range='bytes 1-4/5'),
                         lambda:Response(b'evil',content_range='bytes 0-3/4'),
                         lambda:Response(b'x',content_range='bytes 0-3/4')):
            with tempfile.TemporaryDirectory() as d,patch('noise_v3184_rnnoise.download.time.sleep'):
                p = Path(d)/'a'
                with self.assertRaises((ValueError,RuntimeError)):
                    fetch(row,p,opener=lambda *a,**k:response(),chunk_bytes=4)
                self.assertFalse(p.exists())
                fetch(row,p,opener=lambda *a,**k:Response(b'good',content_range='bytes 0-3/4'),chunk_bytes=4)
                self.assertEqual(p.read_bytes(),b'good')

    def test_selected_archive_members_only_and_restart_repairs_missing_file(self):
        name = '1507000000000-office.raw'
        entries = [('rnnoise_contributions/'+name,b'012345',tarfile.REGTYPE),
            ('rnnoise_contributions/'+name[:-4]+'.txt',b'browser',tarfile.REGTYPE),
            ('rnnoise_contributions/1507000000001-coffee.raw',b'excluded',tarfile.REGTYPE),
            ('rnnoise_contributions/1507000000002-other.raw',b'not used',tarfile.REGTYPE)]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);arc=root/'a.gz';out=root/'selected'
            archive_at(arc,entries)
            a=index_archive(arc,out,sha256(arc),{'1507000000001-coffee.raw'})
            self.assertEqual(len(a['records']),1)
            self.assertEqual(a['records'][0]['user_agent_sha256'],hashlib.sha256(b'browser').hexdigest())
            (out/name).unlink()
            b=index_archive(arc,out,sha256(arc),{'1507000000001-coffee.raw'})
            self.assertEqual(a,b)
            self.assertEqual((out/name).read_bytes(),b'012345')
            self.assertFalse((out/'1507000000002-other.raw').exists())

    def test_archive_path_traversal_links_and_duplicate_names_rejected(self):
        name='1507000000000-office.raw'
        for entries in ([('../'+name,b'xx',tarfile.REGTYPE)],
                        [('rnnoise_contributions/'+name,b'xx',tarfile.SYMTYPE)],
                        [('rnnoise_contributions/'+name,b'xx',tarfile.REGTYPE)]*2):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);arc=root/'a.gz';archive_at(arc,entries)
                with self.assertRaises(ValueError):index_archive(arc,root/'s',sha256(arc),set())
                self.assertFalse((root/name).exists())

    def test_rate_inference_and_real_polyphase_resampling(self):
        for rate in (44100,48000):
            self.assertEqual(infer_rate(rate*60*2),rate)
            x=(5000*np.sin(np.arange(rate,dtype=np.float64)*2*np.pi*440/rate)).astype('<i2')
            y,clipping=decode_raw(x.tobytes(),rate)
            self.assertEqual(len(y),16000)
            self.assertEqual(clipping,0)
            spectrum=np.abs(np.fft.rfft(y))
            self.assertEqual(int(np.argmax(spectrum)),440)
        for size in (0,1,44100*30*2,44100*56*2):
            with self.assertRaises(ValueError):infer_rate(size)

    def test_vad_masks_speech_and_preserves_only_long_contiguous_noise(self):
        probs=np.zeros(500,np.float32);probs[150:180]=.9;probs[181:220]=.9
        n=len(probs)*FRAME
        spans,fraction=safe_intervals(probs,n)
        self.assertGreater(fraction,0)
        for lo,hi in spans:
            self.assertGreaterEqual(hi-lo,48000)
            self.assertTrue(hi<=150*FRAME-4000 or lo>=220*FRAME+4000)
        self.assertEqual(safe_intervals(np.ones(500),n)[0],[])
        self.assertEqual(safe_intervals(np.zeros(500),n)[0],[[0,n]])
        with self.assertRaises(ValueError):safe_intervals(np.array([np.nan]),512)

    def test_batched_vad_keeps_independent_recurrent_states_and_lengths(self):
        import torch
        class Model:
            def reset_states(self):self.state=None
            def __call__(self,x,sr):
                self.state=x.mean(dim=1) if self.state is None else .8*self.state+x.mean(dim=1)
                return torch.sigmoid(self.state).reshape(-1,1)
        screen=object.__new__(SpeechScreen);screen.torch=torch;screen.model=Model()
        waves=[np.random.default_rng(i).normal(size=4096+i*170).astype(np.float32) for i in range(8)]
        expected=[screen.probabilities(wave) for wave in waves]
        together=screen.probabilities_batch(waves)
        for a,b,wave in zip(expected,together,waves):
            np.testing.assert_allclose(a,b,atol=1e-7)
            self.assertEqual(len(b),(len(wave)+511)//512)
        with self.assertRaises(ValueError):screen.probabilities_batch([])

    def test_recordings_sessions_duplicates_and_segments_stay_in_one_split(self):
        rows=[]
        for cat in ('office','coffee'):
            for i in range(12):
                name=f'{1507000000000+i*86400000}-{cat}.raw'
                row=dict(recording=name,timestamp_ms=1507000000000+i*86400000,
                    user_agent_sha256='browser-'+cat,category=cat,source_sha256=name,pcm_sha256=name)
                rows.extend([dict(row,key=name+'-a'),dict(row,key=name+'-b')])
        # A second upload in the same recording session, with different PCM.
        adjacent=dict(rows[0],recording='1507000001000-office.raw',timestamp_ms=1507000001000,
                      source_sha256='different',pcm_sha256='different')
        rows.append(adjacent)
        # Same content uploaded on a different day/browser must also stay together.
        rows.append(dict(rows[0],recording='1607000000000-office.raw',timestamp_ms=1607000000000,
                         user_agent_sha256='another-browser'))
        result=assign_splits(copy.deepcopy(rows))
        original=next(r for r in result if r['recording']==rows[0]['recording'])
        for r in result:
            if r['recording'] in (adjacent['recording'],'1607000000000-office.raw'):
                self.assertEqual(r['source_group'],original['source_group'])
                self.assertEqual(r['split'],original['split'])
        sets={s:{x for r in result if r['split']==s for x in r['original_sources']} for s in ('train','dev')}
        self.assertFalse(sets['train']&sets['dev'])
        self.assertEqual(result,assign_splits(copy.deepcopy(list(reversed(rows))))[::-1])

    def test_screen_cache_uses_real_pcm_and_excludes_speech(self):
        class Screen:
            def probabilities(self,wave):
                probs=np.zeros((len(wave)+511)//512,np.float32)
                probs[200:1000]=.99
                return probs
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for part in ('raw','screening','audio'):(root/part).mkdir()
            rate=44100;name='1507000000000-office.raw'
            raw=(1000*np.sin(np.arange(rate*60)*.011)).astype('<i2').tobytes()
            (root/'raw'/name).write_bytes(raw)
            row=dict(recording=name,file=name,bytes=len(raw),source_sha256=hashlib.sha256(raw).hexdigest(),category='office')
            first=screen_recording(row,root/'raw',root,Screen(),{'test':'v1'})
            self.assertEqual(first['status'],'accepted')
            self.assertGreater(first['blocked_fraction'],.1)
            for record in first['records']:
                x,sr=sf.read(root/record['file'])
                self.assertEqual((len(x),sr),(record['samples'],16000))
                self.assertTrue(record['stop_sample_16k']<=200*512-4000 or record['start_sample_16k']>=1000*512+4000)
            with patch.object(Screen,'probabilities',side_effect=AssertionError('duplicate screening')):
                self.assertEqual(first,screen_recording(row,root/'raw',root,Screen(),{'test':'v1'}))

    def test_new_manifest_uses_existing_training_bank_and_augmentation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=scene_fixture(root/'fixture')
            manifest=json.loads(Path(cfg['scene_noise_manifest']).read_text())
            out=root/'new';out.mkdir();(out/'audio').mkdir()
            import shutil
            rows=[]
            for original in manifest['records']:
                row=dict(original,key=Path(original['file']).stem,attribution='fixture',page='https://example.com')
                source=Path(cfg['scene_noise_manifest']).parent/original['file']
                dest=out/'audio'/source.name;shutil.copyfile(source,dest)
                row['file']=dest.relative_to(out).as_posix();row['seconds']=len(sf.read(dest)[0])/16000
                rows.append(row)
            write_outputs(out,{'fixture':True},rows,{}, {'provenance':{'fixture':True}})
            bound=bind_manifest(out/'manifest.json')
            class Bank:
                scenes=SceneBank(bound['scene_noise_records']['train'])
            x=np.sin(np.arange(32000)*.024).astype(np.float32)*.1
            for category in ('office','coffee','rain','footsteps','keyboard','echo'):
                r=recipe(9,1,'source',0,2,scene_probability=1,rtc_policy='three-profile',scene=category)
                y,meta=generate(x,r,Bank(),lambda mixed,recipe:mixed)
                self.assertEqual(y.shape,x.shape)
                self.assertTrue(np.isfinite(y).all())
                self.assertEqual(r['codec'],'opus')
                self.assertNotIn('stress',r)
                self.assertTrue(meta['pair_eligible'])


if __name__=='__main__':unittest.main()
