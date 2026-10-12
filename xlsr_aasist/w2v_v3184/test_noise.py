"""Noise integrity, source isolation, restart safety and actual PCM conversion."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from .noise import validate_records,bind_manifest,SceneBank,scene_wave,sha256
from .prepare_noise import CATALOG,download,validate_catalog,convert_cached,preparation_lock
from .test_support import scene_fixture


class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):
        super().__init__(data);self.status=status;self.headers=headers or {}


class NoiseTests(unittest.TestCase):
    def test_pinned_catalog_is_small_complete_and_source_disjoint(self):
        catalog=json.loads(CATALOG.read_text(encoding='utf-8'));rows=validate_catalog(catalog)
        self.assertEqual(len(rows),128);self.assertLess(sum(r['bytes'] for r in rows),60_000_000)
        self.assertTrue(all('/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/' in r['url'] for r in rows if r['dataset']=='ESC-50'))
        changed=copy.deepcopy(catalog)
        a=next(r for r in changed['records'] if r['split']=='train')
        b=next(r for r in changed['records'] if r['split']=='dev')
        b['original_sources']=a['original_sources']
        with self.assertRaisesRegex(ValueError,'source overlap'):validate_catalog(changed)

    def test_http_resume_honors_range_and_ignoring_range_restarts(self):
        data=b'0123456789'*100
        row=dict(url='https://example.com/audio.wav',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        for honor in (True,False):
            with self.subTest(honor=honor),tempfile.TemporaryDirectory() as d:
                p=Path(d)/'audio.wav';p.with_suffix('.wav.part').write_bytes(data[:100])
                def opener(request,timeout):
                    self.assertEqual(request.get_header('Range'),'bytes=100-')
                    return Response(data[100:],206,{'Content-Range':f'bytes 100-999/{len(data)}'}) if honor else Response(data)
                download(row,p,opener=opener,attempts=1)
                self.assertEqual(p.read_bytes(),data)
                self.assertEqual(download(row,p,opener=lambda *a,**k:self.fail('cached network')),'cached')

    def test_download_rejects_invalid_bytes_ranges_and_hashes(self):
        row=dict(url='https://example.com/a.wav',bytes=5,sha256=hashlib.sha256(b'right').hexdigest())
        for value in (b'wrong',b'toolong',b'short'[:2]):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as d:
                p=Path(d)/'a.wav'
                with self.assertRaisesRegex(RuntimeError,'Download failed'):
                    download(row,p,opener=lambda *a,**k:Response(value),attempts=1)
                self.assertFalse(p.exists())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.wav';p.with_suffix('.wav.part').write_bytes(b'ri')
            with self.assertRaisesRegex(RuntimeError,'Invalid HTTP range'):
                download(row,p,opener=lambda *a,**k:Response(b'ght',206,{'Content-Range':'bytes 1-4/5'}),attempts=1)
            with self.assertRaises(ValueError):download(dict(row,bytes=100*1024**2),p)
            with self.assertRaises(ValueError):download(dict(row,url='http://example.com/a'),p)

    def test_source_and_pcm_isolation_and_manifest_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=scene_fixture(Path(d));records=cfg['scene_noise_records']
            for field in ('original_sources','pcm_sha256'):
                changed=copy.deepcopy(records);changed['dev'][0][field]=changed['train'][0][field]
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'overlaps'):validate_records(changed)
            Path(records['train'][0]['path']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'changed'):bind_manifest(cfg['scene_noise_manifest'])

    def test_natural_sequence_repeats_length_safely_and_cache_detects_changes(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=scene_fixture(Path(d));bank=SceneBank(cfg['scene_noise_records']['train'],cap_bytes=70000)
            a,hashes,sources=scene_wave(bank,'keyboard',64000,np.random.default_rng(71))
            b,_,_=scene_wave(bank,'keyboard',64000,np.random.default_rng(71))
            np.testing.assert_array_equal(a,b);self.assertEqual(len(a),64000)
            self.assertTrue(hashes and sources);self.assertLessEqual(bank.used,bank.cap)
            p=Path(next(r for r in cfg['scene_noise_records']['train'] if r['category']=='keyboard')['path'])
            with p.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'changed during training'):bank.sample('keyboard',np.random.default_rng())

    def test_os_lock_is_exclusive_and_reusable_after_exception(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'.prepare.lock'
            with self.assertRaisesRegex(RuntimeError,'interrupt'):
                with preparation_lock(p):
                    with self.assertRaisesRegex(RuntimeError,'already running'):
                        with preparation_lock(p):pass
                    raise RuntimeError('interrupt')
            with preparation_lock(p):pass

    def test_actual_ffmpeg_conversion_reuses_finished_pcm(self):
        import imageio_ffmpeg
        binary=imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'stereo.wav';dest=root/'normalized.wav'
            x=np.stack([.1*np.sin(np.arange(44100)*.02),.2*np.sin(np.arange(44100)*.03)],axis=1)
            sf.write(src,x,44100);source_hash=sha256(src);binary_hash=sha256(binary)
            info=convert_cached(src,dest,binary,source_hash,binary_hash)
            pcm,sr=sf.read(dest,dtype='int16');self.assertEqual((pcm.shape,sr),((16000,),16000))
            self.assertEqual(info['pcm_sha256'],hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest())
            with patch('w2v_v3184.prepare_noise.convert',side_effect=AssertionError('duplicate conversion')):
                self.assertEqual(convert_cached(src,dest,binary,source_hash,binary_hash),info)


if __name__=='__main__':unittest.main()
