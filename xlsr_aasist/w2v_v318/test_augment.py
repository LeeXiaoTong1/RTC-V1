"""Real bundled-codec regressions, including non-frame-aligned audio tails."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .augment import Engines,recipe
from .runtime import bundled_ffmpeg


class TestAugment(unittest.TestCase):
    def test_bundled_processing_preserves_full_recording(self):
        engine=Engines()
        for family in ('ffmpeg','light','bypass','g711_mulaw','anlmdn'):
            for length in (3920,16123,48007):
                for codec in (('none',) if family in ('g711_mulaw','anlmdn') else ('none','opus')):
                    with self.subTest(family=family,length=length,codec=codec):
                        wave=(.08*np.sin(np.arange(length)*.094)).astype(np.float32)
                        r=recipe(31801,'regression','synthetic',0,0,family=family);r['codec']=codec
                        y=engine(wave,r)
                        self.assertEqual(y.shape,wave.shape)
                        self.assertTrue(np.isfinite(y).all())
                        if family=='bypass' and codec=='none':self.assertTrue(np.array_equal(y,wave))

    def test_errors_preserve_stage_lengths_and_binary(self):
        engine=Engines();r=dict(family='bypass',codec='opus')
        for bad in (np.zeros(16069,np.float32),np.full(16000,np.nan,np.float32)):
            with self.assertRaises(RuntimeError) as error:engine.checked(bad,16000,'codec',r)
            message=str(error.exception)
            for value in ('family=bypass','codec=opus','stage=codec','expected_samples=16000',engine.rtc.ffmpeg):
                self.assertIn(value,message)
        for bad in (np.zeros((2,100)),np.array([]),np.array([np.nan])):
            with self.assertRaisesRegex(ValueError,'finite nonempty mono'):engine(bad,r)

    def test_default_binary_does_not_inherit_path_or_imageio_override(self):
        with patch.dict(os.environ,{'IMAGEIO_FFMPEG_EXE':'/obsolete/ffmpeg','FFMPEG_BIN':'/obsolete/ffmpeg'}):
            self.assertTrue(Path(bundled_ffmpeg()).is_file())
            self.assertEqual(Engines().rtc.ffmpeg,bundled_ffmpeg())

    def test_training_pins_same_binary_as_preflight_not_data_run(self):
        from .config import parser,configuration
        from .assets import spec
        from .common import atomic_json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pair=root/'pairs.csv';pair.write_text('offline_id,online_id\n')
            assets=root/'assets.json';atomic_json(assets,dict(checkpoint='fixture-3b.pt',sha256='fixture'))
            args=parser().parse_args(['--data-run',d,'--dev-pairs',str(pair),'--omni-assets',str(assets)])
            # Old data includes a legacy executable, which must never become the
            # new run's codec implementation after a successful preflight.
            old=dict(augmentation_runtime=dict(ffmpeg_path='/old/ffmpeg',ffmpeg_sha256='old'),
                noise_records={},augmentation_files={},dev_protocol='fixture-dev',dev_data_path='fixture-root')
            runtime=dict(ffmpeg_path=bundled_ffmpeg(),ffmpeg='fixture version',ffmpeg_sha256='new')
            with patch('w2v_v318.config.data_inputs',return_value=(old,[],[],{})), \
                 patch('w2v_v318.config.validate_assets',return_value=spec('3b')), \
                 patch('w2v_v318.config.torch.cuda.is_available',return_value=True), \
                 patch('w2v_v318.config.torch.cuda.is_bf16_supported',return_value=True), \
                 patch('w2v_v318.augment.augmentation_runtime',return_value=runtime), \
                 patch('w2v_v318.config.runtime_versions',return_value={}), \
                 patch('w2v_v318.runtime.execution_profile',return_value='cu118'), \
                 patch('w2v_v318.runtime.native_abi',return_value={'profile':'cu118'}), \
                 patch('w2v_v318.config.verify_inputs'):
                cfg,_,_=configuration(args)
            self.assertEqual(cfg['ffmpeg'],Engines().rtc.ffmpeg)
            self.assertEqual(cfg['ffmpeg_sha256'],'new')
            self.assertEqual(cfg['augmentation_runtime'],runtime)


if __name__=='__main__':unittest.main()
