"""Signal-level scene ordering, deterministic pairs and old-probe equivalence."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .augment import recipe,generate,NoiseBank,Engines,CATEGORIES,prior_recipe,legacy_generate
from .test_support import scene_fixture,NativeBoundary


class AugmentationTests(unittest.TestCase):
    def test_target_probability_balanced_categories_in_both_views(self):
        counts=[Counter(),Counter()];total=6000
        for occurrence in range(total):
            pair=[recipe(82,occurrence,'source',view,2) for view in (0,1)]
            for v,r in enumerate(pair):counts[v][r.get('category','legacy')]+=1
            if all('category' in r for r in pair):self.assertNotEqual(pair[0]['category'],pair[1]['category'])
        for values in counts:
            self.assertAlmostEqual(values['legacy']/total,.2,delta=.02)
            for category in CATEGORIES:self.assertAlmostEqual(values[category]/total,.8/6,delta=.02)
        self.assertEqual(recipe(82,9,'source',1,2),recipe(82,9,'source',1,2))

    def test_three_profile_marginals_and_no_extra_stress(self):
        for view in (0,1):
            counts=Counter()
            for i in range(2500):
                r=recipe(18,i,'source',view,2,rtc_policy='three-profile')
                counts[r['family']]+=1;self.assertEqual(r['codec'],'opus');self.assertNotIn('stress',r)
            for key,p in (('webrtc',.5),('ffmpeg',.3),('light',.2)):
                self.assertAlmostEqual(counts[key]/2500,p,delta=.04)

    def test_fixed_probes_are_bitwise_old_recipe_and_audio(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=scene_fixture(Path(d));bank=NoiseBank(cfg['scene_noise_records']['train'],1024**2,cfg['scene_noise_records']['train'])
            x=np.sin(np.arange(16000)*.07).astype(np.float32)*.06
            for view in (0,1):
                for family in (None,'anlmdn','g711_mulaw'):
                    args=(91,'probe','same',view,0,1,'train',family,True)
                    old=prior_recipe(*args);new=recipe(*args,rtc_policy='three-profile')
                    self.assertEqual({k:v for k,v in new.items() if k!='rtc_policy'},old)
                    a,_=legacy_generate(x,old,bank,NativeBoundary());b,_=generate(x,new,bank,NativeBoundary())
                    np.testing.assert_array_equal(a,b)

    def test_input_mix_happens_before_rtc_and_requested_snr_is_measured(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=scene_fixture(Path(d));bank=NoiseBank(cfg['scene_noise_records']['train'],1024**2,cfg['scene_noise_records']['train'])
            x=np.sin(np.arange(16000)*.05).astype(np.float32)*.08;original=x.copy();captured=[]
            def rtc(y,r):captured.append(y.copy());return y.copy()*.7
            r=recipe(14,'one','source',0,2,scene='rain');y,meta=generate(x,r,bank,rtc)
            self.assertTrue(meta['pair_eligible']);self.assertAlmostEqual(meta['measured_input_snr_db'],r['snr_db'],places=4)
            self.assertTrue(meta['noise_original_sources']);self.assertEqual(len(y),len(x))
            self.assertGreater(float(np.max(np.abs(captured[0]-x*meta['common_gain']))),0)
            np.testing.assert_array_equal(y,captured[0]*.7);np.testing.assert_array_equal(x,original)

    def test_echo_delay_is_physical_and_input_is_not_modified(self):
        x=np.zeros(6000,np.float32);x[500]=.3;before=x.copy()
        r=recipe(1,2,'source',0,2,scene='echo');r['echo']=dict(delay_samples=1000,gain=.3);r['input_gain_db']=0
        y,meta=generate(x,r,None,NativeBoundary())
        self.assertAlmostEqual(float(y[500]),.3,places=6);self.assertAlmostEqual(float(y[1500]),.09,places=6)
        self.assertEqual(np.count_nonzero(y),2);self.assertIsNone(meta['measured_input_snr_db'])
        np.testing.assert_array_equal(x,before)

    def test_real_filters_and_opus_preserve_lengths_for_all_scenes(self):
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory() as d:
            cfg=scene_fixture(Path(d));bank=NoiseBank(cfg['scene_noise_records']['train'],1024**2,cfg['scene_noise_records']['train'])
            engines=Engines(imageio_ffmpeg.get_ffmpeg_exe());x=np.sin(np.arange(17123)*.02).astype(np.float32)*.08
            for scene in CATEGORIES:
                for family in ('ffmpeg','light'):
                    r=recipe(39,scene,'source',0,2,scene=scene);r.update(family=family,codec='opus')
                    y,meta=generate(x,r,bank,engines)
                    self.assertEqual(y.shape,x.shape);self.assertTrue(np.isfinite(y).all())


if __name__=='__main__':unittest.main()
