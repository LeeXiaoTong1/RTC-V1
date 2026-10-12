"""Deterministic stress recipes, unchanged probes, native FFmpeg mechanisms."""
import unittest
from unittest.mock import patch
import numpy as np
from w2v_v318.augment import recipe as legacy_recipe
from .augment import recipe,Engines,TRAIN_FAMILIES


class AugmentTests(unittest.TestCase):
    def test_probes_and_noisy_a_unchanged_and_b_controlled(self):
        kinds=set();stress=0
        for i in range(400):
            for view in (0,1):
                args=(31801,f'occurrence{i}','source',view,2,0)
                self.assertEqual(recipe(*args,probe=True),legacy_recipe(*args,probe=True))
                self.assertEqual(recipe(*args,stress_probability=0),legacy_recipe(*args))
                r=recipe(*args);self.assertIn(r['family'],TRAIN_FAMILIES)
                if view==0:self.assertEqual(r,legacy_recipe(*args))
                elif r.get('stress'):stress+=1;kinds.add(r['stress']['mechanism'])
                self.assertEqual(r,recipe(*args))
        self.assertTrue(90<stress<180);self.assertEqual(kinds,{'gain_envelope','bandwidth','soft_limit','codec'})
        for family in ('anlmdn','g711_mulaw'):
            self.assertEqual(recipe(3,'x','s',0,0,0,'dev',family,True),legacy_recipe(3,'x','s',0,0,0,'dev',family,True))

    def test_native_bandwidth_gain_limit_and_opus_preserve_length(self):
        engine=Engines()
        x=np.random.default_rng(3).normal(0,.15,17403).astype(np.float32)
        for stress in (dict(mechanism='gain_envelope',seed=4,span_db=6.,period=.5),
                       dict(mechanism='bandwidth',highpass=100,lowpass=3400),
                       dict(mechanism='soft_limit',threshold=.4),dict(mechanism='codec')):
            r=recipe(1,'x','s',1,2,0,stress_probability=0)
            r.update(stress=stress,family='bypass',codec='opus' if stress['mechanism']=='codec' else 'none',bitrate=12000)
            y=engine(x,r)
            self.assertEqual(x.shape,y.shape);self.assertTrue(np.isfinite(y).all());self.assertFalse(np.array_equal(x,y))


if __name__=='__main__':unittest.main()
