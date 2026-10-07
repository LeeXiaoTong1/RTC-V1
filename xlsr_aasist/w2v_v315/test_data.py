import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from w2v_v314.test_core import rows_fixture
from .augment import recipe, settings_grid, common_inputs, generate, noise_wave, seed_for
from .cache import RollingCache
from .data import SourcePlan, validate_batch, TripletCollator


class Bank:
    def sample(self,rng):
        # Shorter than the utterance: must remain usable in every noise mode.
        return np.sin(np.arange(1300)*.13).astype(np.float32),'noise_hash'


class RecordingEngine:
    def __init__(self):self.calls=[]
    def __call__(self,x,r):
        self.calls.append((x.copy(),copy.deepcopy(r)))
        return np.tanh(x*1.2).astype(np.float32)


class DataTests(unittest.TestCase):
    def test_balanced_three_view_source_budget_resume_and_missing_online(self):
        rows=rows_fixture(17);plan=SourcePlan(rows,16,31501)
        for epoch in (0,1,2):
            batches=plan.batches(epoch)
            self.assertEqual(batches[1:],plan.batches(epoch,1))
            for batch in batches:
                examples=[]
                for t in batch:
                    for role,mass in (('ordinary',.9-t.noisy_mass),('reference',.1),('noisy',t.noisy_mass)):
                        examples.append(dict(rows[t.original],role=role,ce_weight=mass/16,pair_occurrence=t.occurrence))
                    self.assertIn(rows[t.ordinary]['condition'],('offline','online'))
                validate_batch(examples,16)
                self.assertEqual(len(examples),48)
        self.assertEqual(sum(plan.coverage()['processing_families'].values()),plan.steps*16)

    def test_settings_disjoint_and_recipes_repeat_without_labels(self):
        for family in ('ffmpeg','webrtc','light'):
            a={json.dumps(x,sort_keys=True) for x in settings_grid(family,'train')}
            b={json.dumps(x,sort_keys=True) for x in settings_grid(family,'dev')}
            self.assertFalse(a&b)
        a=recipe(315,'0:10:2',42,'a'*64)
        self.assertEqual(a,recipe(315,'0:10:2',42,'a'*64))
        self.assertNotEqual(a,recipe(315,'1:10:2',42,'a'*64))
        self.assertNotIn('label',a);self.assertNotIn('language',a)

    def test_shared_gain_measured_local_snr_and_identical_processing_settings(self):
        rng=np.random.default_rng(21);x=rng.normal(0,.2,17031).astype(np.float32)
        noise=rng.normal(0,1,len(x)).astype(np.float32)
        a,b,meta=common_inputs(x,noise,13.,3.)
        self.assertAlmostEqual(meta['measured_input_snr_db'],13.,places=6)
        np.testing.assert_allclose(a,x*meta['common_gain'],rtol=1e-6,atol=1e-7)
        self.assertLessEqual(max(abs(a).max(),abs(b).max()),.970001)
        r=recipe(5,'source',2,'a'*64);engine=RecordingEngine()
        aa,bb,info=generate(x,r,Bank(),engine)
        self.assertEqual(engine.calls[0][1],engine.calls[1][1])
        self.assertEqual(len(aa),len(bb));self.assertTrue(info['same_settings'])
        expected=len(x)+(r['echo']['delay_samples'] if r['echo'] else 0)
        self.assertEqual(len(aa),expected)

    def test_short_noise_usable_for_long_utterance_and_events_remain_local(self):
        for kind in ('continuous','events','mixed'):
            n,used=noise_wave(Bank(),32000,kind,np.random.default_rng(3))
            self.assertEqual(n.shape,(32000,));self.assertTrue(np.isfinite(n).all())
            self.assertGreater(np.count_nonzero(n),0);self.assertEqual(used,['noise_hash'])
            if kind=='events':self.assertLess(np.count_nonzero(n),6000)

    def test_short_silence_is_bounded_recorded_and_does_not_blank_full_clip(self):
        x=np.sin(np.arange(32000)*.13).astype(np.float32)*.2
        r=recipe(1,'a',0,'a'*64);r['echo']=None
        r['dropout']=dict(position=.5,seconds=.1,gain=0.)
        a,b,meta=generate(x,r,Bank(),RecordingEngine())
        lo,hi=meta['erased_spans'][0]
        self.assertLessEqual(hi-lo,len(x)//50)
        self.assertTrue((b[lo:hi]==0).all());self.assertGreater(np.count_nonzero(b),len(b)*.8)

    def test_cache_hard_cap_eviction_corrupt_rebuild_and_owner_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            cache=RollingCache(Path(d)/'cache','run-A',7000)
            a=np.arange(250,dtype=np.float32)
            for i in range(12):
                key=f'{i:064x}';cache.put(key,a,a+1,{'source':i})
                self.assertLessEqual(cache.used_bytes(),7000)
            self.assertIsNone(cache.get(f'{0:064x}'))
            value=cache.get(f'{11:064x}');np.testing.assert_array_equal(value[0],a)
            (Path(d)/'cache'/(f'{11:064x}'+'.npz')).write_bytes(b'corrupt')
            self.assertIsNone(cache.get(f'{11:064x}'))
            with self.assertRaisesRegex(ValueError,'different run'):RollingCache(Path(d)/'cache','run-B',7000)
            important=Path(d)/'cache'/'important.txt';important.write_text('keep')
            with self.assertRaises(ValueError):cache.put(f'{15:064x}',a,a,{})
            self.assertEqual(important.read_text(),'keep')


if __name__=='__main__':unittest.main()
