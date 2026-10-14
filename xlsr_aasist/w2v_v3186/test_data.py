import unittest
from unittest.mock import patch
import numpy as np
from w2v_v3185.views import augment as old_augment
from .views import augment, recipe
from .augment import generate_matched, estimate_delay, reliable_spans
from .data import microbatches
from .data import Waves
from w2v_v3185.data import Waves as OldWaves


class DataTests(unittest.TestCase):
    def test_base_waveforms_unchanged_and_maps_exact(self):
        wave=np.random.default_rng(2).normal(0,.1,96000).astype(np.float32)
        for variant in range(5):
            r=recipe(31801,'source','offline',variant)
            a,meta=augment(wave,r);b,_=old_augment(wave,r)
            np.testing.assert_array_equal(a,b)
            mapping=meta['time_map']
            if variant in (1,2):self.assertEqual(mapping['scale'],1000/r['rate'])
            elif variant==4 and not meta['fallback']:
                start=-int(mapping['offset'])
                np.testing.assert_array_equal(a,wave[start:start+len(a)])
            else:self.assertEqual(mapping['scale'],1.)

    def test_crop_fallback_maps_original_not_attempted_crop(self):
        x=np.zeros(64000,np.float32);x[:2000]=.4
        r=recipe(1,'x','offline',4);r.update(keep=.85,start=1.)
        y,meta=augment(x,r)
        self.assertIsNotNone(meta['fallback'])
        self.assertEqual(meta['time_map']['offset'],0.)
        self.assertEqual(len(x),len(y))

    def test_matched_clean_reuses_gain_recipe_without_second_noise_draw(self):
        x=np.arange(16000,dtype=np.float32)/32000
        rec={'family':'fake','codec':'none','seed':12};calls=[]
        noisy=x*.25+.01
        def engine(w,r):calls.append((w.copy(),r));return w.copy()
        with patch('w2v_v3186.augment.generate',return_value=(noisy,dict(common_gain=.25,pair_eligible=True,recipe=rec))) as generated:
            clean,result,meta=generate_matched(x,rec,None,engine)
        generated.assert_called_once();np.testing.assert_array_equal(noisy,result)
        np.testing.assert_array_equal(clean,x*.25)
        self.assertEqual(calls[0][1],rec);self.assertTrue(meta['fresh_state_per_utterance'])

    def test_scene_echo_perturbs_noisy_only(self):
        x=np.random.default_rng(1).normal(0,.1,16000).astype(np.float32)
        rec=dict(category='echo',echo={'delay_samples':640,'gain':.2},input_gain_db=0.,family='bypass',codec='none')
        calls=[]
        def engine(w,r):calls.append(w.copy());return w.copy()
        clean,noisy,meta=generate_matched(x,rec,None,engine)
        self.assertEqual(len(calls),2)
        np.testing.assert_array_equal(clean,x*meta['common_gain'])
        self.assertFalse(np.array_equal(clean,noisy))
        self.assertFalse(meta['clean_has_scene_echo'])

    def test_actual_delay_and_deleted_spans(self):
        x=np.random.default_rng(1).normal(0,.1,16000).astype(np.float32)
        delay=160;y=np.pad(x,(delay,0))[:len(x)]
        found=estimate_delay(x,y)
        self.assertTrue(found['valid']);self.assertEqual(found['offset'],delay)
        y[4160:6160]=0
        mask=reliable_spans(x,y,[(1000,2000),(4000,6000)],0,delay)
        self.assertEqual(mask,[True,False])
        self.assertFalse(estimate_delay(x,np.zeros_like(x))['valid'])

    def test_phone_mask_downsamples_each_waveform_only_once(self):
        from scipy.signal import resample_poly as actual_resample
        x=np.random.default_rng(2).normal(0,.1,32000).astype(np.float32)
        spans=[(i,i+1000) for i in range(1000,29000,1000)]
        with patch('w2v_v3186.augment.resample_poly',wraps=actual_resample) as sample:
            mask=reliable_spans(x,x,spans)
        self.assertTrue(all(mask));self.assertEqual(sample.call_count,2)

    def test_auxiliary_waves_count_towards_microbatch(self):
        wave=np.zeros(16000,np.float32)
        unit=[{'wave':wave,'pcl_refs':{'clean':{'wave':wave},'official':{'wave':wave}},
            'pcl_pairs':{'noise':[[1,2,1,2]],'platform':[[1,2,1,2]]}}, {'wave':wave}]
        self.assertEqual(len(list(microbatches([unit,unit],4,1000))),2)
        with self.assertRaises(ValueError):list(microbatches([unit],3,1000))

    def test_dataset_preserves_noisy_ce_waves_and_sample_pair_contract(self):
        rng=np.random.default_rng(22);source=rng.normal(0,.1,16000).astype(np.float32)
        meta=dict(source_id='off',group_id='g',source_sha256='g',language='en',label=1,split='train')
        rows=[dict(meta,id='off',condition='offline',audio='off'),dict(meta,id='on',condition='online',audio='on')]
        cfg=dict(seed=31801,warm_epochs=0,rtc_stress_probability=0.,rtc_stress_ramp_epochs=1.,
            scene_probability=1.,rtc_policy='three-profile')
        ticket=dict(index=0,kind='offline',base_variant=0,base_family='original',base_generation=0,
            base_dynamic=False,base_view_id='off-test',occurrence='visit',epoch=0,pcl_enabled=True,
            pcl_noisy_index=1,pcl_noise=True,pcl_online_index=1)
        def rec(*args,**kwargs):
            return dict(category='echo',echo={'delay_samples':640,'gain':.2},input_gain_db=0.,family='bypass',codec='none')
        new=Waves(rows,cfg);old=OldWaves(rows,cfg)
        for ds in (new,old):
            ds.original=lambda r:source.copy();ds.bank=object();ds.engines=lambda w,r:w.copy()
        phones=[dict(token='a',instance=0,start_sample=1000,end_sample=3000,posterior=.9),
            dict(token='b',instance=1,start_sample=6000,end_sample=8000,posterior=.9)]
        new._phones=lambda row:phones
        with patch('w2v_v3186.data.recipe',side_effect=rec),patch('w2v_v3185.data.recipe',side_effect=rec):
            actual=new[ticket];expected=old[ticket]
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a['wave'],b['wave'])
        self.assertEqual(len(actual),2)
        self.assertEqual(set(actual[0]['pcl_refs']),{'clean','official'})
        self.assertEqual(actual[0]['pcl_noisy_index'],1)
        self.assertEqual(len(actual[0]['pcl_pairs']['noise']),2)
        self.assertEqual(actual[0]['pcl_pairs']['platform'],[[1000,3000,1000,3000],[6000,8000,6000,8000]])


if __name__=='__main__':unittest.main()
