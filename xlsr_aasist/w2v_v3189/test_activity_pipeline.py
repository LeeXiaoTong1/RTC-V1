import unittest
from unittest.mock import patch
import numpy as np
from .augment import endpoint,generate,generate_matched,recipe
from .activity import settings
from .test_activity import speech_fixture


class SpyEngine:
    def __init__(self):self.calls=[]
    def __call__(self,wave,rec):
        self.calls.append((wave.copy(),dict(rec)))
        return wave.copy()


class PipelineTests(unittest.TestCase):
    def test_gate_between_processing_and_exactly_one_codec_call(self):
        x=speech_fixture();e=SpyEngine()
        rec=dict(family='webrtc',codec='opus',activity=settings(1,'strong'))
        y,m=endpoint(x,rec,e)
        self.assertEqual([(r['family'],r['codec']) for _,r in e.calls],[('webrtc','none'),('bypass','opus')])
        np.testing.assert_array_equal(e.calls[0][0],x)
        np.testing.assert_array_equal(e.calls[1][0],y)
        self.assertGreater(m['suppressed_samples'],0)

    def test_off_is_exact_legacy_and_matched_states_are_independent(self):
        x=speech_fixture();e=SpyEngine();rec=dict(family='light',codec='opus',activity=settings(1,'off'))
        y,m=endpoint(x,rec,e)
        self.assertEqual(len(e.calls),1);np.testing.assert_array_equal(y,x)
        def base(w,r,bank,receive):
            noisy=receive(w+np.random.default_rng(1).normal(0,.035,len(w)).astype(np.float32),r)
            return noisy,dict(common_gain=1.,pair_eligible=True,recipe=r)
        rec['activity']=settings(1,'strong')
        with patch('w2v_v3189.augment.base_generate',side_effect=base):
            clean,noisy,meta=generate_matched(x,rec,None,e)
        self.assertEqual(clean.shape,noisy.shape)
        self.assertNotEqual(meta['clean_activity']['suppressed_spans'],meta['rtc_activity']['suppressed_spans'])
        self.assertEqual(meta['erased_spans'],meta['rtc_activity']['suppressed_spans'])
        self.assertTrue(meta['same_settings'])

    def test_training_recipe_active_fixed_probe_unchanged_and_control_off(self):
        r=recipe(1,'train','src',0,5,0,rtc_policy='three-profile',activity_profile='strong')
        self.assertEqual(r['activity']['profile'],'strong')
        r=recipe(1,'probe','src',0,5,0,probe=True,rtc_policy='three-profile',activity_profile='strong')
        self.assertEqual(r['activity']['profile'],'off')

    def test_dataset_masks_both_noise_ends_and_platform_clean_end(self):
        from .data import Waves
        source=np.random.default_rng(1).normal(0,.1,8000).astype(np.float32)
        base=dict(source_id='off',group_id='g',language='en',label=1,split='train',audio='unused')
        rows=[dict(base,id='off',condition='offline'),dict(base,id='on',condition='online')]
        cfg=dict(seed=1,warm_epochs=0,rtc_stress_probability=0.,rtc_stress_ramp_epochs=1.,
                 scene_probability=1.,rtc_policy='three-profile')
        ds=Waves(rows,cfg);ds.original=lambda _:source.copy();ds.bank=object();ds.engines=object()
        ds._phones=lambda _:[dict(token='a',instance=0,start_sample=0,end_sample=3200),
                            dict(token='b',instance=1,start_sample=4000,end_sample=8000)]
        ticket=dict(index=0,kind='offline',base_variant=0,base_family='original',base_generation=0,
            base_dynamic=False,base_view_id='view',occurrence='visit',epoch=1,pcl_enabled=True,
            pcl_noise=True,pcl_online_index=1)
        meta=dict(pair_eligible=True,clean_activity={'suppressed_spans':[[0,4000]]},
                  rtc_activity={'suppressed_spans':[[6000,8000]]})
        with patch('w2v_v3189.data.recipe',return_value={}), \
             patch('w2v_v3189.data.generate_matched',return_value=(source.copy(),source.copy(),meta)), \
             patch('w2v_v3189.data.estimate_delay',return_value={'valid':True,'offset':0.}), \
             patch('w2v_v3189.data.reliable_spans',side_effect=lambda a,b,spans,*_: [True]*len(spans)):
            first=ds[ticket][0]
        self.assertEqual(first['pcl_pairs']['noise'],[[4200.,5800.,4200.,5800.]])
        self.assertEqual(first['pcl_pairs']['platform'],[[4200.,8000.,4200.,8000.]])
        self.assertEqual(first['pcl_coverage']['noise_activity_affected_phones'],2)
        self.assertEqual(first['pcl_coverage']['platform_activity_affected_phones'],2)
        r=recipe(1,'train','src',0,5,0,rtc_policy='three-profile',activity_policy='off')
        self.assertEqual(r['activity']['profile'],'off')

    def test_native_codec_preserves_geometry_and_pause_contrast(self):
        from w2v_v318.runtime import bundled_ffmpeg
        from w2v_v3184.augment import Engines
        ffmpeg=bundled_ffmpeg();e=Engines(ffmpeg);x=speech_fixture()
        for family in ('light','ffmpeg'):
            rec=recipe(3,'native','src',0,5,0,rtc_policy='three-profile',activity_profile='strong')
            rec.update(family=family,codec='opus')
            off=dict(rec,activity=settings(3,'off'))
            y,m=endpoint(x,rec,e);baseline,_=endpoint(x,off,e)
            self.assertEqual(len(y),len(x));self.assertTrue(np.isfinite(y).all())
            self.assertGreater(m['suppressed_samples'],0)
            # Real Opus encode/decode, not a mock: pause must become quieter.
            self.assertLess(np.mean(y[13800:14800]**2),np.mean(baseline[13800:14800]**2)*.3)


if __name__=='__main__':unittest.main()
