import unittest
from unittest.mock import patch
import numpy as np
from .views import augment,recipe,FAMILIES
from .data import Waves,ready,prepared,_valid_pairs,microbatches


class DataTests(unittest.TestCase):
    def test_complete_content_and_exact_timing(self):
        x=np.linspace(-.4,.4,96000,dtype=np.float32)
        for variant,family in enumerate(FAMILIES):
            rec=recipe(1,'source','offline',variant);y,meta=augment(x,rec)
            self.assertFalse(meta['content_cropped']);self.assertEqual(meta['time_map']['offset'],0.)
            if family in ('speed_slow','speed_fast'):
                self.assertEqual(len(y),(len(x)*1000+rec['rate']-1)//rec['rate'])
                self.assertEqual(meta['time_map']['scale'],1000/rec['rate'])
            else:self.assertEqual(len(y),len(x))
        rec=recipe(1,'source','offline',4);y,_=augment(x,rec)
        self.assertNotEqual(y[0],0);self.assertNotEqual(y[-1],0)
        self.assertEqual(np.sign(y[0]),np.sign(x[0]));self.assertEqual(np.sign(y[-1]),np.sign(x[-1]))

    def test_reflection_preserves_original_prefix_and_normal_length(self):
        x=np.array([.1,.2,.3,.4],np.float32);y=ready(x)
        self.assertEqual(len(y),3920)
        np.testing.assert_array_equal(y[:10],x[[0,1,2,3,2,1,0,1,2,3]])
        long=np.linspace(-.1,.1,3920,dtype=np.float32)
        self.assertIs(ready(long),long)
        np.testing.assert_array_equal(ready(np.array([.3],np.float32)),np.full(3920,.3,np.float32))
        with self.assertRaises(ValueError):ready(np.array([],np.float32))

    def test_pcl_excludes_receptive_field_crossing_reflection(self):
        row=prepared({},np.zeros(1000,np.float32))
        # Frame centres 200 and 520 have real RFs; centre 840 would need
        # sample1040 and must not contribute although centre840 is original.
        self.assertEqual(row['pcl_valid_frames'],2)
        kept=_valid_pairs([[0,800,0,800],[800,1000,800,1000]],row,row,1)
        self.assertEqual(kept,[[0,800,0,800]])
        tiny=prepared({},np.zeros(399,np.float32))
        self.assertEqual(tiny['pcl_valid_frames'],0)
        self.assertEqual(_valid_pairs([[0,399,0,399]],tiny,tiny,1),[])

    def test_all_dataset_paths_reflect_after_generation_and_keep_metadata(self):
        source=np.linspace(-.1,.1,1000,dtype=np.float32)
        row=dict(id='off',source_id='off',group_id='g',language='en',label=1,split='train',audio='unused')
        cfg=dict(seed=1,warm_epochs=1,rtc_stress_probability=0.,rtc_stress_ramp_epochs=1.,scene_probability=1.,rtc_policy='three-profile')
        ds=Waves([row],cfg);ds.original=lambda r:source.copy();ds.bank=object();ds.engines=object()
        rec=dict(family='bypass',codec='none')
        def generate(x,*args):
            self.assertEqual(len(x),1000)
            return x*.5,{'recipe':rec}
        scheduled=dict(index=0,kind='offline',base_variant=0,base_family='original',base_generation=0,
            base_dynamic=False,base_view_id='view',occurrence='visit',epoch=0,pcl_enabled=False,
            batch_group_quotas=(7,6,5,6),pcl_platform_budget=0)
        with patch('w2v_v3187.data.recipe',return_value=rec),patch('w2v_v3187.data.generate',side_effect=generate),\
                patch('w2v_v3187.data.generate_matched',side_effect=AssertionError('No warmup reference')):
            noise=ds[scheduled]
            probe=ds[dict(index=0,kind='offline',occurrence='probe',epoch=0,probe=True)]
        infer=ds[0]
        online=ds[dict(scheduled,kind='online')]
        for item in noise+probe+infer+online:
            self.assertEqual(item['original_samples'],1000);self.assertEqual(len(item['wave']),3920)
            self.assertEqual(item['pcl_valid_frames'],2)
        self.assertEqual(noise[0]['batch_group_quotas'],(7,6,5,6))
        np.testing.assert_array_equal(noise[0]['wave'][:1000],source*.5)

    def test_short_pcl_pairs_reject_only_artificial_support(self):
        source=np.random.default_rng(2).normal(0,.1,1000).astype(np.float32)
        base=dict(source_id='off',group_id='g',language='en',label=1,split='train',audio='unused')
        rows=[dict(base,id='off',condition='offline'),dict(base,id='on',condition='online')]
        cfg=dict(seed=1,warm_epochs=1,rtc_stress_probability=0.,rtc_stress_ramp_epochs=1.,scene_probability=1.,rtc_policy='three-profile')
        ds=Waves(rows,cfg);ds.original=lambda r:source.copy();ds.bank=object();ds.engines=object()
        phones=[dict(token='a',instance=0,start_sample=0,end_sample=800),dict(token='b',instance=1,start_sample=800,end_sample=1000)]
        ds._phones=lambda row:phones
        ticket=dict(index=0,kind='offline',base_variant=0,base_family='original',base_generation=0,
            base_dynamic=False,base_view_id='view',occurrence='visit',epoch=1,pcl_enabled=True,
            pcl_noisy_index=0,pcl_noise=True,pcl_online_index=1,batch_group_quotas=(6,6,6,6),pcl_platform_budget=1)
        meta={'pair_eligible':True};delay={'valid':True,'offset':0.}
        with patch('w2v_v3187.data.recipe',return_value={}),\
                patch('w2v_v3187.data.generate',return_value=(source.copy(),meta)),\
                patch('w2v_v3187.data.generate_matched',return_value=(source.copy(),source.copy(),meta)),\
                patch('w2v_v3187.data.estimate_delay',return_value=delay),\
                patch('w2v_v3187.data.reliable_spans',return_value=[True,True]):
            units=ds[ticket]
        self.assertEqual(units[0]['pcl_pairs']['noise'],[[0,800,0.,800.]])
        self.assertEqual(units[0]['pcl_pairs']['platform'],[[0,800,0,800]])
        self.assertEqual(units[0]['pcl_coverage']['noise_original_frame_rejected'],1)

    def test_length_packing_retains_atomic_units_and_counts_only_active_refs(self):
        def row(identity,n,group=('en',0)):
            return dict(id=identity,source_id=identity,language=group[0],label=group[1],role='online',wave=np.zeros(n,np.float32))
        units=[[row('a',32000)],[row('b',8000)],[row('c',32000)],[row('d',8000)]]
        batches=list(microbatches(units,2,200))
        self.assertEqual(len(batches),2)
        self.assertEqual({tuple(u[0]['id'] for u in b) for b in batches},{('a','c'),('b','d')})
        main=row('e',16000)
        main.update(pcl_refs={'clean':row('e',16000),'official':row('e',16000)},
            pcl_pairs={'noise':[[1,2,1,2]],'platform':[[1,2,1,2]]})
        unit=[main,row('e',16000)]
        self.assertEqual(len(list(microbatches([unit],2,1000,weights={'noise':0,'platform':0}))),1)
        with self.assertRaisesRegex(ValueError,'complete source/PCL unit'):
            list(microbatches([unit],3,1000))

    def test_length_ties_mix_groups(self):
        units=[]
        for lang,label in (('en',0),('en',1),('zh',0),('zh',1)):
            for i in range(4):units.append([dict(id=f'{lang}{label}{i}',source_id=f'{lang}{label}{i}',
                language=lang,label=label,role='online',wave=np.zeros(16000,np.float32))])
        for batch in microbatches(units,4,1000):
            self.assertEqual(len({(u[0]['language'],u[0]['label']) for u in batch}),4)


if __name__=='__main__':unittest.main()
