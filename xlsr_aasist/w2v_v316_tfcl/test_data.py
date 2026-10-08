"""Actual waveform/feature data path: retained sources, mapping masks, no audio cache."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from transformers import SeamlessM4TFeatureExtractor
from w2v_v315.augment import generate,recipe
from w2v_v3151.test_data import rows_fixture,Bank,Engine,wave_row,echo_loader
from . import data
from .data import SourcePlan,RawWaveCache,Triplets,TripletCollator,TrainingLoader,validate_batch,generate_noisy


class DataTests(unittest.TestCase):
    def test_all_sources_retained_and_each_group_has_full_classification_budget(self):
        rows=rows_fixture(65);plan=SourcePlan(rows,16,31601)
        coverage=plan.coverage()
        self.assertEqual(coverage['available_sources'],260)
        self.assertEqual(coverage['missing_online_sources'],128)
        self.assertEqual(coverage['excluded_sources'],0)
        for epoch in range(2):
            batches=plan.batches(epoch);self.assertEqual(batches[2:5],plan.batches(epoch,2,5))
            audit=plan.coverage(epoch)
            self.assertEqual(audit['missing_online_draws'],sum(t.online is None for b in batches for t in b))
            self.assertEqual(audit['unique_sources'],len({t.original for b in batches for t in b}))
            for batch in batches:
                examples=[]
                for t in batch:
                    weights={'offline':.1,'online':.5,'noisy':.4} if t.online is not None else {'offline':.2,'noisy':.8}
                    for role,mass in weights.items():
                        index=t.online if role=='online' else t.original
                        examples.append(dict(rows[index],role=role,ce_weight=mass/16,pair_occurrence=t.occurrence))
                validate_batch(examples,16)
        broken=copy.deepcopy(rows);next(r for r in broken if r['condition']=='online')['source_sha256']='wrong'
        with self.assertRaises(ValueError):SourcePlan(broken,16,31601)

    def test_unused_clean_rtc_removed_but_noisy_wave_is_bit_exact(self):
        wave=np.sin(np.arange(32001,dtype=np.float32)*.087)*.15
        for phase in range(12):
            r=recipe(31601,'example',phase,'sha',warm=1.)
            oldref,old,meta=generate(wave,r,Bank(),Engine())
            calls=[]
            def engine(x,r):calls.append(len(x));return Engine()(x,r)
            actual,m=generate_noisy(wave,r,Bank(),engine)
            np.testing.assert_array_equal(actual,old);self.assertEqual(len(calls),1)
            self.assertEqual(m['erased_spans'],meta['erased_spans'])
            self.assertEqual(m['original_samples'],len(wave))

    def test_actual_waveforms_and_features_exclude_only_auxiliary_missing_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);rows=rows_fixture(1)
            for i,row in enumerate(rows):
                if row['condition'] not in ('offline','online'):continue
                wave=np.sin(np.arange(32000,dtype=np.float32)*.1+i*.17)*.15
                row.update(wave_row(folder/f'{i}.wav',wave))
                if row['condition']=='offline':
                    for other in rows:
                        if other['source_id']==row['source_id']:other.update(group_id=row['audio_sha256'],source_sha256=row['audio_sha256'])
            plan=SourcePlan(rows,4,31601);ticket=plan.batches(0)[0][0]
            r=json.loads(ticket.recipe_json);r.update(echo=dict(delay_samples=3200,gain=.2),dropout=dict(position=.5,seconds=.1,gain=0.))
            ticket=replace(ticket,recipe_json=json.dumps(r));cfg=dict(source_batch=4,rolling_cache_bytes=0)
            ds=Triplets(rows,cfg,folder);ds.bank,ds.engines,ds.raw=Bank(),Engine(),RawWaveCache(1000000)
            views=ds[ticket];self.assertEqual([r['role'] for r in views],['offline','online','noisy'])
            self.assertEqual([len(r['wave']) for r in views],[32000,32000,35200])
            processor=folder/'processor';SeamlessM4TFeatureExtractor().save_pretrained(processor)
            features=TripletCollator(processor)([views])
            noisy=features[-1];self.assertGreater(noisy['mask'].sum(),noisy['aux_valid'].sum())
            self.assertFalse(noisy['aux_valid'][-4:].any())
            self.assertTrue(features[0]['aux_valid'].all());self.assertTrue(features[1]['aux_valid'].all())
            self.assertFalse((folder/'rolling_pairs').exists())
            missing=ds[replace(ticket,online=None)]
            self.assertEqual([r['role'] for r in missing],['offline','noisy'])
            self.assertEqual([r['ce_weight'] for r in missing],[.2/4,.8/4])
            with self.assertRaisesRegex(ValueError,'Invalid official pair'):ds[replace(ticket,online=ticket.original)]

    def test_persistent_workers_keep_same_process_and_resume_tickets(self):
        plan=SourcePlan(rows_fixture(25),16,31601)
        with patch.object(data,'_loader',side_effect=echo_loader):
            stream=TrainingLoader(plan,{},'.')
            try:
                first=list(stream.segment(0,0,2));second=list(stream.segment(0,2,4))
                self.assertEqual({x[0] for batch in first for x in batch},{x[0] for batch in second for x in batch})
                wanted=[(t.occurrence,t.recipe_json) for b in plan.batches(0,2,4) for t in b]
                self.assertEqual([(x[1],x[2]) for batch in second for x in batch],wanted)
            finally:stream.close()


if __name__=='__main__':unittest.main()
