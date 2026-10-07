import copy
import unittest

import numpy as np

from w2v_v39.metrics import measure
from .diagnostics import treatment_errors, panel_plan, panel_metrics
from .train import acceptance


class DiagnosticTests(unittest.TestCase):
    def test_exact_hash_join_distinguishes_preexisting_and_treatment_induced_errors(self):
        off=[dict(audio_sha256=str(i),source_sha256=str(i),source_id=str(i),condition='offline',
                  language='en',label=1) for i in range(4)]
        processed=[dict(r,condition='seen') for r in off]
        a=np.array([[0,1],[0,1],[1,0],[1,0]])
        b=np.array([[0,1],[1,0],[0,1],[1,0]])
        value=treatment_errors(off,a,processed,b)['groups']['seen/en/real']
        self.assertEqual(value['both_correct'],1);self.assertEqual(value['both_wrong'],1)
        self.assertEqual(value['original_correct_processed_wrong'],1)
        self.assertEqual(value['original_wrong_processed_correct'],1)
        processed[0]['source_sha256']='unverified'
        with self.assertRaisesRegex(ValueError,'No verified'):treatment_errors(off,a,processed,b)

    def test_panel_is_frozen_and_cannot_change_historical_weighted(self):
        rows=[dict(source_id=str(i),group_id=str(i),condition='offline',language=l,label=y)
              for i,(l,y) in enumerate((('en',0),('en',1),('zh',0),('zh',1)))]
        plan=panel_plan(rows,dict(panel_seed=315))
        self.assertEqual(plan,panel_plan(rows,dict(panel_seed=315)))
        z=np.array([[1,0],[0,1],[1,0],[0,1]])
        value=panel_metrics(rows,plan,z)
        self.assertEqual(value['f1'],1.);self.assertFalse(value['included_in_weighted'])
        self.assertNotIn('weighted_f1',value)

    def test_guard_requires_complete_panel_without_changing_checkpoint_ranking(self):
        rows=[];a=[];b=[]
        for c in ('online','seen','heldout'):
            for l in ('en','zh'):
                for y in (0,1):
                    for i in range(100):
                        rows.append(dict(condition=c,language=l,label=y))
                        before=(1-y) if i<5 else y;after=(1-y) if i<3 else y
                        a.append([1,0] if before==0 else [0,1]);b.append([1,0] if after==0 else [0,1])
        base,value=measure(rows,np.array(a)),measure(rows,np.array(b))
        cfg=dict(min_gain=.0002,max_noisy_drop=.0005,max_clean_drop=.003,max_fake_drop=.005,
            max_real_drop=.015,max_auc_drop=.002,max_matched_real_drop=.02,max_panel_drop=.003)
        eligible,reasons=acceptance(base,value,cfg,{'f1':.95},None)
        self.assertFalse(eligible);self.assertIn('robustness_panel_not_measured',reasons)
        self.assertTrue(acceptance(base,value,cfg,{'f1':.95},{'f1':.96})[0])
        self.assertIn('robustness_panel_regression',acceptance(base,value,cfg,{'f1':.95},{'f1':.90})[1])


if __name__=='__main__':unittest.main()
