"""Behavioral checks for targeted version budgets and unchanged stream semantics."""
from collections import Counter,defaultdict
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from w2v_v3182.test_support import metadata
from w2v_v3183r2.sampling import Plan as SourcePlan
from w2v_v3184_aug5.views import recipe as previous_recipe,augment as previous_augment
from w2v_v318.test_core import make_fixture,NativeBoundary
from w2v_v3184.test_support import scene_fixture
from .test_support import tiny_cfg
from .sampling import Plan,DEFAULT_VIEW_FACTORS
from .views import recipe,augment,FAMILIES


class AugmentationTests(unittest.TestCase):
    def test_fixed_library_scope_and_balanced_exposure(self):
        rows=metadata((40,40,40,40),missing=False);plan=Plan(rows)
        for kind,pools in plan.pools.items():
            for group,pool in pools.items():
                factor=plan.view_factors[group];families=Counter()
                for index in pool:
                    specs=[plan.view_spec(index,kind,c) for c in range(10)]
                    for start in range(6):
                        self.assertEqual(sum(s['base_family']=='original' for s in specs[start:start+5]),1)
                    for s in specs[:5]:families[s['base_family']]+=1
                    fixed={s['base_view_id'] for s in specs if not s['base_dynamic']}
                    self.assertEqual(len(fixed),factor)
                    if group==('zh',0):
                        self.assertEqual({s['base_variant'] for s in specs},{0})
                        self.assertEqual(sum(s['base_dynamic'] for s in specs),8)
                        self.assertEqual(len({s['base_view_id'] for s in specs}),9)
                    else:
                        self.assertFalse(any(s['base_dynamic'] for s in specs))
                        self.assertEqual({s['base_variant'] for s in specs},set(range(factor)))
                    if group==('zh',1):
                        self.assertEqual(len(plan.definitions[index]['families']),2)
                        self.assertEqual(len(set(plan.definitions[index]['families'])),2)
                self.assertEqual(families,{family:len(pool) for family in FAMILIES})

    def test_original_source_order_steps_and_replay_preserved(self):
        rows=metadata((17,7,31,11),missing=False)
        plan=Plan(rows,16,3,(4,3,5,4));previous=SourcePlan(rows,16,3,(4,3,5,4))
        self.assertEqual(plan.steps,previous.steps)
        for epoch in range(4):
            for kind,groups in plan.pools.items():
                for group in groups:self.assertEqual(plan.stream(kind,group,epoch),previous.stream(kind,group,epoch))
            batches=list(plan.batches(epoch))
            for batch in batches:
                self.assertEqual(len(batch),32)
                self.assertEqual(len({(t['kind'],t['index']) for t in batch}),32)
                self.assertEqual(Counter((t['kind'],rows[t['index']]['language'],rows[t['index']]['label']) for t in batch),
                    {(k,*g):q for k in plan.pools for g,q in plan.quotas.items()})
            # Reconstructed from epoch number; independent of prior iteration.
            self.assertEqual(batches,list(Plan(rows,16,3,(4,3,5,4)).batches(epoch)))

    def test_coverage_matches_realized_history_without_replay(self):
        rows=metadata((17,7,31,11),missing=False)
        for refresh in (False,True):
            plan=Plan(rows,16,3,(4,3,5,4),refresh_base_variants=refresh)
            history=defaultdict(list)
            for epoch in range(7):
                for batch in plan.batches(epoch):
                    for t in batch:history[(t['kind'],rows[t['index']]['language'],rows[t['index']]['label'])].append(t)
                report=plan.coverage(epoch)
                for key,tickets in history.items():
                    cell=report['cells']['/'.join(map(str,key))]
                    current=[t for t in tickets if t['epoch']==epoch]
                    self.assertEqual(cell['unique_realized_views_through_epoch'],len({t['base_view_id'] for t in tickets}))
                    self.assertEqual(cell['unique_dynamic_views_through_epoch'],len({t['base_view_id'] for t in tickets if t['base_dynamic']}))
                    self.assertEqual(cell['unique_fixed_slots_through_epoch'],len({(t['index'],t['base_variant']) for t in tickets if not t['base_dynamic']}))
                    self.assertEqual(cell['unique_virtual_views_this_epoch'],len({t['base_view_id'] for t in current}))
                    self.assertEqual(cell['original_draws'],sum(t['base_family']=='original' for t in current))
                    self.assertEqual(cell['base_families'],Counter(t['base_family'] for t in current))

    def test_refresh_happens_only_after_nonoriginal_rotation(self):
        rows=metadata((17,7,31,11),missing=False)
        fixed=Plan(rows);refresh=Plan(rows,refresh_base_variants=True)
        for kind,pools in refresh.pools.items():
            for group,pool in pools.items():
                if group==('zh',0):continue
                count=refresh.view_factors[group]-1;index=pool[0]
                augmented=[refresh.view_spec(index,kind,c) for c in range(20)
                           if refresh.view_spec(index,kind,c)['base_family']!='original']
                for generation in range(2):
                    block=augmented[generation*count:(generation+1)*count]
                    self.assertEqual({s['base_variant'] for s in block},set(range(1,count+1)))
                    self.assertEqual({s['base_generation'] for s in block},{generation})
                self.assertTrue(all(fixed.view_spec(index,kind,c)['base_generation']==0 for c in range(20)))

    def test_waveform_family_not_slot_controls_transform(self):
        x=(.1*np.sin(np.arange(64000)*.033)).astype(np.float32)
        for family in FAMILIES[1:]:
            dynamic=recipe(7,'source','online',0,family=family,dynamic=True)
            a,_=augment(x,dynamic);b,_=augment(x,dynamic)
            np.testing.assert_array_equal(a,b)
            self.assertNotEqual(a.tobytes(),x.tobytes())
            self.assertTrue(np.isfinite(a).all())
        for variant in range(5):
            a,_=augment(x,recipe(7,'source','online',variant))
            b,_=previous_augment(x,previous_recipe(7,'source','online',variant))
            np.testing.assert_array_equal(a,b)
        r=recipe(7,'source','offline',1,family='frequency_response')
        self.assertEqual(r['family'],'frequency_response')
        a,_=augment(x,r)
        self.assertEqual(len(a),len(x))
        self.assertNotEqual(recipe(7,'source','online',1)['seed'],recipe(7,'source','online',1,generation=1)['seed'])

    def test_rtc_pair_shares_one_transformation_and_fixed_inputs_unchanged(self):
        from .data import Waves
        from w2v_v3184.data import Waves as OldWaves
        with tempfile.TemporaryDirectory() as d,patch('w2v_v3185.data.Engines',NativeBoundary):
            root=Path(d);cfg,train,dev=make_fixture(root/'audio');cfg.update(tiny_cfg());cfg.update(scene_fixture(root/'scenes'))
            data=Waves(train,cfg);index=next(i for i,r in enumerate(train) if r['condition']=='offline')
            t=dict(index=index,kind='offline',epoch=0,step=0,occurrence='x',base_variant=0,
                base_family='speed_fast',base_dynamic=True,base_generation=2,base_view_id='auditable-dynamic')
            seen=[]
            def generate(x,r,*args):seen.append(x.copy());return x.copy(),dict(recipe=r)
            with patch('w2v_v3185.data.generate',generate),patch('w2v_v3185.data.augment',wraps=augment) as transform:
                views=data[t]
                self.assertEqual(transform.call_count,1)
            expected,_=augment(data.original(train[index]),recipe(cfg['seed'],train[index]['id'],'offline',0,
                family='speed_fast',dynamic=True,generation=2))
            self.assertEqual(len(seen),2)
            self.assertEqual({v['role'] for v in views},{'noisy_a','noisy_b'})
            for wave in seen:np.testing.assert_array_equal(wave,expected)
            self.assertEqual(views[0]['base_augmentation'],views[1]['base_augmentation'])
            self.assertEqual(views[0]['view_id'],'auditable-dynamic')
            index=next(i for i,r in enumerate(train) if r['condition']=='online')
            t.update(index=index,kind='online')
            with patch('w2v_v3185.data.generate') as noisy:
                self.assertEqual(data[t][0]['role'],'online');noisy.assert_not_called()
            probe=dict(index=index,kind='online',epoch=0,step=0,occurrence='probe',probe=True)
            np.testing.assert_array_equal(data[probe][0]['wave'],OldWaves(train,cfg)[probe][0]['wave'])
            self.assertNotIn('base_augmentation',data[probe][0])
            new_dev=Waves(dev,cfg,'dev');old_dev=OldWaves(dev,cfg,'dev')
            np.testing.assert_array_equal(new_dev[0][0]['wave'],old_dev[0][0]['wave'])

    def test_invalid_budgets_rejected(self):
        rows=metadata((17,7,31,11),missing=False)
        for factors in ((5,5,1),(5,5,2,3),(True,5,1,3)):
            with self.assertRaises(ValueError):Plan(rows,view_factors=factors)
        self.assertEqual(DEFAULT_VIEW_FACTORS,(5,5,1,3))


if __name__=='__main__':unittest.main()
