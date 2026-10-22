import copy
import unittest
from unittest.mock import patch
import numpy as np
from .data import Waves
from .sampling import Plan, annotate
from w2v_v5.sampling import Plan as ParentPlan
from w2v_v5.data import Waves as ParentWaves
from w2v_v5.test_integration import inventory, ticket, fake_augment, fake_matched


class DataTests(unittest.TestCase):
    def test_full_epoch_parent_source_order_and_exposure_unchanged(self):
        rows, cfg = inventory()
        cfg['stream_sources'] = 16
        cfg['source_draws_per_stream'] = [16]*4
        cfg['group_quotas'] = (4,)*4
        a, b = ParentPlan(rows,16,50001,quotas=(4,)*4,cfg=cfg), Plan(rows,16,50001,quotas=(4,)*4,cfg=cfg)
        for epoch in (0,1):
            groups = 0
            for old,new in zip(a.batches(epoch),b.batches(epoch)):
                stripped = [{k:v for k,v in t.items() if not k.startswith('hp_')} for t in new]
                self.assertEqual(old,stripped)
                groups += sum(t.get('hp_member_role')=='anchor' for t in new)
                for t in new:
                    if t.get('hp_member_role')=='anchor':
                        selected = [x for x in new if x.get('hp_group')==t['hp_group']]
                        self.assertEqual(len(selected),4)
                        self.assertEqual(len({x['canonical_source'] for x in selected}),4)
            self.assertEqual(groups,a.steps*4)

    def test_small_partial_group_skips_explicitly_without_dropping_ce(self):
        rows,cfg=inventory()
        plan=Plan(rows,cfg['stream_sources'],50001,cfg=cfg)
        batch=next(plan.batches(0))
        self.assertEqual(len(batch),16)
        self.assertEqual(sum(bool(t.get('hp_skip')) for t in batch),4)
        self.assertFalse(any(t.get('hp_member_role')=='anchor' for t in batch))

    def test_second_positive_reuses_source_and_one_noise_pass_no_extra_clean(self):
        rows,cfg=inventory()
        index=next(i for i,r in enumerate(rows) if r['condition']=='offline' and r['label']==1 and r['v5_origin_split']=='dev')
        t=ticket(index)
        t.update(hp_group='A/en/0/0',hp_member_role='anchor')
        wave=np.random.default_rng(42).normal(0,.1,3200).astype(np.float32)
        current=Waves(rows,cfg); parent=ParentWaves(rows,cfg)
        for data in (current,parent):
            data.original=lambda row: wave.copy()
            data.bank,data.engines=object(),object()
            data._phones=lambda row: self.fail('Dev has no phoneme assets')
        def rec(ticket,*args):
            return dict(seed=ticket['occurrence'],lane=ticket.get('noise_recipe_view',0))
        def noisy(x,r,*args):
            return x*.8,dict(rtc_activity={'suppressed_spans':[]},recipe=r)
        with patch('w2v_v55.data.make_recipe',side_effect=rec) as recipes, \
             patch('w2v_v55.data.augment',side_effect=fake_augment), \
             patch('w2v_v55.data.generate_matched_context',side_effect=fake_matched) as matched, \
             patch('w2v_v55.data.generate',side_effect=noisy) as extra:
            unit=current[t]
        with patch('w2v_v5.data.make_recipe',side_effect=rec), \
             patch('w2v_v5.data.augment',side_effect=fake_augment), \
             patch('w2v_v5.data.generate_matched_context',side_effect=fake_matched):
            old=parent[ticket(index)]
        self.assertEqual(matched.call_count,1)
        self.assertEqual(extra.call_count,1)
        self.assertEqual(recipes.call_count,2)
        np.testing.assert_array_equal(old[0]['wave'],unit[0]['wave'])
        np.testing.assert_array_equal(old[0]['pcl_refs']['clean']['wave'],unit[0]['pcl_refs']['clean']['wave'])
        p=unit[0]['hp_positive2']
        self.assertEqual(p['original_samples'],len(wave))
        self.assertEqual(len(p['wave']),48000)
        self.assertEqual(p['label'],1)
        self.assertEqual(p['source_id'],unit[0]['source_id'])
        self.assertEqual(p['input_processing']['policy'],unit[0]['input_processing']['policy'])
        self.assertFalse(np.array_equal(p['wave'],unit[0]['wave']))
        self.assertEqual(unit[0]['pcl_pairs'],old[0]['pcl_pairs'])


if __name__=='__main__':
    unittest.main()
