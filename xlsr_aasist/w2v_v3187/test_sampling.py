import unittest
from collections import Counter
from .sampling import Plan, quota_schedule
from w2v_v3186.test_sampling import rows_for
from w2v_v3185.common import GROUPS


class SamplingTests(unittest.TestCase):
    def test_exact_default_budget_and_partial_final_batch(self):
        plan=Plan(rows_for((21,19,35,23)),cfg={'consistency':'none'})
        self.assertEqual(plan.steps,1403)
        self.assertEqual(sum(plan.schedule[-1]),20)
        counts=Counter()
        for tickets in plan.batches(0):
            actual=tickets[0]['batch_group_quotas']
            for kind in ('online','offline'):
                selected=[t for t in tickets if t['kind']==kind]
                self.assertEqual(len(selected),sum(actual))
                for j,g in enumerate(GROUPS):
                    ids=[t['index'] for t in selected if (plan.rows[t['index']]['language'],plan.rows[t['index']]['label'])==g]
                    self.assertEqual(len(ids),actual[j]);self.assertEqual(len(set(ids)),len(ids))
                    counts[kind,g]+=len(ids)
        for kind in ('online','offline'):
            self.assertEqual([counts[kind,g] for g in GROUPS],[9668,8000,8000,8000])

    def test_resume_cycle_boundary_and_no_duplicate_with_variable_quotas(self):
        rows=rows_for((9,7,13,11));cfg={'source_draws_per_stream':(113,89,89,89),'warm_epochs':0}
        plan=Plan(rows,cfg=cfg)
        expected=list(plan.batches(7));fresh=Plan(rows,cfg=cfg)
        self.assertEqual(expected,list(fresh.batches(7)))
        for epoch in range(8):
            for tickets in plan.batches(epoch):
                for kind in ('online','offline'):
                    ids=[t['index'] for t in tickets if t['kind']==kind]
                    self.assertEqual(len(ids),len(set(ids)))
            for kind in ('online','offline'):
                for group in GROUPS:
                    source_indices=plan.stream(kind,group,epoch)
                    self.assertEqual(len(source_indices),cfg['source_draws_per_stream'][GROUPS.index(group)])

    def test_continuous_source_coverage_and_dynamic_refresh(self):
        rows=rows_for((13,11,31,17));cfg={'source_draws_per_stream':(101,80,80,80),'consistency':'none'}
        plan=Plan(rows,cfg=cfg)
        for kind in ('online','offline'):
            for group in GROUPS:
                n=len(plan.pools[kind][group]);all_values=[]
                for epoch in range(4):all_values.extend(plan.view_stream(kind,group,epoch))
                for start in range(0,len(all_values)-n+1,n):
                    self.assertEqual(len({i for i,_,_ in all_values[start:start+n]}),n)
        i=plan.pools['online'][('en',1)][0]
        specs=[plan.view_spec(i,'online',cycle) for cycle in range(15)]
        self.assertEqual(sum(s['base_family']=='original' for s in specs),3)
        augmented=[s['base_view_id'] for s in specs if s['base_family']!='original']
        self.assertEqual(len(augmented),len(set(augmented)))
        self.assertGreater(max(s['base_generation'] for s in specs),0)
        self.assertNotIn('temporal_context',{s['base_family'] for s in specs})

    def test_warmup_has_no_reference_work_and_platform_budget_is_actual(self):
        rows=rows_for((17,15,23,19),missing=True)
        plan=Plan(rows,cfg={'source_draws_per_stream':(73,60,60,60),'warm_epochs':1,'platform_sources_per_group':99})
        self.assertTrue(all(not t.get('pcl_enabled',False) for batch in plan.batches(0) for t in batch))
        for batch in plan.batches(1):
            for group in GROUPS:
                selected=[t for t in batch if t['kind']=='offline' and (rows[t['index']]['language'],rows[t['index']]['label'])==group]
                expected=sum('pcl_online_index' in t for t in selected)
                self.assertTrue(all(t['pcl_platform_budget']==expected for t in selected))
                self.assertTrue(all(t['pcl_noise'] for t in selected))

    def test_audit_distinguishes_sources_and_generated_noisy_views(self):
        plan=Plan(rows_for((11,9,23,13)),cfg={'source_draws_per_stream':(73,60,60,60),'consistency':'none'})
        audit=plan.audit_budget(3)
        online=audit['cells']['online/en/1'];offline=audit['cells']['offline/en/1']
        self.assertEqual(online['source_count'],9)
        self.assertEqual(online['source_draws'],180)
        self.assertEqual(online['classification_views'],180)
        self.assertEqual(offline['classification_views'],360)
        self.assertTrue(audit['all_source_recordings_seen'])
        self.assertEqual(plan.coverage(0)['online_views'],253)

    def test_no_budget_overfill_when_remainder_less_than_four(self):
        schedule=quota_schedule((101,80,80,76),24)
        self.assertEqual(tuple(map(sum,zip(*schedule))),(101,80,80,76))
        self.assertTrue(all(min(q)>0 and sum(q)<=24 for q in schedule))
        with self.assertRaises(ValueError):quota_schedule((100,1,1,1),24)


if __name__=='__main__':unittest.main()
