import unittest
from collections import Counter
from w2v_v3185.sampling import Plan as OldPlan
from .sampling import Plan, canonical_pairs


def rows_for(counts=(11,7,19,9),missing=False):
    rows=[]
    for group,n in zip((('en',0),('en',1),('zh',0),('zh',1)),counts):
        for j in range(n):
            source=f'{group[0]}-{group[1]}-{j}'
            sha=f'{len(rows):064x}'
            base=dict(source_id=source,group_id=sha,source_sha256=sha,
                language=group[0],label=group[1],split='train',full_length=True)
            rows.append(dict(base,id=source,condition='offline',audio_sha256=sha))
            if not missing or j%3:
                rows.append(dict(base,id=source+'-on',condition='online',audio_sha256=f'{len(rows)+10000:064x}'))
    return rows


class SamplingTests(unittest.TestCase):
    def test_classification_schedule_bitwise_unchanged(self):
        rows=rows_for();new=Plan(rows);old=OldPlan(rows)
        for epoch in range(6):
            aa=list(new.batches(epoch));bb=list(old.batches(epoch))
            self.assertEqual(len(aa),len(bb))
            for actual,expected in zip(aa,bb):
                self.assertEqual([{k:v for k,v in t.items() if not k.startswith('pcl_')} for t in actual],expected)
                counts=Counter((rows[t['index']]['language'],rows[t['index']]['label'])
                    for t in actual if 'pcl_online_index' in t)
                self.assertEqual(set(counts.values()),{2})
                for t in actual:
                    if t['kind']=='offline':self.assertEqual(t['pcl_noisy_index'],t['source_cycle']%2)

    def test_ce_only_has_no_references_or_pair_requirement(self):
        rows=rows_for();new=Plan(rows,cfg={'consistency':'none'})
        for a,b in zip(new.batches(0),OldPlan(rows).batches(0)):self.assertEqual(a,b)

    def test_missing_official_pair_never_replaces_classification(self):
        rows=rows_for((14,14,14,14),True);new=Plan(rows);old=OldPlan(rows)
        for a,b in zip(new.batches(0),old.batches(0)):
            self.assertEqual([t['index'] for t in a],[t['index'] for t in b])
            for t in a:
                if t['kind']=='offline':self.assertTrue(t['pcl_noise'])

    def test_exact_coverage_and_fixed_factors(self):
        rows=rows_for();plan=Plan(rows);audit=plan.audit_budget(6)
        for kind in ('online','offline'):
            for lang,label,factor in (('en',0,5),('en',1,5),('zh',0,1),('zh',1,3)):
                key=f'{kind}/{lang}/{label}';cell=audit['cells'][key]
                visited=set();draws=Counter()
                for epoch in range(6):
                    for tix in plan.batches(epoch):
                        for t in tix:
                            r=rows[t['index']]
                            if t['kind']==kind and (r['language'],r['label'])==(lang,label):
                                draws[t['index']]+=1
                                if not t['base_dynamic']:visited.add((t['index'],t['base_variant']))
                self.assertEqual(cell['fixed_view_factor'],factor)
                self.assertEqual(cell['fixed_views_seen'],len(visited))
                self.assertEqual(cell['source_draws'],sum(draws.values()))
                self.assertEqual(cell['fixed_views_unseen'],cell['available_fixed_views']-len(visited))

    def test_conflicting_canonical_pair_fails(self):
        rows=rows_for();rows[1]['label']=1
        with self.assertRaises(ValueError):canonical_pairs(rows)


if __name__=='__main__':unittest.main()
