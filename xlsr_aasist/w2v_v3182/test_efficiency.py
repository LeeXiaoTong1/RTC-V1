"""Equivalence checks for batched objectives, telemetry and source planning."""
from collections import defaultdict
import math
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from .common import GROUPS, capture_rng, restore_rng, seed_all
from .objective import contrastive
from .sampling import Plan
from .step import backward_objective, schedule
from .test_core import Toy
from .test_support import metadata, tiny_cfg, units_from_plan


def reference_contrastive(features, units, temperature=.1, negative_mass=4.):
    """Scalar definition retained only in tests, independent of batched indexing."""
    h = F.normalize(features.float(), dim=-1)
    total, positive, negative = [h.sum()*0 for _ in range(3)]
    for i, unit in enumerate(units):
        row = unit[0]
        mass = row['source_mass']/4
        for a, p in ((0,1),(1,0),(0,2),(2,0)):
            pos = (h[3*i+a]*h[3*i+p]).sum()
            weighted_scores = [pos/temperature]
            neg_cos = h.sum()*0
            for lang in ('en','zh'):
                candidates = [j for j,u in enumerate(units) if u[0]['label'] != row['label']
                    and u[0]['language'] == lang and u[0]['group_id'] != row['group_id']]
                if not candidates:
                    raise ValueError('Missing opposite-label language')
                weight = negative_mass/(2*len(candidates))
                for j in candidates:
                    cosine = (h[3*i+a]*h[3*j+p]).sum()
                    weighted_scores.append(cosine/temperature+math.log(weight))
                    neg_cos = neg_cos + weight*cosine/negative_mass
            total = total + mass*(torch.logsumexp(torch.stack(weighted_scores),0)-pos/temperature)
            positive = positive+mass*pos
            negative = negative+mass*neg_cos
    return total, (positive, negative)


class EfficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_vectorized_loss_and_gradients_match_scalar_definition(self):
        for batch,temp,mass,zero in ((4,.1,4.,False),(16,.07,3.5,False),
                                     (32,.3,8.,False),(16,.1,4.,True)):
            with self.subTest(batch=batch,temperature=temp,negative_mass=mass,zero=zero):
                seed_all(14)
                units=units_from_plan(Plan(metadata((13,7,21,11)),batch),samples=8)
                h=(torch.zeros if zero else torch.randn)(3*batch,160,requires_grad=True)
                other=h.detach().clone().requires_grad_()
                expected,detail=reference_contrastive(h,units,temp,mass)
                actual,stats=contrastive(other,units,temp,mass)
                expected.backward();actual.backward()
                torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-6)
                torch.testing.assert_close(other.grad,h.grad,rtol=2e-5,atol=2e-6)
                self.assertAlmostEqual(stats['positive_cosine'],float(detail[0]),places=6)
                self.assertAlmostEqual(stats['negative_cosine'],float(detail[1]),places=6)

    def test_same_source_exclusion_and_missing_language_guard(self):
        units=units_from_plan(Plan(metadata(),16),samples=8)
        fake=next(i for i,u in enumerate(units) if u[0]['label']==0)
        real=next(i for i,u in enumerate(units) if u[0]['label']==1)
        for row in units[real]:row['group_id']=units[fake][0]['group_id']
        h=torch.randn(48,160,requires_grad=True)
        expected,_=reference_contrastive(h,units)
        actual,_=contrastive(h,units)
        torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-6)
        for unit in units:
            for row in unit:row['language']='en'
        with self.assertRaisesRegex(ValueError,'both languages'):contrastive(h,units)

    def test_telemetry_matches_independent_predictions(self):
        seed_all(29)
        cfg=tiny_cfg();cfg.update(stream_sources=16,microbatch=4)
        units=units_from_plan(Plan(metadata(),16),samples=8)
        rows=[r for u in units for r in u]
        model=Toy().train();rng=capture_rng()
        with torch.no_grad():
            z=torch.cat([model([r['wave'] for r in rows[i:i+4]]) for i in range(0,len(rows),4)])
            labels=torch.tensor([r['label'] for r in rows])
            raw=F.cross_entropy(z,labels,reduction='none')
        restore_rng(rng)
        stats=backward_objective(model,units,cfg,0.)
        self.assertAlmostEqual(stats['raw_ce_sum'],float(raw.sum()),places=5)
        self.assertEqual(stats['max_example_ce'],float(raw.max()))
        for label,name in ((0,'fake'),(1,'real')):
            ix=labels==label
            self.assertEqual(stats['predicted_fake_by_true_class'][name],
                [int((z[ix].argmax(1)==0).sum()),int(ix.sum())])
        cells=defaultdict(list)
        for i,r in enumerate(rows):
            role='noisy' if r['role'].startswith('noisy') else r['role']
            cells[f'{role}/{r["language"]}/{r["label"]}'].append(i)
        for key,ix in cells.items():
            self.assertEqual(stats['cells'][key]['views'],len(ix))
            self.assertEqual(stats['cells'][key]['correct'],int((z[ix].argmax(1)==labels[ix]).sum()))
            self.assertAlmostEqual(stats['cells'][key]['raw_ce_sum'],sum(float(raw[i]) for i in ix))

    def test_epoch_preparation_reuses_identical_source_streams(self):
        plan=Plan(metadata(),16)
        for epoch in (0,1,5):
            expected=(plan.coverage(epoch),list(plan.batches(epoch)))
            with patch.object(plan,'stream',wraps=plan.stream) as stream:
                actual=plan.prepare_epoch(epoch)
            self.assertEqual(actual,expected)
            self.assertEqual(stream.call_count,len(GROUPS))

    def test_six_epoch_schedule_has_one_warmup_and_complete_joint_decay(self):
        from .config import parser,validate
        args=parser().parse_args([]);validate(args)
        cfg=tiny_cfg();cfg.update(epochs=args.epochs,warm_epochs=args.warm_epochs)
        optimizer=torch.optim.AdamW([dict(params=[nn.Parameter(torch.ones(1))],name='aasist',
            lr=cfg['head_lr'],initial_lr=cfg['head_lr'])])
        values=[[schedule(optimizer,cfg,e,s,10) for s in range(10)] for e in range(cfg['epochs'])]
        self.assertEqual(values[0],[0.]*10)
        self.assertAlmostEqual(values[1][0],.005)
        self.assertAlmostEqual(values[1][-1],.05)
        self.assertTrue(all(abs(v-.05)<1e-12 for epoch in values[2:] for v in epoch))
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],cfg['head_lr']*.1)
        self.assertEqual((cfg['epochs'],cfg['warm_epochs']),(6,1))


if __name__=='__main__':unittest.main()
