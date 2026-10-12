import copy
import unittest
from collections import Counter,defaultdict
import numpy as np
import torch
from torch import nn
from unittest.mock import patch
from w2v_v318.objective import unit_loss as old_loss
from .test_support import tiny_cfg,tiny_model,metadata,units
from .common import GROUPS,partial_state,apply_partial,seed_all
from .sampling import Plan,validate_units
from .objective import unit_loss,micro_losses
from .model import optimizer_for
from .step import train_step,schedule


class Toy(nn.Module):
    def __init__(self):super().__init__();self.linear=nn.Linear(8,2)
    def forward(self,waves):return self.linear(torch.as_tensor(np.stack(waves)))


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)



    def test_quota_coverage_and_no_cycle_boundary_duplicates(self):
        plan=Plan(metadata((7,3,11,5),missing=False),16,quotas=(4,2,7,3))
        for epoch in range(8):
            draws=Counter()
            for batch in plan.batches(epoch):
                cells=defaultdict(list)
                for t in batch:
                    row=plan.rows[t['index']];cells[t['kind'],row['language'],row['label']].append(t['index']);draws[t['index']]+=1
                for (kind,lang,label),values in cells.items():
                    self.assertEqual(len(values),plan.quotas[lang,label]);self.assertEqual(len(values),len(set(values)))
            for kind,pools in plan.pools.items():
                for g,pool in pools.items():
                    counts=[draws[i] for i in pool]
                    self.assertLessEqual(max(counts)-min(counts),2)
        report=plan.coverage(0)
        self.assertEqual(report['stream_loss_mass'],{'online':.5,'noisy':.5})
        self.assertEqual(report['class_view_counts']['fake'],plan.steps*33)

    def test_c3_weight_mass_and_equal_quota_legacy_gradient(self):
        plan=Plan(metadata((8,4,12,6),missing=False),16)
        batch=units(plan,samples=8);validate_units(batch,[4,2,7,3]);masses=defaultdict(float)
        for unit in batch:
            z=torch.zeros(len(unit),2,requires_grad=True)
            loss,_,_=unit_loss(z,unit)
            masses[unit[0]['language'],unit[0]['label']]+=float(loss)/np.log(2)
            other=z.detach().clone().requires_grad_()
            expected=old_loss(z,unit,16)[0];actual=unit_loss(other,unit,[4]*4)[0]
            expected.backward();actual.backward();torch.testing.assert_close(z.grad,other.grad)
        for v in masses.values():self.assertAlmostEqual(v,.25,places=6)
        bad=copy.deepcopy(batch);bad[0][0]['role']='offline'
        with self.assertRaises(ValueError):validate_units(bad,[4,2,7,3])

    def test_microbatch_equivalence_and_single_forward_budget(self):
        cfg=tiny_cfg();cfg.update(stream_sources=16,group_quotas=[4,2,7,3],frame_budget=1000)
        batch=units(Plan(metadata((8,4,12,6),missing=False),16),samples=8)
        seed_all(20);a=Toy();b=copy.deepcopy(a)
        oa=torch.optim.SGD(a.parameters(),lr=.01);ob=torch.optim.SGD(b.parameters(),lr=.01)
        for opt in (oa,ob):opt.param_groups[0]['name']='toy'
        with patch.object(a,'forward',wraps=a.forward) as forward:
            x=train_step(a,oa,batch,dict(cfg,microbatch=2),.25)
            self.assertEqual(sum(len(call.args[0]) for call in forward.call_args_list),48)
        y=train_step(b,ob,batch,dict(cfg,microbatch=48),.25)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-6)
        self.assertAlmostEqual(x['loss'],y['loss'],places=6)
        self.assertEqual(x['predicted_fake_by_true_class'],y['predicted_fake_by_true_class'])

    def test_batched_ce_matches_independent_c3_formula_and_gradients(self):
        batch=units(Plan(metadata((8,4,12,6),missing=False),16),samples=8)
        z=torch.randn(48,2,requires_grad=True);other=z.detach().clone().requires_grad_()
        expected=0.;offset=0
        for unit in batch:
            expected=expected+unit_loss(z[offset:offset+len(unit)],unit)[0];offset+=len(unit)
        actual=sum(loss for loss,raw in micro_losses(other,batch,[4,2,7,3],.25,.02))
        expected.backward();actual.backward()
        torch.testing.assert_close(expected,actual,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(z.grad,other.grad,rtol=1e-6,atol=1e-7)




if __name__=='__main__':unittest.main()
