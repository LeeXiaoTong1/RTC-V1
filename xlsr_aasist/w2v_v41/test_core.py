"""Logical-gradient equivalence, dual-PCL graph and masking contracts."""
import copy
import unittest
import numpy as np
import torch
from torch import nn
from .step import train_step,components
from .gradients import coordinate
from .pcl import phoneme_mse,expand,ramp
from .data import simulated_matches


def cfg(**kwargs):
    value=dict(device='cpu',backend_precision='fp32',microbatch=4,frame_budget=9000,
        label_smoothing=.02,grad_clip=100.,seed=4001,gradient_coordination='pcgrad',consistency='dual',
        warm_epochs=1,pcl_ramp_updates=2,pcl_gradient_target=.05,pcl_max_weight=1.)
    value.update(kwargs);return value


class Tiny(nn.Module):
    def __init__(self,cfg=None):
        super().__init__();self.fusion=nn.Linear(3,5);self.head=nn.Linear(5,2)
    def encode(self,waves):
        result=[]
        for wave in waves:
            x=torch.as_tensor(wave,dtype=torch.float32)[:3840].reshape(12,320)
            f=torch.stack((x.mean(1),x.square().mean(1),x.abs().mean(1)),1)
            result.append(self.fusion(f).tanh())
        return result
    def classify_sequences(self,sequences):return self.head(torch.stack([h.mean(0) for h in sequences]))
    def forward(self,waves):return self.classify_sequences(self.encode(waves))
    def set_phase(self,joint):pass


def fixture():
    units=[]
    for role in ('online','noisy'):
        for lang in ('en','zh'):
            for label in (0,1):
                i=len(units);x=np.random.default_rng(i+5).normal(0,.2+label*.3,4240).astype(np.float32)
                row=dict(wave=x,role=role,language=lang,label=label,split='train',canonical_source=str(i),original_samples=len(x))
                off=dict(row,wave=(x*.7+.03*np.sin(np.arange(len(x))*.1)).astype(np.float32),role='offline_ref')
                row['pcl']=dict(kind='official' if role=='online' else 'noise',reference=off,
                    received=dict(row,role='online_ref') if role=='online' else None,reuse_received=True,matches=[[0,3000,0,3000]])
                units.append([row])
    return units


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_pcgrad_original_vectors_symmetric_and_auxiliary_untouched(self):
        c=torch.tensor([1.,0.]);n=torch.tensor([-1.,1.]);a=torch.tensor([.1,.2])
        merged,stats=coordinate([c],[n],[a]);expected=c-(c@n)/(n@n)*n+n-(c@n)/(c@c)*c+a
        torch.testing.assert_close(merged,expected);self.assertTrue(stats['applied'])
        scaled,other=coordinate([256*c],[256*n],[256*a],scale=256)
        torch.testing.assert_close(scaled/256,merged);self.assertEqual(stats,other)
    def test_nonfinite_gradient_is_never_coordinated(self):
        merged,stats=coordinate([torch.tensor([float('inf')])],[torch.ones(1)],[torch.zeros(1)])
        self.assertFalse(stats['finite']);self.assertFalse(torch.isfinite(merged).all())
    def test_no_coordination_matches_direct_joint_loss(self):
        a=Tiny();b=copy.deepcopy(a);units=fixture();settings=cfg(gradient_coordination='none')
        oa=torch.optim.SGD(a.parameters(),lr=.05);ob=torch.optim.SGD(b.parameters(),lr=.05)
        expected=0.
        for _,ce,aux,_ in components(a,units,settings,True):
            loss=ce+sum(.03*v for v in aux.values());loss.backward();expected+=float(loss.detach())
        oa.step();actual=train_step(b,ob,units,settings,dict(official=.03,noise=.03),2,0,2)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)
        self.assertAlmostEqual(expected,actual['loss'],places=6)
    def test_logical_pcgrad_independent_of_microbatch_partition(self):
        a=Tiny();b=copy.deepcopy(a);units=fixture()
        oa=torch.optim.SGD(a.parameters(),lr=.05);ob=torch.optim.SGD(b.parameters(),lr=.05)
        sa=train_step(a,oa,units,cfg(microbatch=3),dict(official=.02,noise=.03),2,0,2)
        sb=train_step(b,ob,units,cfg(microbatch=8),dict(official=.02,noise=.03),2,0,2)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)
        self.assertEqual(sa['pcl_pairs'],{'official':4,'noise':4})
        self.assertEqual(sa['views'],8);self.assertEqual(sa['reference_views'],8)
        self.assertGreater(sa['pcl_gradient_norms']['noise'],0)
    def test_pcl_both_ends_receive_gradients_on_bert_grid(self):
        a=torch.zeros(12,3,requires_grad=True);b=torch.ones(12,3,requires_grad=True)
        loss,n=phoneme_mse(a,b,[[0,1040,0,1040]],1040,1040);loss.backward()
        self.assertEqual(n,1);self.assertTrue(a.grad[:2].abs().sum()>0);self.assertTrue(b.grad[:2].abs().sum()>0)
        self.assertEqual(float(a.grad[2:].abs().sum()),0)
        self.assertAlmostEqual(ramp(cfg(pcl_ramp_updates=300),1,299,1403),1.)
    def test_warmup_does_not_encode_references(self):
        model=Tiny();opt=torch.optim.SGD(model.parameters(),lr=.01)
        stats=train_step(model,opt,fixture(),cfg(),dict(official=1.,noise=1.),0,0,2)
        self.assertEqual(stats['reference_views'],0);self.assertEqual(stats['pcl_pairs'],{})
    def test_wrong_source_or_augmented_official_reuse_rejected(self):
        row=fixture()[0][0];row['pcl']['reference']['canonical_source']='bad'
        with self.assertRaises(ValueError):expand(row,True)
        row=fixture()[0][0];row['wave']=row['wave']*.5
        with self.assertRaises(ValueError):expand(row,True)
    def test_simulated_time_map_delay_silence_and_gate(self):
        from scipy.signal import resample_poly
        x=np.random.default_rng(21).normal(0,.1,16000).astype(np.float32)
        base=resample_poly(x,11,10).astype(np.float32);delay=160
        y=np.pad(base,(delay,0))[:len(base)]
        spans=[[2000,5000,2000,5000],[8000,11000,8000,11000]]
        matches,audit=simulated_matches(spans,x,base,y,dict(scale=1.1,offset=0.),dict(pair_eligible=True,erased_spans=[]))
        self.assertTrue(audit['delay']['valid']);self.assertEqual(len(matches),2)
        self.assertAlmostEqual(matches[0][2],2000*1.1+delay,delta=4)
        masked,_=simulated_matches(spans,x,base,y,dict(scale=1.1,offset=0.),dict(pair_eligible=True,erased_spans=[[0,len(y)]]))
        self.assertFalse(masked)
        silent,_=simulated_matches(spans,x,base,np.zeros_like(y),dict(scale=1.1,offset=0.),dict(pair_eligible=True))
        self.assertFalse(silent)


if __name__=='__main__':unittest.main()
