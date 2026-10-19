"""Optimization behavior, padding isolation and inherited-checkpoint guarantees."""
import copy
import unittest
import numpy as np
import torch
from torch import nn
from w2v_v31813.test_core import Tiny, fixture, cfg as old_cfg
from w2v_v31813.step import train_step as old_step
from w2v_v31810.test_training import settings
from w2v_v318.test_core import TinyFrontend, TinyLayout
from w2v_v31810.model import Detector as ParentDetector
from w2v_v31810.common import partial_state as parent_state
from .adapters import MultiConvAdapter
from .model import Detector, optimizer_for
from .common import partial_state, apply_partial
from .step import train_step
from .gradients import coordinate


def cfg(**kwargs):
    return old_cfg(gradient_coordination='pcgrad',**kwargs)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj=nn.Linear(16,16);self.k_proj=nn.Linear(16,16)
        self.v_proj=nn.Linear(16,16);self.output_proj=nn.Linear(16,16)
    def forward(self,x,layout,**kwargs):
        score=self.q_proj(x)@self.k_proj(x).transpose(-1,-2)/4
        valid=torch.arange(x.shape[1])[None,:]<torch.tensor(layout.seq_lens)[:,None]
        score=score.masked_fill(~valid[:,None,:],-1e9)
        return self.output_proj(score.softmax(-1)@self.v_proj(x))


class Layer(nn.Module):
    def __init__(self):super().__init__();self.self_attn=Attention();self.norm=nn.LayerNorm(16)
    def forward(self,x,layout,bias):return self.norm(x+self.self_attn(x,layout))


class Encoder(nn.Module):
    def __init__(self):super().__init__();self.layers=nn.ModuleList([Layer(),Layer()])
    def forward(self,x,layout):
        for layer in self.layers:x=layer(x,layout,None)
        return x


def actual_cfg():
    c,_=settings();c.update(adapter_layers=2,adapter_dim=8,adapter_lr=.001,**cfg())
    return c


def backbone():
    with torch.random.fork_rng():
        torch.manual_seed(12345)
        return TinyFrontend(),Encoder()


def actual_model(c):return Detector(*backbone(),c,TinyLayout)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_zero_adapter_and_padding_cannot_change_valid_features(self):
        module=MultiConvAdapter(16,8)
        short=torch.randn(1,12,16,requires_grad=True)
        torch.testing.assert_close(module(short,[12]),short,rtol=0,atol=0)
        with torch.no_grad():module.up.weight.normal_(0,.1);module.up.bias.fill_(.2)
        mixed=torch.randn(2,37,16);mixed[0,:12]=short.detach();mixed[0,12:]=1000
        alone=module(short,[12]);batched=module(mixed,[12,37])
        torch.testing.assert_close(alone[0],batched[0,:12],atol=2e-6,rtol=2e-6)
        self.assertTrue(torch.equal(batched[0,12:],mixed[0,12:]))
        alone.sum().backward();self.assertGreater(float(short.grad.abs().sum()),0)

    def test_pcgrad_original_vectors_symmetric_and_auxiliary_untouched(self):
        c=torch.tensor([1.,0.]);n=torch.tensor([-1.,1.]);a=torch.tensor([.1,.2])
        merged,stats=coordinate([c],[n],[a])
        expected=c-(c@n)/(n@n)*n+n-(c@n)/(c@c)*c+a
        torch.testing.assert_close(merged,expected)
        self.assertTrue(stats['applied'])
        self.assertGreaterEqual(float((merged-a)@c),0)
        self.assertGreaterEqual(float((merged-a)@n),0)
        scaled,other=coordinate([256*c],[256*n],[256*a],scale=256)
        torch.testing.assert_close(scaled/256,merged)
        self.assertEqual(stats,other)

    def test_zero_nonconflicting_and_nonfinite_gradients(self):
        for c,n in (([0.,0.],[1.,2.]),([1.,0.],[1.,1.])):
            c,n=torch.tensor(c),torch.tensor(n)
            result,stats=coordinate([c],[n],[torch.zeros(2)])
            torch.testing.assert_close(result,c+n);self.assertFalse(stats['applied'])
        result,stats=coordinate([torch.tensor([float('inf')])],[torch.ones(1)],[torch.zeros(1)])
        self.assertFalse(stats['finite']);self.assertFalse(bool(torch.isfinite(result).all()))

    def test_no_coordination_matches_existing_eight_cell_ce_and_auxiliary_gradients(self):
        a=Tiny();b=copy.deepcopy(a);units=fixture()
        oa=torch.optim.SGD(a.parameters(),lr=.05);ob=torch.optim.SGD(b.parameters(),lr=.05)
        settings=old_cfg(gradient_coordination='none')
        sa=old_step(a,oa,units,settings,dict(prior=.1,pcl=.01),3,0,2)
        sb=train_step(b,ob,units,settings,dict(prior=.1,pcl=.01),3,0,2)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)
        self.assertAlmostEqual(sa['loss'],sb['loss'],places=6)

    def test_logical_batch_pcgrad_does_not_depend_on_microbatch_partition(self):
        a=Tiny();b=copy.deepcopy(a);units=fixture()
        oa=torch.optim.SGD(a.parameters(),lr=.05);ob=torch.optim.SGD(b.parameters(),lr=.05)
        sa=train_step(a,oa,units,cfg(microbatch=2),dict(prior=.1,pcl=.01),3,0,2)
        sb=train_step(b,ob,units,cfg(microbatch=8),dict(prior=.1,pcl=.01),3,0,2)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)
        self.assertEqual(sa['coordination']['applied'],sb['coordination']['applied'])

    def test_parent_prediction_preserved_and_frozen_lora_survives_adapter_updates(self):
        c=actual_cfg();parent=ParentDetector(*backbone(),c,TinyLayout).eval()
        with torch.no_grad():
            for n,p in parent.named_parameters():
                if n.endswith('lora_b'):p.normal_(0,.03)
        model=actual_model(c).eval();apply_partial(model,parent_state(parent),parent=True)
        waves=[np.random.default_rng(7).normal(size=n).astype('float32') for n in (4240,8080)]
        with torch.no_grad():torch.testing.assert_close(model(waves),parent(waves),rtol=0,atol=0)
        inherited={n:p.detach().clone() for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b'))}
        before=partial_state(model);model.train();model.set_phase(True);opt=optimizer_for(model,c)
        stats=train_step(model,opt,fixture(),c,dict(prior=.03,pcl=.001),3,0,2)
        self.assertFalse(stats['amp']['skipped_update'])
        after=partial_state(model)
        self.assertTrue(any(not torch.equal(value,after[n]) for n,value in before.items() if n.startswith('adapters.')))
        for name,value in inherited.items():
            self.assertTrue(torch.equal(value,dict(model.named_parameters())[name]));self.assertFalse(dict(model.named_parameters())[name].requires_grad)
        restored=actual_model(c).eval();apply_partial(restored,after)
        model.eval()
        with torch.no_grad():torch.testing.assert_close(model(waves),restored(waves),rtol=0,atol=0)
        with self.assertRaises(ValueError):apply_partial(restored,parent_state(parent))

    def test_attention_adapter_recomputed_gradients_match_without_checkpointing(self):
        c=actual_cfg();a=actual_model(c).eval()
        with torch.no_grad():
            for adapter in a.adapters.values():adapter.up.weight.normal_(0,.02)
        b=actual_model(dict(c,checkpointing=False,fusion_checkpointing=False)).eval();apply_partial(b,partial_state(a))
        # The tiny backbone represents the same immutable official weights.
        b.frontend.load_state_dict(a.frontend.state_dict());b.encoder.load_state_dict(a.encoder.state_dict())
        waves=[fixture()[0][0]['wave']]
        a(waves).sum().backward();b(waves).sum().backward()
        for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(name,other)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=1e-4)


if __name__=='__main__':unittest.main()
