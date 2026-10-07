"""Real tiny w2v-BERT gradients, masking, microbatching and inference isolation."""
import copy
import unittest

import numpy as np
import torch

from w2v_v3.test_model_step import tiny_detector
from w2v_v32.model import install_runtime
from .objectives import TFCL, channel_cka, fusion_frames
from .step import train_step
from .data import validate_batch

torch.set_num_threads(1)


def examples():
    rows=[]
    for i,(language,label) in enumerate((('en',0),('en',1),('zh',0),('zh',1))):
        for role,mass in (('ordinary',.3),('reference',.1),('noisy',.6)):
            length=14+i*2
            rows.append(dict(id=str(i),source_id=str(i),group_id=str(i),source_sha256=str(i),
                language=language,label=label,role=role,condition=role,split='train',pair_occurrence=str(i),
                pair_eligible=True,ce_weight=mass/4,features=torch.randn(1,length,160),
                mask=torch.ones(1,length,dtype=torch.long),aux_valid=np.ones(length,dtype=bool)))
    return rows


def config(**kw):
    return dict(dict(source_batch=4,microbatch=8,frame_budget=2000,device='cpu',amp='none',
        tfcl_time_weight=.15,tfcl_structure_weight=.045,max_grad_norm=1.),**kw)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(315)

    def test_both_pair_branches_and_both_auxiliary_modules_receive_gradients(self):
        aux=TFCL(8,2,21)
        a=torch.randn(27,8,requires_grad=True);b=torch.randn(32,8,requires_grad=True)
        t,s=aux(a,b,torch.ones(27,dtype=torch.bool),torch.ones(32,dtype=torch.bool))
        (t+.3*s).backward()
        self.assertGreater(float(a.grad.norm()),0);self.assertGreater(float(b.grad.norm()),0)
        self.assertGreater(float(aux.attention.in_proj_weight.grad.norm()),0)
        self.assertGreater(float(aux.time_projection.weight.grad.norm()),0)

    def test_padding_and_erased_frames_cannot_change_aux_loss_or_receive_gradient(self):
        aux=TFCL(8,2,21)
        a=torch.randn(40,8);b=torch.randn(40,8)
        mask=torch.ones(40,dtype=torch.bool);mask[3:6]=False;mask[29:]=False
        first=aux(a,b,mask,mask)
        aa=a.clone();bb=b.clone();aa[~mask]=999;bb[~mask]=-777
        aa.requires_grad_();bb.requires_grad_()
        second=aux(aa,bb,mask,mask)
        for x,y in zip(first,second):torch.testing.assert_close(x,y,rtol=0,atol=0)
        sum(second).backward()
        self.assertTrue(bool((aa.grad[~mask]==0).all() and (bb.grad[~mask]==0).all()))

    def test_structure_is_per_source_and_has_no_batch_coupling(self):
        a=torch.randn(128,201);b=torch.randn(128,201)
        torch.testing.assert_close(channel_cka(a,a*2),torch.tensor(0.),atol=3e-7,rtol=0)
        aux=TFCL(8,2,21)
        a=torch.randn(29,8);b=torch.randn(41,8)
        maska=torch.ones(29,dtype=torch.bool);maskb=torch.ones(41,dtype=torch.bool)
        t,s=aux(a,b,maska,maskb)
        self.assertTrue(bool(torch.isfinite(t+s)))

    def test_real_encoder_auxiliary_gradients_and_physical_microbatch_equivalence(self):
        a=install_runtime(tiny_detector(checkpointing=False));a.configure_trainable_layers(1)
        # Conformer has a separate default dropout. Turn it off ONLY in this
        # numerical test; stochastic masks naturally differ across batch shapes.
        for module in a.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
        b=copy.deepcopy(a);ta=TFCL(8,2,21);tb=copy.deepcopy(ta)
        samples=examples()
        oa=torch.optim.SGD([p for p in a.parameters() if p.requires_grad]+list(ta.parameters()),lr=.02)
        ob=torch.optim.SGD([p for p in b.parameters() if p.requires_grad]+list(tb.parameters()),lr=.02)
        before={k:v.clone() for k,v in a.state_dict().items()}
        sa=train_step(a,ta,oa,samples,config(microbatch=2,frame_budget=50),1.,audit=True)
        sb=train_step(b,tb,ob,samples,config(microbatch=8),1.)
        self.assertTrue(all(v>0 for v in sa['aux_gradient_audit'].values()))
        self.assertEqual(sa['eligible_pairs'],4)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5)
        for p,q in zip(ta.parameters(),tb.parameters()):torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5)
        self.assertAlmostEqual(sa['classification_loss'],sb['classification_loss'],places=6)
        changed=[k for k,v in a.state_dict().items() if not torch.equal(v,before[k])]
        self.assertTrue(any(k.startswith('backbone.encoder.layers.1.') for k in changed))
        self.assertFalse(any(k.startswith('backbone.encoder.layers.0.') for k in changed))

    def test_capture_and_auxiliary_do_not_modify_the_inference_function(self):
        model=install_runtime(tiny_detector(checkpointing=False)).eval()
        x=torch.randn(1,20,160);m=torch.ones(1,20,dtype=torch.long)
        with torch.inference_mode():
            before=model(x,m)[0]
            with fusion_frames(model) as features:
                after=model(x,m)[0]
                self.assertEqual(len(features),1)
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertEqual(len(model.head.blocks[0]._forward_pre_hooks),0)
        self.assertFalse(any('attention.in_proj' in n for n in model.state_dict()))

    def test_pair_zero_mask_remains_finite_and_dev_never_trains(self):
        aux=TFCL(8,2,21)
        a=torch.randn(10,8,requires_grad=True)
        loss=sum(aux(a,a,torch.zeros(10,dtype=torch.bool),torch.ones(10,dtype=torch.bool)))
        loss.backward();self.assertEqual(float(loss),0)
        rows=examples();rows[0]['split']='dev'
        with self.assertRaisesRegex(ValueError,'cannot enter'):validate_batch(rows,4)


if __name__=='__main__':unittest.main()
