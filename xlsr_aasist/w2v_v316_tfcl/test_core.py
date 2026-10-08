"""Gradient direction, missing correspondences, initialization and real tiny SSL steps."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from w2v_v3151.test_step import examples as old_examples,model_fixture as old_model
from w2v_v3.test_model_step import small_head
from .alignment import align,monotone_pairs
from .objectives import TFCL,importance,local_cka
from .model import RuntimeDetector,stage,load_model
from .step import train_step,source_batches
from .data import validate_batch


def examples(missing=False):
    result=old_examples()
    for r in result:
        if r['role']=='reference':r['role']='offline'
        r['sample_hop']=320.
    if missing:
        result=[r for r in result if not (r['pair_occurrence']=='1' and r['role']=='online')]
        for r in result:
            if r['pair_occurrence']=='1':r['ce_weight']*=2
    return result


def model_fixture():
    model=old_model();model.__class__=RuntimeDetector;model.head_only=False
    return model


def config(**changes):
    result=dict(source_batch=4,microbatch=12,frame_budget=1000,device='cpu',amp='none',
        tfcl_mode='weighted',trainable_layers=1,tfcl_time_weight=.15,tfcl_structure_weight=.045,
        max_grad_norm=1.,reference_probability=.8,importance_uniform_mass=.25,importance_cap=4.,
        alignment_max_bins=192,alignment_min_cosine=.6,alignment_drop_cost=.18,
        alignment_unique_margin=.02,alignment_local_radius=2,alignment_min_coverage=.1,
        structure_window_frames=50,structure_min_frames=8,time_tolerance=.02,structure_tolerance=.02,
        head_warmup_updates=0,objective_ramp_epochs=.25,structure_delay_epochs=.125,
        init_mode='parent',variant='offline_reference_tfcl_v1',pretrained_fingerprints={})
    result.update(changes);return result


class CoreTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(31601);torch.set_num_threads(1)

    def test_monotone_all_unmatched_and_dropped_endpoints(self):
        self.assertEqual(monotone_pairs(np.ones((4,7)),.1),[])
        c=np.ones((8,5));c[np.arange(2,7),np.arange(5)]=0
        self.assertEqual(monotone_pairs(c,.1),list(zip(range(2,7),range(5))))

    def test_real_alignment_handles_cut_beginning_without_forcing_missing_frames(self):
        a=torch.eye(30);b=a[5:25].clone()
        m=align(a,b,np.ones(30,bool),np.ones(20,bool),config())
        self.assertEqual(m['reference'].tolist(),list(range(5,25)))
        self.assertEqual(m['processed'].tolist(),list(range(20)))
        self.assertFalse(m['confidence'].requires_grad)

    def test_ambiguous_content_abstains_and_known_missing_remains_unmatched(self):
        a=torch.ones(32,12)
        m=align(a,a,np.ones(32,bool),np.ones(32,bool),config())
        self.assertEqual(m['coverage'],0)
        a=torch.randn(32,12);b=a.clone();am=np.ones(32,bool);bm=am.copy();bm[8:12]=False
        b[8:12]=float('nan')
        m=align(a,b,am,bm,config(),known_mapping=True)
        self.assertFalse(set(range(8,12)).intersection(m['processed'].tolist()))
        self.assertTrue(torch.isfinite(m['confidence']).all())

    def test_auxiliary_has_no_reference_content_or_importance_gradient(self):
        a=torch.randn(40,8,requires_grad=True);b=torch.randn(40,8,requires_grad=True)
        content=torch.randn(40,16,requires_grad=True);weight=torch.rand(40,8,requires_grad=True)+.25
        valid=np.ones(40,bool);valid[10:12]=False
        loss=TFCL(8,2,21)
        t,s,meta=loss(a,b,content,content,valid,valid,weight,config(),True)
        grads=torch.autograd.grad(t+s,(a,b,content,weight),allow_unused=True)
        self.assertIsNone(grads[0]);self.assertGreater(grads[1].norm().item(),0)
        self.assertIsNone(grads[2]);self.assertIsNone(grads[3])
        self.assertEqual(grads[1][10:12].count_nonzero().item(),0)
        self.assertGreater(meta['windows'],0);self.assertEqual(sum(p.numel() for p in loss.parameters()),0)

    def test_rejected_pairs_and_short_disjoint_windows_do_not_force_structure(self):
        a=torch.randn(24,8,requires_grad=True);b=torch.randn(24,8,requires_grad=True)
        content=torch.randn(24,16);valid=np.ones(24,bool);valid[3::4]=False
        t,s,meta=TFCL(8,2,21)(a,b,content,content,valid,valid,torch.ones_like(a),config(),True)
        self.assertEqual(meta['windows'],0);self.assertEqual(float(s),0)
        v=np.zeros(24,bool)
        t,s,meta=TFCL(8,2,21)(a,b,content,content,v,v,torch.ones_like(a),config(),True)
        self.assertTrue(meta['rejected']);(t+s).backward()
        self.assertIsNone(a.grad);self.assertEqual(b.grad.count_nonzero().item(),0)

    def test_cka_identity_collapse_and_detached_bounded_importance(self):
        a=torch.randn(35,8)
        self.assertLess(abs(float(local_cka(a,3*a,torch.ones(8)))),1e-6)
        self.assertEqual(float(local_cka(a,torch.zeros_like(a),torch.ones(8))),1.)
        g=torch.randn_like(a,requires_grad=True);a.requires_grad_(True)
        w=importance(a,g,np.ones(35,bool))
        self.assertFalse(w.requires_grad);self.assertGreaterEqual(float(w.min()),.25)
        self.assertLessEqual(float(w.max()),3.25)

    def test_full_source_batches_keep_all_frames_and_missing_online_ce_budget(self):
        rows=examples(True);groups=list(validate_batch(rows,4).values());seen=[]
        for batch,x,mask in source_batches(groups,6,120):
            for i,r in enumerate(batch):
                n=r['features'].shape[1];torch.testing.assert_close(x[i,:n],r['features'][0],atol=0,rtol=0)
                self.assertEqual(int(mask[i].sum()),n);self.assertEqual(x[i,n:].count_nonzero().item(),0)
                seen.append(r['id'])
        self.assertEqual(sorted(seen),sorted(r['id'] for r in rows))
        self.assertAlmostEqual(sum(r['ce_weight'] for r in rows),1.)

    def test_unreliable_reference_still_trains_all_ce_views(self):
        model=model_fixture();before=copy.deepcopy(model.state_dict());aux=TFCL(8,2,21)
        opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.02)
        stats=train_step(model,aux,opt,examples(),config(reference_probability=1.),1.,audit=True)
        self.assertEqual(stats['reference_eligible'],0);self.assertEqual(stats['weighted_time_loss'],0)
        self.assertGreater(stats['classification_loss'],0)
        self.assertTrue(any(not torch.equal(v,before[n]) for n,v in model.state_dict().items()))

    def test_auxiliary_changes_detector_and_microbatch_accumulation_matches(self):
        rows=examples(True);model=model_fixture();other=copy.deepcopy(model);ce_model=copy.deepcopy(model)
        def run(m,microbatch,mode):
            aux=TFCL(8,2,21,mode);opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=.02)
            return train_step(m,aux,opt,rows,config(microbatch=microbatch,tfcl_mode=mode,
                profile_force_reference=True,alignment_min_cosine=-1.,alignment_unique_margin=-2.,
                alignment_drop_cost=2.,alignment_min_coverage=0.),1.,audit=True)
        a=run(model,12,'weighted');b=run(other,3,'weighted');run(ce_model,12,'ce')
        self.assertGreater(a['aux_gradient_audit']['aux_feature_grad'],0)
        for (n,p),q in zip(model.named_parameters(),other.parameters()):
            torch.testing.assert_close(p,q,atol=3e-6,rtol=5e-5,msg=n)
        self.assertTrue(any(not torch.equal(p,q) for p,q in zip(model.parameters(),ce_model.parameters())))
        self.assertEqual(a['reference_total'],4);self.assertEqual(a['reference_eligible'],4)

    def test_head_warmup_is_no_ssl_graph_then_trainable_suffix_updates(self):
        m=model_fixture();cfg=config(head_warmup_updates=200)
        warm,structure=stage(m,cfg,0,100)
        self.assertTrue(m.head_only);self.assertEqual((warm,structure),(0.,0.))
        rows=examples();aux=TFCL(8,2,21);opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=.02)
        train_step(m,aux,opt,rows,cfg,warm)
        self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
        warm,structure=stage(m,cfg,26,100);self.assertFalse(m.head_only)
        train_step(m,aux,opt,rows,cfg,warm)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.backbone.encoder.layers[-1].parameters()))
        self.assertTrue(all(p.grad is None for p in m.backbone.encoder.layers[0].parameters()))

    def test_original_official_ssl_loading_reproducible_fresh_head_without_rng_change(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture=model_fixture();fixture.backbone.save_pretrained(folder)
            head=asdict(small_head());head.pop('input_dim')
            cfg=config(init_mode='pretrained',pretrained_path=folder,seed=31601,
                adapter_hidden=4,adapter_max_ratio=.1,checkpointing=False,new_head_config=head)
            rng=torch.get_rng_state().clone();a=load_model(cfg);b=load_model(cfg)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for p,q in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(p,q))
            for p,q in zip(a.backbone.parameters(),fixture.backbone.parameters()):self.assertTrue(torch.equal(p,q))
            self.assertTrue(any(not torch.equal(p,q) for p,q in zip(a.head.blocks[0].parameters(),fixture.head.blocks[0].parameters())))


if __name__=='__main__':unittest.main()
