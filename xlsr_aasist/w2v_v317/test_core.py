"""Behavioral checks for frozen bases, the shared path and source budgets."""
from dataclasses import asdict
import math
import unittest
import numpy as np
import torch
from w2v_v3.test_model_step import tiny_detector
from .model import Detector,LoRALinear,optimizer_for,inventory
from .objectives import TFCL
from .data import SourcePlan,validate_batch
from .monitor import diagnostics
from .state import partial_state,apply_partial,promote


def config():
    return dict(lora_layers=1,lora_rank=2,lora_alpha=4.,lora_dropout=0.,feature_dim=8,
        head_expansion=32,head_blocks=2,block_dropout=0.,classifier_dropout=0.,
        fusion_chunk_layers=2,checkpointing=False,lora_lr=1e-3,head_lr=1e-3,tfcl_lr=1e-3,weight_decay=.01)


def model_fixture():
    return Detector(tiny_detector().backbone,config())


class CoreTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(317)

    def test_lora_starts_at_exact_base_and_base_stays_frozen(self):
        base=torch.nn.Linear(8,8);layer=LoRALinear(base,2,4.,0.)
        x=torch.randn(3,4,8)
        self.assertTrue(torch.equal(layer(x),base(x)))
        before={k:v.clone() for k,v in base.state_dict().items()}
        optimizer=torch.optim.AdamW([layer.lora_A,layer.lora_B],lr=.01)
        for _ in range(2):
            optimizer.zero_grad();layer(x).square().mean().backward();optimizer.step()
        self.assertTrue(layer.lora_B.abs().sum()>0)
        self.assertTrue(all(torch.equal(v,base.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.grad is None for p in base.parameters()))

    def test_tfcl_reaches_multiconv_fusion_and_lora_b(self):
        model=model_fixture();aux=TFCL(8,2,21)
        x=torch.randn(2,19,160);mask=torch.ones(2,19,dtype=torch.long)
        _,frames=model(x,mask)
        t,s,valid=aux.forward_batch([frames[0]],[frames[1]],[mask[0].bool()],[mask[1].bool()])
        names=[('fusion',model.head.projection.weight),('multiconv',model.head.blocks[0].up.weight)]
        names.extend((n,p) for n,p in model.backbone.named_parameters() if n.endswith('lora_B'))
        grads=torch.autograd.grad(t.sum()+s.sum(),[p for _,p in names],allow_unused=True)
        for (name,_),grad in zip(names,grads):
            self.assertIsNotNone(grad,name);self.assertTrue(torch.isfinite(grad).all(),name)
            self.assertGreater(float(grad.abs().sum()),0.,name)
        self.assertTrue(valid.all())

    def test_no_classifier_bypass_and_masked_padding_is_inert(self):
        model=model_fixture().eval();x=torch.randn(2,20,160)
        mask=torch.ones(2,20,dtype=torch.long);mask[0,13:]=0
        z,_=model(x,mask);changed=x.clone();changed[0,13:]=10000.
        actual,_=model(changed,mask);torch.testing.assert_close(z,actual)
        handle=model.head.forensic.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
        try:
            z,_=model(x,mask)
            torch.testing.assert_close(z[0],z[1],rtol=0.,atol=0.)
        finally:handle.remove()

    def test_real_optimizer_updates_only_lora_and_head_and_roundtrips(self):
        model=model_fixture();aux=TFCL(8,2,21);opt=optimizer_for(model,config(),aux)
        before={n:p.detach().clone() for n,p in model.backbone.named_parameters() if not p.requires_grad}
        x=torch.randn(2,20,160);mask=torch.ones(2,20,dtype=torch.long)
        for _ in range(2):
            opt.zero_grad();logits,frames=model(x,mask)
            loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1]))
            t,s,_=aux.forward_batch([frames[0]],[frames[1]],[mask[0].bool()],[mask[1].bool()])
            (loss+.15*(t.sum()+s.sum())).backward();opt.step()
        for n,p in model.backbone.named_parameters():
            if n in before:self.assertTrue(torch.equal(before[n],p),n)
        state=partial_state(model)
        self.assertTrue(all(n.startswith('head.') or n.endswith(('lora_A','lora_B')) for n in state))
        # Rebuild with the exact same frozen backbone, then load only trainables.
        import copy
        other=copy.deepcopy(model);apply_partial(other,state)
        model.eval();other.eval();torch.testing.assert_close(model(x,mask)[0],other(x,mask)[0],rtol=0,atol=0)
        self.assertGreater(inventory(model,aux)['lora'],0)
        self.assertLess(inventory(model,aux)['lora'],sum(p.numel() for p in model.backbone.parameters()))

    def test_checkpointed_lora_gradients_and_padded_training_match(self):
        import copy
        model=model_fixture()
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
        checked=copy.deepcopy(model);checked.backbone.encoder.gradient_checkpointing=True
        x=torch.randn(2,23,160);mask=torch.ones(2,23,dtype=torch.long);mask[0,17:]=0
        z,_=model(x,mask);other,_=checked(x,mask)
        torch.testing.assert_close(z,other,atol=1e-6,rtol=1e-5)
        z.square().sum().backward();other.square().sum().backward()
        for (n,p),(m,q) in zip(model.named_parameters(),checked.named_parameters()):
            if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=1e-4)
        model.eval();batched=model(x,mask)[0]
        exact=model(x[:1,:17],mask[:1,:17])[0][0]
        torch.testing.assert_close(batched[0],exact,atol=2e-5,rtol=2e-4)

    def test_microbatch_split_preserves_weighted_objective_with_missing_online(self):
        import copy
        from .step import train_step
        model=model_fixture();aux=TFCL(8,2,21)
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
        other,other_aux=copy.deepcopy(model),copy.deepcopy(aux)
        rows=[];serial=0
        for (language,label),quota in {('en',0):4,('en',1):2,('zh',0):7,('zh',1):3}.items():
            for i in range(quota):
                key=f'{language}/{label}/{i}';length=12+serial%4;serial+=1
                full=torch.randn(1,length,160)
                roles={'offline':.1,'online':.5,'noisy':.4} if i else {'offline':.2,'noisy':.8}
                for role,mass in roles.items():
                    rows.append(dict(split='train',language=language,label=label,source_id=key,
                        group_id=key,source_sha256=key,pair_occurrence=key,role=role,
                        source_mass=.25/quota,ce_weight=.25/quota*mass,
                        features=full+torch.randn_like(full)*.01,mask=torch.ones(1,length,dtype=torch.long),
                        aux_valid=np.ones(length,dtype=bool)))
        cfg=dict(device='cpu',amp='none',source_batch=16,microbatch=3,frame_budget=2000,
                 max_grad_norm=1e9,tfcl_time_weight=.15,tfcl_structure_weight=.15)
        def run(m,a,c):
            parameters=[p for mod in (m,a) for p in mod.parameters() if p.requires_grad]
            optimizer=torch.optim.SGD(parameters,lr=0.)
            return train_step(m,a,optimizer,rows,c,1.)
        a=run(model,aux,cfg);b=run(other,other_aux,dict(cfg,microbatch=48))
        for key in ('classification_loss','weighted_time_loss','weighted_structure_loss','total_loss'):
            self.assertAlmostEqual(a[key],b[key],places=5,msg=key)
        self.assertEqual(a['bridge_pairs'],12);self.assertEqual(a['matched_pairs'],16)
        for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
            if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=3e-4)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA is unavailable locally')
    def test_cuda_bf16_checkpointed_forward_and_backward(self):
        if not torch.cuda.is_bf16_supported():self.skipTest('BF16 unsupported')
        model=model_fixture().cuda();model.backbone.encoder.gradient_checkpointing=True
        aux=TFCL(8,2,21).cuda();mask=torch.ones(2,30,dtype=torch.long,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z,h=model(torch.randn(2,30,160,device='cuda'),mask)
            t,s,_=aux.forward_batch([h[0]],[h[1]],[mask[0].bool()],[mask[1].bool()])
            loss=z.float().square().mean()+t.sum()+s.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for n,p in model.named_parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),n)

    def test_balanced_diagnostic_is_not_dominated_by_majority_or_ties(self):
        rows=[dict(condition='online',language=l,label=y) for l,y in [('en',0),('en',1),('zh',0),('zh',1)]]
        result=diagnostics(rows,np.zeros((4,2)))
        self.assertAlmostEqual(result['conditions']['online']['balanced_ce'],math.log(2))
        self.assertEqual(result['groups']['online/en/0']['recall'],1.)
        self.assertEqual(result['groups']['online/en/1']['recall'],0.)

    def test_best_never_uses_untrained_or_historical_baseline(self):
        state=dict(best_metrics=None,best_tag=None)
        m=dict(complete=True,weighted_f1=.91,noisy_f1=.90,clean_f1=.94)
        self.assertTrue(promote(state,'epoch_1_step_10',{},m,{'version':'3.17'}))
        self.assertFalse(promote(state,'epoch_2_step_10',{},dict(m,weighted_f1=.9),{'version':'3.17'}))
        with self.assertRaises(ValueError):promote(state,'bad',{},dict(m,complete=False),{})


if __name__=='__main__':unittest.main()
