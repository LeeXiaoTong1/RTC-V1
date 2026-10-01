"""Source budgets, true-source pairs, shared gradients and one-pass execution."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from w2v_v3.model import diversity_cka
from w2v_v3.test_model_step import tiny_detector
from .model import install_runtime
from .losses import (source_components,local_tokens,aligned_temporal_loss,
                     pair_objective,reference_objective)
from .step import supervised_step

torch.set_num_threads(1)


def examples(count=3,dim=160):
    rows=[]
    for source in range(count):
        conditions=['offline','noisy_a','noisy_b']+(['online'] if source%2==0 else [])
        for c,condition in enumerate(conditions):
            for view,length,weight in (('full',13+source*2+c,.7),('short',9+source,.3)):
                rows.append(dict(id=f'{source}/{condition}',source_id=f'offline/en/{source}.wav',
                    source_group=f'{source}:{condition}',condition=condition,view=view,
                    view_weight=weight,label=source%2,noisy=condition.startswith('noisy'),
                    features=torch.randn(1,length,dim),mask=torch.ones(1,length).long()))
    return rows


def model():
    m=install_runtime(tiny_detector(checkpointing=True)).configure_trainable_layers(1).train()
    for module in m.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0
    return m


class SourceLossTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(333)

    def test_class_weight_once_and_condition_means_with_missing_online(self):
        rows=examples();weights=torch.tensor([.7,1.8]);parts=source_components(rows,weights,'cpu')
        self.assertEqual(parts['source_count'],3)
        self.assertEqual(len(parts['offline_full']),3)
        for source in range(3):
            indices=[i for i,e in enumerate(rows) if e['source_id']==f'offline/en/{source}.wav']
            expected=weights[source%2]/3
            torch.testing.assert_close(parts['coefficients'][indices].sum(),expected)
            ordinary=[i for i in indices if not rows[i]['noisy']]
            noisy=[i for i in indices if rows[i]['noisy']]
            torch.testing.assert_close(parts['coefficients'][ordinary].sum(),expected*.5)
            torch.testing.assert_close(parts['coefficients'][noisy].sum(),expected*.5)
        with self.assertRaisesRegex(ValueError,'separate noisy'):
            source_components(rows,weights,'cpu',noisy_class_weights=torch.ones(2))

    def test_invalid_pairs_and_source_budgets_rejected_before_forward(self):
        for mutation in ('missing_noisy','wrong_label','duplicate_full','bad_view_weight'):
            with self.subTest(mutation=mutation):
                rows=examples();m=model();before=copy.deepcopy(m.state_dict())
                if mutation=='missing_noisy':rows=[r for r in rows if not(r['source_id'].endswith('0.wav') and r['condition']=='noisy_b')]
                if mutation=='wrong_label':rows[2]['label']=1
                if mutation=='duplicate_full':rows[1]['view']='full'
                if mutation=='bad_view_weight':rows[0]['view_weight']=.9
                optimizer=torch.optim.AdamW(m.parameters(),lr=2e-6)
                seen=[];hook=m.backbone.register_forward_pre_hook(lambda *_:seen.append(1))
                with self.assertRaises(ValueError):supervised_step(m,rows,optimizer,torch.ones(2),'cpu',amp='none')
                hook.remove();self.assertFalse(seen);self.assertFalse(optimizer.state)
                for k,v in before.items():torch.testing.assert_close(m.state_dict()[k],v,rtol=0,atol=0)

    def test_masked_soft_alignment_gradients_reach_both_sides_and_ignore_poison(self):
        x=torch.randn(21,8,requires_grad=True);y=torch.randn(27,8,requires_grad=True)
        left=local_tokens(x,torch.ones(21).bool(),max_tokens=12)
        mask=torch.arange(27)<19
        poison=torch.where(mask[:,None],y,torch.full_like(y,float('nan')))
        right=local_tokens(poison,mask,max_tokens=12)
        loss,*_=aligned_temporal_loss(left,right);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(x.grad.abs().sum()),0.)
        self.assertGreater(float(y.grad[:19].abs().sum()),0.)
        self.assertEqual(float(y.grad[19:].abs().sum()),0.)
        self.assertEqual(len(left[0]),12)

    def test_silence_is_valid_and_explicit_unusable_frames_skip_only_pair(self):
        frames=torch.zeros(15,8,requires_grad=True)
        tokens=local_tokens(frames,torch.ones(15).bool(),max_tokens=8)
        self.assertIsNotNone(tokens)
        self.assertIsNone(local_tokens(frames,torch.ones(15).bool(),8,torch.zeros(15).bool()))
        rows=examples(count=1);parts=source_components(rows,torch.ones(2),'cpu')
        records={i:(torch.ones(13,8,requires_grad=True),torch.ones(13).bool(),None)
                 for i,e in enumerate(rows) if e['view']=='full'}
        loss,stats=pair_objective(records,parts,max_tokens=8)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(stats['pair_valid_count'],3)
        self.assertEqual(float(stats['pair_collapsed_count']),3.)
        self.assertEqual(float(stats['pair_variance']),0.)
        target=parts['groups'][rows[0]['source_id']]['full']['online']
        records[target]=(records[target][0],records[target][1],torch.zeros(13).bool())
        _,stats=pair_objective(records,parts,max_tokens=8)
        self.assertEqual(stats['pair_valid_count'],2);self.assertEqual(stats['pair_skipped_count'],1)
        self.assertEqual(stats['pair_en_fake_sources'],1)
        self.assertEqual(stats['pair_en_fake_valid_pairs'],2)
        self.assertEqual(stats['pair_en_fake_skipped_pairs'],1)
        self.assertEqual(stats['pair_en_real_sources'],0)

    def test_explicit_internal_gap_preserves_time_positions_and_rejects_poison(self):
        frames=torch.randn(30,8,requires_grad=True)
        audible=torch.ones(30).bool();audible[10:20]=False
        poisoned=frames.masked_fill(~audible[:,None],float('nan'))
        tokens=local_tokens(poisoned,torch.ones(30).bool(),max_tokens=30,audible_mask=audible)
        self.assertEqual(len(tokens[0]),20)
        self.assertGreater(float(tokens[1][10]-tokens[1][9]),.3)
        self.assertTrue(torch.isfinite(tokens[0]).all())
        tokens[0].square().sum().backward()
        self.assertEqual(float(frames.grad[10:20].abs().sum()),0.)

    def test_pair_mean_is_within_source_not_number_of_available_conditions(self):
        rows=examples(count=2);parts=source_components(rows,torch.tensor([.7,1.4]),'cpu')
        records={};expected=[]
        for source,group in parts['groups'].items():
            reference=torch.randn(11,8,requires_grad=True)
            target=torch.randn(11,8,requires_grad=True)
            for condition,index in group['full'].items():
                records[index]=(reference if condition=='offline' else target,torch.ones(11).bool(),None)
            value=aligned_temporal_loss(local_tokens(reference,torch.ones(11).bool()),
                local_tokens(target,torch.ones(11).bool()))[0]
            expected.append(value*parts['weights'][group['label']]/2)
        actual,stats=pair_objective(records,parts)
        torch.testing.assert_close(actual,sum(expected),rtol=1e-6,atol=1e-7)
        self.assertEqual(stats['pair_valid_count'],5)


class StepTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(337)

    def test_joint_loss_gradients_and_one_encoder_pass_per_view(self):
        reference=model();actual=copy.deepcopy(reference);rows=examples();weights=torch.tensor([.7,1.4])
        logits=[];blocks=[];frames={}
        for i,e in enumerate(rows):
            z,h,f,m=reference.forward_training(e['features'],e['mask'])
            logits.append(z);blocks.append(h)
            if e['view']=='full':frames[i]=(f[0],m[0],None)
        z=torch.cat(logits);h=torch.cat(blocks)
        loss,expected=reference_objective(z,h,rows,weights,frames,cka_weight=.01,pair_weight=.02,aux_max_tokens=16)
        loss.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
        torch.optim.SGD(reference.parameters(),lr=2e-4).step()
        seen=[];hook=actual.backbone.register_forward_pre_hook(lambda _m,_a,kw:seen.append(len(kw['input_features'])),with_kwargs=True)
        stats,scores=supervised_step(actual,rows,torch.optim.SGD(actual.parameters(),lr=2e-4),weights,
            'cpu',amp='none',cka_weight=.01,pair_weight=.02,aux_max_tokens=16,microbatch=4,frame_budget=80,
            diagnose_aux_grad=True,offload_activations=False)
        hook.remove();self.assertEqual(sum(seen),len(rows));self.assertEqual(len(seen),stats['encoder_forward_microbatches'])
        torch.testing.assert_close(scores,z.detach(),rtol=3e-5,atol=3e-6)
        self.assertAlmostEqual(stats['loss'],float(loss),places=5)
        self.assertEqual(stats['source_count'],3);self.assertEqual(stats['cka_full_views'],3)
        self.assertEqual(stats['pair_valid_count'],8)
        self.assertEqual(stats['pair_en_real_sources'],1)
        self.assertEqual(stats['pair_en_real_valid_pairs'],2)
        self.assertEqual(stats['pair_en_fake_valid_pairs'],6)
        self.assertGreater(stats['pair_shared_feature_grad_norm'],0.)
        self.assertGreater(stats['ce_shared_feature_grad_norm'],0.)
        for (name,a),(_,b) in zip(actual.named_parameters(),reference.named_parameters()):
            torch.testing.assert_close(a,b,rtol=4e-5,atol=4e-6,msg=name)
            if a.grad is None:self.assertIsNone(b.grad)
            else:torch.testing.assert_close(a.grad,b.grad,rtol=6e-5,atol=5e-6,msg=name)

    def test_pair_only_reaches_shared_encoder_and_head_without_classifier_params(self):
        m=model();rows=examples(count=1);parts=source_components(rows,torch.ones(2),'cpu');records={}
        for i,e in enumerate(rows):
            if e['view']!='full':continue
            _,_,f,mask=m.forward_training(e['features'],e['mask']);records[i]=(f[0],mask[0],None)
        loss,_=pair_objective(records,parts,max_tokens=16);loss.backward()
        self.assertGreater(float(m.head.blocks[-1].down.weight.grad.abs().sum()),0.)
        encoder_grad=sum(float(p.grad.abs().sum()) for p in m.backbone.encoder.layers[-1].parameters() if p.grad is not None)
        self.assertGreater(encoder_grad,0.)
        self.assertTrue(all(p.grad is None for p in m.head.classifier.parameters()))

    def test_zero_pair_arm_has_same_source_objective_and_no_pair_gradient_diagnostics(self):
        m=model();rows=examples();stats,_=supervised_step(m,rows,torch.optim.SGD(m.parameters(),lr=0.),torch.ones(2),
            'cpu',amp='none',pair_weight=0.,cka_weight=.01,diagnose_aux_grad=True)
        self.assertEqual(stats['pair_loss'],0.)
        self.assertEqual(stats['pair_valid_count'],0)
        self.assertNotIn('pair_shared_feature_grad_norm',stats)
        self.assertEqual(stats['source_count'],3)

    def test_nonfinite_pair_loss_does_not_advance_optimizer_or_weights(self):
        m=model();before=copy.deepcopy(m.state_dict());rows=examples(count=1)
        optimizer=torch.optim.AdamW(m.parameters(),lr=2e-6)
        from . import step as training_step
        original=training_step.pair_objective
        def invalid(*args,**kwargs):
            loss,stats=original(*args,**kwargs)
            return loss*float('nan'),stats
        with patch.object(training_step,'pair_objective',side_effect=invalid):
            with self.assertRaisesRegex(FloatingPointError,'not advanced'):
                supervised_step(m,rows,optimizer,torch.ones(2),'cpu',amp='none',aux_max_tokens=16)
        self.assertFalse(optimizer.state)
        for name,value in before.items():torch.testing.assert_close(m.state_dict()[name],value,rtol=0,atol=0)

    def test_real_dataset_collator_source_weights_and_audibility_groups(self):
        import numpy as np
        import soundfile as sf
        from .test_data import fixture
        from .data import PairedEpochPlan,loader
        with tempfile.TemporaryDirectory() as directory:
            cfg,records,sources,cache=fixture(Path(directory),per_group=1,samples=11200)
            cfg.update(source_batch=4,short_min_seconds=.4,short_max_seconds=.45,
                       short_prefix_probability=.5)
            # This input remains supervised by CE. Only its genuinely absent
            # speech information is excluded from auxiliary alignment.
            en_real=next(s for s in sources if s['language']=='en' and s['label']==1)
            sf.write(en_real['audio'],np.zeros(11200,np.float32),16000,subtype='FLOAT')
            plan=PairedEpochPlan(sources,cache,4,cfg['seed'])
            batch=next(iter(loader(plan,cfg,training=True,epoch=1,batches=plan.batches(1))))
            self.assertTrue(hasattr(batch,'batches'))
            self.assertTrue(all(e['audibility_mask'].shape==e['mask'][0].shape for e in batch))
            parts=source_components(batch,plan.weights,'cpu')
            self.assertEqual(parts['source_count'],4)
            self.assertEqual(len(parts['offline_full']),4)
            m=model();seen=[]
            hook=m.backbone.register_forward_pre_hook(lambda _m,_a,kw:seen.append(len(kw['input_features'])),with_kwargs=True)
            stats,scores=supervised_step(m,batch,torch.optim.AdamW(m.parameters(),lr=2e-6),plan.weights,
                'cpu',amp='none',pair_weight=.02,aux_max_tokens=16,
                microbatch=cfg['microbatch'],frame_budget=cfg['frame_budget'],diagnose_aux_grad=True)
            hook.remove()
            self.assertEqual(sum(seen),len(batch));self.assertEqual(scores.shape,(len(batch),2))
            self.assertEqual(stats['source_count'],4)
            self.assertEqual(stats['pair_en_real_sources'],1)
            self.assertEqual(stats['pair_en_real_valid_pairs'],0)
            self.assertEqual(stats['pair_en_real_skipped_pairs'],2)
            self.assertGreater(stats['pair_zh_real_valid_pairs'],0)
            self.assertTrue(torch.isfinite(torch.tensor(stats['loss'])))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA BF16 pair smoke requires CUDA')
    def test_cuda_bf16_pair_backward_is_finite(self):
        m=model().cuda();rows=examples(count=2)
        stats,_=supervised_step(m,rows,torch.optim.AdamW(m.parameters(),lr=2e-6),torch.ones(2),
            'cuda',amp='bf16',pair_weight=.02,aux_max_tokens=16,gpu_activation_gib=.1)
        self.assertTrue(torch.isfinite(torch.tensor(stats['loss'])))
        self.assertGreater(stats['pair_valid_count'],0)


if __name__=='__main__':unittest.main()
