"""Gradient direction, real/condition isolation, actual fine-tuning and atomic state."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from w2v_v3.test_model_step import tiny_detector
from w2v_v32.model import install_runtime
from .data import SourcePlan, loss_weights, probe_split
from .model import FeatureClassifier, Reverse, LanguageAdversary, optimizer_for
from .state import (partial_state, apply_partial, atomic_save, capture_rng, restore_rng,
                    to_cpu, storage_budget, identity, SCHEMA, load_resume)
from .train import train_step, acceptance
from .probes import run_probes, compare

torch.set_num_threads(1)


def config():
    return dict(device='cpu',seed=31001,amp='none',encoder_lr=1e-4,head_lr=2e-4,
        adapter_lr=1e-3,adversary_lr=1e-3,layer_decay=.8,weight_decay=1e-4,
        max_grad_norm=1.,microbatch=3,frame_budget=2400,probe_sources_per_language=4,
        probe_steps=35,probe_hidden=8,probe_min_drop=.01,disk_margin_bytes=0,
        gradient_protection=True,retention_weight=.2,retention_min_margin=1.,retention_cap=4.,retention_slack=.5)


def rows(count=24):
    return [dict(source_id=f'{language}_{label}_{i}',group_id=f'{language}_{label}_{i}',
        id=f'{language}_{label}_{i}_{condition}',language=language,label=label,
        condition=condition,view='full',split='train',retention_target=1.)
        for language in ('en','zh') for label in (0,1) for i in range(count)
        for condition in ('offline','online','noisy_a','noisy_b')]


def detector():
    torch.manual_seed(7)
    model = install_runtime(tiny_detector(checkpointing=True))
    model.head.classifier = FeatureClassifier(model.head.classifier,8)
    model.configure_trainable_layers(1)
    return model


class GradientTests(unittest.TestCase):
    def test_reversal_flips_only_feature_gradient_not_reader_learning(self):
        x = torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
        reader = nn.Linear(2,2)
        target = torch.tensor([0,1])
        a = nn.functional.cross_entropy(reader(x),target)
        gx, gw = torch.autograd.grad(a,(x,reader.weight))
        b = nn.functional.cross_entropy(reader(Reverse.apply(x,.3)),target)
        rx, rw = torch.autograd.grad(b,(x,reader.weight))
        torch.testing.assert_close(rx,-.3*gx)
        torch.testing.assert_close(rw,gw)

    def test_only_real_same_condition_heads_get_gradients_and_microbatch_objective_matches(self):
        _, adv_weights = loss_weights(rows(1))
        data = rows(1)
        h = torch.randn(len(data),6,requires_grad=True)
        critic = LanguageAdversary(6,4)
        loss = critic.loss(h,data,adv_weights,.2)
        grad = torch.autograd.grad(loss,h,retain_graph=True)[0]
        self.assertEqual(float(grad[[i for i,r in enumerate(data) if r['label']==0]].abs().sum()),0.)
        self.assertGreater(float(grad.abs().sum()),0.)
        parts = sum(critic.loss(h[i:i+3],data[i:i+3],adv_weights[i:i+3],.2) for i in range(0,len(data),3))
        torch.testing.assert_close(parts,loss)
        torch.testing.assert_close(torch.autograd.grad(parts,h)[0],grad,atol=2e-8,rtol=1e-5)

    def test_zero_adapter_preserves_original_forward_and_true_model_layers_update(self):
        model = detector().eval()
        x, mask = torch.randn(2,21,160), torch.ones(2,21,dtype=torch.long)
        with torch.no_grad():
            z,_ = model(x,mask)
            h = model.head.classifier.features
            torch.testing.assert_close(z,model.head.classifier[-1](h),rtol=0,atol=0)
            stats = torch.randn(3,model.head.classifier[0].in_features)
            plain = model.head.classifier[2](model.head.classifier[1](model.head.classifier[0](stats)))
            torch.testing.assert_close(model.head.classifier(stats),plain,rtol=0,atol=0)
        critic = LanguageAdversary(model.head.classifier[-1].in_features,8)
        optimizer = optimizer_for(model,critic,config())
        before = {k:p.detach().clone() for k,p in model.named_parameters()}
        samples = [dict(r,features=torch.randn(1,20+i%3,160),mask=torch.ones(1,20+i%3,dtype=torch.long))
                   for i,r in enumerate(rows(1))]
        value = train_step(model,critic,optimizer,samples,config(),.1)
        self.assertTrue(np.isfinite(value['classification_loss']))
        updated = {k for k,p in model.named_parameters() if not torch.equal(p,before[k])}
        self.assertTrue(any(k.startswith('backbone.encoder.layers.1.') for k in updated))
        self.assertTrue(any(k.startswith('head.blocks.') for k in updated))
        self.assertTrue(any(k.startswith('head.classifier.adapter.') for k in updated))
        self.assertFalse(any(k.startswith('backbone.encoder.layers.0.') for k in updated))
        for name,p in model.named_parameters():
            if not p.requires_grad:
                self.assertIsNone(p.grad,name)
        self.assertIsNone(model.head.classifier.features)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA validation runs on the training server')
    def test_cuda_bf16_finetuning_and_partial_deployment(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest('GPU lacks BF16 support')
        cfg = dict(config(),device='cuda:0',amp='bf16')
        model = detector().to('cuda:0')
        critic = LanguageAdversary(model.head.classifier[-1].in_features,8).to('cuda:0')
        optimizer = optimizer_for(model,critic,cfg)
        batch = [dict(r,features=torch.randn(1,20+i%3,160),mask=torch.ones(1,20+i%3,dtype=torch.long))
                 for i,r in enumerate(rows(1))]
        result = train_step(model,critic,optimizer,batch,cfg,.1)
        self.assertTrue(np.isfinite(result['gradient_norm']))
        restored = detector().to('cuda:0')
        apply_partial(restored,partial_state(model))
        model.eval(); restored.eval()
        x,mask = torch.randn(2,21,160,device='cuda:0'),torch.ones(2,21,dtype=torch.long,device='cuda:0')
        with torch.no_grad():
            torch.testing.assert_close(model(x,mask)[0],restored(x,mask)[0],atol=2e-5,rtol=1e-5)


class DataTests(unittest.TestCase):
    def test_balanced_source_budget_resume_order_and_holdout_source_separation(self):
        training, probes, ids, split = probe_split(rows(),config())
        self.assertFalse({r['group_id'] for r in training} & {r['group_id'] for r in probes})
        self.assertFalse(set(split['fit_groups']) & set(split['test_groups']))
        self.assertEqual(len(probes),32)
        plan = SourcePlan(training,16,55)
        for epoch in (0,1):
            full = plan.batches(epoch)
            self.assertEqual(plan.batches(epoch,2),full[2:])
            for batch in full:
                r = [training[i] for i in batch]
                ce,adv = loss_weights(r)
                for lang in ('en','zh'):
                    for label in (0,1):
                        self.assertAlmostEqual(sum(w for w,s in zip(ce,r) if s['language']==lang and s['label']==label),.25)
                    for condition in ('offline','online','noisy_a','noisy_b'):
                        self.assertAlmostEqual(sum(w for w,s in zip(adv,r) if s['language']==lang and s['condition']==condition),.125)
        self.assertNotEqual(plan.batches(0),plan.batches(1))

    def test_missing_one_language_in_online_excludes_that_adversary_only(self):
        data = [r for r in rows(1) if not (r['language']=='en' and r['condition']=='online')]
        ce,adv = loss_weights(data)
        self.assertAlmostEqual(sum(ce),1.)
        self.assertAlmostEqual(sum(adv),1.)
        self.assertEqual(sum(w for w,r in zip(adv,data) if r['condition']=='online'),0.)

    def test_fresh_probes_detect_language_and_do_not_consume_training_rng(self):
        _,probe,_,split = probe_split(rows(40),dict(config(),probe_sources_per_language=8))
        signal = np.array([[5 if r['language']=='zh' else -5,0,0,1] for r in probe],dtype=np.float32)
        torch.manual_seed(14)
        before = torch.get_rng_state().clone()
        measured = run_probes(signal,probe,split,config())
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertGreater(measured['readability'],.95)
        empty = run_probes(np.zeros_like(signal),probe,split,config())
        self.assertLessEqual(empty['readability'],.55)
        # A collapsed detector does not qualify as successful language debiasing.
        self.assertFalse(compare(measured,empty,config())['evidence'])


class StateTests(unittest.TestCase):
    def test_partial_resume_reproduces_next_optimizer_update_and_frozen_prefix_is_absent(self):
        cfg = config()
        model = detector()
        critic = LanguageAdversary(model.head.classifier[-1].in_features,8)
        optimizer = optimizer_for(model,critic,cfg)
        batch = [dict(r,features=torch.randn(1,20,160),mask=torch.ones(1,20,dtype=torch.long)) for r in rows(1)]
        train_step(model,critic,optimizer,batch,cfg,.05)
        state = dict(schema=SCHEMA,identity=identity(cfg),model=partial_state(model),adversary=to_cpu(critic.state_dict()),
                     optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng())
        self.assertFalse(any(k.startswith('backbone.encoder.layers.0.') for k in state['model']))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'last.pt'
            atomic_save(path,state,0)
            saved = load_resume(path,cfg)
            restore_rng(state['rng'])
            train_step(model,critic,optimizer,batch,cfg,.1)
            reference = partial_state(model)
            model2 = detector()
            critic2 = LanguageAdversary(model2.head.classifier[-1].in_features,8)
            optimizer2 = optimizer_for(model2,critic2,cfg)
            apply_partial(model2,saved['model'])
            critic2.load_state_dict(saved['adversary'])
            optimizer2.load_state_dict(saved['optimizer'])
            restore_rng(saved['rng'])
            train_step(model2,critic2,optimizer2,batch,cfg,.1)
            for k,v in partial_state(model2).items():
                torch.testing.assert_close(v,reference[k],rtol=0,atol=0)
            with self.assertRaises(ValueError):
                apply_partial(model2,dict(saved['model'],unknown=torch.ones(1)))
            with self.assertRaises(ValueError):
                load_resume(path,dict(cfg,seed=2))

    def test_failed_write_keeps_previous_state_and_disk_budget_covers_atomic_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'last.pt'
            atomic_save(path,{'a':torch.ones(2)},0)
            before = path.read_bytes()
            with patch('w2v_v311.state.os.replace',side_effect=OSError('simulated full disk')):
                with self.assertRaises(OSError):
                    atomic_save(path,{'a':torch.zeros(2)},0)
            self.assertEqual(before,path.read_bytes())
            self.assertFalse(path.with_name('last.pt.tmp').exists())
            budget = storage_budget(detector(),LanguageAdversary(32,8),config())
            self.assertEqual(budget['peak_new_run_bytes'],2*budget['estimated_last_bytes'])
            self.assertEqual(budget['new_audio_bytes'],0)


class AcceptanceTests(unittest.TestCase):
    def test_en_real_gain_must_protect_fake_and_ranking(self):
        cfg = dict(config(),min_gain=.001,max_clean_drop=.001,max_fake_drop=.002,
                   max_real_drop=.005,max_auc_drop=.0005,min_en_real_gain=.005,max_matched_real_drop=.005)
        groups = {c+'/'+lang:dict(recall=[.996,.8],class_counts=[5000,1000],auc=.98)
                  for c in ('online','seen','heldout') for lang in ('en','zh')}
        baseline = dict(complete=True,clean_f1=.98,noisy_f1=.95,weighted_f1=.959,groups=groups,
                        matched={g:dict(real_recall=.85) for g in groups})
        candidate = copy.deepcopy(baseline)
        candidate.update(noisy_f1=.954,weighted_f1=.9618)
        for c in ('online','seen','heldout'):
            candidate['groups'][c+'/en']['recall'][1] += .01
        self.assertTrue(acceptance(baseline,candidate,cfg)[0])
        worse_fake = copy.deepcopy(candidate)
        worse_fake['groups']['seen/en']['recall'][0] -= .003
        self.assertFalse(acceptance(baseline,worse_fake,cfg)[0])
        worse_ranking = copy.deepcopy(candidate)
        worse_ranking['groups']['heldout/en']['auc'] -= .001
        self.assertFalse(acceptance(baseline,worse_ranking,cfg)[0])
        no_real_gain = copy.deepcopy(candidate)
        for c in ('online','seen','heldout'):
            no_real_gain['groups'][c+'/en']['recall'][1] -= .01
        self.assertFalse(acceptance(baseline,no_real_gain,cfg)[0])


if __name__ == '__main__':
    unittest.main()
