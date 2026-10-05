"""Test actual update directions, preservation exclusions and bounded stopping."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from w2v_v39.common import atomic_json, digest
from .config import configuration, parser
from .control import stopping
from .data import loss_weights
from .model import ProtectedReverse, LanguageAdversary, optimizer_for
from .objectives import reference_rows, retention_loss
from .test_core import config, rows, detector
from .test_workflow import fixture
from .train import train_step, schedule


class ProtectionTests(unittest.TestCase):
    def test_conflict_removed_beneficial_component_kept_and_first_order_ce_protected(self):
        # Real CE direction is [1,0]. Gradient descent moves opposite the gradient.
        h = torch.tensor([[.2, 1.], [.3, 1.]], requires_grad=True)
        raw_reader_gradient = torch.tensor([[2., 3.], [-2., 3.]])
        audit = []
        out = ProtectedReverse.apply(h, .5, torch.tensor([1., 0.]), True, audit)
        (out*raw_reader_gradient).sum().backward()
        torch.testing.assert_close(h.grad, torch.tensor([[0., -1.5], [1., -1.5]]), rtol=0, atol=0)
        linear = nn.Linear(2,2)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[1.,0.],[0.,0.]])); linear.bias.zero_()
            before = nn.functional.cross_entropy(linear(h), torch.ones(2,dtype=torch.long),reduction='none')
            after = nn.functional.cross_entropy(linear(h-.001*h.grad), torch.ones(2,dtype=torch.long),reduction='none')
        self.assertTrue(bool((after <= before).all()))
        self.assertEqual(int(audit[0][0]), 2)
        self.assertEqual(int(audit[0][1]), 1)
        self.assertGreater(float(audit[0][3]), 0.)

    def test_critic_gradient_unchanged_fake_rows_isolated_and_microbatches_match(self):
        torch.manual_seed(19)
        data = rows(1)
        _, weights = loss_weights(data)
        h = torch.randn(len(data),6,requires_grad=True)
        critic = LanguageAdversary(6,8)
        parameters = list(critic.parameters())
        direction = torch.randn(6)
        plain = critic.loss(h,data,weights,.05)
        projected = critic.loss(h,data,weights,.05,direction,True,[])
        raw = torch.autograd.grad(plain,[h]+parameters,retain_graph=True)
        guarded = torch.autograd.grad(projected,[h]+parameters,retain_graph=True)
        for a,b in zip(raw[1:],guarded[1:]):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        real = [i for i,r in enumerate(data) if r['label']==1]
        fake = [i for i,r in enumerate(data) if r['label']==0]
        self.assertTrue(bool(((guarded[0][real]*direction).sum(-1) >= -1e-8).all()))
        self.assertEqual(float(guarded[0][fake].abs().sum()),0.)
        pieces = sum(critic.loss(h[i:i+3],data[i:i+3],weights[i:i+3],.05,direction,True,[])
                     for i in range(0,len(data),3))
        partial = torch.autograd.grad(pieces,[h]+parameters)
        for a,b in zip(guarded,partial):
            torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-8)

    def test_zero_pressure_and_explicit_ablation_do_not_remove_task_updates(self):
        for strength, protect in ((0.,True),(.5,False)):
            h = torch.ones(2,3,requires_grad=True)
            ProtectedReverse.apply(h,strength,torch.tensor([1.,0.,0.]),protect,[]).sum().backward()
            torch.testing.assert_close(h.grad,torch.full_like(h,-strength),rtol=0,atol=0)

    def test_known_wrong_reference_is_free_to_correct_and_correct_margin_is_one_sided(self):
        data = [dict(r,label=label) for r,label in zip(rows(1)[:5],[1,1,0,0,1])]
        z = np.array([[3.,0.],[0.,.2],[8.,0.],[1.5,0.],[0.,2.]],dtype=np.float32)
        augmented, inventory = reference_rows(data,z,config())
        targets = torch.tensor([r['retention_target'] for r in augmented])
        torch.testing.assert_close(targets,torch.tensor([0.,0.,4.,1.,1.5]))
        self.assertEqual(inventory['protected'],3)
        labels = torch.tensor([r['label'] for r in data])
        # Only rows 2 and 4 erode a known correct margin.
        current = torch.tensor([[2.,0.],[0.,.1],[2.,0.],[2.,0.],[0.,.5]],requires_grad=True)
        weights = torch.ones(5)/5
        loss, breaches = retention_loss(current,labels,weights,targets)
        gradient = torch.autograd.grad(loss,current)[0]
        self.assertEqual(breaches,2)
        self.assertTrue(torch.equal(gradient[:2],torch.zeros(2,2)))
        self.assertTrue(torch.equal(gradient[3],torch.zeros(2)))
        self.assertLess(float(gradient[2,0]),0.)
        self.assertLess(float(gradient[4,1]),0.)
        pieces = sum(retention_loss(current[i:i+1],labels[i:i+1],weights[i:i+1],targets[i:i+1])[0]
                     for i in range(5))
        torch.testing.assert_close(pieces,loss)
        with self.assertRaises(ValueError):
            reference_rows([dict(r,split='dev') for r in data],z,config())
        with self.assertRaises(ValueError):
            reference_rows(data,z[:-1],config())

    def test_one_encoder_forward_per_microbatch_and_bounded_gradient_diagnostics(self):
        model = detector()
        critic = LanguageAdversary(model.head.classifier[-1].in_features,8)
        cfg = dict(config(), microbatch=4)
        batch = [dict(r,features=torch.randn(1,20,160),mask=torch.ones(1,20,dtype=torch.long)) for r in rows(1)]
        calls = []
        hook = model.register_forward_hook(lambda *args: calls.append(1))
        value = train_step(model,critic,optimizer_for(model,critic,cfg),batch,cfg,.05)
        hook.remove()
        self.assertEqual(len(calls),len(batch)//4)
        self.assertEqual(value['language_gradient_rows'],8)
        self.assertLessEqual(value['removed_language_gradient_energy'],value['language_gradient_energy']+1e-8)
        self.assertGreater(value['language_gradient_energy'],0.)


class ControlTests(unittest.TestCase):
    def test_improving_unpromoted_candidate_is_not_a_plateau_and_catastrophic_drop_stops(self):
        cfg = dict(epochs=4,minimum_epochs=2,patience=4,progress_min_delta=.0002,catastrophic_weighted_drop=.02)
        baseline = dict(weighted_f1=.963)
        history = []
        for i,score in enumerate((.962,.9635,.964,.965)):
            value = dict(weighted_f1=score)
            control = stopping(history,value,(i+1)*50,100,baseline,cfg)
            self.assertFalse(control['stop'])
            history.append(dict(metrics=value,promoted=False))
        for i in range(4):
            value = dict(weighted_f1=.9649)
            control = stopping(history,value,250+50*i,100,baseline,cfg)
            history.append(dict(metrics=value,promoted=False))
        self.assertEqual(control['reason'],'candidate_progress_stalled')
        self.assertTrue(control['stop'])
        self.assertEqual(stopping([],dict(weighted_f1=.90),50,100,baseline,cfg)['reason'],'catastrophic_regression')

    def test_default_four_epochs_progressive_pressure_and_verified_v310_fallback_source(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp)
            source,cfg,_,_ = fixture(root)
            actual = configuration(parser().parse_args(['--source-run',str(source),'--device','cpu']))
            self.assertEqual(actual['epochs'],4)
            self.assertEqual(actual['trainable_layers'],8)
            self.assertTrue(actual['gradient_protection'])
            self.assertEqual(actual['seed'],31001)
            parameter=nn.Parameter(torch.zeros(1))
            optim=torch.optim.AdamW([dict(params=[parameter],lr=.01,initial_lr=.01,name='feature_adapter')])
            self.assertEqual(schedule(optim,0,400,100,actual),0.)
            self.assertEqual(schedule(optim,10,400,100,actual),0.)
            self.assertAlmostEqual(schedule(optim,55,400,100,actual),.025)
            self.assertAlmostEqual(schedule(optim,100,400,100,actual),.05)
            # V3.10 selected baseline is provenance, never its rejected last.pt.
            from w2v_v310.state import SCHEMA, identity, atomic_save
            prior=root/'v310';prior.mkdir()
            prior_cfg=dict(cfg,version='3.10')
            checkpoint=dict(schema=SCHEMA,identity=identity(prior_cfg),config=prior_cfg,model=None,selected='baseline')
            atomic_save(prior/'best.pt',checkpoint,0)
            done=dict(version='3.10',status='complete',selected='baseline',baseline_fallback=True,
                checkpoint_sha256=digest(prior/'best.pt'),base_checkpoint_sha256=cfg['base_checkpoint_sha256'])
            atomic_json(prior/'completed.json',done)
            result=configuration(parser().parse_args(['--source-run',str(prior),'--device','cpu']))
            self.assertEqual(result['source_version'],'3.10')
            self.assertEqual(result['base_checkpoint_sha256'],cfg['base_checkpoint_sha256'])
            self.assertNotIn(str(prior/'last.pt'),result['source_fingerprints'])
            # Do not silently replace a genuinely selected V3.10 candidate with the old base.
            checkpoint.update(model={},selected='epoch_1')
            atomic_save(prior/'best.pt',checkpoint,0)
            done.update(selected='epoch_1',baseline_fallback=False,checkpoint_sha256=digest(prior/'best.pt'))
            atomic_json(prior/'completed.json',done)
            with self.assertRaisesRegex(ValueError,'discard'):
                configuration(parser().parse_args(['--source-run',str(prior),'--device','cpu']))


if __name__ == '__main__':
    unittest.main()
