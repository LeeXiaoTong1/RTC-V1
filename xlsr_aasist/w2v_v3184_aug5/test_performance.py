"""Numerical/state parity for geometry reuse and single checkpoint snapshots."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from w2v_v318.regions import windows,coverage_mass
from .test_support import tiny_cfg,tiny_model
from .common import partial_state,seed_all
from .model import optimizer_for
from .train import save


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_cached_pooling_logits_and_all_gradients_match_uncached(self):
        seed_all(57);cached=tiny_model(tiny_cfg()).head
        with torch.no_grad():
            cached.attention_delta[-1].weight.normal_(0,.05)
            cached.aasist.out_layer.weight[:,160:].normal_(0,.05)
        original=copy.deepcopy(cached)
        def uncached(length,device):
            ranges=windows(length,original.window,original.hop)
            return ranges,coverage_mass(length,ranges,device)
        original.window_plan=uncached
        sequences=[torch.randn(n,128) for n in (12,128,129,211,300,300)]
        for training in (False,True):
            cached.train(training);original.train(training)
            actual_inputs=[x.clone().requires_grad_() for x in sequences]
            expected_inputs=[x.clone().requires_grad_() for x in sequences]
            seed_all(83);a=cached(actual_inputs);a.square().sum().backward()
            seed_all(83);b=original(expected_inputs);b.square().sum().backward()
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            for x,y in zip(actual_inputs,expected_inputs):torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
            for (n,p),(m,q) in zip(cached.named_parameters(),original.named_parameters()):
                self.assertEqual(n,m)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
            for name,value in cached.state_dict().items():
                torch.testing.assert_close(value,original.state_dict()[name],rtol=0,atol=0)
            cached.zero_grad();original.zero_grad()

    def test_geometry_is_bounded_nonpersistent_and_training_safe_after_inference(self):
        head=tiny_model(tiny_cfg()).head
        with patch('w2v_v3184_aug5.regions.coverage_mass',wraps=coverage_mass) as build:
            with torch.inference_mode():first=head.window_plan(211,'cpu')
            again=head.window_plan(211,torch.device('cpu'))
            self.assertIs(first[0],again[0]);self.assertIs(first[1],again[1])
            self.assertEqual(build.call_count,1)
            self.assertFalse(torch.is_inference(first[1]))
            evidence=torch.randn(len(first[0]),160,requires_grad=True)
            head.weights(evidence,first[1]).square().sum().backward()
            self.assertTrue(torch.isfinite(evidence.grad).all())
        for n in range(12,350):head.window_plan(n,'cpu')
        self.assertLessEqual(len(head._geometry),128)
        self.assertLessEqual(head._geometry_bytes,256*1024)
        self.assertFalse(any('geometry' in key for key in head.state_dict()))
        head.to('cpu');self.assertEqual(len(head._geometry),0)
        with self.assertRaises(ValueError):head.window_plan(11,'cpu')

    def test_single_snapshot_commit_preserves_best_last_and_optimizer(self):
        cfg=tiny_cfg();model=tiny_model(cfg);optimizer=optimizer_for(model,cfg)
        snapshot=partial_state(model)
        state=dict(best_tag='e1',last_tag='e1',history=[],best_model=snapshot)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('w2v_v3184_aug5.train.partial_state',side_effect=AssertionError('Snapshot must not repeat')):
                save(Path(tmp),cfg,state,model,optimizer,snapshot)
            stored=torch.load(Path(tmp)/'last.pt',weights_only=True)
        self.assertIs(state['model'],snapshot);self.assertIs(state['best_model'],snapshot)
        for name,value in snapshot.items():
            torch.testing.assert_close(stored['model'][name],value,rtol=0,atol=0)
            torch.testing.assert_close(stored['best_model'][name],value,rtol=0,atol=0)
        self.assertEqual(stored['optimizer']['param_groups'],optimizer.state_dict()['param_groups'])
        self.assertIn('rng',stored)


if __name__=='__main__':unittest.main()
