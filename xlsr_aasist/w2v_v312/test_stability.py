"""Exercise the V3.11 scale failure and transactional rejection of unsafe states."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from w2v_v39.common import atomic_json, read_json, digest
from .model import FeatureClassifier, LanguageAdversary, optimizer_for, ProtectedReverse
from .objectives import reference_rows
from .probes import compare
from .stability import StabilityStop, scale_loss, summarize, compare_diagnostics
from .train import train_step, run_experiment
from .test_core import config, rows, detector
from .test_workflow import fixture, synthetic_expanded_bundles


class StabilityTests(unittest.TestCase):
    def test_overflow_is_reportable_without_nan_or_infinity_in_stop_json(self):
        import json
        for h,z in ((torch.full((1,6),1e38),torch.zeros(1,2)),
                    (torch.ones(1,6),torch.tensor([[3e38,-3e38]]))):
            with self.assertRaises(StabilityStop) as raised:
                scale_loss(h,z,rows(1)[:1],torch.ones(1),config())
            json.dumps(raised.exception.details,allow_nan=False)

    def test_adapter_is_identity_at_start_and_remains_bounded_under_large_weights(self):
        torch.manual_seed(51)
        plain = torch.nn.Sequential(torch.nn.Linear(12,8),torch.nn.SELU(),torch.nn.Linear(8,2))
        adapter = FeatureClassifier(plain,4,.2)
        x = torch.randn(5,12,requires_grad=True)
        torch.testing.assert_close(adapter(x),plain(x),rtol=0,atol=0)
        with torch.no_grad():
            adapter.adapter[-1].weight.fill_(10000.)
            adapter.adapter[-1].bias.fill_(10000.)
        adapter(x).sum().backward()
        self.assertLessEqual(float(adapter.monitor['adapter_ratio'].max()),.200001)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in adapter.parameters()))

    def test_penalties_push_back_scale_do_not_lock_wrong_predictions_and_microbatch_match(self):
        data = rows(1)[:2]
        h = torch.full((2,6),2.,requires_grad=True)
        z = torch.tensor([[12.,0.],[-12.,0.]],requires_grad=True)
        weights = torch.tensor([.4,.6])
        loss,_ = scale_loss(h,z,data,weights,config())
        gh,gz = torch.autograd.grad(loss,(h,z),retain_graph=True)
        self.assertTrue(bool((gh>0).all()))
        self.assertGreater(float(gz[0,0]),0.)
        # Row 1 is fake: pulling its negative margin back toward zero agrees with CE.
        self.assertLess(float(gz[1,0]),0.)
        pieces = sum(scale_loss(h[i:i+1],z[i:i+1],data[i:i+1],weights[i:i+1],config())[0] for i in range(2))
        torch.testing.assert_close(pieces,loss)
        for a,b in zip(torch.autograd.grad(pieces,(h,z)),(gh,gz)):
            torch.testing.assert_close(a,b)

    def test_training_watchdog_fires_before_any_parameter_or_optimizer_change(self):
        model = detector()
        critic = LanguageAdversary(model.head.classifier[-1].in_features,8)
        opt = optimizer_for(model,critic,config())
        batch = [dict(r,features=torch.randn(1,20,160),mask=torch.ones(1,20,dtype=torch.long)) for r in rows(1)]
        train_step(model,critic,opt,batch,config(),.05)
        before = {k:v.detach().clone() for k,v in model.state_dict().items()}
        old_critic = {k:v.detach().clone() for k,v in critic.state_dict().items()}
        old_opt = copy.deepcopy(opt.state_dict())
        bad = [dict(r,reference_rms=1e-8) for r in batch]
        with self.assertRaises(StabilityStop):
            train_step(model,critic,opt,bad,config(),.05)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        for name,value in critic.state_dict().items():
            torch.testing.assert_close(value,old_critic[name],rtol=0,atol=0)
        for key,state in opt.state_dict()['state'].items():
            for name,value in state.items():
                torch.testing.assert_close(value,old_opt['state'][key][name],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_v311_explosion_never_counts_as_language_evidence(self):
        base = dict(readability=.911,representation_variance=.0577578,
                    condition_variances={'online':.1,'noisy_a':.2})
        for ratio in (85.1975,9123.3,520881.6):
            current = dict(base,readability=.694,representation_variance=base['representation_variance']*ratio)
            result = compare(base,current,config())
            self.assertFalse(result['stable'])
            self.assertFalse(result['evidence'])
        current = dict(base,readability=.8,condition_variances={'online':.1,'noisy_a':1.})
        self.assertFalse(compare(base,current,config())['stable'])
        self.assertTrue(compare(base,dict(base,readability=.8),config())['evidence'])

    def test_validation_catches_small_group_extremes_even_if_overall_score_improves(self):
        data = rows(1)
        z = np.tile([10.,0.],(len(data),1))
        rms = np.ones(len(data))
        base = summarize(data,rms,rms,z,np.zeros(len(data)))
        self.assertTrue(compare_diagnostics(base,base,config())['stable'])
        z[-1,0] = -11049.05
        current = summarize(data,rms,rms,z,np.zeros(len(data)))
        result = compare_diagnostics(base,current,config())
        self.assertFalse(result['stable'])
        self.assertTrue(any('noisy_b/zh/1/abs_margin' in r for r in result['reasons']))

    def test_reference_uses_existing_vectors_only_and_rejects_nonfinite(self):
        data = rows(1)
        z = np.tile([10.,0.],(len(data),1)).astype(np.float32)
        x = np.full((len(data),6),.25,dtype=np.float32)
        linked,inventory = reference_rows(data,z,config(),x)
        self.assertEqual(linked[0]['reference_rms'],.25)
        self.assertEqual(linked[0]['margin_ceiling'],17.)
        self.assertEqual(inventory['derived_scalar_bytes'],len(data)*12)
        self.assertEqual(inventory['new_teacher_forward_passes'],0)
        x[0,0] = np.nan
        with self.assertRaises(ValueError):
            reference_rows(data,z,config(),x)

    def test_parallel_radial_axis_does_not_create_nan_and_leaves_other_directions(self):
        h = torch.tensor([[2.,0.,0.],[0.,0.,0.]],requires_grad=True)
        ProtectedReverse.apply(h,.5,torch.tensor([1.,0.,0.]),True,[]).sum().backward()
        torch.testing.assert_close(h.grad,torch.tensor([[0.,-.5,-.5],[0.,-.5,-.5]]),rtol=0,atol=0)


class SafetyWorkflowTests(unittest.TestCase):
    def test_training_stop_preserves_prior_committed_cursor_and_selected_original(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            _,cfg,_,_ = fixture(root)
            run = root/'safe'; run.mkdir(); atomic_json(run/'config.json',cfg)
            calls = 0
            def fail_after_boundary(*args,**kwargs):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise StabilityStop(dict(stage='training',reason='injected_scale_explosion'))
                return train_step(*args,**kwargs)
            with patch('w2v_v312.train.bundles',synthetic_expanded_bundles), \
                 patch('w2v_v312.train.train_step',side_effect=fail_after_boundary):
                done = run_experiment(cfg,run)
            self.assertTrue(done['baseline_fallback'])
            self.assertEqual(done['committed_updates'],3)
            self.assertEqual(done['stability_stop']['attempted_update'],4)
            last = torch.load(run/'last.pt',map_location='cpu',weights_only=True)
            self.assertEqual(last['cursor'],3)
            self.assertEqual(len(last['history']),1)
            self.assertEqual(read_json(run/'stability_stop.json')['reason'],'injected_scale_explosion')

    def test_validation_instability_cannot_promote_or_commit_even_when_accuracy_gate_passes(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            _,cfg,_,_ = fixture(root)
            run = root/'safe'; run.mkdir(); atomic_json(run/'config.json',cfg)
            with patch('w2v_v312.train.bundles',synthetic_expanded_bundles), \
                 patch('w2v_v312.train.acceptance',return_value=(True,[],.02)), \
                 patch('w2v_v312.train.compare_diagnostics',return_value=dict(stable=False,reasons=['injected_scale_explosion'])):
                done = run_experiment(cfg,run)
            self.assertTrue(done['baseline_fallback'])
            self.assertEqual(done['committed_updates'],0)
            self.assertEqual(done['discarded_uncommitted_updates'],3)
            self.assertFalse((run/'last.pt').exists())
            entry = read_json(run/'training_history.json')[0]
            self.assertFalse(entry['eligible'])
            self.assertFalse(entry['promoted'])
            self.assertFalse(entry['committed'])
            self.assertFalse(entry['language_evidence']['evidence'])


if __name__ == '__main__':
    unittest.main()
