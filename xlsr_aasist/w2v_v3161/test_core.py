"""Real tiny SSL gradients, optimizer continuity, selection and pair diagnosis."""
import copy
import io
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from w2v_v316_tfcl.test_core import model_fixture,examples,config
from .objectives import TFCL,ssl_frames,channel_cka
from .step import train_step
from .model import restore_detector_optimizer
from .state import promote,metric_key
from .diagnostics import paired_treatment
from .arguments import parser,validate


class CoreTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(316101);torch.set_num_threads(1)

    def test_matches_published_single_source_math_and_both_side_gradients(self):
        a,b=torch.randn(21,16,requires_grad=True),torch.randn(21,16,requires_grad=True)
        aux=TFCL(16,2,21);valid=torch.ones(21,dtype=torch.bool)
        t,s=aux(a,b,valid,valid)
        ab=aux.attention(a[None],b[None],b[None],need_weights=False)[0][0]
        ba=aux.attention(b[None],a[None],a[None],need_weights=False)[0][0]
        expected_t=.5*((1-F.cosine_similarity(ab,a,dim=-1)).mean()+(1-F.cosine_similarity(ba,b,dim=-1)).mean())
        x,y=aux.time_projection(a.T),aux.time_projection(b.T)
        h=torch.eye(16)-torch.ones(16,16)/16
        k,l=x@x.T,y@y.T
        cka=torch.trace(k@h@l@h)/(torch.trace(k@h@k@h)*torch.trace(l@h@l@h)).sqrt()
        torch.testing.assert_close(t,expected_t)
        torch.testing.assert_close(s,1-cka,atol=2e-6,rtol=2e-6)
        (t+s).backward()
        self.assertGreater(a.grad.norm(),0);self.assertGreater(b.grad.norm(),0)
        self.assertGreater(aux.attention.in_proj_weight.grad.norm(),0)
        self.assertGreater(aux.time_projection.weight.grad.norm(),0)

    def test_known_missing_and_padding_do_not_contribute(self):
        aux=TFCL(16,2,21)
        a,b=torch.randn(31,16,requires_grad=True),torch.randn(27,16,requires_grad=True)
        am,bm=torch.ones(31,dtype=torch.bool),torch.ones(27,dtype=torch.bool)
        am[5:10]=False;bm[18:]=False
        t,s=aux(a,b,am,bm);expected=aux(a[am],b[bm],am[am],bm[bm])
        torch.testing.assert_close(t,expected[0]);torch.testing.assert_close(s,expected[1])
        (t+s).backward()
        self.assertEqual(a.grad[~am].count_nonzero(),0);self.assertEqual(b.grad[~bm].count_nonzero(),0)
        with torch.no_grad():a[~am]=float('nan');b[~bm]=float('nan')
        t2,s2=aux(a,b,am,bm);torch.testing.assert_close(t,t2);torch.testing.assert_close(s,s2)

    def test_full_ssl_features_and_no_confidence_rejection(self):
        m=model_fixture();a=TFCL(m.backbone.config.hidden_size,2,21)
        observed=[]
        original=a.forward_batch
        def observe(left,right,*args):
            observed.extend([x.shape[-1] for x in left]);return original(left,right,*args)
        a.forward_batch=observe
        opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad]+list(a.parameters()),lr=.01)
        cfg=config(reference_probability=1.,alignment_min_cosine=1.,tfcl_structure_weight=.15)
        stats=train_step(m,a,opt,examples(),cfg,1.,audit=True)
        self.assertTrue(observed);self.assertEqual(set(observed),{16})
        self.assertEqual(m.head.config.projection,8)
        self.assertEqual(stats['bridge_pairs'],4);self.assertEqual(stats['matched_pairs'],4)
        self.assertGreater(stats['feature_gradient']['tfcl_norm'],0)

    def test_microbatch_does_not_change_objective_or_update(self):
        m=model_fixture();other=copy.deepcopy(m);a=TFCL(16,2,21);b=copy.deepcopy(a);rows=examples()
        def opt(model,aux):return torch.optim.SGD([p for p in model.parameters() if p.requires_grad]+list(aux.parameters()),lr=.01)
        x=train_step(m,a,opt(m,a),rows,config(microbatch=3,tfcl_structure_weight=.15),1.)
        y=train_step(other,b,opt(other,b),rows,config(microbatch=12,tfcl_structure_weight=.15),1.)
        self.assertAlmostEqual(x['total_loss'],y['total_loss'],places=5)
        for left,right in zip(list(m.parameters())+list(a.parameters()),list(other.parameters())+list(b.parameters())):
            torch.testing.assert_close(left,right,atol=2e-6,rtol=3e-5)

    def test_adam_moments_restored_but_new_rates_and_aux_preserved(self):
        p=torch.nn.Parameter(torch.ones(3));old=torch.optim.AdamW([dict(params=[p],name='detector',lr=.01,initial_lr=.01)])
        p.sum().backward();old.step();saved=copy.deepcopy(old.state_dict())
        q=torch.nn.Parameter(p.detach().clone());aux=torch.nn.Parameter(torch.ones(2))
        new=torch.optim.AdamW([dict(params=[q],name='detector',lr=.001,initial_lr=.001),
                              dict(params=[aux],name='training_only_tfcl',lr=.0001,initial_lr=.0001)])
        report=restore_detector_optimizer(new,saved)
        self.assertEqual(report['detector_parameters_with_moments'],1)
        self.assertEqual(new.param_groups[0]['initial_lr'],.001)
        torch.testing.assert_close(new.state[q]['exp_avg'],old.state[p]['exp_avg'])
        self.assertNotIn(aux,new.state)
        with self.assertRaisesRegex(ValueError,'no Adam'):restore_detector_optimizer(new,{})

    def test_best_ignores_old_guards_and_rejects_incomplete_metrics(self):
        def metric(w,n=.9):return dict(complete=True,weighted_f1=w,noisy_f1=n,clean_f1=.95)
        state=dict(best_metrics=metric(.91),best_tag='source:epoch_4_step_2')
        self.assertTrue(promote(state,'epoch_5_step_2',{},metric(.92),dict(version='3.16.1')))
        self.assertFalse(promote(state,'epoch_6_step_2',{},metric(.90),dict(version='3.16.1')))
        self.assertEqual(state['best_tag'],'epoch_5_step_2')
        for bad in (dict(metric(.92),complete=False),metric(float('nan'))):
            with self.assertRaises(ValueError):metric_key(bad)

    def test_official_dev_is_not_guessed_from_simulated_source_hash(self):
        off=[dict(audio_sha256='offline',language='en',label=1)]
        dev=[dict(audio_sha256='online',source_sha256='online',condition='online',language='en',label=1),
             dict(audio_sha256='sim',source_sha256='offline',condition='seen',language='en',label=1)]
        a=np.array([[0.,1.]]);b=np.array([[1.,0.],[0.,1.]])
        result=paired_treatment(off,a,dev,b)
        self.assertEqual(result['official']['status'],'unavailable')
        self.assertEqual(set(result['simulated']['groups']),{'seen/en/real'})
        paired=paired_treatment(off,a,dev,b,[dict(offline_sha256='offline',online_sha256='online')])
        self.assertEqual(paired['official']['groups']['online/en/real']['original_correct_processed_wrong'],1)

    def test_additional_epoch_defaults_and_invalid_budgets(self):
        args=parser().parse_args([]);validate(args);self.assertEqual(args.epochs,4)
        for flags in (['--epochs','0'],['--microbatch','2'],['--structure-weight','nan']):
            with self.assertRaises(ValueError):validate(parser().parse_args(flags))

    def test_console_keeps_dev_block_but_redirects_inference_counters(self):
        import tempfile
        from pathlib import Path
        from . import output
        from .test_workflow import score
        terminal=io.StringIO()
        def noisy(*args):print('irrelevant inference counter 200/1434');return np.zeros((1,2)),None
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(terminal):
            with patch.object(output,'full_infer',side_effect=noisy):output.infer(None,[],{},'Dev',folder)
            output.print_metrics('epoch_5_step_2',score(.91))
            detail=(Path(folder)/'details.log').read_text(encoding='utf-8')
        self.assertNotIn('irrelevant inference',terminal.getvalue())
        self.assertIn('Weighted=91.000',terminal.getvalue())
        self.assertEqual(terminal.getvalue().count('Recall(fake/real)'),6)
        self.assertIn('irrelevant inference',detail)


if __name__=='__main__':unittest.main()
