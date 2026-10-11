"""Numerical and behavioral checks for the new sampling and gradient contracts."""
import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from .common import GROUPS, seed_all, capture_rng, restore_rng, partial_state
from .sampling import Plan, validate_units
from .data import Waves, loader, close, TicketSampler, microbatches
from .objective import classification, contrastive
from .step import backward_objective, schedule
from .metrics import summarize
from .test_support import tiny_cfg,tiny_model,metadata,units_from_plan,make_fixture,NativeBoundary


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(8,160); self.bn=nn.BatchNorm1d(160)
        self.drop=nn.Dropout(.3); self.out=nn.Linear(160,2)
    def train(self,mode=True):
        super().train(mode); self.bn.eval(); return self
    def forward(self,waves,return_features=False):
        x=torch.as_tensor(np.stack(waves))
        h=self.drop(self.bn(self.fc(x))).tanh()
        z=self.out(self.drop(h))
        return (z,h) if return_features else z


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_default_contract_and_invalid_options(self):
        from .config import parser,validate
        a=parser().parse_args([]);validate(a)
        self.assertEqual((a.epochs,a.warm_epochs,a.stream_sources,a.variant),(6,1,16,'C1'))
        self.assertEqual((a.contrastive_weight,a.source_sampling_power),(.05,.5))
        for flags in (['--contrastive-temperature','0'],['--source-sampling-power','nan'],
                      ['--contrastive-weight','-1'],['--warm-epochs','0'],
                      ['--init-warmup','x','--resume','y']):
            with self.assertRaises(ValueError): validate(parser().parse_args(flags))

    def test_resume_does_not_silently_ignore_objective_overrides(self):
        from .config import parser,validate_locked_cli
        for flags in (['--resume','run','--ce-only'],['--resume','run','--epochs','20'],
                      ['--init-warmup','run','--microbatch','8']):
            with self.assertRaises(ValueError):validate_locked_cli(parser().parse_args(flags),flags)
        flags=['--init-warmup','run','--ce-only']
        validate_locked_cli(parser().parse_args(flags),flags)

    def test_production_quotas(self):
        p=Plan(metadata((7499,1500,23698,5963)))
        self.assertEqual(list(p.quotas.values()),[4,2,7,3])
        report=p.coverage(0)
        self.assertEqual(p.steps,2417)
        self.assertAlmostEqual(report['cells']['en/1']['mean_repeats'],4834/1500)
        self.assertAlmostEqual(sum(report['condition_loss_mass'].values()),1.)

    def test_cycles_no_duplicates_at_epoch_or_shuffle_boundaries(self):
        p=Plan(metadata(),16,3)
        for e in range(9):
            for b in p.batches(e):
                self.assertEqual(len({p.rows[t['index']]['group_id'] for t in b}),16)
        for g,pool in p.pools.items():
            stream=sum([p.stream(g,e) for e in range(9)],[])
            for start in range(0,len(stream)-len(pool)+1,len(pool)):
                self.assertEqual(set(stream[start:start+len(pool)]),set(pool))
        self.assertEqual(list(p.batches(3)),list(p.batches(3)))

    def test_pair_identity_conflicts_and_orphans_rejected(self):
        rows=metadata(missing=False)
        broken=copy.deepcopy(rows);broken[1]['label']=1-broken[1]['label']
        with self.assertRaises(ValueError): Plan(broken)
        broken=copy.deepcopy(rows);broken[-1]['audio_sha256']=broken[0]['audio_sha256']
        with self.assertRaises(ValueError): Plan(broken)
        with self.assertRaises(ValueError): Plan(rows[1:])
        broken=copy.deepcopy(rows);broken[1]['source_id']='invented'
        with self.assertRaises(ValueError): Plan(broken)

    def test_physical_batches_use_all_slots_without_losing_source_pairs(self):
        units=units_from_plan(Plan(metadata(),16))
        batches=list(microbatches(units,4,10000))
        self.assertEqual(len(batches),12)
        flat=[r for batch in batches for unit in batch for r in unit]
        self.assertEqual([id(r) for r in flat],[id(r) for unit in units for r in unit])
        self.assertTrue(all(sum(map(len,b))==4 for b in batches))

    def test_group_mass_not_raw_sample_counts(self):
        p=Plan(metadata(),16);units=units_from_plan(p,samples=8);validate_units(units,16)
        rows=[r for u in units for r in u]
        z=torch.zeros(len(rows),2,requires_grad=True)
        loss,_,_=classification(z,rows)
        loss.backward()
        self.assertAlmostEqual(loss.item(),np.log(2),places=6)
        for g in GROUPS:
            ix=[i for i,r in enumerate(rows) if (r['language'],r['label'])==g]
            self.assertAlmostEqual(sum(rows[i]['ce_weight'] for i in ix),.25)
            self.assertAlmostEqual(float(z.grad[ix].abs().sum()),.245,places=6)
        repeated=copy.deepcopy(units);repeated[-1]=repeated[0]
        with self.assertRaises(ValueError): validate_units(repeated,16)

    def test_missing_online_retains_three_views_and_noise_is_fresh(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows,_=make_fixture(Path(d));removed=next(r for r in rows if r['condition']=='online')
            rows=[r for r in rows if r is not removed];p=Plan(rows,4)
            source=removed['group_id']; t=p.ticket(source,0,0,0)
            with patch('w2v_v3182.data.Engines',NativeBoundary):
                dataset=Waves(rows,cfg)
                a=dataset[t];b=dataset[t];c=dataset[p.ticket(source,1,0,0)]
            self.assertEqual([r['role'] for r in a],['offline','noisy_a','noisy_b'])
            self.assertEqual([r['ce_weight']/r['source_mass'] for r in a],[.1,.45,.45])
            self.assertTrue(np.array_equal(a[1]['wave'],b[1]['wave']))
            self.assertFalse(np.array_equal(a[1]['wave'],a[2]['wave']))
            self.assertFalse(np.array_equal(a[1]['wave'],c[1]['wave']))

    def test_contrastive_masks_and_language_negative_mass(self):
        p=Plan(metadata(),16);units=units_from_plan(p,samples=8)
        # Perfect class separation has tiny pair loss despite 11/5 source counts.
        h=torch.tensor([[1.,0.] if r['label']==0 else [-1.,0.] for u in units for r in u],requires_grad=True)
        loss,stats=contrastive(h,units)
        self.assertLess(loss.item(),1e-6)
        self.assertEqual(stats['same_label_negatives'],0)
        self.assertEqual(stats['candidate_sources'],16)
        self.assertEqual(stats['directed_positive_pairs'],64)
        zero=torch.ones_like(h,requires_grad=True)
        constant,_=contrastive(zero,units)
        self.assertAlmostEqual(constant.item(),np.log(5),places=5)
        constant.backward();self.assertTrue(torch.isfinite(zero.grad).all())

    def test_gradcache_matches_direct_graph_with_dropout_and_bn(self):
        seed_all(19);cfg=tiny_cfg();p=Plan(metadata(),16)
        units=units_from_plan(p,samples=8); cfg.update(stream_sources=16,microbatch=4)
        a=Toy().train();b=copy.deepcopy(a);rng=capture_rng()
        direct=backward_objective(a,units,cfg,.05,replay=False);end_a=capture_rng()
        restore_rng(rng);cached=backward_objective(b,units,cfg,.05,replay=True);end_b=capture_rng()
        self.assertAlmostEqual(direct['loss'],cached['loss'],places=6)
        for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(n,m);self.assertTrue(torch.allclose(p.grad,q.grad,atol=2e-6,rtol=2e-5),n)
        self.assertTrue(torch.equal(a.bn.running_mean,b.bn.running_mean))
        restore_rng(end_a);x=torch.rand(5);restore_rng(end_b);self.assertTrue(torch.equal(x,torch.rand(5)))

    def test_replay_rejects_updating_bn(self):
        cfg=tiny_cfg(); units=units_from_plan(Plan(metadata(),4),samples=8)
        m=Toy().train();m.bn.train()
        with self.assertRaises(ValueError): backward_objective(m,units,cfg,.05)

    def test_real_aasist_pair_gradient_reaches_lora(self):
        seed_all(4);cfg=tiny_cfg();m=tiny_model(cfg);m.set_phase(True);m.train()
        units=units_from_plan(Plan(metadata(),4))
        before=partial_state(m)
        outputs=[m([r['wave'] for r in u],return_features=True) for u in units]
        h=torch.cat([v[1] for v in outputs]);self.assertEqual(h.shape,(12,160))
        loss,_=contrastive(h,units);loss.backward()
        grads=[p.grad for n,p in m.named_parameters() if n.endswith('.lora_b')]
        self.assertTrue(any(g is not None and g.abs().sum()>0 for g in grads))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if n.startswith('head.aasist.') and 'out_layer' not in n))
        self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if '.base.' in n))
        for name,value in m.named_buffers():
            if name.startswith('head.'):self.assertTrue(torch.equal(value.cpu(),before[name]),name)

    def test_lambda_warmup_and_ramp(self):
        cfg=tiny_cfg();optim=torch.optim.Adam([dict(params=[nn.Parameter(torch.ones(1))],name='aasist',initial_lr=.001)])
        self.assertEqual(schedule(optim,cfg,0,0,10),0)
        self.assertAlmostEqual(schedule(optim,cfg,1,0,10),.005)
        self.assertAlmostEqual(schedule(optim,cfg,1,9,10),.05)

    def test_metric_drift_explained_by_precision_and_matched_recall(self):
        rows=[dict(condition='seen',language='en',label=c) for c in [0]*2500+[1]*500]
        def scores(tp_fake,tp_real):
            margin=np.array([1]*tp_fake+[-1]*(2500-tp_fake)+[-1]*tp_real+[1]*(500-tp_real))
            return np.stack((margin,-margin),axis=1)
        a=summarize(rows,scores(1918,439))['groups']['seen/en']
        b=summarize(rows,scores(2168,406))['groups']['seen/en']
        self.assertLess(b['recall'][1],a['recall'][1]);self.assertGreater(b['f1'][1],a['f1'][1])
        self.assertGreater(b['precision'][1],a['precision'][1])
        self.assertGreaterEqual(b['matched_95']['actual_fake_recall'],.95)
        self.assertNotIn('threshold',b['matched_99'])


if __name__=='__main__': unittest.main()
