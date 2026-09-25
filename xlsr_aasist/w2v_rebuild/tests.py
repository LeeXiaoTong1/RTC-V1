"""CPU regression tests. No download, no speech data, no fairseq dependency.

ToyEncoder is a test double, NOT an evaluation of pretrained w2v-BERT quality.
Optional HF tests instantiate the real library's small random Wav2Vec2BertModel.
"""
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import warnings
import zipfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from . import SCHEMA
from .model import AASISTSampleNorm, Detector, forward_chunks, ResidualBlock
from .core import (class_weights, pair_loss, objective, Metrics, Schedule, build_optimizer,
                   rng_state, restore_rng, atomic_save, load_checkpoint, gradient_audit, finish_audit)
from .data import SeededDataset, FeatureCollator, combine_batches
from .evaluate import package_scores
from .train import train_step


class ToyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm, self.linear = nn.LayerNorm(d), nn.Linear(d, d)
    def forward(self, x):
        return x + .1 * F.gelu(self.linear(self.norm(x)))


class ToyEncoder(nn.Module):
    def __init__(self, layers=24, checkpointing=False):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.feature_projection = nn.Linear(160, 16)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([ToyBlock(16) for _ in range(layers)])
        self.checkpointing = checkpointing
    def forward(self, input_features, attention_mask, **kwargs):
        x = self.feature_projection(input_features)
        for layer in self.encoder.layers:
            x = checkpoint(layer, x, use_reentrant=False) if self.checkpointing and self.training else layer(x)
        return SimpleNamespace(last_hidden_state=x)


def no_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.
    return model


class RandomDataset(torch.utils.data.Dataset):
    def __len__(self): return 4
    def __getitem__(self, i): return torch.tensor([np.random.rand(), random.random(), float(torch.rand(()))])


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def setUp(self): torch.manual_seed(234)

    def test_nonreentrant_frozen_input(self):
        x, layer = torch.randn(3, 5), nn.Linear(5, 2)
        z = checkpoint(layer, x, use_reentrant=False)
        z.sum().backward()
        self.assertIsNotNone(layer.weight.grad)
        with warnings.catch_warnings(record=True) as seen:
            y = checkpoint(layer, x, use_reentrant=True)
        self.assertFalse(y.requires_grad)
        self.assertTrue(any('requires_grad' in str(w.message) for w in seen))

    def test_checkpoint_gradient_equivalence(self):
        a, b = nn.Linear(8, 8), nn.Linear(8, 8)
        b.load_state_dict(a.state_dict())
        x = torch.randn(4, 8)
        a(x).square().sum().backward()
        checkpoint(b, x, use_reentrant=False).square().sum().backward()
        torch.testing.assert_close(a.weight.grad, b.weight.grad)

    def test_no_batchnorm_in_model(self):
        m = Detector(ToyEncoder())
        self.assertFalse(any(isinstance(x, nn.modules.batchnorm._BatchNorm) for x in m.modules()))

    def test_head_shapes(self):
        z, h = AASISTSampleNorm(16)(torch.randn(3, 30, 16))
        self.assertEqual(tuple(z.shape), (3, 2))
        self.assertEqual(tuple(h.shape), (3, 160))

    def test_sample_independent_normalization(self):
        m = no_dropout(AASISTSampleNorm(16)).train()
        x = torch.randn(3, 30, 16)
        a, _ = m(x)
        b, _ = m(x[:1])
        torch.testing.assert_close(a[:1], b, rtol=1e-4, atol=2e-5)
        m.eval()
        c, _ = m(x)
        torch.testing.assert_close(a, c)

    def test_bn_buffer_changes_without_optimizer(self):
        # Reproduces a plausible state mechanism, not a diagnosis of user checkpoints.
        m = nn.BatchNorm1d(4)
        before = m.running_mean.clone()
        m(torch.randn(8, 4) + 9)
        self.assertFalse(torch.equal(before, m.running_mean))

    def test_residual_preactivation_receives_gradients(self):
        m = ResidualBlock(32, 32)
        m(torch.randn(2, 32, 8, 12)).square().mean().backward()
        self.assertIsNotNone(m.pre[0].weight.grad)
        self.assertGreater(float(m.pre[0].weight.grad.abs().sum()), 0)

    def test_logical_microbatch_gradient_equivalence(self):
        a = no_dropout(Detector(ToyEncoder(2))).eval()
        b = copy.deepcopy(a)
        x, mask = torch.randn(6, 24, 160), torch.ones(6, 24, dtype=torch.long)
        y = torch.tensor([0, 1, 0, 1, 0, 1])
        za, ha = forward_chunks(a, x, mask, 6)
        zb, hb = forward_chunks(b, x, mask, 2)
        la, _ = objective(za, ha, y, 2, 2, 0, torch.tensor([.7, 1.8]), .1, 0)
        lb, _ = objective(zb, hb, y, 2, 2, 0, torch.tensor([.7, 1.8]), .1, 0)
        la.backward(); lb.backward()
        torch.testing.assert_close(la, lb, rtol=2e-4, atol=1e-5)
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertIsNotNone(pa.grad)
            torch.testing.assert_close(pa.grad, pb.grad, rtol=.003, atol=3e-4)

    def test_fixed_shape_eval_padding_and_companion_independence(self):
        m = no_dropout(Detector(ToyEncoder(2))).eval()
        x = torch.randn(3, 24, 160)
        mask = torch.ones(3, 24, dtype=torch.long)
        group, _ = forward_chunks(m, x, mask, 4, pad_last=True)
        single, _ = forward_chunks(m, x[:1], mask[:1], 4, pad_last=True)
        self.assertEqual(tuple(group.shape), (3, 2))
        torch.testing.assert_close(group[:1], single, rtol=1e-5, atol=1e-6)

    def test_full24_actual_updates(self):
        m = Detector(ToyEncoder(24, checkpointing=True))
        m.backbone.eval()  # HF from_pretrained may start this child in eval mode.
        opt = build_optimizer(m, 1e-6, 1e-4, 1e-4)
        sched = Schedule(opt, 10)
        args = SimpleNamespace(device='cpu', amp='none', microbatch=2, stage=3, grad_clip=1.)
        b = {'features': torch.randn(10, 24, 160), 'mask': torch.ones(10, 24, dtype=torch.long),
             'labels': torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])}
        stats, _, _ = train_step(m, opt, sched, b, (2, 2, 2), torch.tensor([.7, 1.8]), args, 0, 10, True)
        self.assertEqual(len(stats['gradient_audit']), 26)
        self.assertTrue(m.backbone.training)
        self.assertTrue(all(x['sampled_update_norm'] > 0 for x in stats['gradient_audit'].values()))

    def test_optimizer_coverage_and_no_norm_decay(self):
        m = Detector(ToyEncoder(2))
        o = build_optimizer(m, 1e-6, 1e-4, 1e-4)
        ps = [p for g in o.param_groups for p in g['params']]
        self.assertEqual(len(ps), len({id(p) for p in ps}))
        self.assertEqual({id(p) for p in ps}, {id(p) for p in m.parameters() if p.requires_grad})
        for g in o.param_groups:
            if g['weight_decay']:
                self.assertTrue(all(p.ndim > 1 for p in g['params']))

    def test_inverse_frequency_and_duplicate_guard(self):
        w, n = class_weights(['a','b','c','d'], dict(a=0,b=0,c=0,d=1))
        torch.testing.assert_close(w, torch.tensor([2/3, 2.]))
        self.assertEqual(n.tolist(), [3, 1])
        with self.assertRaises(ValueError): class_weights(['a','a'], {'a':0})

    def test_stage1_ce_equivalence(self):
        z, y, w = torch.randn(5, 2), torch.tensor([0,1,0,0,1]), torch.tensor([.7,1.8])
        l, _ = objective(z, torch.randn(5,16), y, 5, 0, 0, w, 0, 0)
        torch.testing.assert_close(l, F.cross_entropy(z,y,weight=w))

    def test_pair_ce_does_not_reweight_real(self):
        z, h = torch.zeros(6,2), torch.randn(6,16)
        y = torch.tensor([0,1,0,1,0,1])
        _, stats = objective(z,h,y,2,2,0,torch.tensor([1.,9.]),0,0)
        self.assertAlmostEqual(stats['ce_real_pair'], math_log2(), places=6)
        self.assertAlmostEqual(sum(stats['coefficients'].values()),1.)

    def test_v2_ce_coefficients(self):
        y = torch.tensor([0,1]*20)
        _, s = objective(torch.randn(40,2),torch.randn(40,16),y,24,4,4,torch.ones(2),.1,.1)
        expected = [ .7*24/36, .7*8/36, .7*4/36, .3 ]
        for x, v in zip(s['coefficients'].values(), expected): self.assertAlmostEqual(x,v)

    def test_pair_loss_symmetric(self):
        a,b,y = torch.randn(4,16),torch.randn(4,16),torch.tensor([0,0,1,1])
        torch.testing.assert_close(pair_loss(a,b,y),pair_loss(b,a,y))
        with self.assertRaises(ValueError): pair_loss(a,b,torch.zeros(4,dtype=torch.long))

    def test_pair_loss_matches_reference(self):
        a,b,y = torch.randn(4,8),torch.randn(4,8),torch.tensor([0,1,0,1])
        an,bn=F.normalize(a,dim=1),F.normalize(b,dim=1)
        s=an@bn.T/.1
        vals=[]
        for i in range(4):
            idx=[j for j in range(4) if j==i or y[i]!=y[j]]
            vals.extend([torch.logsumexp(s[i,idx],0)-s[i,i],torch.logsumexp(s[idx,i],0)-s[i,i]])
        torch.testing.assert_close(pair_loss(a,b,y),torch.stack(vals).mean())

    def test_metric_ties_and_balanced_ce(self):
        m=Metrics(); z=torch.tensor([[0.,0.],[3.,0.],[0.,3.],[2.,0.]]); y=torch.tensor([0,0,1,1])
        m.update(z,y); r=m.result()
        self.assertEqual(r['confusion'],[[2,0],[1,1]])
        self.assertEqual(r['recall'],[1.,.5])
        self.assertAlmostEqual(r['macro_f1'],(.8+2/3)/2)
        self.assertAlmostEqual(r['balanced_ce'],float(F.cross_entropy(z,y)),places=6)

    def test_plateau_preserves_adam_state(self):
        p=nn.Parameter(torch.ones(2)); opt=torch.optim.AdamW([{'params':[p],'base_lr':1e-3,'lr':1e-3}])
        p.sum().backward(); opt.step()
        moment=opt.state[p]['exp_avg'].clone()
        s=Schedule(opt,0,patience=2)
        s.validate(1.,10); self.assertFalse(s.validate(1.1,11)); self.assertTrue(s.validate(1.2,12))
        self.assertEqual(s.scale,.5)
        torch.testing.assert_close(moment,opt.state[p]['exp_avg'])

    def test_warmup_and_floor(self):
        p=nn.Parameter(torch.ones(2)); opt=torch.optim.AdamW([{'params':[p],'base_lr':1e-3,'lr':1e-3}])
        s=Schedule(opt,10,patience=1,floor=.1)
        s.before_step(0); self.assertGreater(opt.param_groups[0]['lr'],0)
        s.before_step(10); self.assertAlmostEqual(opt.param_groups[0]['lr'],1e-3)
        for i in range(40): s.validate(1+i,10+i)
        self.assertEqual(s.scale,.1)

    def test_rng_roundtrip_and_seeded_dataset(self):
        state=rng_state(); expected=(random.random(),np.random.rand(),torch.rand(3))
        restore_rng(state); actual=(random.random(),np.random.rand(),torch.rand(3))
        self.assertEqual(expected[:2],actual[:2]); torch.testing.assert_close(expected[2],actual[2])
        ds=SeededDataset(RandomDataset(),123)
        a=ds[0]; torch.testing.assert_close(a,ds[0])
        ds.epoch[0]=1; self.assertFalse(torch.equal(a,ds[0]))
        before=torch.get_rng_state(); ds[1]; torch.testing.assert_close(before,torch.get_rng_state())

    def test_optimizer_resume_replays_next_step(self):
        m=nn.Sequential(nn.Linear(4,4),nn.Dropout(.2),nn.Linear(4,2))
        opt=torch.optim.AdamW(m.parameters(),lr=.01)
        x,y=torch.randn(3,4),torch.tensor([0,1,0])
        def step(mod,optim):
            optim.zero_grad(); loss=F.cross_entropy(mod(x),y); loss.backward(); optim.step(); return loss.detach()
        step(m,opt)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'last.pt'
            atomic_save({'schema':SCHEMA,'model':m.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state()},p)
            target=step(m,opt); expected=copy.deepcopy(m.state_dict())
            clone=nn.Sequential(nn.Linear(4,4),nn.Dropout(.2),nn.Linear(4,2))
            co=torch.optim.AdamW(clone.parameters(),lr=.01)
            saved=load_checkpoint(p); clone.load_state_dict(saved['model']); co.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
            torch.testing.assert_close(target,step(clone,co),rtol=0,atol=0)
            for k,v in clone.state_dict().items(): torch.testing.assert_close(v,expected[k],rtol=0,atol=0)

    def test_legacy_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'old.pt'; torch.save({'x':torch.ones(1)},p)
            with self.assertRaises(ValueError): load_checkpoint(p)

    def test_submission_zip(self):
        with tempfile.TemporaryDirectory() as d:
            txt,z=package_scores(['p/a.wav','p/b.wav'],[.2,.8],d)
            with zipfile.ZipFile(z) as f:
                self.assertEqual(f.namelist(),['scores.txt'])
                self.assertTrue(f.read('scores.txt').startswith(b'p/a.wav 0.2000000000'))
            with self.assertRaises(ValueError): package_scores(['a','a'],[.2,.8],d)
            with self.assertRaises(ValueError): package_scores(['a'],[float('nan')],d)

    def test_feature_collator_layout_and_mask(self):
        class FakeExtractor:
            def __call__(self,waves,**kwargs):
                self.kwargs=kwargs
                return {'input_features':torch.stack([torch.full((13,160),float(w[0])) for w in waves]),
                        'attention_mask':torch.tensor([[1]*12+[0]]*len(waves))}
        c=FeatureCollator('unused','pair'); c.extractor=FakeExtractor()
        b=c([(torch.ones(64600),torch.ones(64600)*3,0),(torch.ones(64600)*2,torch.ones(64600)*4,1)])
        self.assertEqual(b['labels'].tolist(),[0,1,0,1])
        self.assertEqual(b['features'][:,0,0].tolist(),[1,2,3,4])
        self.assertEqual(tuple(b['features'].shape),(4,12,160))
        self.assertEqual(c.extractor.kwargs['sampling_rate'],16000)

    def test_three_stage_engine_and_preflight(self):
        from unittest.mock import patch
        import sys
        from . import train as engine
        class Bundle:
            def __init__(self, args):
                self.args=args; self.steps=1; self.weights=torch.ones(2); self.counts=torch.tensor([20,20])
                self.fingerprints={'fixture':'unchanged'}; self.completed=0
                self.base_fingerprints = self.fingerprints.copy()
                self.fingerprints['cache'] = 'new' if args.ordinary_sampling == 'balanced' else 'old'
                n={1:40,2:32,3:24}[args.stage]
                g=torch.Generator().manual_seed(222)
                def batch(size, kind='ordinary'):
                    b={'features':torch.randn(size,24,160,generator=g),
                       'mask':torch.ones(size,24,dtype=torch.long),'labels':torch.arange(size)%2}
                    if kind=='pair': b['pairs']=size//2
                    else: b['ids']=['online/'+str(i) if i<4 else 'offline/'+str(i) for i in range(size)]
                    return b
                self.train=[[batch(n)]]
                if args.stage>1: self.train.append([batch(8,'pair')])
                if args.stage==3: self.train.append([batch(8,'pair')])
                clean=batch(8); seen=batch(8); seen['bands']=[0,0,1,1,2,2,3,3]
                self.dev={'clean':[clean],'seen':[seen],'heldout':[seen]}
            def begin(self, epoch): pass
            def end(self, steps): self.completed+=1
            def sampler_state(self): return {'completed':self.completed}
            def load_sampler_state(self,state): self.completed=state['completed']
        def build(*a,**kw):
            model=Detector(ToyEncoder(24,checkpointing=False))
            model.backbone.config.to_dict=lambda: {'hidden_size':16,'num_hidden_layers':24}
            return model
        with tempfile.TemporaryDirectory() as d, patch.object(engine,'DataBundle',Bundle), \
             patch.object(engine.Detector,'load',side_effect=build), patch.object(engine,'source_hashes',return_value={}):
            common=['test','--device','cpu','--amp','none','--microbatch','4','--num_workers','0','--epochs','2']
            for name in ('train_data_path','dev_data_path','train_protocol','dev_protocol','rtc_pairs',
                         'train_noise_manifest','train_noisy_cache','dev_noisy_cache','dev_heldout_cache','ssl_path'):
                common += ['--'+name,'fixture']
            for stage in (1,2,3):
                target=str(Path(d)/f'stage{stage}')
                argv=common+['--stage',str(stage),'--out',target]
                if stage>1: argv += ['--init',str(Path(d)/f'stage{stage-1}'/'best_model.pt')]
                if stage==1:
                    with patch.object(sys,'argv',argv+['--preflight']): engine.main()
                    self.assertTrue((Path(target)/'preflight.json').is_file())
                with patch.object(sys,'argv',argv): engine.main()
                state=load_checkpoint(Path(target)/'last.pt')
                self.assertEqual(state['stage'],stage)
                self.assertEqual(state['global_step'],2)
                self.assertIn('optimizer',state)
                self.assertTrue((Path(target)/'completed.json').is_file())
            # A same-stage completed resume must safely preserve the existing best artifact.
            target=str(Path(d)/'stage3')
            argv=common+['--stage','3','--out',target,'--resume',str(Path(target)/'last.pt')]
            with patch.object(sys,'argv',argv): engine.main()
            # New Stage3 fine-tuning may replace caches, but never the base data or baseline artifact.
            from .core import sha256
            baseline = str(Path(target)/'best_model.pt')
            before = sha256(baseline)
            adapted = str(Path(d)/'adapted')
            args = common+['--stage','3','--out',adapted,'--epochs','1', '--ordinary_sampling','balanced',
                          '--noisy_bank_policy','mixed','--consistency_weight','.02']
            with patch.object(sys,'argv',args+['--finetune_from',baseline]): engine.main()
            self.assertEqual(before, sha256(baseline))
            self.assertTrue((Path(adapted)/'baseline_dev.json').is_file())
            with patch.object(sys,'argv',args+['--resume',str(Path(adapted)/'last.pt')]): engine.main()
            state = load_checkpoint(Path(adapted)/'last.pt')
            self.assertEqual(state['config']['init_sha256'], before)
            self.assertTrue(state['config']['adaptation'])

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'HF library unavailable in offline test container')
    def test_real_hf_random_small_encoder(self):
        from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
        cfg=Wav2Vec2BertConfig(hidden_size=32,num_hidden_layers=3,num_attention_heads=4,intermediate_size=64,
                              feature_projection_input_dim=160,layerdrop=0.,apply_spec_augment=False)
        enc=Wav2Vec2BertModel(cfg)
        enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        m=Detector(enc)
        z,_=m(torch.randn(2,24,160),torch.ones(2,24,dtype=torch.long))
        F.cross_entropy(z,torch.tensor([0,1])).backward()
        self.assertEqual(len(gradient_audit(m)[0]),5)


def math_log2(): return float(np.log(2))

if __name__=='__main__': unittest.main(verbosity=2)
