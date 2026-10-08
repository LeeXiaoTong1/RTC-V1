"""CPU contract tests; real fairseq2/7B smoke tests live in preflight.py."""
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import Detector,LoRALinear,optimizer_for
from .objectives import TFCL
from .data import auxiliary_mask,grouped,TrainingLoader
from .step import train_step
from .state import partial_state,apply_partial,to_cpu,capture_rng,restore_rng,choose,budget
from .metrics import ap_eer,measure
from .cleanup import inventory,apply,sha,safe,version
from .performance import select_execution


class Layout:
    def __init__(self,shape,seq_lens,device): self.seq_lens=seq_lens


class Frontend(nn.Module):
    def __init__(self):
        super().__init__(); self.conv=nn.Conv1d(1,16,400,stride=320)
    def forward(self,x,layout):
        y=self.conv(x[:,None]).transpose(1,2)
        return y,Layout(y.shape,[(n-400)//320+1 for n in layout.seq_lens],x.device)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj=nn.Linear(16,16); self.k_proj=nn.Linear(16,16)
        self.v_proj=nn.Linear(16,16); self.output_proj=nn.Linear(16,16)
        self.norm=nn.LayerNorm(16)
    def forward(self,x,layout,cache):
        valid=torch.arange(x.shape[1])[None]<torch.tensor(layout.seq_lens)[:,None]
        q,k,v=[m(x)[:,None] for m in (self.q_proj,self.k_proj,self.v_proj)]
        out=F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None])[:,0]
        return self.norm(x+self.output_proj(out))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); self.layers=nn.ModuleList([Layer(),Layer()]); self.norm=nn.LayerNorm(16)
    def forward(self,x,layout):
        for layer in self.layers: x=layer(x,layout,None)
        return self.norm(x)


def fixture():
    cfg=dict(lora_layers=1,lora_rank=2,lora_alpha=4.,feature_layers=[0,1],amp='none',device='cpu',
        checkpointing=True,lora_lr=.001,head_lr=.001,tfcl_lr=.001,weight_decay=0.,source_batch=4,
        microbatch=4,frame_budget=500,max_grad_norm=1.,tfcl_time_weight=.15,tfcl_structure_weight=.045,
        disk_margin_bytes=1024,maximum_new_peak_bytes=4*1024**3,free_reserve_bytes=10*1024**3)
    model=Detector(Frontend(),Encoder(),cfg,Layout,dim=16,
        head_kwargs=dict(projection=8,expansion=64,blocks=2,dropout=0.))
    aux=TFCL(8,2,7)
    return cfg,model,aux,optimizer_for(model,aux,cfg)


def examples():
    rng=np.random.default_rng(33); rows=[]
    for i,(language,label) in enumerate((('en',0),('en',1),('zh',0),('zh',1))):
        wave=rng.normal(0,.1,1800+i*320).astype(np.float32)
        for role,mass in (('ordinary',.3),('reference',.1),('noisy',.6)):
            rows.append(dict(split='train',language=language,label=label,role=role,
                source_id=str(i),group_id=str(i),source_sha256=str(i),pair_occurrence=str(i),
                pair_eligible=True,ce_weight=mass/4,wave=wave.copy(),aux_spans=[]))
    return rows


class WorkerDataset(torch.utils.data.Dataset):
    def __init__(self,*args): pass
    def __len__(self): return 4
    def __getitem__(self,index): return [dict(index=index,pid=os.getpid(),wave=np.ones(400,dtype=np.float32))]


class WorkerPlan:
    rows=[]
    def batches(self,epoch,start=0): return [[epoch*10+i] for i in range(start,4)]


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def setUp(self): torch.manual_seed(31)

    def test_zero_lora_exact_base_and_gradient(self):
        base=nn.Linear(8,7); m=LoRALinear(base,2,4.); x=torch.randn(3,8)
        self.assertTrue(torch.equal(m(x),base(x)))
        m(x).square().sum().backward()
        self.assertGreater(m.b.grad.norm(),0); self.assertEqual(m.a.grad.norm(),0)
        self.assertTrue(all(p.grad is None for p in base.parameters()))

    def test_exact_length_frontend_and_padding_equivalence(self):
        _,m,_,_=fixture(); m.eval()
        waves=[r['wave'] for r in examples()[::3]]
        with torch.no_grad():
            batched=m(waves)[0]; separate=torch.cat([m([w])[0] for w in waves])
        self.assertTrue(torch.allclose(batched,separate,atol=1e-5,rtol=1e-5))

    def test_checkpoint_recomputation_same_gradients(self):
        _,a,_,_=fixture(); b=copy.deepcopy(a)
        a.set_phase(True,True); b.set_phase(True,False)
        waves=[r['wave'] for r in examples()[::3]]
        a(waves)[0].square().sum().backward(); b(waves)[0].square().sum().backward()
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            if pa.requires_grad: self.assertTrue(torch.allclose(pa.grad,pb.grad,atol=2e-5,rtol=2e-5),na)

    def test_two_sided_tfcl_batched_matches_separate(self):
        aux=TFCL(8,2,7)
        pairs=[]
        for n,k in ((7,5),(4,8)):
            a=torch.randn(n,8,requires_grad=True); b=torch.randn(k,8,requires_grad=True)
            am=torch.ones(n,dtype=torch.bool); am[1]=False
            bm=torch.ones(k,dtype=torch.bool); bm[-1]=False
            pairs.append((a,b,am,bm))
        ta,sd=aux(pairs); single=[aux([p]) for p in pairs]
        self.assertTrue(torch.allclose(ta,torch.cat([v[0] for v in single]),atol=1e-6))
        self.assertTrue(torch.allclose(sd,torch.cat([v[1] for v in single]),atol=1e-6))
        (ta+sd).sum().backward()
        for a,b,am,bm in pairs:
            self.assertGreater(a.grad[am].norm(),0); self.assertGreater(b.grad[bm].norm(),0)
            self.assertEqual(a.grad[~am].norm(),0); self.assertEqual(b.grad[~bm].norm(),0)

    def test_known_missing_masks_frontend_receptive_field(self):
        row=dict(wave=np.zeros(3600),aux_spans=[[1300,1500]])
        mask=auxiliary_mask(row,11,'cpu')
        self.assertFalse(mask[3]); self.assertTrue(mask[9])
        with self.assertRaises(ValueError): auxiliary_mask(row,10,'cpu')

    def test_pairs_never_split_or_audio_truncated(self):
        rows=examples(); pairs=[[r for r in rows if r['pair_occurrence']==str(i) and r['role']!='ordinary'] for i in range(4)]
        batches=list(grouped(pairs,2,1))
        self.assertEqual([len(x) for x in batches],[2]*4)
        self.assertEqual(sum(len(r['wave']) for b in batches for r in b),sum(len(r['wave']) for p in pairs for r in p))

    def test_full_update_lora_head_only_and_total_loss(self):
        cfg,m,aux,opt=fixture()
        frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
        stats=train_step(m,aux,opt,examples(),cfg,1.,True)
        self.assertAlmostEqual(stats['total_loss'],stats['classification_loss']+.15*stats['time_loss']+.045*stats['structure_loss'],places=6)
        self.assertEqual(stats['eligible_pairs'],4)
        self.assertGreater(stats['aux_gradient_audit']['lora_last_output_B'],0)
        self.assertTrue(all(torch.equal(p,frozen[n]) for n,p in m.named_parameters() if n in frozen))

    def test_physical_batch_does_not_change_update(self):
        cfg,a,aa,oa=fixture(); b,ab=copy.deepcopy(a),copy.deepcopy(aa); ob=optimizer_for(b,ab,cfg)
        sa=train_step(a,aa,oa,examples(),dict(cfg,microbatch=2),1.)
        sb=train_step(b,ab,ob,examples(),dict(cfg,microbatch=8),1.)
        self.assertAlmostEqual(sa['total_loss'],sb['total_loss'],places=5)
        for k,v in partial_state(a).items(): self.assertTrue(torch.allclose(v,partial_state(b)[k],atol=3e-5,rtol=3e-5),k)

    def test_partial_resume_matches_uninterrupted_next_update(self):
        cfg,a,aa,oa=fixture()
        b,ab=copy.deepcopy(a),copy.deepcopy(aa); ob=optimizer_for(b,ab,cfg)
        train_step(a,aa,oa,examples(),cfg,1.)
        weights,aux,opt,rng=partial_state(a),to_cpu(aa.state_dict()),to_cpu(oa.state_dict()),capture_rng()
        train_step(a,aa,oa,examples(),cfg,1.)
        apply_partial(b,weights); ab.load_state_dict(aux); ob.load_state_dict(opt); restore_rng(rng)
        train_step(b,ab,ob,examples(),cfg,1.)
        for k,v in partial_state(a).items(): self.assertTrue(torch.equal(v,partial_state(b)[k]),k)

    def test_head_warmup_no_encoder_graph(self):
        cfg,m,aux,opt=fixture(); m.set_phase(False,True)
        train_step(m,aux,opt,examples(),cfg,0.)
        self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if not n.startswith('head.')))

    def test_no_dev_gradients(self):
        cfg,m,aux,opt=fixture(); rows=examples(); rows[0]['split']='dev'
        with self.assertRaises(ValueError): train_step(m,aux,opt,rows,cfg,1.)

    def test_metric_ties_and_direction(self):
        self.assertEqual(ap_eer([0,1],[1,0]),dict(ap=1.,eer=0.))
        self.assertEqual(ap_eer([0,1],[0,0]),dict(ap=.5,eer=.5))
        self.assertEqual(ap_eer([0,0],[1,0]),dict(ap=None,eer=None))
        self.assertEqual(ap_eer([0,1],[0,1]),dict(ap=.5,eer=1.))

    def test_selection_independent_and_no_silent_fallback(self):
        state=dict(selections=dict(best_weighted=None,best_guarded=None),scores=dict(best_weighted=-1.,best_guarded=-1.),candidates={})
        choose(state,'a',dict(weighted_f1=.96),False,{'w':torch.ones(1)})
        self.assertIsNone(state['selections']['best_guarded'])
        choose(state,'b',dict(weighted_f1=.95),True,{'w':torch.zeros(1)})
        self.assertEqual(state['selections'],dict(best_weighted='a',best_guarded='b'))
        choose(state,'c',dict(weighted_f1=.97),True,{'w':torch.zeros(1)})
        self.assertEqual(set(state['candidates']),{'c'})

    def test_worker_pool_reused_across_validation_segments(self):
        with tempfile.TemporaryDirectory() as d,patch('w2v_v316.data.Triplets',WorkerDataset):
            pool=TrainingLoader(WorkerPlan(),dict(workers=1,seed=316),d)
            try:
                first=list(pool.segment(0,0,2)); second=list(pool.segment(1,2,4))
                self.assertEqual(first[0][0]['pid'],second[0][0]['pid'])
                self.assertNotEqual(first[0][0]['pid'],os.getpid())
                self.assertEqual([x[0]['index'] for x in second],[12,13])
                self.assertIsInstance(second[0][0]['wave'],np.ndarray)
            finally: pool.close()

    def test_profiler_rolls_back_moments_rng_after_oom(self):
        cfg,m,aux,opt=fixture(); train_step(m,aux,opt,examples(),cfg,1.)
        cfg.update(autotune=True,gpu_reserve_bytes=1024)
        original=partial_state(m); original_aux=to_cpu(aux.state_dict()); moments=to_cpu(opt.state_dict()); rng=capture_rng()
        def probe(model,auxiliary,optimizer,rows,configuration,warm):
            optimizer.zero_grad(set_to_none=True)
            sum(p.sum()*torch.rand(()) for g in optimizer.param_groups for p in g['params']).backward()
            optimizer.step()
            if configuration['microbatch']>=8: raise torch.cuda.OutOfMemoryError('simulated')
        with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
            for name in ('synchronize','reset_peak_memory_stats','empty_cache'):
                stack.enter_context(patch('torch.cuda.'+name))
            stack.enter_context(patch('torch.cuda.max_memory_reserved',return_value=1024))
            stack.enter_context(patch('torch.cuda.memory_reserved',return_value=1024))
            stack.enter_context(patch('torch.cuda.mem_get_info',return_value=(1024**3,2*1024**3)))
            stack.enter_context(patch('w2v_v316.performance.train_step',side_effect=probe))
            chosen=select_execution(m,aux,opt,examples(),cfg,Path(d))
            self.assertLess(chosen['microbatch'],8)
            for k,v in original.items(): self.assertTrue(torch.equal(v,partial_state(m)[k]))
            for k,v in original_aux.items(): self.assertTrue(torch.equal(v,aux.state_dict()[k]))
            for key,value in moments['state'].items():
                for name,tensor in value.items(): self.assertTrue(torch.equal(tensor,opt.state_dict()['state'][key][name]))
            self.assertTrue(torch.equal(rng['torch'],capture_rng()['torch']))


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        (self.root/'exp').mkdir()
    def tearDown(self): self.tmp.cleanup()
    def run_folder(self,v):
        path=self.root/'exp'/('w2v_v3'+str(v)+'_test'); path.mkdir()
        (path/'config.json').write_text(json.dumps(dict(version='3.'+str(v))))
        (path/'best.pt').write_bytes(b'best'); (path/'last.pt').write_bytes(b'last')
        return path
    def test_version_315_is_not_decimal_3point15(self):
        self.assertEqual(version(self.run_folder(5)),(3,5)); self.assertEqual(version(self.run_folder(15)),(3,15))
    def test_references_protect_v312_last_and_v315(self):
        old=self.run_folder(12); new=self.run_folder(15); trash=self.run_folder(5)
        (new/'config.json').write_text(json.dumps(dict(version='3.15',starting_checkpoint=str(old/'last.pt'))))
        plan=inventory(self.root)
        self.assertEqual({v['path'] for v in plan['candidates']},{str((trash/'last.pt').resolve())})
        result=apply(plan)
        self.assertTrue((old/'last.pt').exists()); self.assertTrue((new/'last.pt').exists())
        self.assertTrue((trash/'best.pt').exists()); self.assertEqual(result['freed_bytes'],4)
    def test_embedded_best_in_last_is_retained(self):
        run=self.run_folder(14)
        (run/'best_weighted.json').write_text(json.dumps(dict(checkpoint='last.pt',tag='epoch_1')))
        self.assertEqual(inventory(self.root)['candidates'],[])
    def test_unknown_run_and_no_best_are_retained(self):
        run=self.run_folder(5); (run/'best.pt').unlink()
        self.assertEqual(inventory(self.root)['candidates'],[])
    def test_metadata_change_aborts_before_delete(self):
        run=self.run_folder(5); plan=inventory(self.root)
        (run/'config.json').write_text('{}')
        with self.assertRaises(ValueError): apply(plan)
        self.assertTrue((run/'last.pt').exists())
    def test_candidate_change_aborts_before_delete(self):
        run=self.run_folder(5); plan=inventory(self.root); (run/'last.pt').write_bytes(b'new checkpoint')
        with self.assertRaises(ValueError): apply(plan)
    def test_escape_rejected(self):
        with self.assertRaises(ValueError): safe(self.root/'..'/'external.pt',self.root)

    def test_verified_feature_arrays_retired_metadata_kept(self):
        run=self.run_folder(7); folder=run/'features'/'train'; folder.mkdir(parents=True)
        (folder/'owner.json').write_text(json.dumps(dict(format='rtc_v36_frozen_vectors_v1',mode='eval-fp32-exact-length-full-wave-final-linear-input')))
        (folder/'rows.json').write_text('[]'); (folder/'x.npy').write_bytes(b'derived')
        (folder/'complete.json').write_text(json.dumps(dict(format='rtc_v36_frozen_vectors_v1',owner_sha256=sha(folder/'owner.json'),
            files={'rows.json':sha(folder/'rows.json'),'x.npy':sha(folder/'x.npy')})))
        plan=inventory(self.root); apply(plan)
        self.assertFalse((folder/'x.npy').exists()); self.assertTrue((folder/'rows.json').exists())

    def test_owned_train_audio_removed_fixed_dev_retained(self):
        train=self.root/'data'/'rtc_v35'/'train'; train.mkdir(parents=True)
        train=train.resolve()
        (train/'owner.json').write_text(json.dumps(dict(format='rtc_v35_epoch_full_views_v1',role='train',root=str(train))))
        (train/'recipe.json').write_text('{}')
        folder=train/'epoch_001'; folder.mkdir()
        (folder/'generation.json').write_text(json.dumps(dict(format='rtc_v35_epoch_full_views_v1',role='train',root=str(train),epoch=1,recipe_sha256=sha(train/'recipe.json'))))
        (folder/'a.wav').write_bytes(b'derived')
        dev=train.parent/'dev'/'epoch_000'; dev.mkdir(parents=True); (dev/'a.wav').write_bytes(b'fixed')
        plan=inventory(self.root); apply(plan)
        self.assertFalse((folder/'a.wav').exists()); self.assertTrue((dev/'a.wav').exists())


if __name__=='__main__': unittest.main()
