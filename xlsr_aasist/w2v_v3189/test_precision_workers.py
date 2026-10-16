"""Precision boundaries, scaled accumulation/restart and deterministic CPU queues."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .precision import scaler_for,unscale_and_clip,commit
from .workers import settings,loader
from .config import parser,validate
from .data import close
from .test_model import tiny_cfg,tiny_model
from .test_step import fixture,CountingDetector,optimizer
from .step import train_step


class IndexedWaves(torch.utils.data.Dataset):
    def __len__(self):return 9
    def __getitem__(self,index):
        rng=np.random.default_rng(index+123)
        return dict(index=index,wave=rng.normal(size=40).astype(np.float32))


class PrecisionWorkersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_scaling_accumulates_then_unscales_once_and_clips(self):
        cfg,units=fixture();cfg.update(consistency='none')
        torch.manual_seed(55);a=CountingDetector();b=copy.deepcopy(a)
        oa,ob=optimizer(a),optimizer(b)
        ob._v3189_scaler=torch.amp.GradScaler('cpu',init_scale=256.)
        sa=train_step(a,oa,units,cfg,0.,audit=True)
        sb=train_step(b,ob,units,cfg,0.,audit=True)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertAlmostEqual(sa['grad_norm'],sb['grad_norm'],places=6)
        self.assertFalse(sb['amp']['skipped_update'])
        self.assertEqual(sb['amp']['scale'],256.)

    def test_overflow_skips_whole_update_and_recovers(self):
        p=torch.nn.Parameter(torch.tensor([1.]));opt=torch.optim.SGD([p],lr=.1)
        scaler=torch.amp.GradScaler('cpu',init_scale=256.)
        scaler.scale(p.sum()*float('inf')).backward()
        norm=unscale_and_clip(opt,scaler,1.);status=commit(opt,scaler)
        self.assertIsNone(norm);self.assertTrue(status['skipped_update'])
        self.assertEqual(p.item(),1.);self.assertEqual(scaler.get_scale(),128.)
        opt.zero_grad(set_to_none=True);scaler.scale(p.sum()).backward()
        unscale_and_clip(opt,scaler,1.);self.assertFalse(commit(opt,scaler)['skipped_update'])
        self.assertLess(p.item(),1.)

    def test_scaler_restart_matches_uninterrupted_growth_and_parameters(self):
        p=torch.nn.Parameter(torch.tensor([1.]));a=torch.optim.AdamW([p],lr=.01)
        s=torch.amp.GradScaler('cpu',init_scale=32.,growth_interval=2)
        def update(p,opt,scale):
            opt.zero_grad(set_to_none=True);scale.scale(p.square().sum()).backward()
            unscale_and_clip(opt,scale,1.);commit(opt,scale)
        update(p,a,s)
        q=torch.nn.Parameter(p.detach().clone());b=torch.optim.AdamW([q],lr=.01)
        b.load_state_dict(copy.deepcopy(a.state_dict()))
        t=torch.amp.GradScaler('cpu');t.load_state_dict(s.state_dict())
        update(p,a,s);update(q,b,t)
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        self.assertEqual(s.state_dict(),t.state_dict());self.assertEqual(s.get_scale(),64.)

    def test_real_backend_fp16_math_with_fp32_pooling_and_gradients(self):
        # Exercise actual FP16 kernels on CPU as a logic/numerics test. Production
        # enables them only on CUDA, whose native preflight must still run.
        cfg=tiny_cfg();cfg.update(backend_precision='fp16')
        model=tiny_model(cfg);model.train();seen=[];pooled=[]
        h=model.head.aasist.out_layer.register_forward_hook(lambda _m,_a,y:seen.append(y.dtype))
        j=model.head.aasist.out_layer.register_forward_pre_hook(lambda _m,x:pooled.append(x[0].dtype))
        x=[torch.randn(n,128,requires_grad=True) for n in (12,25,140)]
        with patch('w2v_v3189.regions.backend_context',side_effect=lambda *_:torch.autocast('cpu',dtype=torch.float16)):
            z=model.classify_sequences(x)
        self.assertEqual(seen,[torch.float16]);self.assertEqual(pooled,[torch.float32])
        self.assertEqual(z.dtype,torch.float32);self.assertEqual(z.shape,(3,2))
        opt=torch.optim.SGD(model.head.parameters(),lr=.001)
        scaler=torch.amp.GradScaler('cpu',init_scale=256.)
        scaler.scale(torch.nn.functional.cross_entropy(z,torch.tensor([0,1,1]))).backward()
        unscale_and_clip(opt,scaler,1.);self.assertFalse(commit(opt,scaler)['skipped_update'])
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in x))
        self.assertTrue(all(p.dtype==torch.float32 for p in model.head.parameters()))
        h.remove();j.remove()

    def test_training_pause_resume_restores_enabled_scaler(self):
        from .test_training import TrainingTests
        original=torch.amp.GradScaler
        def cpu_scaler(*args,**kwargs):
            return original('cpu',enabled=True,init_scale=256.,growth_interval=2)
        case=TrainingTests('test_mid_joint_epoch_pause_keeps_best_and_resume_is_exact')
        with patch('w2v_v3189.precision.torch.amp.GradScaler',side_effect=cpu_scaler):
            case.test_mid_joint_epoch_pause_keeps_best_and_resume_is_exact()

    def test_new_precision_reevaluates_parent_once(self):
        from .test_single_noisy import parent_fixture
        from .parent import initialize
        with tempfile.TemporaryDirectory() as folder:
            _,cfg,_,dev,partition,_=parent_fixture(Path(folder))
            cfg['backend_precision']='fp16';model=tiny_model(cfg)
            z=np.zeros((len(dev),2),np.float32);z[:,0]=1.
            with patch('w2v_v3189.inference.infer',return_value=(z,dev)) as infer:
                baseline,_,logits,audit=initialize(model,cfg,dev,partition)
            infer.assert_called_once()
            self.assertTrue(audit['baseline_reevaluated'])
            np.testing.assert_array_equal(logits.numpy(),z)
            self.assertIn('Re-evaluated',baseline['score_provenance'])

    def test_auto_workers_bound_queues_and_aggregate_caches(self):
        args=parser().parse_args([]);validate(args)
        self.assertEqual(args.backend_precision,'fp16')
        old=dict(workers=4,raw_audio_cache_mib=128,noise_cache_mib=128,stream_sources=24)
        host=dict(usable_cpu_cores=32,available_memory_bytes=64*1024**3)
        result=settings(args,old,host)
        self.assertEqual((result['workers'],result['eval_workers'],result['prefetch_factor']),(8,4,1))
        self.assertEqual(result['worker_resource_audit']['queued_train_batches'],8)
        self.assertEqual(8*sum(result[k] for k in ('raw_audio_cache_mib','noise_cache_mib','scene_cache_mib')),4*(128+128+32))
        result=settings(args,old,dict(host,usable_cpu_cores=4))
        self.assertEqual(result['workers'],2)
        result=settings(args,old,dict(host,available_memory_bytes=2*1024**3))
        self.assertEqual(result['workers'],0)
        explicit=parser().parse_args(['--workers','6','--eval-workers','2'])
        self.assertEqual(settings(explicit,old,dict(host,usable_cpu_cores=2))['workers'],6)

    def test_workers_zero_and_two_keep_order_and_values(self):
        outputs=[]
        for count in (0,2):
            batches=loader(IndexedWaves(),dict(workers=count,seed=10,prefetch_factor=1),batch_size=3)
            try:
                self.assertEqual(batches.prefetch_factor,1 if count else None)
                outputs.append([r for batch in batches for r in batch])
            finally:close(batches)
        self.assertEqual([r['index'] for r in outputs[1]],list(range(9)))
        for a,b in zip(*outputs):np.testing.assert_array_equal(a['wave'],b['wave'])


if __name__=='__main__':unittest.main()
