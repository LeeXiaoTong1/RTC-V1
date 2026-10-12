"""Exact learned-weight migration and phase schedule, including frozen warm LoRA."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch,Mock
import numpy as np
import torch
from w2v_v3183.test_support import tiny_cfg as old_cfg,tiny_model as old_model
from w2v_v3183.common import partial_state as old_state
from .test_support import tiny_cfg,tiny_model,metadata,units
from .parent import initialize,baseline_check
from .step import schedule,train_step
from .model import optimizer_for
from .common import digest,atomic_json


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self,mode='parent-transfer'):
        prior=old_cfg();old=old_model(prior)
        with torch.no_grad():
            for name,value in old.named_parameters():
                if name.endswith('.lora_b'):value.normal_(0,.1)
            for name,value in old.named_buffers():
                if name.endswith('running_mean'):value.fill_(3.)
                if name.endswith('running_var'):value.fill_(4.)
                if name.endswith('num_batches_tracked'):value.fill_(7)
        cfg=tiny_cfg();cfg['initialization']=dict(mode=mode,run='fixture',tag='epoch_6_step_3',
            checkpoint_sha256='fixture-hash',files={})
        meta=dict(tag='epoch_6_step_3',checkpoint_sha256='fixture-hash')
        return cfg,prior,old_state(old),meta

    def test_parent_transfer_moves_projection_preserves_weights_and_resets_only_bn_statistics(self):
        cfg,prior,values,meta=self.fixture();model=tiny_model(cfg)
        with patch('w2v_v3183r2.parent.inspect_parent',return_value=(prior,values,meta)):
            audit=initialize(model,cfg)
        actual=model.state_dict()
        for source,destination in audit['copied_tensors'].items():
            if source.endswith(('running_mean','running_var','num_batches_tracked')):continue
            torch.testing.assert_close(actual[destination],values[source],rtol=0,atol=0)
        self.assertFalse(audit['old_model_prediction_identity'])
        for name,value in model.named_buffers():
            if name.endswith(('running_mean','num_batches_tracked')):self.assertEqual(int(torch.count_nonzero(value)),0)
            if name.endswith('running_var'):torch.testing.assert_close(value,torch.ones_like(value))
        self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b'))))
        from .model import LoRALinear
        self.assertTrue(all(m.enabled for m in model.modules() if isinstance(m,LoRALinear)))
        self.assertEqual(set(audit['discarded_parent_fusion']),{'fusion.gate','fusion.mix_logits'})

    def test_pretrained_does_not_copy_learned_parameters(self):
        cfg,prior,values,meta=self.fixture('pretrained');model=tiny_model(cfg)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        with patch('w2v_v3183r2.parent.inspect_parent',return_value=(prior,values,meta)):
            audit=initialize(model,cfg)
        self.assertEqual(audit['copied_tensors'],{})
        for n,p in model.named_parameters():torch.testing.assert_close(p,before[n],rtol=0,atol=0)

    def test_malformed_parent_fails_before_mutating_model(self):
        cfg,prior,values,meta=self.fixture();model=tiny_model(cfg)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        values['head.aasist.LL.weight']=values['head.aasist.LL.weight'].double()
        with patch('w2v_v3183r2.parent.inspect_parent',return_value=(prior,values,meta)):
            with self.assertRaisesRegex(ValueError,'Invalid parent transfer tensor'):initialize(model,cfg)
        for n,p in model.named_parameters():torch.testing.assert_close(p,before[n],rtol=0,atol=0)
        values.pop('fusion.gate')
        with patch('w2v_v3183r2.parent.inspect_parent',return_value=(prior,values,meta)):
            with self.assertRaisesRegex(ValueError,'inventory mismatch'):initialize(model,cfg)

    def test_cached_best_reference_never_runs_new_model_or_uses_last(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=root/'parent';run=root/'run';parent.mkdir();run.mkdir()
            best=torch.tensor([[2.,-2.],[-2.,2.],[3.,-3.],[-3.,3.]])
            torch.save(dict(best_tag='epoch_4_step_3',best_logits=best,last_logits=-best),parent/'last.pt')
            cfg=tiny_cfg();cfg['initialization']=dict(run=str(parent),tag='epoch_4_step_3',checkpoint_sha256=digest(parent/'last.pt'))
            measure=lambda rows,z:dict(weighted_f1=float(z[0,0]),noisy_f1=.5,clean_f1=.5)
            infer=Mock(side_effect=AssertionError('No inference expected'))
            result=baseline_check(tiny_model(cfg),cfg,run,[{}]*4,{'select':[0,2]},infer,measure)
            infer.assert_not_called();self.assertTrue(result['reference_only']);self.assertFalse(result['new_model_reproduces_parent'])
            np.testing.assert_array_equal(np.load(run/'parent_baseline_scores.npz')['logits'],best.numpy())

    def test_schedule_freezes_warm_lora_and_ramps_joint_without_head_reset(self):
        cfg=tiny_cfg();cfg.update(epochs=6,warm_epochs=1,lr_warmup_updates=200)
        opt=optimizer_for(tiny_model(cfg),cfg)
        rates=lambda:{g['name']:g['lr'] for g in opt.param_groups}
        schedule(opt,cfg,0,0,1000)
        self.assertEqual(rates()['lora'],0.)
        self.assertAlmostEqual(rates()['fusion'],cfg['warm_fusion_lr']/200)
        schedule(opt,cfg,0,999,1000)
        self.assertAlmostEqual(rates()['aasist'],cfg['warm_lr'])
        schedule(opt,cfg,1,0,1000)
        self.assertAlmostEqual(rates()['aasist'],cfg['head_lr'])
        self.assertAlmostEqual(rates()['lora'],cfg['lora_lr']/200)
        schedule(opt,cfg,1,199,1000)
        self.assertGreater(rates()['lora'],cfg['lora_lr']*.99)
        schedule(opt,cfg,5,999,1000)
        for group in opt.param_groups:self.assertAlmostEqual(group['lr'],group['initial_lr']*cfg['lr_floor'])
        with self.assertRaises(ValueError):schedule(opt,cfg,6,0,1000)

    def test_cli_default_parent_pointer_full_depth_and_dependency_preservation(self):
        from .config import parser,validate,configuration,verify_inputs
        from .parent import inspect_parent
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'exp').mkdir();parent=root/'parent';parent.mkdir()
            (root/'exp'/'.latest_v3183_run').write_text(str(parent)+'\n')
            package=root/'w2v_v3183r2';package.mkdir();module=package/'fusion.py';module.write_text('# fixture\n')
            rows=metadata((8,4,12,6),missing=False)
            atomic_json(parent/'train_rows.json',rows);atomic_json(parent/'dev_rows.json',[])
            old=old_cfg();old.update(omni_arch='3b',encoder_layers=60,
                code_fingerprints={'v318/model.py':'v318-hash','v3183/model.py':'producer-hash','rtc_noisy/simulator.py':'rtc-hash'},
                data_files={'train-source.json':'data-hash'})
            initial=dict(run=str(parent),tag='epoch_4_step_2417',kind='best',checkpoint_sha256='fixture',
                         files={'parent-last.pt':'checkpoint-hash'})
            args=parser().parse_args([]);validate(args)
            self.assertEqual((args.epochs,args.warm_epochs,args.init_mode),(6,1,'parent-transfer'))
            self.assertEqual((args.lora_lr,args.head_lr,args.fusion_lr,args.layer_mix_lr),(2e-5,1e-4,3e-4,1e-3))
            with patch('w2v_v3183r2.config.ROOT',root), \
                 patch('w2v_v3183r2.config.inspect_parent',return_value=(old,{},initial)) as inspect, \
                 patch('w2v_v3183r2.config.verify_inputs') as verify, \
                 patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.is_bf16_supported',return_value=True):
                cfg,actual,_=configuration(args)
            inspect.assert_called_once_with(str(parent),None);verify.assert_called_once_with(cfg)
            self.assertEqual(cfg['fusion_layers'],list(range(1,61)))
            self.assertEqual(cfg['fusion_dim'],128);self.assertNotIn('fusion_max_residual',cfg)
            self.assertEqual(cfg['initialization']['kind'],'best');self.assertEqual(actual,rows)
            for name,value in old['code_fingerprints'].items():self.assertEqual(cfg['code_fingerprints'][name],value)
            self.assertEqual(cfg['code_fingerprints'][str(module)],digest(module))
            self.assertEqual(cfg['data_files'],dict(old['data_files'],**initial['files']))
            self.assertEqual(old['version'],'3.18.3')
            with self.assertRaisesRegex(ValueError,'Not V3.18.3-R2'):verify_inputs(dict(version='3.18.3'))
            wrong=dict(old,version='3.18')
            with patch('w2v_v3183r2.parent.selected',return_value=(wrong,{},{})):
                with self.assertRaisesRegex(ValueError,'completed V3.18.3'):inspect_parent(parent)

    def test_actual_update_audit_matches_parameters_and_reports_frozen_zero(self):
        from .sampling import Plan
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(8,2)
                self.adapter=torch.nn.Parameter(torch.ones(3),requires_grad=False)
            def forward(self,waves):return self.linear(torch.as_tensor(np.stack(waves)))
        cfg=tiny_cfg();cfg.update(stream_sources=16,group_quotas=[4,2,7,3],frame_budget=1000,microbatch=48)
        batch=units(Plan(metadata((8,4,12,6),missing=False),16),samples=8);model=Toy()
        opt=torch.optim.AdamW([dict(name='aasist',params=list(model.linear.parameters())),
                              dict(name='lora',params=[model.adapter])],lr=.01)
        previous=[p.detach().clone() for p in model.linear.parameters()]
        stats=train_step(model,opt,batch,cfg,.25,audit=True)
        update=float(torch.stack([(p-old).square().sum() for p,old in zip(model.linear.parameters(),previous)]).sum().sqrt())
        norm=float(torch.stack([p.square().sum() for p in previous]).sum().sqrt())
        self.assertAlmostEqual(stats['module_update_norms']['aasist'],update,places=7)
        self.assertAlmostEqual(stats['module_relative_updates']['aasist'],update/norm,places=7)
        self.assertEqual(stats['module_update_norms']['lora'],0.)
        self.assertEqual(stats['module_relative_updates']['lora'],0.)
        stats=train_step(model,opt,batch,cfg,.25,audit=False)
        self.assertEqual(stats['module_update_norms'],{});self.assertEqual(stats['module_relative_updates'],{})
        original_step=opt.step
        def invalid_step():
            original_step()
            with torch.no_grad():model.linear.weight.fill_(float('inf'))
        with patch.object(opt,'step',invalid_step):
            with self.assertRaisesRegex(FloatingPointError,'module update audit'):
                train_step(model,opt,batch,cfg,.25,audit=True)


if __name__=='__main__':unittest.main()
