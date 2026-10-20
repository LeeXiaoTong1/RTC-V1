from contextlib import ExitStack,redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .common import atomic_json,read_json,partial_state,atomic_save,seed_all
from w2v_v31810.test_meta import assert_state_equal
from .test_core import Tiny,fixture,cfg
from .train import run_experiment
from .step import train_step
from .evaluate import selected
from . import VERSION,SCHEMA


class FixturePlan:
    def __init__(self,rows,cfg,phones):self.steps=2;self.audit={}
    def epoch_batches(self,epoch):
        return [[dict(row=i,epoch=epoch,pcl_official=True,pcl_noise=True) for i in range(8)] for _ in range(2)]


class FixtureWaves:
    def __init__(self,rows,cfg):self.units=fixture()
    def __getitem__(self,t):
        unit=copy.deepcopy(self.units[t['row']])
        if t['epoch']==0:
            unit[0].pop('pcl',None)
        return unit


class FixtureLoader:
    def __init__(self,dataset,sampler):self.dataset=dataset;self.sampler=sampler
    def __iter__(self):
        for tickets in self.sampler.batches:yield [self.dataset[t] for t in tickets]


def optimizer(model,cfg):
    return torch.optim.AdamW([dict(params=list(model.parameters()),lr=.01,initial_lr=.01,role='head')])


class Evaluator:
    def __init__(self,test,run,parent_best=False):self.test=test;self.run=run;self.calls=[];self.parent_best=parent_best
    def __call__(self,model,protocol,cfg,**kwargs):
        tag=kwargs['checkpoint_tag'];self.calls.append((kwargs['split'],tag))
        if kwargs['split']=='audit':
            state=torch.load(self.run/'last.pt',weights_only=True)
            for name,value in partial_state(model).items():torch.testing.assert_close(value,state['best_model'][name],rtol=0,atol=0)
        score=.99 if tag=='parent' and self.parent_best else .8 if tag=='epoch_1' else .7
        return dict(score=score,panels={name:dict(macro_f1=score) for name in ('official_online','regular_noisy','heldout_noisy')},groups={})


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def patches(self,stack):
        stack.enter_context(patch('w2v_v41.train.Plan',FixturePlan))
        stack.enter_context(patch('w2v_v41.train.Waves',FixtureWaves))
        stack.enter_context(patch('w2v_v41.train.loader',lambda d,c,*,batch_sampler:FixtureLoader(d,batch_sampler)))
        stack.enter_context(patch('w2v_v41.train.close'))
        stack.enter_context(redirect_stdout(io.StringIO()))
    def make(self,root,name,settings):
        run=root/name;run.mkdir();atomic_json(run/'config.json',settings)
        atomic_json(run/'pcl_pairs.json',dict(official={},noise={}))
        seed_all(42);atomic_save(run/'init.pt',partial_state(Tiny()));return run
    def test_full_pause_resume_and_best_audit_are_exact(self):
        settings=cfg(version=VERSION,recipe=SCHEMA,epochs=3,warm_epochs=1,lr_floor=.1,lr_warmup_updates=2,free_reserve_bytes=0)
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            self.patches(stack);root=Path(directory)
            full=self.make(root,'full',settings);resume=self.make(root,'resume',settings)
            aev=Evaluator(self,full);bev=Evaluator(self,resume)
            run_experiment(settings,full,[],{},model_factory=Tiny,optimizer_factory=optimizer,evaluator=aev)
            def stop(model,opt,units,cfg,weights,epoch,step,steps):
                result=train_step(model,opt,units,cfg,weights,epoch,step,steps)
                if epoch==1:atomic_json(resume/'stop_requested.json',dict(test=True))
                return result
            with patch('w2v_v41.train.train_step',stop):
                result=run_experiment(settings,resume,[],{},model_factory=Tiny,optimizer_factory=optimizer,evaluator=bev)
            self.assertEqual(result['status'],'paused')
            state=torch.load(resume/'last.pt',weights_only=True);self.assertEqual((state['epoch'],state['next_step']),(1,1))
            run_experiment(settings,resume,[],{},model_factory=Tiny,optimizer_factory=optimizer,evaluator=bev)
            a=torch.load(full/'last.pt',weights_only=True);b=torch.load(resume/'last.pt',weights_only=True)
            for key in ('model','best_model','optimizer','rng','grad_scaler','weights'):assert_state_equal(self,a[key],b[key])
            self.assertEqual(a['cursor'],6);self.assertEqual(a['history'],b['history'])
            self.assertEqual(aev.calls,bev.calls);self.assertEqual(aev.calls[-1],('audit','epoch_1'))
            _,weights,done=selected(resume);self.assertEqual(done['best_tag'],'epoch_1')
            assert_state_equal(self,weights,b['best_model'])
    def test_untrained_model_is_never_exported_as_parent_best(self):
        settings=cfg(version=VERSION,recipe=SCHEMA,epochs=1,warm_epochs=1,lr_floor=.1,free_reserve_bytes=0)
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            self.patches(stack);run=self.make(Path(directory),'run',settings);ev=Evaluator(self,run,True)
            done=run_experiment(settings,run,[],{},model_factory=Tiny,optimizer_factory=optimizer,evaluator=ev)
            self.assertEqual(done['best_tag'],'epoch_1');self.assertNotIn(('select','parent'),ev.calls)
    def test_pause_before_any_epoch_has_no_exportable_best(self):
        settings=cfg(version=VERSION,recipe=SCHEMA,epochs=2,free_reserve_bytes=0)
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            self.patches(stack);run=self.make(Path(directory),'run',settings)
            atomic_json(run/'stop_requested.json',dict(test=True))
            done=run_experiment(settings,run,[],{},model_factory=Tiny,optimizer_factory=optimizer,evaluator=Evaluator(self,run))
            self.assertEqual(done['status'],'paused')
            with self.assertRaisesRegex(ValueError,'No trained epoch'):selected(run)

    def test_pre_optimizer_oom_retries_same_update_without_losing_rng_or_adam(self):
        from types import SimpleNamespace
        from .step import components
        class Recoverable(Tiny):
            def __init__(self,cfg=None):
                super().__init__(cfg);self.encoder=SimpleNamespace(layers=[None,None]);self.joint=False
            def set_phase(self,joint):self.joint=joint
            def set_checkpoint_count(self,count):self.checkpoint_override=count
        settings=cfg(version=VERSION,recipe=SCHEMA,epochs=3,warm_epochs=1,lr_floor=.1,
            lr_warmup_updates=2,free_reserve_bytes=0,checkpoint_policy='fixed',checkpoint_layers=0)
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            self.patches(stack);root=Path(directory)
            full=self.make(root,'full',settings);retry=self.make(root,'retry',settings)
            run_experiment(settings,full,[],{},model_factory=Recoverable,optimizer_factory=optimizer,evaluator=Evaluator(self,full))
            failed=[False]
            def fail_once(model,*args,**kwargs):
                for i,part in enumerate(components(model,*args,**kwargs)):
                    yield part
                    if model.joint and model.training and not failed[0]:
                        failed[0]=True;torch.rand(7)
                        raise torch.cuda.OutOfMemoryError('test pre-optimizer allocation failure')
            with patch('w2v_v41.step.components',fail_once):
                run_experiment(settings,retry,[],{},model_factory=Recoverable,optimizer_factory=optimizer,evaluator=Evaluator(self,retry))
            self.assertTrue(failed[0]);self.assertTrue((retry/'execution_memory_fallback.json').exists())
            a=torch.load(full/'last.pt',weights_only=True);b=torch.load(retry/'last.pt',weights_only=True)
            for key in ('model','best_model','optimizer','rng','grad_scaler','weights','history','cursor'):
                assert_state_equal(self,a[key],b[key])
            self.assertEqual(b['checkpoint_layers_effective'],2)

if __name__=='__main__':unittest.main()
