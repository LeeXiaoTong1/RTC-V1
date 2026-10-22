import copy
from contextlib import ExitStack
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import train, workflow, delivery
from .common import RECIPE, partial_state


def cfg():
    return dict(version='5.5',recipe=RECIPE,training_recipe_base='5.1',v5_stage='final',
        parent_init=dict(kind='v5_submitted_final'),lora_layers=16,retained_lora_layers=60,
        experiment='E0',robust_arm='baseline',seed=123,sampling_seed=123,epochs=2,
        v5_tail_snapshots=4,v5_tail_start=.75,checkpoint_interval=2,device='cpu',
        backend_precision='fp32',classification_ce_weights=dict(online=.25,clean_rtc=.25,noisy=.5))


class Model(torch.nn.Module):
    def __init__(self,cfg=None):
        super().__init__()
        self.encoder=torch.nn.Linear(2,2)
        self.encoder.requires_grad_(False)
        self.head=torch.nn.Linear(2,1)


class Plan:
    steps=8
    def batches(self,epoch):
        return [[epoch,i] for i in range(self.steps)]
    def coverage(self,epoch):
        return dict(epoch=epoch,steps=self.steps)


class Loader:
    def __init__(self,sampler): self.sampler=sampler
    def __iter__(self): yield from self.sampler.batches


def initialize(model,cfg):
    with torch.no_grad():
        model.head.weight.fill_(.1);model.head.bias.fill_(.2)
    return partial_state(model)


def step(model,opt,ticket,cfg,audit=False):
    x=torch.rand(1,2)+random.random()*.1+np.random.random()*.1
    loss=(model.head(model.encoder(x))-.5).square().mean()
    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return dict(loss=float(loss.detach()),classification_loss=float(loss.detach()),encoded_views=1,
        amp=dict(skipped_update=False),cells={},hard_positive_gradient_audit={},meta={},
        hard_positive=dict(groups=0,guidance_weighted=0.,triplet_weighted=0.,extra_encoded_views=0,
            sides={s:dict(records=[],skips={}) for s in ('A','B')}))


class LifecycleTests(unittest.TestCase):
    def patches(self):
        stack=ExitStack()
        for name,value in (('make_plan',lambda rows,cfg:Plan()),('Waves',lambda rows,cfg:rows),
                ('loader',lambda rows,cfg,batch_sampler:Loader(batch_sampler)),
                ('optimizer_for',lambda m,c:torch.optim.AdamW(m.head.parameters(),lr=.01)),
                ('close',lambda _:None),('adapter_inventory',lambda _:{}),('schedule',lambda *args:None)):
            stack.enter_context(patch.object(train,name,side_effect=value))
        return stack

    def run_stage(self,root,step_fn=step):
        return train.run_stage(cfg(),root/'stage',[],model_factory=Model,step_fn=step_fn,initializer=initialize)

    def equal_states(self,left,right):
        for category in ('model','tail_model'):
            self.assertEqual(set(left[category]),set(right[category]))
            for name,value in left[category].items():
                torch.testing.assert_close(value,right[category][name],rtol=0,atol=0)
        self.assertEqual(left['cursor'],right['cursor'])
        self.assertEqual(left['tail_count'],right['tail_count'])
        for key,state in left['optimizer']['state'].items():
            for name,value in state.items():
                torch.testing.assert_close(value,right['optimizer']['state'][key][name],rtol=0,atol=0)

    def test_pause_resume_exact_adam_rng_tail_and_no_second_initialization(self):
        with tempfile.TemporaryDirectory() as directory,self.patches():
            root=Path(directory)
            _,expected=self.run_stage(root/'whole')
            count=0
            def pausing(*args,**kwargs):
                nonlocal count
                result=step(*args,**kwargs);count+=1
                if count==13:(root/'paused'/'STOP_REQUESTED').touch()
                return result
            _,paused=self.run_stage(root/'paused',pausing)
            self.assertEqual(paused['status'],'paused')
            self.assertEqual(paused['cursor'],13)
            (root/'paused'/'STOP_REQUESTED').unlink()
            _,actual=self.run_stage(root/'paused')
            self.equal_states(expected,actual)
            self.run_stage(root/'paused',lambda *a,**k:self.fail('complete run trained again'))
            lines=[json.loads(v) for v in (root/'paused'/'stage'/'steps.jsonl').read_text().splitlines()]
            self.assertEqual([x['update'] for x in lines],list(range(1,17)))

    def test_crash_trims_uncommitted_logs_and_replays_deterministically(self):
        with tempfile.TemporaryDirectory() as directory,self.patches():
            root=Path(directory)
            _,expected=self.run_stage(root/'whole')
            count=0
            def crashing(*args,**kwargs):
                nonlocal count
                count+=1
                if count==6:raise KeyboardInterrupt()
                return step(*args,**kwargs)
            with self.assertRaises(KeyboardInterrupt):self.run_stage(root/'crash',crashing)
            _,actual=self.run_stage(root/'crash')
            self.equal_states(expected,actual)
            lines=[json.loads(v) for v in (root/'crash'/'stage'/'steps.jsonl').read_text().splitlines()]
            self.assertEqual([x['update'] for x in lines],list(range(1,17)))


if __name__=='__main__':
    unittest.main()
