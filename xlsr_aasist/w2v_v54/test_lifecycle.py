"""Actual repair optimizer/phase and real checkpoint driver, with tiny fixtures."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import train
from .test_integration import fixture
from .common import partial_state, apply_partial
from w2v_v53.test_lifecycle import TinyLoader


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_committed_resume_across_warmup_and_tail_mean_is_exact(self):
        cfg,units,initial,_=fixture('paired')
        cfg.update(sampling_seed=7,v5_stage='final',v5_tail_snapshots=4,v5_tail_start=.5,
            checkpoint_interval=1,workers=0,repair_warmup_max=2,repair_warmup_fraction=.25)
        class Plan:
            steps=8
            def batches(self,epoch):return [units]*8
            def coverage(self,epoch):return dict(steps=8)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root=Path(directory)
            stack.enter_context(patch.object(train,'schema_for',return_value='v54_lifecycle_test'))
            stack.enter_context(patch.object(train,'make_plan',return_value=Plan()))
            stack.enter_context(patch.object(train,'Waves',side_effect=lambda rows,cfg:rows))
            stack.enter_context(patch.object(train,'loader',side_effect=lambda d,c,batch_sampler:TinyLoader(batch_sampler)))
            stack.enter_context(patch.object(train,'close'))
            factory=lambda c:copy.deepcopy(initial)
            initializer=lambda m,c:partial_state(m)
            _,reference=train.run_stage(cfg,root/'full'/'final',[],model_factory=factory,initializer=initializer)
            calls=[]
            def pause(model,opt,rows,settings,**kwargs):
                stats=train.train_step(model,opt,rows,settings,**kwargs)
                calls.append(1)
                if len(calls)==2:(root/'split'/'STOP_REQUESTED').touch()
                return stats
            _,first=train.run_stage(cfg,root/'split'/'final',[],model_factory=factory,initializer=initializer,step_fn=pause)
            self.assertEqual(first['cursor'],2)
            self.assertEqual(first['status'],'paused')
            (root/'split'/'STOP_REQUESTED').unlink()
            _,resumed=train.run_stage(cfg,root/'split'/'final',[],model_factory=factory,
                initializer=lambda *a:self.fail('Parent reinitialized'))
            self.assertEqual(resumed['cursor'],8)
            self.assertEqual(resumed['tail_count'],4)
            for section in ('model','tail_model'):
                for name,value in reference[section].items():
                    torch.testing.assert_close(resumed[section][name],value,rtol=0,atol=0)
            for key,state in reference['optimizer']['state'].items():
                for name,value in state.items():
                    torch.testing.assert_close(resumed['optimizer']['state'][key][name],value,rtol=0,atol=0)
            events=[json.loads(line) for line in (root/'split'/'final'/'hard_examples.jsonl').read_text().splitlines()]
            self.assertEqual([e['update'] for e in events],list(range(1,9)))
            self.assertTrue(all(e['validated'] for e in events))
            with self.assertRaisesRegex(ValueError,'incompatible resume'):
                train.run_stage(dict(cfg,sampling_seed=8),root/'split'/'final',[],model_factory=factory,initializer=initializer)


if __name__=='__main__':unittest.main()
