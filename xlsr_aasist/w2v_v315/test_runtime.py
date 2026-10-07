from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from w2v_v3.test_model_step import tiny_detector
from w2v_v32.model import install_runtime
from w2v_v313.test_workflow import assert_nested_equal
from w2v_v39.common import atomic_json, read_json, digest
from .objectives import TFCL
from .performance import select_execution
from .state import identity, partial_state, capture_rng, SCHEMA, load_selected, atomic_save
from .cleanup import cleanup


def completed_fixture(directory):
    root=Path(directory);source=root/'protected_last.pt';torch.save({'source':torch.ones(4)},source)
    run=root/'v315';run.mkdir()
    cfg=dict(version='3.15',source_fingerprints={},data_fingerprints={},base_checkpoint=str(source),
        base_checkpoint_sha256=digest(source),starting_checkpoint=str(source),starting_checkpoint_sha256=digest(source),
        disk_margin_bytes=0,rolling_cache_bytes=10000)
    execution={'test':True};atomic_json(run/'execution_plan.json',execution)
    h=digest(run/'execution_plan.json');selection=dict(best_weighted='joint',best_guarded='starting_last')
    weights={'weight':torch.randn(2,4)}
    state=dict(schema=SCHEMA,identity=identity(cfg),cursor=2,last_tag='joint',selections=selection,
        model=weights,candidates={'joint':dict(kind='partial',state=weights)},
        history=[dict(tag='joint',cursor=2,committed=True)],execution_plan_sha256=h,
        optimizer={'large_state':torch.ones(20000)},rng=capture_rng(),auxiliary={'discard':torch.ones(200)})
    atomic_save(run/'last.pt',state,0);atomic_json(run/'config.json',cfg)
    done=dict(version='3.15',status='complete',state_file='last.pt',checkpoint_sha256=digest(run/'last.pt'),
        selections=selection,last_tag='joint',committed_updates=2,execution_plan_sha256=h,
        base_checkpoint_sha256=cfg['base_checkpoint_sha256'],starting_checkpoint_sha256=cfg['starting_checkpoint_sha256'])
    atomic_json(run/'completed.json',done)
    for kind in ('best_guarded','best_weighted','last'):atomic_json(run/(kind+'.json'),{'checkpoint':'last.pt'})
    return run,cfg


class RuntimeTests(unittest.TestCase):
    def test_tuner_rolls_back_parameters_moments_rng_including_after_oom(self):
        model=install_runtime(tiny_detector(checkpointing=False));model.configure_trainable_layers(1)
        aux=TFCL(8,2,21)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]+list(aux.parameters()),lr=.01)
        original=partial_state(model);original_aux=copy.deepcopy(aux.state_dict());rng=capture_rng()
        def simulated_step(model,auxiliary,optim,examples,cfg,warm):
            random.random();np.random.rand();torch.rand(3)
            optim.zero_grad(set_to_none=True)
            loss=sum(p.square().mean() for p in model.parameters() if p.requires_grad)
            loss=loss+sum(p.square().mean() for p in auxiliary.parameters())
            loss.backward();optim.step()
            if cfg['microbatch']==16 and not cfg['checkpointing']:
                raise torch.cuda.OutOfMemoryError('fixture peak')
        cfg=dict(device='cuda:0',autotune=True,microbatch=16,frame_budget=9600,source_batch=16,gpu_reserve_bytes=6*1024**3)
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()), \
             patch('w2v_v315.performance.probe_examples',return_value=[]), \
             patch('w2v_v315.performance.train_step',side_effect=simulated_step), \
             patch('torch.cuda.get_device_name',return_value='mock GPU'), \
             patch('torch.cuda.synchronize'),patch('torch.cuda.reset_peak_memory_stats'),patch('torch.cuda.empty_cache'), \
             patch('torch.cuda.mem_get_info',return_value=(30*1024**3,40*1024**3)), \
             patch('torch.cuda.memory_reserved',return_value=8*1024**3), \
             patch('torch.cuda.max_memory_reserved',return_value=12*1024**3), \
             patch('torch.cuda.max_memory_allocated',return_value=10*1024**3):
            result=select_execution(model,aux,optimizer,None,cfg,d)
            self.assertTrue(result['rollback_verified'])
            self.assertTrue(any(v['status']=='out_of_memory' for v in result['attempts']))
            self.assertEqual(optimizer.state_dict()['state'],{})
            assert_nested_equal(self,original,partial_state(model))
            assert_nested_equal(self,original_aux,aux.state_dict());assert_nested_equal(self,rng,capture_rng())

    def test_checkpoint_disk_failure_preserves_previous_transaction(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'last.pt';atomic_save(path,{'x':torch.ones(20)},0);before=digest(path)
            with patch('w2v_v313.state.shutil.disk_usage',return_value=type('Disk',(),{'free':0})()):
                with self.assertRaises(OSError):atomic_save(path,{'x':torch.zeros(20)},0)
            self.assertEqual(digest(path),before)

    def test_compaction_interruption_preserves_valid_old_then_new_exports(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            run,cfg=completed_fixture(d);protected=digest(cfg['starting_checkpoint'])
            real=atomic_json
            def interrupted(path,value):
                if Path(path).name=='completed.json':raise InterruptedError('manifest switch')
                return real(path,value)
            with patch('w2v_v315.cleanup.atomic_json',side_effect=interrupted):
                with self.assertRaises(InterruptedError):cleanup(run,apply=True)
            load_selected(run,'last');self.assertTrue((run/'last.pt').exists())
            cleanup(run,apply=True)
            self.assertFalse((run/'last.pt').exists())
            for kind in ('best_guarded','best_weighted','last'):load_selected(run,kind)
            self.assertEqual(protected,digest(cfg['starting_checkpoint']))

    def test_unowned_cache_and_linked_outputs_are_never_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            run,cfg=completed_fixture(d);cache=run/'rolling_pairs';cache.mkdir()
            atomic_json(cache/'owner.json',{'format':'unknown','identity':'other'})
            important=cache/'keep';important.write_text('keep')
            before=digest(run/'last.pt')
            with self.assertRaises(ValueError):cleanup(run,True,True)
            self.assertEqual(important.read_text(),'keep');self.assertEqual(digest(run/'last.pt'),before)
            real=Path.is_symlink
            target=(run/'inference.pt').resolve()
            with patch.object(Path,'is_symlink',lambda p:p in (run/'inference.pt',target) or real(p)):
                with self.assertRaises(ValueError):cleanup(run,True)
            self.assertEqual(digest(run/'last.pt'),before)

    def test_export_rejects_mutated_execution_plan(self):
        with tempfile.TemporaryDirectory() as d:
            run,cfg=completed_fixture(d)
            atomic_json(run/'execution_plan.json',{'changed':True})
            with self.assertRaisesRegex(ValueError,'execution plan differs'):load_selected(run)


if __name__=='__main__':unittest.main()
