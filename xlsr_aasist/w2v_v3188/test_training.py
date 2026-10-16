"""Checkpoint/resume and inference-order regression tests on real tiny models."""
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayout, NativeBoundary, make_fixture
from w2v_v3184.test_support import scene_fixture
from w2v_v3184.records import dev_partition
from w2v_v3186.test_support import fixture_phones
from .test_model import tiny_cfg as model_cfg
from .common import RECIPE, atomic_json, read_json,partial_state,digest,fingerprint
from .model import Detector
from .inference import infer
from .sampling import Plan
from .train import run_experiment
from .step import train_step


def tiny_model(cfg):
    return Detector(TinyFrontend(), TinyEncoder(), cfg, TinyLayout)


def fixture(root):
    base, train, dev = make_fixture(root/'audio')
    cfg=model_cfg()
    cfg.update({k:base[k] for k in ('dev_pairs','official_dev_protocol','official_dev_root','noise_records')})
    cfg.update(scene_fixture(root/'scenes'))
    cfg.update(version='3.18.8',recipe=RECIPE,omni_arch='3b',epochs=2,warm_epochs=1,
        stream_sources=8,group_quotas=[2,2,2,2],source_draws_per_stream=[6,5,5,5],
        refresh_base_variants=True,view_factors=[5,5,1,3],base_original_fraction=.2,
        consistency='dual',pcl_noise_weight=.15,pcl_platform_weight=.15,
        platform_sources_per_group=1,pcl_ramp_updates=2,pcl_min_frames=1,
        en_real_confidence_weight=.01,en_real_confidence_margin=3.5,
        microbatch=8,frame_budget=2000,checkpoint_layers=1,
        gradient_audit_interval=100,diagnostic_interval=5,
        initialization=dict(mode='pretrained',old_detector_loaded=False),
        data_source=dict(run=str(root/'data_source')))
    off={row['group_id']:row for row in train if row['condition']=='offline'}
    for row in train:
        parent=off[row['group_id'].removeprefix('online:')]
        identity=hashlib.sha256(parent['id'].encode()).hexdigest()
        row.update(source_id=parent['id'],group_id=identity,source_sha256=identity)
    source=root/'data_source';source.mkdir()
    atomic_json(source/'config.json',cfg)
    atomic_json(source/'train_rows.json',train);atomic_json(source/'dev_rows.json',dev)
    atomic_json(source/'dev_partition.json',dev_partition(dev,cfg['dev_pairs'],cfg['seed'],cfg['calibration_fraction']))
    return cfg,train,dev


def make_run(root,name,cfg,train,dev):
    run=root/name;run.mkdir()
    atomic_json(run/'config.json',cfg)
    atomic_json(run/'train_rows.json',train);atomic_json(run/'dev_rows.json',dev)
    return run


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_mid_joint_epoch_pause_keeps_best_and_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('w2v_v3188.data.Engines',NativeBoundary), \
             patch('w2v_v3188.data.Waves._phones',fixture_phones), \
             patch('w2v_v3188.train.render'),redirect_stdout(io.StringIO()):
            root=Path(directory);cfg,train,dev=fixture(root)
            full=make_run(root,'full',cfg,train,dev)
            resumed=make_run(root,'resumed',cfg,train,dev)
            plan=Plan(train,cfg['stream_sources'],cfg['seed'],cfg['group_quotas'],cfg['view_factors'],True,cfg=cfg)
            self.assertEqual([sum(q) for q in plan.schedule],[8,8,5])
            run_experiment(cfg,full,train,dev,tiny_model)
            def pause_after_first_joint_update(model,optimizer,*args,**kwargs):
                result=train_step(model,optimizer,*args,**kwargs)
                if optimizer._v3188_update==plan.steps+1:
                    atomic_json(resumed/'stop_requested.json',dict(reason='test optimizer-boundary pause'))
                return result
            with patch('w2v_v3188.train.train_step',pause_after_first_joint_update):
                paused=run_experiment(cfg,resumed,train,dev,tiny_model)
            self.assertEqual(paused['status'],'paused')
            partial=torch.load(resumed/'last.pt',weights_only=True)
            self.assertEqual((partial['epoch'],partial['next_step']),(1,1))
            self.assertEqual(partial['best_tag'],'epoch_1_step_3')
            self.assertTrue(any(not torch.equal(partial['model'][k],v) for k,v in partial['best_model'].items()))
            self.assertEqual(partial['epoch_accumulator']['totals']['updates'],1)
            self.assertTrue((resumed/'calibration_best.json').is_file())
            with (resumed/'steps.jsonl').open('a') as stream:stream.write('{"update":999}\n{"broken":')
            run_experiment(cfg,resumed,train,dev,tiny_model)
            a=torch.load(full/'last.pt',weights_only=True);b=torch.load(resumed/'last.pt',weights_only=True)
            for field in ('model','best_model'):
                for name,value in a[field].items():
                    torch.testing.assert_close(value,b[field][name],rtol=0,atol=0)
            for key,values in a['optimizer']['state'].items():
                for name,value in values.items():
                    if isinstance(value,torch.Tensor):
                        torch.testing.assert_close(value,b['optimizer']['state'][key][name],rtol=0,atol=0)
            self.assertEqual(a['best_tag'],b['best_tag'])
            self.assertEqual(a['cursor'],6)
            self.assertEqual([e['phase'] for e in b['history']],['head_warmup','joint'])
            self.assertEqual([json.loads(line)['update'] for line in (resumed/'steps.jsonl').read_text().splitlines()],list(range(1,7)))
            for before,after in zip(a['history'],b['history']):
                self.assertEqual(before['training']['views'],42)
                self.assertEqual(before['training']['loss'],after['training']['loss'])
                self.assertEqual(before['training']['en_real_confidence'],after['training']['en_real_confidence'])
            self.assertFalse((root/'data_source'/'last.pt').exists())

    def test_inference_restores_row_order_after_length_packing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,_,dev=fixture(root)
            # Descending packing must undo this deliberately mixed order.
            rows=[dev[i] for i in (8,0,4,17,9,2,14,7,1)]
            torch.manual_seed(81);model=tiny_model(cfg)
            packed,observed=infer(model,rows,dict(cfg,eval_batch=9,microbatch=9))
            single,_=infer(model,rows,dict(cfg,eval_batch=1,microbatch=4))
            self.assertEqual([r['id'] for r in observed],[r['id'] for r in rows])
            np.testing.assert_allclose(packed,single,rtol=1e-5,atol=1e-6)

    def test_preflight_saves_fresh_state_without_dev_inference(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('w2v_v3188.train.infer',side_effect=AssertionError('Fresh preflight must not infer Dev')), \
             redirect_stdout(io.StringIO()):
            root=Path(directory);cfg,train,dev=fixture(root)
            run=make_run(root,'preflight',cfg,train,dev)
            run_experiment(cfg,run,train,dev,tiny_model,baseline_only=True)
            state=torch.load(run/'last.pt',weights_only=True)
            self.assertIsNone(state['best_model'])
            self.assertIsNone(state['best_tag'])
            self.assertFalse((run/'completed.json').exists())
            self.assertEqual(read_json(run/'initialization_audit.json')['detector_checkpoint_loaded'],False)

    def test_stop_requested_during_model_startup_is_not_discarded(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('w2v_v3188.data.Engines',NativeBoundary), \
             patch('w2v_v3188.data.Waves._phones',fixture_phones), \
             redirect_stdout(io.StringIO()):
            root=Path(directory);cfg,train,dev=fixture(root)
            run=make_run(root,'startup_stop',cfg,train,dev)
            def factory(settings):
                model=tiny_model(settings)
                atomic_json(run/'stop_requested.json',dict(reason='stop while initial model loads'))
                return model
            outcome=run_experiment(cfg,run,train,dev,factory)
            self.assertEqual(outcome['status'],'paused')
            saved=torch.load(run/'last.pt',weights_only=True)
            self.assertEqual((saved['epoch'],saved['next_step']),(0,1))
            self.assertIsNone(saved['best_tag'])
            self.assertFalse((run/'calibration_best.json').exists())

    def test_stop_during_dev_commits_epoch_without_starting_next_update(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('w2v_v3188.data.Engines',NativeBoundary), \
             patch('w2v_v3188.data.Waves._phones',fixture_phones), \
             redirect_stdout(io.StringIO()):
            root=Path(directory);cfg,train,dev=fixture(root)
            run=make_run(root,'eval_stop',cfg,train,dev)
            def request_during_inference(*args,**kwargs):
                atomic_json(run/'stop_requested.json',dict(reason='stop during Dev'))
                return infer(*args,**kwargs)
            with patch('w2v_v3188.train.infer',request_during_inference):
                outcome=run_experiment(cfg,run,train,dev,tiny_model)
            saved=torch.load(run/'last.pt',weights_only=True)
            self.assertEqual((outcome['status'],saved['epoch'],saved['next_step']),('paused',1,0))
            self.assertEqual(saved['cursor'],3)
            self.assertIsNotNone(saved['best_tag'])
            self.assertTrue((run/'warmup.pt').is_file())
            self.assertTrue(saved['model']['pcl_scale.frozen'].item())

    def test_ce_control_inherits_exact_warmup_weights_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('w2v_v3188.data.Engines',NativeBoundary), \
             patch('w2v_v3188.data.Waves._phones',fixture_phones), \
             patch('w2v_v3188.train.render'),redirect_stdout(io.StringIO()):
            root=Path(directory);cfg,train,dev=fixture(root)
            parent=make_run(root,'parent',cfg,train,dev)
            def stop_after_dev(*args,**kwargs):
                atomic_json(parent/'stop_requested.json',{})
                return infer(*args,**kwargs)
            with patch('w2v_v3188.train.infer',stop_after_dev):
                run_experiment(cfg,parent,train,dev,tiny_model)
            warm=torch.load(parent/'warmup.pt',weights_only=True)
            settings=dict(cfg,consistency='none',pcl_noise_weight=0.,pcl_platform_weight=0.,pcl_variance_weight=0.,
                warm_start=dict(run=str(parent),checkpoint=str(parent/'warmup.pt'),sha256=digest(parent/'warmup.pt'),parent_identity=fingerprint(cfg)))
            branch=make_run(root,'control',settings,train,dev);checked=[]
            def verify_then_step(model,optimizer,units,*args,**kwargs):
                if not checked:
                    for name,value in partial_state(model).items():
                        torch.testing.assert_close(value,warm['model'][name],rtol=0,atol=0)
                    for key,values in warm['optimizer']['state'].items():
                        for name,value in values.items():
                            if isinstance(value,torch.Tensor):
                                torch.testing.assert_close(value,optimizer.state_dict()['state'][key][name],rtol=0,atol=0)
                    torch.testing.assert_close(torch.get_rng_state(),warm['rng']['torch'],rtol=0,atol=0)
                    self.assertTrue(model.pcl_scale.is_frozen)
                    checked.append(True)
                self.assertTrue(all('pcl_refs' not in u[0] for u in units))
                result=train_step(model,optimizer,units,*args,**kwargs)
                self.assertEqual(result['auxiliary_reference_views'],0)
                return result
            with patch('w2v_v3188.train.train_step',verify_then_step):
                run_experiment(settings,branch,train,dev,tiny_model)
            self.assertEqual(checked,[True])
            final=torch.load(branch/'last.pt',weights_only=True)
            self.assertEqual(final['epoch'],2)
            self.assertEqual(final['history'][0],warm['history'][0])

    def test_failed_resume_loading_retains_valid_paused_best_acknowledgement(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root=Path(directory);cfg,train,dev=fixture(root)
            run=make_run(root,'failed_resume',cfg,train,dev)
            paused=dict(status='paused',best_tag='epoch_1',checkpoint_sha256='unchanged')
            atomic_json(run/'paused.json',paused)
            def failed_factory(_):
                raise RuntimeError('simulated unavailable model asset')
            with self.assertRaisesRegex(RuntimeError,'unavailable model'):
                run_experiment(cfg,run,train,dev,failed_factory)
            self.assertEqual(read_json(run/'paused.json'),paused)


if __name__=='__main__':unittest.main()
