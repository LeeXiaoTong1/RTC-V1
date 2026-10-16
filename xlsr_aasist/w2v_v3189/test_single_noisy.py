"""Single-view, source-weighting and exact best-transfer regression tests."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .config import parser,validate,validate_locked_cli,end_learning_rates,configuration
from .data import Waves
from .objective import micro_losses
from .sampling import Plan,validate_units
from .common import RECIPE,atomic_json,digest,fingerprint,partial_state,read_json
from .parent import metadata,initialize
from .test_training import fixture,tiny_model,make_run
from .train import run_experiment
from .evaluate import selected
from .monitor import measure,Dashboard
from w2v_v3188.common import RECIPE as OLD_RECIPE,schema_for as old_schema
from w2v_v3186.test_sampling import rows_for
from w2v_v3183r2.common import GROUPS
from w2v_v318.test_core import NativeBoundary
from w2v_v3186.test_support import fixture_phones


def parent_fixture(root):
    cfg,train,dev=fixture(root)
    parent=root/'parent_v3188';parent.mkdir()
    for name,rows in (('train_rows.json',train),('dev_rows.json',dev)):
        atomic_json(parent/name,rows)
    partition=read_json(Path(cfg['data_source']['run'])/'dev_partition.json')
    atomic_json(parent/'dev_partition.json',partition)
    cfg.update(version='3.18.8',recipe=OLD_RECIPE,epochs=20,encoder_layers=cfg['lora_layers'],code_fingerprints={},
        saved_manifests={name:digest(parent/name) for name in ('train_rows.json','dev_rows.json')})
    atomic_json(parent/'config.json',cfg)
    model=tiny_model(cfg)
    model.pcl_scale.observe(torch.stack((torch.zeros(128),torch.ones(128))),'first')
    model.pcl_scale.observe(torch.stack((torch.zeros(128),torch.ones(128)*2)),'second')
    model.pcl_scale.freeze()
    best=partial_state(model);last=copy.deepcopy(best)
    last['head.aasist.out_layer.bias'].add_(10.)
    z=torch.randn(len(dev),2)
    indices=partition['select'];metric=measure([dev[i] for i in indices],z.numpy()[indices])
    state=dict(schema=old_schema(cfg),identity=fingerprint(cfg),epoch=4,next_step=0,
        best_tag='epoch_3_step_3',last_tag='epoch_4_step_3',best_model=best,model=last,
        best_logits=z,last_logits=z+10,best_metrics=metric,
        history=[dict(epoch=3,tag='epoch_3_step_3',phase='joint')])
    torch.save(state,parent/'last.pt')
    old,origin,train,dev=metadata(parent)
    new=dict(cfg,version='3.18.9',recipe=RECIPE,epochs=1,warm_epochs=0,parent_init=origin,
        source_epoch_offset=3,initialization={'mode':'v3188_best'},
        data_source={'run':str(parent)},pcl_ramp_updates=1,lr_warmup_updates=1)
    return parent,new,train,dev,partition,best


class SingleNoisyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_defaults_and_locked_resume(self):
        args=parser().parse_args([]);validate(args)
        self.assertEqual((args.epochs,args.warm_epochs,args.microbatch,args.frame_budget,args.window_batch),(6,0,12,9000,16))
        for options in (['--microbatch','8'],['--warm-epochs','1'],['--epochs','0']):
            with self.assertRaises(ValueError):validate(parser().parse_args(options))
        args=parser().parse_args(['--resume','run'])
        with self.assertRaisesRegex(ValueError,'Resume uses saved'):
            validate_locked_cli(args,['--resume','run','--frame-budget','6000'])

    def test_parent_epoch_zero_baseline_and_cumulative_source_offset(self):
        with tempfile.TemporaryDirectory() as folder:
            parent,_,train,dev,_,_=parent_fixture(Path(folder))
            old=read_json(parent/'config.json');old.update(source_epoch_offset=5)
            atomic_json(parent/'config.json',old)
            state=torch.load(parent/'last.pt',weights_only=True)
            state.update(identity=fingerprint(old),best_tag='parent_v3187_epoch_5_step_1403')
            state['history']=[dict(epoch=0,tag=state['best_tag'],phase='parent_baseline')]
            torch.save(state,parent/'last.pt')
            with patch('w2v_v3189.config.torch.cuda.is_available',return_value=True), \
                 patch('w2v_v3189.config.torch.cuda.is_bf16_supported',return_value=True), \
                 patch('w2v_v3189.config.runtime_versions',return_value={'fixture':'cpu'}), \
                 patch('w2v_v3189.config.verify_inputs'):
                cfg,_,_=configuration(parser().parse_args(['--parent-run',str(parent)]))
            self.assertEqual(cfg['source_epoch_offset'],5)
            self.assertEqual(cfg['parent_init']['epoch'],0)
            self.assertEqual(cfg['rtc_activity_policy'],'mixed')
            for key,value in end_learning_rates(old,0,3).items():self.assertEqual(value,old[key])
            state['best_tag']='epoch_2_step_3';state['history']=[dict(epoch=2,tag=state['best_tag'],phase='joint')]
            torch.save(state,parent/'last.pt')
            with patch('w2v_v3189.config.torch.cuda.is_available',return_value=True), \
                 patch('w2v_v3189.config.torch.cuda.is_bf16_supported',return_value=True), \
                 patch('w2v_v3189.config.runtime_versions',return_value={'fixture':'cpu'}), \
                 patch('w2v_v3189.config.verify_inputs'):
                cfg,_,_=configuration(parser().parse_args(['--parent-run',str(parent),'--rtc-activity','off']))
            self.assertEqual(cfg['source_epoch_offset'],7)
            self.assertEqual(cfg['rtc_activity_policy'],'off')

    def test_same_half_stream_and_quarter_group_mass(self):
        quotas=(7,6,5,6);units=[]
        for kind in ('online','noisy'):
            for group,quota in zip(GROUPS,quotas):
                for i in range(quota):
                    units.append([dict(role=kind,language=group[0],label=group[1],split='train',
                        source_id=f'{kind}-{group}-{i}')])
        validate_units(units,quotas)
        logits=torch.zeros(len(units),2,requires_grad=True)
        parts=micro_losses(logits,units,quotas,smoothing=.02)
        for role in ('online','noisy'):
            mass=sum(v for (v,_),u in zip(parts,units) if u[0]['role']==role)
            self.assertAlmostEqual(float(mass.detach()),np.log(2)*.5,places=6)
        for group in GROUPS:
            mass=sum(v for (v,_),u in zip(parts,units) if (u[0]['language'],u[0]['label'])==group)
            self.assertAlmostEqual(float(mass.detach()),np.log(2)*.25,places=6)
        with self.assertRaisesRegex(ValueError,'raw Offline'):
            validate_units([[dict(units[0][0],role='offline')]]+units[1:],quotas)

    def test_new_run_configuration_preserves_assets_and_architecture(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            parent,_,train,dev,_,_=parent_fixture(Path(folder))
            old=read_json(parent/'config.json')
            with patch('w2v_v3189.config.torch.cuda.is_available',return_value=True), \
                 patch('w2v_v3189.config.torch.cuda.is_bf16_supported',return_value=True), \
                 patch('w2v_v3189.config.runtime_versions',return_value={'fixture':'cpu'}), \
                 patch('w2v_v3189.config.verify_inputs') as verify:
                cfg,actual_train,actual_dev=configuration(parser().parse_args(['--parent-run',str(parent)]))
            self.assertEqual((cfg['epochs'],cfg['warm_epochs'],cfg['source_epoch_offset']),(6,0,3))
            self.assertEqual((cfg['microbatch'],cfg['frame_budget'],cfg['window_batch']),(12,9000,16))
            self.assertEqual(cfg['noisy_views_per_source'],1)
            self.assertTrue(cfg['window_resource_audit']['valid'])
            self.assertEqual(cfg['checkpoint_layers'],old['encoder_layers'])
            self.assertFalse(cfg['initialization']['old_optimizer_loaded'])
            for key in ('lora_layers','window','hop','pooling','consistency','source_draws_per_stream',
                        'view_factors','refresh_base_variants','scene_probability','noise_records'):
                self.assertEqual(cfg[key],old[key])
            for key,rate in end_learning_rates(old,3,3).items():self.assertEqual(cfg[key],rate)
            self.assertEqual((actual_train,actual_dev),(train,dev))
            verify.assert_called_once_with(cfg)
            self.assertNotIn('saved_manifests',cfg)

    def test_exact_default_views_and_fresh_noise_lanes(self):
        plan=Plan(rows_for((21,19,35,23)),cfg={'warm_epochs':0,'source_epoch_offset':5})
        coverage=plan.coverage(0)
        self.assertEqual((coverage['steps'],coverage['online_views'],coverage['noisy_views']),(1403,33668,33668))
        self.assertEqual(coverage['last_stream_sources'],20)
        tickets=[t for batch in plan.batches(0) for t in batch if t['kind']=='offline']
        self.assertEqual({t['pcl_noisy_index'] for t in tickets},{0})
        self.assertEqual({t['noise_recipe_view'] for t in tickets},{0,1})
        self.assertEqual(len({t['occurrence'] for t in tickets}),33668)

    def test_single_synthesis_not_generate_two_then_drop(self):
        wave=np.random.default_rng(1).normal(0,.1,16000).astype(np.float32)
        row=dict(id='off',source_id='off',group_id='g',language='en',label=1,split='train')
        cfg=dict(seed=1,warm_epochs=0,rtc_stress_probability=0.,rtc_stress_ramp_epochs=1.,
                 scene_probability=1.,rtc_policy='three-profile')
        ds=Waves([row],cfg);ds.original=lambda _:wave.copy();ds.bank=object();ds.engines=object()
        ticket=dict(index=0,kind='offline',base_variant=0,base_family='original',base_generation=0,
            base_dynamic=False,base_view_id='view',occurrence='visit',epoch=0,pcl_enabled=False,
            noise_recipe_view=1)
        with patch('w2v_v3189.data.recipe',return_value={}) as recipe, \
             patch('w2v_v3189.data.generate',return_value=(wave.copy(),{})) as generate:
            result=ds[ticket]
        self.assertEqual(len(result),1);self.assertEqual(result[0]['role'],'noisy')
        self.assertEqual(result[0]['noise_recipe_view'],1);generate.assert_called_once()
        self.assertEqual(recipe.call_args.args[3],1)

    def test_parent_best_not_last_and_no_classifier_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            parent,cfg,train,dev,partition,best=parent_fixture(Path(folder))
            model=tiny_model(cfg)
            baseline,values,logits,audit=initialize(model,cfg,dev,partition)
            self.assertEqual(baseline['epoch'],0);self.assertEqual(audit['parent_epoch'],3)
            self.assertEqual(audit['classifier_input_dim'],384)
            self.assertTrue(audit['pcl_scale_frozen']);self.assertFalse(audit['optimizer_loaded'])
            for key,value in values.items():torch.testing.assert_close(value,best[key],rtol=0,atol=0)
            rates=end_learning_rates(read_json(parent/'config.json'),3,3)
            self.assertTrue(all(0<rate<=read_json(parent/'config.json')[key] for key,rate in rates.items()))
            with (parent/'last.pt').open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Parent changed'):initialize(model,cfg,dev,partition)

    def test_parent_baseline_survives_training_and_exports_without_extra_baseline_inference(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()), \
             patch('w2v_v3189.data.Engines',NativeBoundary), \
             patch('w2v_v3189.data.Waves._phones',fixture_phones):
            root=Path(folder);parent,cfg,train,dev,partition,best=parent_fixture(root)
            run=make_run(root,'new',cfg,train,dev)
            with patch('w2v_v3189.train.infer',side_effect=AssertionError('No parent Dev rerun')):
                run_experiment(cfg,run,train,dev,tiny_model,baseline_only=True)
            before=torch.load(run/'last.pt',weights_only=True)
            self.assertEqual(before['epoch'],0);self.assertEqual(before['optimizer']['state'],{})
            with patch('w2v_v3189.train.promote',return_value=False):
                done=run_experiment(cfg,run,train,dev,tiny_model)
            self.assertTrue(done['baseline_fallback']);self.assertEqual(done['best_origin'],'v3188_parent')
            self.assertEqual(done['completed_epochs'],1)
            settings,values,meta=selected(run,'best')
            self.assertTrue(meta['baseline_fallback'])
            for key,value in values.items():torch.testing.assert_close(value,best[key],rtol=0,atol=0)
            history=read_json(run/'training_history.json')
            self.assertEqual([e['epoch'] for e in history],[0,1])
            self.assertEqual(history[1]['training']['views'],42)
            self.assertTrue((run/'diagnostics'/'curves.html').is_file())

    def test_namespaced_cuda_error_reaches_terminal(self):
        state=Dashboard(None)
        state.line('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 88 MiB')
        state.exit_code=1
        self.assertIn('CUDA out of memory',state.progress())
        state.line('[Train] V3.18.9 epoch 2/6 phase=joint updates=1403')
        self.assertEqual((state.epoch,state.epochs,state.total),(2,6,1403))

    def test_oom_records_context_without_overwriting_safe_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()), \
             patch('w2v_v3189.data.Engines',NativeBoundary), \
             patch('w2v_v3189.data.Waves._phones',fixture_phones):
            root=Path(folder);_,cfg,train,dev,_,_=parent_fixture(root)
            run=make_run(root,'oom',cfg,train,dev)
            run_experiment(cfg,run,train,dev,tiny_model,baseline_only=True)
            checksum=digest(run/'last.pt')
            with patch('w2v_v3189.train.train_step',side_effect=torch.OutOfMemoryError('fixture OOM')):
                with self.assertRaises(torch.OutOfMemoryError):
                    run_experiment(cfg,run,train,dev,tiny_model)
            self.assertEqual(digest(run/'last.pt'),checksum)
            failure=read_json(run/'oom_context.json')
            self.assertEqual((failure['epoch'],failure['step'],failure['committed_epochs']),(1,1,0))
            self.assertFalse((run/'completed.json').exists())


if __name__=='__main__':unittest.main()
