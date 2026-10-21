"""Parent provenance and repaired final export are validated independently."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import config, parent, delivery, workflow
from .common import RECIPE, schema_for, atomic_json, read_json, digest, fingerprint, partial_state
from .test_integration import fixture as detector_fixture
from w2v_v53.test_parent import fixture as old_fixture
from w2v_v53.common import RECIPE as V53_RECIPE, schema_for as v53_schema
from w2v_v53.test_lifecycle import TinyDetector


def parent_fixture(root):
    run,cfg,weights,rows=old_fixture(root)
    cfg.update(version='5.3',recipe=V53_RECIPE,training_recipe_base='5.1',contrastive_weight=0.,
        adaptive_fusion_mode='dynamic',adaptive_fusion_width=128,adaptive_fusion_seed=53001,
        adaptive_fusion_condition='frozen_frontend_masked_mean_std_v1',
        v53_parent_pcl_weights=dict(noise=.15,platform=.045))
    weights.update({'fusion.conditioner.'+k:v for k,v in TinyDetector().fusion.conditioner.state_dict().items()})
    for name in ('config.json','final/config.json','final_config.json'):atomic_json(run/name,cfg)
    state=torch.load(run/'final.pt',weights_only=True)
    state.update(schema=v53_schema(cfg),identity=fingerprint(cfg),model=weights)
    torch.save(state,run/'final.pt')
    done=read_json(run/'completed.json')
    done.update(version='5.3',checkpoint_sha256=digest(run/'final.pt'),final_config_sha256=digest(run/'final_config.json'))
    atomic_json(run/'completed.json',done)
    return run,cfg,weights,rows


def child_cfg(old,origin,arm='paired'):
    args=config.parser().parse_args(['--workers','0','--repair-arm',arm])
    config.validate(args)
    with patch.object(config,'runtime_versions',return_value={}):
        return config.training_configuration(old,origin,args)


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_verified_v53_final_is_only_parent_and_keeps_recipe_with_failed_upload(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            run,_,expected,_=parent_fixture(Path(directory))
            old,origin,rows,dev=parent.metadata(run)
            cfg=child_cfg(old,origin)
            actual=parent.load(run,origin)[1]['best_model']
            self.assertEqual(set(actual),set(expected))
            for name,value in expected.items():torch.testing.assert_close(actual[name],value,rtol=0,atol=0)
            self.assertEqual(dev,[])
            self.assertEqual(cfg['classification_ce_weights'],old['classification_ce_weights'])
            self.assertEqual(cfg['pcl_noise_weight'],0)
            self.assertEqual(cfg['pcl_platform_weight'],.045)
            self.assertEqual(cfg['pcl_variance_reference_weight'],.15)
            self.assertEqual(cfg['adaptive_fusion_seed'],old['adaptive_fusion_seed'])
            self.assertEqual(cfg['v5_context_seconds'],old['v5_context_seconds'])
            self.assertAlmostEqual(cfg['lora_lr'],old['lora_lr']*old['lr_floor'])
            self.assertEqual(cfg['repair_lr'],1e-4)
            self.assertEqual(cfg['lora_dropout'],0)
            self.assertEqual((cfg['lora_layers'],cfg['retained_lora_layers']),(16,60))
            baseline=child_cfg(old,origin,'baseline')
            self.assertEqual(baseline['pcl_noise_weight'],.15)
            (run/'completed.json').unlink()
            with self.assertRaises(FileNotFoundError):parent.metadata(run)

    def test_parent_tamper_and_incompatible_resume_settings_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            run,_,_,_=parent_fixture(Path(directory))
            _,origin,_,_=parent.metadata(run)
            with (run/'final.pt').open('ab') as target:target.write(b'changed')
            with self.assertRaises(ValueError):parent.load(run,origin)
        argv=['--resume','run','--repair-arm','ce_only']
        with self.assertRaises(ValueError):config.validate(config.parser().parse_args(argv),argv)

    def test_export_reloads_repair_branch_and_retry_keeps_identical_scores(self):
        from .model import parent_anchor
        from w2v_v5.weights import average_states,invariant_keys
        from w2v_v53.common import partial_state as old_state
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory)
            source,old,_,_=parent_fixture(root)
            _,origin,_,_=parent.metadata(source)
            cfg=child_cfg(old,origin)
            cfg['device']='cpu'
            tiny,units,model,original=detector_fixture()
            model.eval()
            with torch.no_grad():model.repair.out.bias.add_(torch.linspace(-1.,1.,128))
            trained=partial_state(model)
            final=average_states([parent_anchor(model,old_state(original)),trained],[.5,.5],constant_keys=invariant_keys(model,trained))
            run=root/'child'; run.mkdir()
            atomic_json(run/'config.json',cfg)
            lock=workflow.continuation_lock(run,cfg)
            cfg['v5_recipe_lock_sha256']=digest(run/'recipe_lock.json')
            atomic_json(run/'final_config.json',cfg)
            tag='final_anchor_tail_50'
            torch.save(dict(schema=schema_for(cfg),identity=fingerprint(cfg),model=final,tag=tag,recipe=lock),run/'final.pt')
            atomic_json(run/'completed.json',dict(version='5.4',status='complete',selected=tag,
                checkpoint_sha256=digest(run/'final.pt'),final_config_sha256=digest(run/'final_config.json'),
                recipe_sha256=digest(run/'recipe_lock.json')))
            protocol=root/'progress.txt'; protocol.write_text('one.wav\ntwo.wav\n')
            audio=root/'audio';audio.mkdir()
            for name in ('one.wav','two.wav'):(audio/name).write_bytes(b'fixture')
            waves=[u[0]['wave'] for u in units[:2]]
            captured=[]
            def inference(loaded,rows,options):
                self.assertGreater(float(loaded.repair.out.bias.norm()),0.)
                z=loaded(waves).detach().cpu().numpy();captured.append(z)
                return z,rows
            out=root/'export'
            delivery.export(run,out,protocol=protocol,audio_root=audio,verify=False,
                model_factory=lambda c:copy.deepcopy(model),inference=inference)
            with patch.object(delivery,'apply_partial',side_effect=AssertionError('unnecessary second inference')):
                delivery.export(run,out,protocol=protocol,audio_root=audio,verify=False)
            probs=torch.tensor(captured[0]).softmax(-1)[:,0]
            lines=(out/'scores.txt').read_text().splitlines()
            for index,line in enumerate(lines):self.assertAlmostEqual(float(line.split()[1]),float(probs[index]),places=9)
            self.assertFalse(delivery.check_archive(out)[1]['independent_prediction_ensemble'])

    def test_no_evaluation_input_read_until_final_is_complete(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with patch.object(delivery,'read_protocol',side_effect=AssertionError('early test access')) as read:
                with self.assertRaises(FileNotFoundError):delivery.export(Path(directory)/'run',Path(directory)/'out',protocol='missing',audio_root='missing')
                read.assert_not_called()

    def test_final_fit_freezes_the_repaired_parameter_average_without_dev_selection(self):
        from w2v_v53.common import partial_state as old_state
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);source,old,_,_=parent_fixture(root)
            _,origin,_,_=parent.metadata(source)
            cfg=child_cfg(old,origin)
            run=root/'new';(run/'final').mkdir(parents=True)
            workflow.continuation_lock(run,cfg)
            final_cfg=dict(cfg,v5_recipe_lock_sha256=digest(run/'recipe_lock.json'))
            atomic_json(run/'final'/'config.json',final_cfg)
            _,units,model,original=detector_fixture()
            from .test_audit import fixed_pair_file
            fixed_pair_file(model,final_cfg,units,run/'final'/'fixed_pairs.pt')
            model.repair.out.bias.data.fill_(.2)
            trained=partial_state(model)
            state=dict(status='complete',model=trained,tail_model=trained)
            with patch.object(workflow,'read_stage',return_value=(final_cfg,[])), patch.object(workflow,'run_stage',return_value=(model,state)), patch.object(workflow,'load_parent',return_value=(old,dict(best_model=old_state(original)),None,None)):
                self.assertTrue(workflow.final_fit(run,cfg))
            _,weights,_=delivery.selected(run)
            torch.testing.assert_close(weights['repair.out.bias'],torch.full((128,),.1),rtol=0,atol=0)
            self.assertTrue(read_json(run/'export_repair_audit.json')['frame_scale_equal'])
            audit=read_json(run/'export_repair_audit.json')
            self.assertTrue(audit['strict_reloaded'])
            self.assertEqual(audit['functional_probe_pairs'],1)
            self.assertLess(audit['functional_changes'][0]['exported_correction_ratio'],
                            audit['functional_changes'][0]['trained_correction_ratio'])


if __name__=='__main__':unittest.main()
