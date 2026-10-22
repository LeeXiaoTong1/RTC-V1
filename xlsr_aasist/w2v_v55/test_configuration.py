import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import config, parent, train
from .common import RECIPE, atomic_json, digest, fingerprint, read_json, partial_state, schema_for
from w2v_v5.common import RECIPE as V51_RECIPE, schema_for as v51_schema
from w2v_v5.processing import policy
from w2v_v5.test_integration import inventory
from w2v_v31810.identity import build_registry, registry_config

LR_KEYS = ('lora_early_lr', 'lora_middle_lr', 'lora_lr', 'head_lr', 'fusion_lr',
           'layer_mix_lr', 'classifier_lr', 'evidence_lr', 'pool_lr')


def submitted_parent(root):
    run = root/'submitted'
    (run/'final').mkdir(parents=True)
    rows, cfg = inventory()
    registry = build_registry(rows, [], {})
    cfg.update(version='5', recipe=V51_RECIPE, v5_stage='final', epochs=1,
        lora_layers=16, retained_lora_layers=60, encoder_layers=60, robust_arm='baseline',
        source_registry=registry_config(registry), lr_floor=.5, source_epoch_offset=6,
        v5_tail_snapshots=4, v5_tail_start=.75, data_files={}, runtime_versions={},
        classification_ce_weights=dict(online=.25, clean_rtc=.25, noisy=.5),
        pcl_platform_weight=.045, **{k: (i+1)*1e-5 for i,k in enumerate(LR_KEYS)})
    lock = dict(input=cfg['v5_input'], weights=dict(name='anchor_tail_50', source='tail', parent_weight=.5))
    atomic_json(run/'recipe_lock.json', lock)
    cfg['v5_recipe_lock_sha256'] = digest(run/'recipe_lock.json')
    for name,value in {'train_rows.json':rows,'source_registry.json':registry}.items():
        atomic_json(run/'final'/name,value)
    cfg['saved_manifests'] = {n:digest(run/'final'/n) for n in ('train_rows.json','source_registry.json')}
    for name in ('config.json','final/config.json','final_config.json'):
        atomic_json(run/name,cfg)
    weights={'head.weight':torch.full((1,2),3.),'head.bias':torch.tensor([.5])}
    tag='final_anchor_tail_50'
    torch.save(dict(schema=v51_schema(cfg),identity=fingerprint(cfg),model=weights,
                    tag=tag,recipe=lock),run/'final.pt')
    atomic_json(run/'completed.json',dict(version='5',status='complete',selected=tag,
        checkpoint_sha256=digest(run/'final.pt'),final_config_sha256=digest(run/'final_config.json'),
        recipe_sha256=digest(run/'recipe_lock.json')))
    (run/'final'/'last.pt').write_bytes(b'not-the-submitted-checkpoint')
    return run,cfg,weights


class ConfigurationTests(unittest.TestCase):
    def test_exact_v51_final_no_new_version_settings_or_parent_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            run,old,expected=submitted_parent(Path(directory))
            old,origin,rows,dev=parent.metadata(run)
            self.assertEqual(dev,[])
            self.assertEqual(origin['checkpoint'],str(run/'final.pt'))
            _,payload,_,_=parent.load(run,origin)
            for k,v in expected.items():
                torch.testing.assert_close(payload['best_model'][k],v,rtol=0,atol=0)
            args=config.parser().parse_args(['--workers','0'])
            with patch.object(config,'runtime_versions',return_value={'cpu':'fixture'}):
                cfg=config.training_configuration(old,origin,args)
            for key in ('classification_ce_weights','v5_input','pcl_noise_weight','pcl_platform_weight',
                        'consistency','source_draws_per_stream','v5_context_seconds','v5_pcl_scope'):
                self.assertEqual(cfg[key],old[key],key)
            for key in LR_KEYS:
                self.assertAlmostEqual(cfg[key],old[key]*old['lr_floor'])
            self.assertEqual(cfg['lora_layers'],16)
            self.assertEqual(cfg['retained_lora_layers'],60)
            self.assertEqual(cfg['training_recipe_base'],'5.1')
            self.assertFalse(cfg['initialization']['old_optimizer_loaded'])
            self.assertEqual(schema_for(cfg),'rtc_v55_v51_hard_positive_v1')
            self.assertFalse(any(k.startswith(('adaptive_','repair_','contrastive_')) for k in cfg))
            for version in ('5.2','5.3','5.4','3.18.9'):
                with self.assertRaises(ValueError):
                    config.training_configuration(dict(old,version=version),origin,args)
            with self.assertRaises(ValueError):
                config.training_configuration(dict(old,adaptive_fusion_mode='dynamic'),origin,args)
            # Upload errors are not training failures; the valid final remains usable.
            atomic_json(run/'report_status.json',dict(status='failed'))
            parent.load(run,origin)
            (run/'final.pt').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                parent.load(run,origin)

    def test_metadata_rejects_newer_runs_and_final_last_path(self):
        with tempfile.TemporaryDirectory() as directory:
            run,cfg,_=submitted_parent(Path(directory))
            for version in ('5.2','5.3','5.4'):
                atomic_json(run/'config.json',dict(cfg,version=version))
                with self.assertRaises(ValueError):
                    parent.metadata(run)
            with self.assertRaises(ValueError):
                parent.metadata(run/'final')

    def test_resume_cannot_change_objective_or_parent(self):
        args=['--resume','run','--submit-best']
        config.validate(config.parser().parse_args(args),args)
        for bad in (['--resume','run','--triplet-weight','.5'],
                    ['--resume','run','--epochs','2'],['--microbatch','12'],
                    ['--guidance-margin','nan'],['--hp-ramp-updates','0'],
                    ['--upload-submission']):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                config.validate(config.parser().parse_args(bad),bad)

    def test_runtime_dependency_tree_excludes_newer_models_and_losses(self):
        files=config.code_inventory()
        self.assertTrue(any('w2v_v3191' in p and p.endswith('model.py') for p in files))
        self.assertFalse(any(v in Path(p).parts for v in ('w2v_v52','w2v_v53','w2v_v54') for p in files))


if __name__=='__main__':
    unittest.main()
