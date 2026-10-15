"""Fresh-data provenance and command-line recipe contracts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .common import atomic_json,RECIPE
from .config import parser,validate,validate_locked_cli,training_configuration,code_inventory
from .source import metadata,resolve_data_run
from .sampling import quota_schedule


class ConfigTests(unittest.TestCase):
    def test_default_source_budgets_and_official_initialization(self):
        args=parser().parse_args([]);validate(args)
        self.assertEqual((args.epochs,args.warm_epochs,args.microbatch),(20,1,16))
        self.assertTrue(args.refresh_base_variants)
        self.assertEqual(args.checkpoint_layers,60)
        self.assertTrue(args.temporal_stats)
        self.assertEqual(args.en_real_confidence_weight,0.)
        self.assertAlmostEqual(args.pcl_platform_weight/args.pcl_noise_weight,.3)
        quotas=quota_schedule(tuple(args.source_draws_per_stream),24)
        self.assertEqual((len(quotas),sum(quotas[-1])),(1403,20))
        old=dict(encoder_layers=60,pooling='asp',omni_arch='3b',window=128,hop=64,window_batch=8,
                 initialization={'files':{'old/last.pt':'hash'}},data_files={'old/last.pt':'hash'},
                 code_fingerprints={'unused.py':'hash'},saved_manifests={'x':'y'})
        source=dict(run='data/run',files={'data/run/train_rows.json':'hash'})
        with patch('w2v_v3187.config.runtime_versions',return_value={}):
            cfg=training_configuration(old,args,source)
        self.assertEqual(cfg['recipe'],RECIPE)
        self.assertEqual(cfg['initialization']['mode'],'pretrained')
        self.assertFalse(cfg['initialization']['old_detector_loaded'])
        self.assertEqual(cfg['initialization']['files'],{})
        self.assertEqual(cfg['data_files'],source['files'])
        self.assertNotIn('saved_manifests',cfg)
        self.assertEqual(cfg['window_batch'],32)
        self.assertEqual((cfg['lora_layers'],cfg['lora_rank'],cfg['lora_alpha']),(60,16,32))
        self.assertEqual(cfg['normalization'],'group_layer')
        self.assertLess(cfg['lora_early_lr'],cfg['lora_middle_lr'])
        self.assertLess(cfg['lora_middle_lr'],cfg['lora_lr'])
        self.assertEqual(old['window_batch'],8)

    def test_cli_rejects_short_budget_or_resume_recipe_changes(self):
        for flags in (['--epochs','6'],['--warm-epochs','20'],['--microbatch','3'],
                      ['--warm-epochs','0'],['--checkpoint-layers','61'],['--pcl-variance-target','nan'],
                      ['--en-real-confidence-weight','nan'],['--source-draws-per-stream','1','8000','8000','8000']):
            with self.subTest(flags=flags),self.assertRaises(ValueError):
                validate(parser().parse_args(flags))
        args=parser().parse_args(['--resume','run'])
        validate_locked_cli(args,['--resume','run'])
        with self.assertRaisesRegex(ValueError,'Resume uses saved'):
            validate_locked_cli(args,['--resume','run','--microbatch','24'])
        validate(parser().parse_args(['--consistency','none','--warm-epochs','0']))
        args=parser().parse_args(['--warm-start-run','run','--consistency','none'])
        validate_locked_cli(args,['--warm-start-run','run','--consistency','none'])
        with self.assertRaisesRegex(ValueError,'Warm-start control'):
            validate_locked_cli(args,['--warm-start-run','run','--lora-lr','0.1'])

    def test_manifest_source_needs_no_detector_checkpoint_or_completed_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'exp'/'w2v_v3184_aug5_test';run.mkdir(parents=True)
            atomic_json(run/'config.json',dict(version='3.18.4',omni_arch='3b',pooling='asp'))
            for name in ('train_rows.json','dev_rows.json','dev_partition.json'):
                atomic_json(run/name,[])
            self.assertEqual(resolve_data_run(root=root),run.resolve())
            old,source=metadata(run)
            self.assertFalse(source['checkpoint_loaded'])
            self.assertEqual(len(source['files']),4)
            self.assertFalse((run/'last.pt').exists())

    def test_code_inventory_follows_relative_and_external_repository_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'w2v_v3187').mkdir();(root/'shared').mkdir()
            (root/'w2v_v3187'/'module.py').write_text('from . import local\nfrom shared.ops import go\n')
            (root/'w2v_v3187'/'local.py').write_text('import math\n')
            (root/'shared'/'ops.py').write_text('def go(): pass\n')
            (root/'w2v_v3187'/'test_unused.py').write_text('raise RuntimeError\n')
            names={Path(p).name for p in code_inventory(root)}
            self.assertEqual(names,{'module.py','local.py','ops.py'})
