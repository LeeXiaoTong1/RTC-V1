"""Fresh pretraining identity, bounded deployment configuration, and protected references."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from . import config, pretrained
from w2v_aasist.runtime import sha256


class ConfigurationTests(unittest.TestCase):
    def source(self, root):
        baseline = root/'original.pt'; baseline.write_bytes(b'original')
        reference = root/'best_model.pt'; reference.write_bytes(b'reference')
        noise = root/'noise'; noise.mkdir()
        (noise/'train.jsonl').write_text('{}\n')
        (noise/'dev.jsonl').write_text('{}\n')
        cfg = dict(baseline=str(baseline), baseline_sha256=sha256(baseline),
            ssl_path=str(root/'ssl'), train_noise_manifest=str(noise/'train.jsonl'),
            train_data_path=str(root/'train'), dev_data_path=str(root/'dev'),
            dev_noisy_cache=str(root/'old_seen'), dev_heldout_cache=str(root/'old_heldout'),
            train_caches=[str(root/'old_train')], arms=['control','candidate'],
            source_data_fingerprints={'legacy': 'hash'})
        return dict(config=cfg, checkpoint=str(reference), checkpoint_sha256=sha256(reference),
                    checkpoint_tag='baseline', provenance={'checkpoint_tag':'baseline'})

    def build(self, root, extra=()):
        source = self.source(root)
        args = config.parser().parse_args(['--cache-root', str(root/'new_cache'), *extra])
        with patch.object(config, 'resolve_source', return_value=source), \
             patch.object(config, 'BASELINE_SHA256', source['config']['baseline_sha256']):
            return config.configuration(args)

    def test_recipe_reference_is_not_training_initialization(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self.build(Path(td))
            self.assertEqual(cfg['version'], '3.5')
            self.assertEqual((cfg['head_epochs'],cfg['joint_epochs']), (1,5))
            self.assertEqual(cfg['trainable_layers'],24)
            self.assertEqual((cfg['offline_weight'],cfg['online_weight'],cfg['noisy_weight']),(.1,.3,.6))
            self.assertEqual((cfg['cka_weight'],cfg['pair_weight'],cfg['short_loss_weight']), (0.,0.,0.))
            self.assertEqual(cfg['evals_per_epoch'],1)
            self.assertEqual(cfg['reference_checkpoint'],cfg['warm_checkpoint'])
            self.assertIsNone(cfg['pretrained_path'])
            self.assertNotIn('arms',cfg)
            self.assertNotIn('source_data_fingerprints',cfg)

    def test_invalid_lr_or_budgets_rejected_before_source(self):
        for extra in (['--encoder-lr','nan'],['--head-lr','0'],['--layer-decay','1.1'],
                      ['--joint-epochs','6'],['--head-epochs','0'],['--source-chunk','0']):
            args = config.parser().parse_args(extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                config.configuration(args)

    def test_existing_data_cannot_be_cache_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(ValueError,'overlap'):
                self.build(root, ['--cache-root',str(root/'train'/'cache')])

    def test_train_dev_noise_must_be_separate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaisesRegex(ValueError,'Distinct'):
                self.build(root,['--dev-noise-manifest',str(root/'noise'/'train.jsonl')])


class PretrainedTests(unittest.TestCase):
    def test_wrong_weights_rejected_even_with_matching_model_name(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'config.json').write_text('{"_name_or_path":"facebook/w2v-bert-2.0"}')
            (root/'preprocessor_config.json').write_text('{}')
            (root/'model.safetensors').write_bytes(b'fine-tuned')
            with self.assertRaisesRegex(ValueError,'not the pinned'):
                pretrained.validate_directory(root)

    def test_explicit_unknown_folder_never_downloads_silently(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'needs config'):
                pretrained.prepare({'pretrained_path':td})

    def test_actual_generic_initialization_and_new_head(self):
        from w2v_aasist.tests import tiny_detector
        from w2v_v3.model import HeadConfig
        with tempfile.TemporaryDirectory() as td:
            torch.set_num_threads(1)
            generic=tiny_detector().backbone
            generic.save_pretrained(td, safe_serialization=True)
            cfg=dict(pretrained_path=td,production_layout=False,checkpointing=True,
                     head_config=asdict(HeadConfig(input_dim=16,projection=8,expansion=32,kernels=(3,7),merge_kernel=3)))
            model=pretrained.initialize(cfg)
            for key,expected in generic.state_dict().items():
                torch.testing.assert_close(model.backbone.state_dict()[key],expected,rtol=0,atol=0)
            self.assertEqual(model.head.config.projection,8)
            self.assertEqual(model.backbone.config.layerdrop,0.)
            self.assertTrue(model.backbone.is_gradient_checkpointing)

    def test_pretrained_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.safetensors';path.write_bytes(b'a')
            cfg={'pretrained_fingerprints':{str(path):sha256(path)}}
            path.write_bytes(b'b')
            with self.assertRaisesRegex(ValueError,'changed'):
                pretrained.verify(cfg)


if __name__ == '__main__':
    unittest.main()
