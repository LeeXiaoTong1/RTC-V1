"""Defaults, immutable resume and the explicit legacy comparison route."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .config import parser, validate, training_configuration, code_inventory, compact_configuration
from .common import schema_for
from w2v_v3189.test_training import fixture


class ConfigurationTests(unittest.TestCase):
    def test_legacy_parent_has_same_v52_recipe_and_fixed_dynamic_control_budget(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            old, _, _ = fixture(Path(folder))
            old.update(encoder_layers=60, lora_layers=60, epochs=10)
            origin = dict(epoch=6, steps=1403, run=folder, manifest_files={}, tag='best')
            args = parser().parse_args(['--workers', '0'])
            validate(args)
            with patch('w2v_v53.config.runtime_versions', return_value={}):
                dynamic = training_configuration(old, origin, args)
                static = training_configuration(old, origin, parser().parse_args(['--workers', '0', '--fusion-mode', 'static']))
            self.assertEqual(dynamic['pcl_noise_weight'], old['pcl_noise_weight']*.2)
            self.assertEqual(dynamic['classification_ce_weights'], dict(online=.6, clean_rtc=.2, noisy=.2))
            self.assertEqual(dynamic['adaptive_fusion_mode'], 'dynamic')
            self.assertEqual(dynamic['epochs'], 1)
            self.assertEqual(dynamic['contrastive_weight'], .01)
            for key in dynamic:
                if key != 'adaptive_fusion_mode':
                    self.assertEqual(dynamic[key], static[key], key)
            schema_for(dynamic)
            self.assertIn('adaptive_fusion_condition', compact_configuration(dynamic)['settings'])

    def test_resume_rejects_structural_and_objective_changes(self):
        argv = ['--resume', 'run', '--submit-best']
        validate(parser().parse_args(argv), argv)
        for override in (['--fusion-mode','static'], ['--fusion-width','64'], ['--epochs','2'],
                         ['--contrastive-weight','.02'], ['--parent-run','parent']):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate(parser().parse_args(argv+override), argv+override)
        for bad in (['--fusion-width','0'], ['--contrastive-weight','nan']):
            with self.assertRaises(ValueError):
                validate(parser().parse_args(bad))

    def test_code_fingerprint_includes_new_fusion_and_inherited_objectives(self):
        files = code_inventory()
        suffixes = {str(Path(name)).replace('\\','/') for name in files}
        for suffix in ('w2v_v53/fusion.py', 'w2v_v53/model.py', 'w2v_v53/step.py',
                       'w2v_v52/side.py', 'w2v_v52/data.py', 'w2v_v52/contrastive.py'):
            self.assertTrue(any(name.endswith(suffix) for name in suffixes), suffix)
        self.assertFalse(any('/test_' in name for name in suffixes))


if __name__ == '__main__':
    unittest.main()
