"""Recipe bounds, independent outputs and no cache mutation in V3.1."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import config, workflow


class WorkflowTests(unittest.TestCase):
    def source(self, root):
        source = root/'old'; source.mkdir()
        value = {'algorithm':'w2v-BERT + MultiConv', 'full_noisy':True,
                 'baseline':str(root/'original.pt'), 'baseline_sha256':config.BASELINE_SHA256,
                 'train_caches':['existing_train'], 'dev_noisy_cache':'fixed_seen',
                 'dev_heldout_cache':'fixed_heldout', 'seed':1234}
        (source/'config.json').write_text(json.dumps(value), encoding='utf-8')
        (source/'best_model.pt').write_bytes(b'existing best')
        return source

    def digest(self, path):
        return config.BASELINE_SHA256 if Path(path).name == 'original.pt' else 'a'*64

    def test_recipe_preserves_cache_paths_and_selects_safe_warm_weights(self):
        with tempfile.TemporaryDirectory() as td:
            source = self.source(Path(td))
            args = config.parser().parse_args(['--source-run',str(source)])
            before = (source/'config.json').read_bytes()
            with patch.object(config,'sha256',side_effect=self.digest):
                c = config.configuration(args)
            self.assertEqual(c['train_caches'], ['existing_train'])
            self.assertEqual(c['dev_noisy_cache'], 'fixed_seen')
            self.assertEqual(Path(c['warm_checkpoint']), (source/'best_model.pt').resolve())
            self.assertEqual((c['head_epochs'],c['joint_epochs']), (0,2))
            self.assertEqual(c['trainable_layers'],4)
            self.assertEqual(c['noisy_weight_start'],c['noisy_weight'])
            self.assertEqual(c['short_loss_weight'],.3)
            self.assertEqual(c['clean_tolerance'],.002)
            self.assertEqual(before,(source/'config.json').read_bytes())

    def test_invalid_recipe_and_original_mismatch_fail_before_start(self):
        with tempfile.TemporaryDirectory() as td:
            source = self.source(Path(td))
            for argv in (['--epochs','0'],['--epochs','3'],['--short-loss-weight','nan'],
                         ['--short-loss-weight','1'],['--short-min-seconds','.1'],
                         ['--short-min-seconds','7'],['--head-lr','0'],
                         ['--encoder-lr','inf'],['--workers','-1']):
                args=config.parser().parse_args(['--source-run',str(source),*argv])
                with self.subTest(argv=argv), patch.object(config,'sha256',side_effect=self.digest):
                    with self.assertRaises(ValueError): config.configuration(args)
            args=config.parser().parse_args(['--source-run',str(source)])
            with patch.object(config,'sha256',return_value='wrong'):
                with self.assertRaisesRegex(ValueError,'Protected original'): config.configuration(args)

    def test_workflow_reuses_cache_and_exports_no_models(self):
        from contextlib import nullcontext
        import io
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'exp').mkdir()
            source=self.source(root)
            args=config.parser().parse_args(['--source-run',str(source),'--device','cpu','--upload-temp'])
            with patch.object(config,'sha256',side_effect=self.digest):
                cfg=config.configuration(args)
            child=Namespace(stdout=io.StringIO('training output\n'),wait=lambda:0)
            with patch.object(workflow,'ROOT',root), \
                 patch.object(workflow,'parser') as parser_mock, \
                 patch.object(workflow,'configuration',return_value=cfg), \
                 patch.object(workflow,'check_environment'), \
                 patch.object(workflow,'ensure_idle'), \
                 patch.object(workflow,'run_lock',side_effect=lambda _:nullcontext()), \
                 patch.object(workflow.subprocess,'Popen',return_value=child) as process, \
                 patch.object(workflow,'export_report') as export:
                parser_mock.return_value.parse_args.return_value=args
                workflow.main()
            run=Path((root/'exp'/'.latest_v31_run').read_text().strip())
            self.assertNotEqual(run,source)
            self.assertEqual(process.call_args.args[0][3], 'w2v_v31.train')
            self.assertTrue((run/'source.json').is_file())
            export.assert_called_once_with(run,args.download_dir,True)
            self.assertEqual((source/'best_model.pt').read_bytes(),b'existing best')
            self.assertFalse((root/'exp'/'.latest_v3_run').exists())


if __name__ == '__main__':
    unittest.main()
