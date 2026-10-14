"""Completed ASP parent -> whole-utterance PCL training -> exact resume/export."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from w2v_v318.test_core import NativeBoundary
from .test_support import tiny_cfg, tiny_model, scene_fixture, completed_v3185, completed_aug5, save_inputs, fixture_phones
from .common import read_json, atomic_json, digest
from .parent import inspect_parent
from .train import run_experiment, save
from .evaluate import export, selected


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parent_migration_joint_pcl_resume_and_delivery(self):
        self.check_parent_workflow(completed_v3185,'3.18.5',[5,5,1,3])

    def test_aug5_parent_migration_joint_pcl_resume_and_delivery(self):
        self.check_parent_workflow(completed_aug5,'3.18.4',[5,5,5,5])

    def check_parent_workflow(self,parent_factory,parent_version,parent_factors):
        with tempfile.TemporaryDirectory() as directory, \
             patch('w2v_v3184.data.Engines', NativeBoundary), \
             patch('w2v_v3185.data.Engines', NativeBoundary), \
             patch('w2v_v3186.data.Engines', NativeBoundary), \
             patch('w2v_v3186.data.Waves._phones', fixture_phones), redirect_stdout(io.StringIO()):
            root = Path(directory)
            parent, prior, train, dev, parent_best = parent_factory(root)
            original_hash = digest(parent / 'last.pt')
            _, _, initialization = inspect_parent(parent)
            self.assertEqual(initialization['tag'], 'epoch_1_step_3')
            self.assertEqual(initialization['parent_version'],parent_version)
            self.assertEqual(initialization['parent_view_factors'],parent_factors)
            cfg = tiny_cfg()
            for key in ('dev_pairs', 'official_dev_protocol', 'official_dev_root', 'noise_records'):
                cfg[key] = prior[key]
            cfg.update(scene_fixture(root / 'scenes'))
            cfg['initialization'] = dict(initialization, mode='parent-transfer', old_optimizer_loaded=False)
            full = root / 'full'
            save_inputs(full, cfg, train, dev)
            run_experiment(cfg, full, train, dev, tiny_model, baseline_only=True)
            initial = torch.load(full / 'last.pt', weights_only=True)
            for key, current in initial['model'].items():
                expected = parent_best[key]
                if key == 'head.aasist.out_layer.weight':
                    expected = expected[:, :160]
                torch.testing.assert_close(current, expected, rtol=0, atol=0)
            audit = read_json(full / 'initialization_audit.json')
            self.assertFalse(audit['initial_function_preserved'])
            self.assertFalse(audit['external_window_pooling'])
            self.assertEqual(audit['bn_statistics_reset'], [])
            self.assertEqual(audit['parent_version'],parent_version)
            self.assertEqual(audit['training_view_factors'],[5,5,1,3])
            self.assertFalse(any('attention_delta' in key or 'contribution' in key for key in initial['model']))
            self.assertFalse((full / 'completed.json').exists())
            with self.assertRaises(FileNotFoundError):
                selected(full, 'best')
            run_experiment(cfg, full, train, dev, tiny_model)
            resumed = root / 'resumed'
            save_inputs(resumed, cfg, train, dev)

            def interrupt(*args):
                save(*args)
                if args[2]['epoch'] == 1:
                    raise RuntimeError('test interruption')

            with patch('w2v_v3186.train.save', interrupt), self.assertRaisesRegex(RuntimeError, 'test interruption'):
                run_experiment(cfg, resumed, train, dev, tiny_model)
            with (resumed / 'steps.jsonl').open('a') as stream:
                stream.write('{"update":999}\n{"broken":')
            run_experiment(cfg, resumed, train, dev, tiny_model)
            a = torch.load(full / 'last.pt', weights_only=True)
            b = torch.load(resumed / 'last.pt', weights_only=True)
            for name, value in a['model'].items():
                torch.testing.assert_close(value, b['model'][name], rtol=0, atol=0)
            for key, values in a['optimizer']['state'].items():
                for name, value in values.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, b['optimizer']['state'][key][name], rtol=0, atol=0)
            self.assertEqual(a['best_tag'], b['best_tag'])
            self.assertEqual([entry['phase'] for entry in a['history']], ['joint', 'joint'])
            self.assertEqual([json.loads(line)['update'] for line in (resumed / 'steps.jsonl').read_text().splitlines()],
                             list(range(1, b['cursor'] + 1)))
            self.assertTrue(any(not torch.equal(initial['model'][name], value)
                                for name, value in a['model'].items() if name.endswith('.lora_b')))
            for entry in a['history']:
                self.assertEqual(entry['scenes']['count'], 24)
                self.assertEqual(entry['dev']['count'], len(dev))
                aux = entry['training']['auxiliary']
                self.assertGreater(aux['mean_per_update']['auxiliary_reference_views'], 0.)
                self.assertGreater(aux['losses']['noise']['mean_weighted_loss_per_update'], 0.)
                self.assertGreater(aux['losses']['platform']['mean_weighted_loss_per_update'], 0.)
                self.assertGreater(sum(value['valid_sources'] for value in aux['coverage'].values()), 0)
            self.assertEqual(digest(parent / 'last.pt'), original_hash)
            atomic_json(resumed / 'training_history.json', [])
            run_experiment(cfg, resumed, train, dev, tiny_model)
            self.assertEqual(len(read_json(resumed / 'training_history.json')), 2)
            out = root / 'dev-export'
            export(resumed, out, dev=True, device='cpu', workers=0, model_factory=tiny_model, verify=False)
            metadata = read_json(out / 'validation_meta.json')
            self.assertTrue(metadata['whole_utterance'])
            self.assertFalse(metadata['local_evidence'])
            self.assertFalse(metadata['inference_requires_phonemes'])
            protocol = root / 'eval.txt'
            ids = [row['id'] for row in dev if row['condition'] == 'online']
            protocol.write_text('\n'.join(ids) + '\n')
            from .delivery import prepare_request, finish_request
            from .config import parser
            from functools import partial
            args = parser().parse_args(['--submit-best', '--upload-submission', '--submission-protocol', str(protocol),
                '--submission-audio-root', str(root / 'audio' / 'dev'), '--submission-out', str(root / 'automatic-best')])
            request = prepare_request(args, resumed, cfg)
            with patch('w2v_v3186.delivery.ROOT', root), \
                 patch('w2v_v3186.delivery.upload_archive', side_effect=RuntimeError('network fixture')):
                with self.assertRaisesRegex(RuntimeError, 'Retry without GPU'):
                    finish_request(resumed, cfg, request, partial(export, model_factory=tiny_model, verify=False))
            self.assertEqual([line.split()[0] for line in (root / 'automatic-best' / 'scores.txt').read_text().splitlines()], ids)
            with patch('w2v_v3186.delivery.ROOT', root), \
                 patch('w2v_v3186.delivery.upload_archive', return_value='https://temp.sh/test/submission.zip'), \
                 patch('w2v_v3186.evaluate.export', side_effect=AssertionError('Inference must not repeat')):
                delivered = finish_request(resumed, cfg, request)
            self.assertEqual(delivered['selected'], b['best_tag'])
            self.assertTrue(delivered['uploaded'])
            self.assertEqual(digest(parent / 'last.pt'), original_hash)
            with self.assertRaisesRegex(ValueError, 'Expected parent'):
                inspect_parent(parent, 'epoch_999_step_3')
            # The additional family must not weaken its original integrity checks.
            old_config=read_json(parent/'config.json')
            changed_config=dict(old_config,seed=old_config['seed']+1)
            atomic_json(parent/'config.json',changed_config)
            with self.assertRaisesRegex(ValueError,'checkpoint/config mismatch'):
                inspect_parent(parent)
            atomic_json(parent/'config.json',old_config)
            with (parent/'last.pt').open('ab') as stream:stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Checkpoint hash changed'):
                inspect_parent(parent)


if __name__ == '__main__':
    unittest.main()
