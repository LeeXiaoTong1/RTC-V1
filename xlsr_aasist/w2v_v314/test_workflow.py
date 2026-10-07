"""Real tiny w2v-BERT waveform training, exact recovery and all export paths."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from w2v_v39.common import atomic_json, digest, read_json
from w2v_v312.test_workflow import fixture, synthetic_expanded_bundles
from w2v_v312.train import run_experiment as run_v312
from w2v_v313.test_workflow import synthetic_paired_bundles, assert_nested_equal
from .config import configuration, parser, verify_inputs
from .model import load_model, optimizer_for
from .state import atomic_save, apply_partial, apply_candidate, load_selected
from .train import run_experiment, infer

torch.set_num_threads(1)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        with redirect_stdout(io.StringIO()):
            _, old, _, cls.records = fixture(cls.root)
            cls.source = cls.root/'v312'
            cls.source.mkdir()
            atomic_json(cls.source/'config.json', old)
            with patch('w2v_v312.train.bundles', synthetic_expanded_bundles):
                run_v312(old, cls.source)
            cls.source_state = torch.load(cls.source/'last.pt', map_location='cpu', weights_only=True)
            cls.cfg = configuration(parser().parse_args([
                '--source-run', str(cls.source), '--device', 'cpu', '--workers', '0', '--epochs', '1']))
            cls.cfg.update(trainable_layers=1, microbatch=4, eval_batch=4, feature_batch=4,
                feature_commit_rows=64, head_steps=3, head_chunk=128, disk_margin_bytes=0,
                feature_free_margin_bytes=0, min_gain=2., patience=8, catastrophic_weighted_drop=2.)
            with synthetic_paired_bundles(cls.cfg) as (train, _):
                cls.train_rows = train['rows']
            cls.protected = {str(p):digest(p) for p in cls.root.rglob('*') if p.is_file()}

    def new_run(self, name):
        run = self.root/name
        run.mkdir()
        atomic_json(run/'config.json', self.cfg)
        return run

    def run_model(self, run):
        with patch('w2v_v314.train.bundles', synthetic_paired_bundles):
            return run_experiment(self.cfg, run)

    def test_train_final_linear_then_real_joint_updates_without_new_audio_and_export_three_choices(self):
        from .evaluate import export
        from .cleanup import cleanup
        from .workflow import report
        with redirect_stdout(io.StringIO()):
            run = self.new_run('v314_complete')
            verify_inputs(self.cfg)
            source = load_model(self.cfg, training=False)
            source_parameters = {k:v.detach().cpu().clone() for k,v in source.named_parameters() if v.requires_grad}
            done = self.run_model(run)
            self.assertEqual(done['selections']['best_guarded'], 'starting_last')
            self.assertEqual(done['committed_updates'], done['planned_updates'])
            report(run)
            saved = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
            changed = {k for k,v in saved['model'].items() if not torch.equal(v, source_parameters[k])}
            self.assertTrue(any(k.startswith('head.classifier.2.') for k in changed))
            self.assertTrue(any(k.startswith('head.blocks.') for k in changed))
            self.assertTrue(any(k.startswith('backbone.encoder.layers.1.') for k in changed))
            self.assertFalse(any(k.startswith('backbone.encoder.layers.0.') for k in saved['model']))
            self.assertGreater(read_json(run/'stage_a_report.json')['parameter_displacement_l2'], 0.)
            self.assertTrue(read_json(run/'stage_a_report.json')['train_only'])
            self.assertEqual(read_json(run/'startup_replay.json')['status'], 'passed')
            self.assertFalse(list(run.rglob('*.wav')))
            self.assertFalse(list(run.glob('epoch_*.pt')))
            self.assertTrue((run/'features/train/x.npy').exists())
            self.assertEqual(np.load(run/'features/train/x.npy', mmap_mode='r').shape,
                             (len(self.train_rows), 512))
            sample = read_json(run/'sampling_epoch_1.json')
            self.assertAlmostEqual(sample['ce_mass']['noisy'], .5)
            self.assertAlmostEqual(sample['ce_mass']['ordinary'], .5)
            self.assertTrue(all(e['committed'] for e in saved['history']))
            self.assertEqual(read_json(run/'training_design.json')['auxiliary_losses'], [])
            self.assertEqual(self.protected, {p:digest(p) for p in self.protected})

            rows = [self.records['dev'][i] for i in (9, 0, 6, 3)]
            protocol = run/'test_protocol.txt'
            protocol.write_text('\n'.join(r['id'] for r in rows)+'\n', encoding='utf-8')
            before = {}
            for kind in ('best_guarded', 'best_weighted', 'last'):
                checkpoint, meta = load_selected(run, kind)
                detector = load_model(self.cfg, training=False)
                apply_candidate(detector, checkpoint['candidate'])
                expected, _ = infer(detector, rows, self.cfg, 'test expected '+kind)
                output = self.root/('export_'+kind)
                archive = export(run, protocol, self.root/'audio', output, 'cpu', 0, kind)
                with zipfile.ZipFile(archive) as z:
                    self.assertEqual(z.namelist(), ['scores.txt'])
                    values = [line.split() for line in z.read('scores.txt').decode().splitlines()]
                self.assertEqual([v[0] for v in values], [r['id'] for r in rows])
                np.testing.assert_allclose([float(v[1]) for v in values],
                    torch.from_numpy(expected).softmax(-1)[:,0].numpy(), rtol=0, atol=1e-9)
                exported = read_json(output/'submission_meta.json')
                self.assertEqual(exported['selected'], meta['selected'])
                self.assertEqual(exported['checkpoint_kind'], kind)
                before[kind] = expected
            old_hash = digest(run/'last.pt')
            cleanup(run, apply=False, remove_features=True)
            self.assertEqual(digest(run/'last.pt'), old_hash)
            cleanup(run, apply=True, remove_features=True)
            self.assertFalse((run/'last.pt').exists())
            self.assertFalse((run/'features').exists())
            self.assertTrue((run/'inference.pt').is_file())
            for kind in before:
                checkpoint, _ = load_selected(run, kind)
                detector = load_model(self.cfg, training=False)
                apply_candidate(detector, checkpoint['candidate'])
                after, _ = infer(detector, rows, self.cfg, 'test compacted '+kind)
                np.testing.assert_array_equal(before[kind], after)
            cleanup(run, apply=True, remove_features=True)
            self.assertEqual(self.protected, {p:digest(p) for p in self.protected})

    def test_resume_after_half_epoch_reproduces_exact_weights_adam_and_rng(self):
        with redirect_stdout(io.StringIO()):
            full = self.new_run('v314_full')
            stopped = self.new_run('v314_resume')
            self.run_model(full)
            did_interrupt = False
            def interrupt(path, value, margin):
                nonlocal did_interrupt
                atomic_save(path, value, margin)
                if Path(path).name == 'last.pt' and value['cursor'] > 0 and not did_interrupt:
                    did_interrupt = True
                    raise InterruptedError('test exit after a committed half epoch')
            with patch('w2v_v314.train.atomic_save', side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    self.run_model(stopped)
            cache_hashes = {str(p):digest(p) for p in (stopped/'features').rglob('*') if p.is_file()}
            with patch('w2v_v314.train.fit_head', side_effect=AssertionError('Head must not refit on resume')):
                self.run_model(stopped)
            a = torch.load(full/'last.pt', map_location='cpu', weights_only=True)
            b = torch.load(stopped/'last.pt', map_location='cpu', weights_only=True)
            self.assertEqual(a['cursor'], b['cursor'])
            for key in ('model', 'optimizer', 'rng', 'selections', 'candidates'):
                assert_nested_equal(self, a[key], b[key])
            self.assertEqual(cache_hashes, {p:digest(p) for p in cache_hashes})
            self.assertFalse((stopped/'validation_pending.json').exists())

    def test_starting_output_function_and_optimizer_inventory_and_wrong_hash_rejected(self):
        with redirect_stdout(io.StringIO()):
            model = load_model(self.cfg, training=False)
            old = load_model(self.cfg, training=False)
            self.assertTrue(all(p.requires_grad for p in model.head.classifier[-1].parameters()))
            for name, value in self.source_state['model'].items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            optimizer = optimizer_for(model, self.cfg)
            group = next(g for g in optimizer.param_groups if g['name'] == 'final_binary_projection')
            self.assertEqual({id(p) for p in group['params']}, {id(p) for p in model.head.classifier[-1].parameters()})
            with self.assertRaises(ValueError):
                load_model(dict(self.cfg, starting_checkpoint_sha256='0'*64))
            self.assertEqual(configuration(parser().parse_args(['--source-run',str(self.source),'--device','cpu']))['epochs'], 2)


if __name__ == '__main__':
    unittest.main()
