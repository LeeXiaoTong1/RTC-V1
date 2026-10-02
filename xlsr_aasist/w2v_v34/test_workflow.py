"""Submission identity, one-run launch, read-only caches and retained terminal metrics."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
from w2v_aasist.runtime import atomic_json, atomic_save, sha256
from . import SCHEMA, workflow, evaluate
from .select_checkpoint import select
from .test_train import fixture


class WorkflowTests(unittest.TestCase):
    def test_one_training_call_no_cache_generator_and_diagnostic_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cfg, _, _, _ = fixture(root)
            cfg.update(version='3.4')
            args = SimpleNamespace(device='cpu', resume=None, prepare_only=False,
                                   smoke_steps=0, download_dir=str(root/'reports'), upload_temp=False)
            initial_hash = sha256(cfg['warm_checkpoint'])
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'parser') as parser, \
                 patch.object(workflow, 'configuration', return_value=cfg), \
                 patch.object(workflow, 'check_environment'), patch.object(workflow, 'ensure_idle'), \
                 patch.object(workflow, 'publish'), patch('w2v_v34.train.train') as training, \
                 patch('w2v_v33.cache.prepare_cache', side_effect=AssertionError('must reuse cache')):
                parser.return_value.parse_args.return_value = args
                workflow.main()
            self.assertEqual(training.call_count, 1)
            passed_cfg, run, resume, smoke = training.call_args.args
            self.assertEqual(passed_cfg, cfg); self.assertIsNone(resume); self.assertEqual(smoke, 0)
            self.assertEqual(sha256(cfg['warm_checkpoint']), initial_hash)
            self.assertFalse((run/'control').exists()); self.assertFalse((run/'candidate').exists())
            self.assertEqual(Path((root/'exp'/'.latest_v34_run').read_text().strip()), run)
            with zipfile.ZipFile(root/'reports'/(run.name+'_report.zip')) as archive:
                self.assertIn('source.json', archive.namelist())
                self.assertFalse(any(name.endswith(('.pt', '.wav')) for name in archive.namelist()))

    def test_protected_checkpoint_and_provenance_change_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cfg, _, _, _ = fixture(root)
            workflow.verify_protected(cfg)
            extra = root/'submission_meta.json'; extra.write_text('{}')
            cfg['source_provenance']['file_fingerprints'][str(extra)] = sha256(extra)
            extra.write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError, 'provenance changed'):
                workflow.verify_protected(cfg)

    def test_default_selection_matches_completed_weights_and_refuses_stale_tag(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cfg, _, _, _ = fixture(root)
            cfg['version'] = '3.4'
            from w2v_v3.train import read_state
            state = read_state(cfg['warm_checkpoint'])
            state.update(schema=SCHEMA, config=cfg)
            run = root/'run'; run.mkdir()
            atomic_json(run/'config.json', cfg)
            atomic_json(run/'completed.json', {'selection': {'best_safe': {'tag': 'baseline'}}})
            atomic_save(run/'best_model.pt', state)
            self.assertEqual(select(run), (run/'best_model.pt').resolve())
            atomic_json(run/'completed.json', {'selection': {'best_safe': {'tag': 'epoch_2'}}})
            with self.assertRaisesRegex(ValueError, 'completed V3.4 selection'):
                select(run)

    def test_real_cpu_submission_order_probabilities_and_source_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cfg, _, fingerprints, _ = fixture(root)
            from w2v_v3.train import read_state
            state = read_state(cfg['warm_checkpoint'])
            state.update(schema=SCHEMA, config=cfg, data_fingerprints=fingerprints)
            checkpoint = root/'v34_best.pt'; atomic_save(checkpoint, state)
            protocol = root/'progress.txt'; protocol.write_text('3.wav\n0.wav\n7.wav\n')
            destination = root/'submission'
            args = ['evaluate', '--checkpoint', str(checkpoint), '--protocol', str(protocol),
                    '--audio-root', str(root), '--out', str(destination), '--device', 'cpu', '--workers', '0']
            with patch('sys.argv', args):
                evaluate.main()
            with zipfile.ZipFile(destination/'submission.zip') as archive:
                self.assertEqual(archive.namelist(), ['scores.txt'])
                rows = [line.split() for line in archive.read('scores.txt').decode().splitlines()]
            self.assertEqual([row[0] for row in rows], ['3.wav', '0.wav', '7.wav'])
            self.assertTrue(all(0 <= float(row[1]) <= 1 for row in rows))
            meta = json.loads((destination/'submission_meta.json').read_text())
            self.assertEqual(meta['warm_checkpoint_sha256'], cfg['warm_checkpoint_sha256'])
            self.assertEqual(meta['checkpoint_sha256'], sha256(checkpoint))
            self.assertTrue(meta['baseline_fallback'])

    def test_viewer_retains_validation_and_v34_title(self):
        from v33_console import watch
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); log = root/'run.log'; state = root/'progress.json'
            log.write_text('V32_EVENT={"kind":"phase","label":"V3.4 validating epoch_1"}\n'
                           'Dev epoch_1 Clean=98 Seen=94 Heldout=95 Noisy=94.5 Weighted=95.55 promoted=True action=continue\n'
                           'offline/en recall [fake,real]=[0.99,0.91]\n'
                           'online/en recall [fake,real]=[0.99,0.84]\n'
                           'seen/en recall [fake,real]=[0.98,0.75]\n'
                           'heldout/en recall [fake,real]=[0.98,0.76]\n')
            state.write_text('{"status":"complete","label":"V3.4 complete"}')
            stream = io.StringIO()
            watch(log, state, once=True, stream=stream, version='V3.4')
            output = stream.getvalue()
            self.assertIn('V3.4 console', output); self.assertIn('epoch_1 (epoch complete)', output)
            self.assertIn('Weighted=95.550', output)


if __name__ == '__main__':
    unittest.main()
