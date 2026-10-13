"""Early export uses the committed AUG5 best and leaves training resumable."""
import copy
import io
from contextlib import ExitStack, redirect_stdout
from functools import partial
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from finish_w2v_v3184_aug5 import authenticate_supervisor, deliver, seal, validate_state
from w2v_v318.test_core import NativeBoundary
from w2v_v3183r2.monitor import measure
from w2v_v3184.test_workflow import completed_r2
from w2v_v3184_aug5.common import digest, read_json
from w2v_v3184_aug5.evaluate import export, selected
from w2v_v3184_aug5.parent import inspect_parent
from w2v_v3184_aug5.sampling import Plan
from w2v_v3184_aug5.test_support import scene_fixture, tiny_cfg, tiny_model
from w2v_v3184_aug5.test_workflow import save_inputs
from w2v_v3184_aug5.train import run_experiment, save


class EarlyExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name).resolve()
        with ExitStack() as stack:
            stack.enter_context(patch('w2v_v3184.data.Engines', NativeBoundary))
            stack.enter_context(patch('w2v_v3184_aug5.data.Engines', NativeBoundary))
            stack.enter_context(redirect_stdout(io.StringIO()))
            parent, prior, cls.train, cls.dev, _ = completed_r2(cls.root)
            _, _, initialization = inspect_parent(parent)
            cls.cfg = tiny_cfg()
            for key in ('dev_pairs', 'official_dev_protocol', 'official_dev_root', 'noise_records'):
                cls.cfg[key] = prior[key]
            cls.cfg.update(scene_fixture(cls.root / 'scenes'))
            cls.cfg.update(epochs=3, initialization=dict(initialization, mode='parent-transfer', old_optimizer_loaded=False))
            cls.interrupted = cls.root / 'interrupted'
            save_inputs(cls.interrupted, cls.cfg, cls.train, cls.dev)

            def interrupt(*args):
                save(*args)
                if args[2]['epoch'] == 2:
                    raise RuntimeError('fixture interruption after atomic epoch')

            try:
                with patch('w2v_v3184_aug5.train.save', interrupt):
                    run_experiment(cls.cfg, cls.interrupted, cls.train, cls.dev, tiny_model)
            except RuntimeError as exc:
                if str(exc) != 'fixture interruption after atomic epoch':
                    raise
            else:
                raise AssertionError('Fixture did not stop after its atomic epoch')
            cls.complete = cls.root / 'complete'
            save_inputs(cls.complete, cls.cfg, cls.train, cls.dev)
            run_experiment(cls.cfg, cls.complete, cls.train, cls.dev, tiny_model)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.case = self.root / self._testMethodName
        self.case.mkdir()
        self.run = self.case / 'run'
        shutil.copytree(self.interrupted, self.run)
        self.stack = ExitStack()
        self.stack.enter_context(patch('w2v_v3184.data.Engines', NativeBoundary))
        self.stack.enter_context(patch('w2v_v3184_aug5.data.Engines', NativeBoundary))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.addCleanup(self.stack.close)

    def hashes(self, run):
        return {p.relative_to(run).as_posix(): digest(p) for p in run.rglob('*') if p.is_file()}

    def force_earlier_best(self):
        state = torch.load(self.run / 'last.pt', weights_only=True)
        partition = read_json(self.run / 'dev_partition.json')
        indices = partition['select']
        rows = [self.dev[i] for i in indices]
        best = torch.full((len(self.dev), 2), -2.)
        last = torch.zeros_like(best)
        for i, row in enumerate(self.dev):
            best[i, row['label']] = 2.
        steps = Plan(self.train, self.cfg['stream_sources'], self.cfg['seed'],
                     self.cfg['group_quotas'], self.cfg['view_factor']).steps
        first = measure(rows, best.numpy()[indices])
        second = measure(rows, last.numpy()[indices])
        state.update(best_tag=f'epoch_1_step_{steps}', last_tag=f'epoch_2_step_{steps}',
                     best_metrics=first, best_logits=best, last_logits=last)
        state['history'][0]['selection_metrics'] = first
        state['history'][1]['selection_metrics'] = second
        # The forced fixture deliberately has different best and last weights.
        key = 'head.aasist.out_layer.bias'
        state['best_model'] = copy.deepcopy(state['best_model'])
        state['best_model'][key] = state['model'][key].clone() + .01
        self.assertFalse(torch.equal(state['best_model'][key], state['model'][key]))
        torch.save(state, self.run / 'last.pt')
        return state

    def test_atomic_snapshot_preserves_source_and_exact_resume(self):
        before = self.hashes(self.run)
        pointer = self.case / 'export_run'
        snapshot = seal(self.run, verify=False, pointer=pointer)
        self.assertNotEqual(snapshot, self.run)
        self.assertEqual(pointer.read_text().strip(), str(snapshot))
        self.assertFalse((self.run / 'completed.json').exists())
        self.assertEqual(before, self.hashes(self.run))
        done = read_json(snapshot / 'completed.json')
        self.assertEqual(done['completed_epochs'], 2)
        self.assertEqual(done['planned_epochs'], 3)
        self.assertTrue(done['stopped_early'])
        self.assertEqual(done['completion_reason'], 'user_sealed_last_committed_epoch')
        snapshot_checkpoint = digest(snapshot / 'last.pt')
        self.assertEqual(snapshot_checkpoint, before['last.pt'])
        self.assertEqual(seal(self.run, verify=False), snapshot)
        run_experiment(self.cfg, self.run, self.train, self.dev, tiny_model)
        resumed = torch.load(self.run / 'last.pt', weights_only=True)
        expected = torch.load(self.complete / 'last.pt', weights_only=True)
        self.assertEqual(resumed['epoch'], 3)
        self.assertEqual(resumed['best_tag'], expected['best_tag'])
        for key, value in expected['model'].items():
            torch.testing.assert_close(resumed['model'][key], value, rtol=0, atol=0)
        self.assertEqual(digest(snapshot / 'last.pt'), snapshot_checkpoint)
        self.assertEqual(seal(self.run, verify=False), self.run)

    def test_best_not_last_validation_and_snapshot_tampering(self):
        state = self.force_earlier_best()
        validate_state(self.run, self.cfg, state)
        for name, change in (
            ('zero_epochs', lambda s: s.update(epoch=0)),
            ('partial_cursor', lambda s: s.update(cursor=s['cursor'] + 1)),
            ('wrong_best', lambda s: s.update(best_tag=s['last_tag'])),
            ('nan_scores', lambda s: s['best_logits'].__setitem__((0, 0), float('nan'))),
            ('inconsistent_scores', lambda s: s['best_logits'].zero_()),
        ):
            with self.subTest(name=name):
                broken = copy.deepcopy(state)
                change(broken)
                with self.assertRaises(ValueError):
                    validate_state(self.run, self.cfg, broken)
        snapshot = seal(self.run, verify=False)
        _, weights, meta = selected(snapshot, 'best')
        self.assertEqual(meta['selected'], state['best_tag'])
        self.assertNotEqual(meta['selected'], state['last_tag'])
        for key, value in state['best_model'].items():
            torch.testing.assert_close(weights[key], value, rtol=0, atol=0)
        self.assertEqual(read_json(snapshot / 'calibration_best.json')['selected'], state['best_tag'])
        (snapshot / 'dev_rows.json').write_text('[]')
        with self.assertRaises(Exception):
            seal(self.run, verify=False)

    def test_delivery_failure_still_reports_and_retry_reuses_inference(self):
        state = self.force_earlier_best()
        protocol = self.case / 'progress.txt'
        ids = [r['id'] for r in self.dev if r['condition'] == 'online']
        protocol.write_text('\n'.join(ids) + '\n')
        args = SimpleNamespace(submission_protocol=str(protocol), submission_audio_root=str(self.root / 'audio' / 'dev'),
                               download_dir=str(self.case / 'downloads'), upload_temp=True, upload_submission=True,
                               submit_best=True, submission_out=None, device='cpu', workers=0)
        from w2v_v3184_aug5.delivery import prepare_request
        prepare_request(args, self.run, self.cfg)
        before = self.hashes(self.run)
        snapshot = seal(self.run, verify=False)
        exporter = partial(export, model_factory=tiny_model, verify=False)
        with patch('w2v_v3184_aug5.delivery.ROOT', self.case), \
             patch('w2v_v3184_aug5.delivery.upload_archive', side_effect=RuntimeError('network fixture')), \
             patch('w2v_v3184_aug5.reports.upload_archive', return_value='https://temp.sh/test/report.zip'):
            with self.assertRaises(RuntimeError):
                deliver(snapshot, self.run, args, exporter=exporter)
        request = read_json(snapshot / 'submission_request.json')
        out = Path(request['out'])
        self.assertTrue((out / 'submission.zip').is_file())
        self.assertEqual([s.split()[0] for s in (out / 'scores.txt').read_text().splitlines()], ids)
        self.assertEqual(read_json(out / 'submission_meta.json')['selected'], state['best_tag'])
        self.assertEqual(read_json(snapshot / 'report_status.json')['url'], 'https://temp.sh/test/report.zip')
        with patch('w2v_v3184_aug5.delivery.ROOT', self.case), \
             patch('w2v_v3184_aug5.delivery.upload_archive', return_value='https://temp.sh/test/submission.zip'), \
             patch('w2v_v3184_aug5.reports.upload_archive', return_value='https://temp.sh/test/report.zip'), \
             patch('w2v_v3184_aug5.evaluate.export', side_effect=AssertionError('Inference must not repeat')):
            links = deliver(snapshot, self.run, args)
        self.assertEqual(links['submission_url'], 'https://temp.sh/test/submission.zip')
        self.assertEqual(links['report_url'], 'https://temp.sh/test/report.zip')
        self.assertEqual(before, self.hashes(self.run))
        protocol.write_text('\n'.join(reversed(ids)) + '\n')
        with patch('w2v_v3184_aug5.delivery.ROOT', self.case), \
             patch('w2v_v3184_aug5.reports.upload_archive', return_value='https://temp.sh/test/report.zip'), \
             patch('w2v_v3184_aug5.delivery.finish_request', side_effect=AssertionError('Changed protocol reached inference')) as finish:
            with self.assertRaises(RuntimeError):
                deliver(snapshot, self.run, args)
            finish.assert_not_called()


class SupervisorIdentityTests(unittest.TestCase):
    def test_supervisor_is_bound_to_script_run_log_owner_and_session(self):
        # Synthetic /proc only. This test never sends a signal.
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary).resolve()
            entry = proc / '123'
            root = entry / 'cwd'
            (root / 'exp').mkdir(parents=True)
            run = root / 'exp' / 'aug5'
            log = root / 'exp' / 'aug5.log'
            for name, value in (('run', run), ('log', log)):
                (root / 'exp' / ('.latest_v3184_aug5_' + name)).write_text(str(value))
            args = ['bash', str(root / 'v3184_aug5_supervise.sh'), str(log)]
            (entry / 'cmdline').write_bytes(('\0'.join(args) + '\0').encode())
            log.write_text('V3184_RUN=' + str(run) + '\n')
            with patch('os.getuid', return_value=entry.stat().st_uid, create=True), \
                 patch('os.getpgid', return_value=123, create=True):
                self.assertEqual(authenticate_supervisor(root, run, 123, proc), args)
                (entry / 'cmdline').write_bytes(b'bash\0another_script.sh\0')
                with self.assertRaises(ValueError):
                    authenticate_supervisor(root, run, 123, proc)
                (entry / 'cmdline').write_bytes(('\0'.join(args) + '\0').encode())
                log.write_text('different-run')
                with self.assertRaises(ValueError):
                    authenticate_supervisor(root, run, 123, proc)
                log.write_text('V3184_RUN=' + str(run) + '\n')
                with patch('os.getpgid', return_value=124, create=True), self.assertRaises(ValueError):
                    authenticate_supervisor(root, run, 123, proc)
                with patch('os.getuid', return_value=entry.stat().st_uid + 1, create=True), self.assertRaises(ValueError):
                    authenticate_supervisor(root, run, 123, proc)


if __name__ == '__main__':
    unittest.main()
