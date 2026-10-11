"""CPU behavior tests; real AASIST, tiny encoder, native audio boundary mocked."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from torch import nn
from w2v_v318.test_core import tiny_cfg, tiny_model, make_fixture, NativeBoundary
from w2v_v318.common import (seed_all, partial_state, atomic_json, read_json, digest,
    capture_rng, restore_rng)
from w2v_v318.step import train_step
from w2v_v318.model import optimizer_for
from w2v_v318.train import run_experiment
from v318_model_audit.measurements import (StepObserver, bn_buffers, recalibration_mode,
    check_bn_only, adapter_delta, tensor_digest)
from v318_model_audit.runner import execute, train_batches, reset_copy, checkpoint
from v318_model_audit.report import render, package, REPORT_FILES


class AuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); seed_all(71)

    def test_bn_changes_only_buffers_and_restores_modes_on_error(self):
        model = tiny_model(tiny_cfg()); model.set_phase(True); model.train()
        before = partial_state(model); original_modes = [m.training for m in model.modules()]
        params = tensor_digest(dict(model.named_parameters()))
        with recalibration_mode(model):
            self.assertFalse(any(m.training for m in model.modules() if isinstance(m, nn.Dropout)))
            self.assertTrue(all(m.training for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)))
            model([np.random.randn(8000).astype(np.float32) for _ in range(3)])
        result = check_bn_only(before, partial_state(model), bn_buffers(model))
        self.assertGreater(result['changed_count'], 0)
        self.assertEqual(params, tensor_digest(dict(model.named_parameters())))
        self.assertEqual(original_modes, [m.training for m in model.modules()])
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with recalibration_mode(model): raise RuntimeError('fixture')
        self.assertEqual(original_modes, [m.training for m in model.modules()])
        self.assertTrue(all(m.momentum == .1 for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)))

    def test_bn_guard_detects_weight_change(self):
        model = tiny_model(tiny_cfg()); before = partial_state(model); after = copy.deepcopy(before)
        key = next(k for k in after if k.endswith('lora_a')); after[key].add_(1)
        with self.assertRaisesRegex(RuntimeError, 'changed weights'):
            check_bn_only(before, after, bn_buffers(model))

    def test_effective_adapter_update_matches_dense_product(self):
        a = torch.randn(3, 17, dtype=torch.float64); b = torch.randn(12, 3, dtype=torch.float64)
        a1 = a+torch.randn_like(a)*1e-4; b1 = b+torch.randn_like(b)*1e-4
        expected = float((2*(b1@a1-b@a)).norm())
        self.assertAlmostEqual(adapter_delta(a,b,a1,b1,2), expected, places=8)
        self.assertEqual(adapter_delta(a,torch.zeros_like(b),a,b,2), float((2*b@a).norm()))

    def test_observer_preserves_exact_real_optimizer_update_and_rng(self):
        with tempfile.TemporaryDirectory() as d, patch('w2v_v318.data.Engines', NativeBoundary):
            cfg, train, _ = make_fixture(Path(d)); model = tiny_model(cfg); model.set_phase(True)
            batches, _ = train_batches(train, cfg, cfg['epochs'], 1)
            units = next(iter(batches))
            clone = copy.deepcopy(model)
            first, second = optimizer_for(model,cfg), optimizer_for(clone,cfg)
            rng = capture_rng()
            expected = train_step(model, first, units, cfg, .25); expected_rng = capture_rng()
            restore_rng(rng); observer = StepObserver(clone, second)
            original_clip = torch.nn.utils.clip_grad_norm_
            with observer.active(): actual = train_step(clone, second, units, cfg, .25)
            self.assertIs(torch.nn.utils.clip_grad_norm_, original_clip)
            result = observer.result()
            self.assertEqual(expected, actual)
            for name, value in partial_state(model).items(): self.assertTrue(torch.equal(value,partial_state(clone)[name]),name)
            for index, state in first.state_dict()['state'].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, second.state_dict()['state'][index][name]))
            self.assertTrue(torch.equal(expected_rng['torch'], capture_rng()['torch']))
            for group in result['gradients'].values(): self.assertLessEqual(group['post'], group['pre']+1e-8)
            self.assertGreater(result['updates']['lora']['norm'],0)
            self.assertTrue(result['adapters'])

    def test_observer_restores_clip_and_hooks_when_step_fails(self):
        model = tiny_model(tiny_cfg()); opt = optimizer_for(model,tiny_cfg())
        original = torch.nn.utils.clip_grad_norm_
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with StepObserver(model,opt).active(): raise RuntimeError('fixture')
        self.assertIs(torch.nn.utils.clip_grad_norm_, original)
        self.assertEqual(len(opt._optimizer_step_pre_hooks),0)
        self.assertEqual(len(opt._optimizer_step_post_hooks),0)

    def test_full_pipeline_independent_copies_original_preserved_and_metrics_only_zip(self):
        with tempfile.TemporaryDirectory() as d, patch('w2v_v318.data.Engines', NativeBoundary), contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); cfg, train, dev = make_fixture(root)
            cfg.update(epochs=2, warm_epochs=1)
            run = root/'source'; run.mkdir()
            atomic_json(run/'train_rows.json',train); atomic_json(run/'dev_rows.json',dev)
            cfg['saved_manifests'] = {n:digest(run/n) for n in ('train_rows.json','dev_rows.json')}
            atomic_json(run/'config.json',cfg)
            run_experiment(cfg,run,train,dev,tiny_model)
            before = {str(p):digest(p) for p in run.rglob('*') if p.is_file()}
            out = root/'audit'; out.mkdir()
            request = dict(run=str(run), workers=0, device='cpu', bn_batches=2, update_steps=2)
            result = execute(request,out,model_factory=tiny_model,verify=False)
            self.assertEqual(result['status'],'complete',result.get('errors'))
            self.assertTrue(result['original_unchanged'])
            self.assertLess(result['baseline_reproduction']['max_logit_difference'],1e-6)
            summary = read_json(out/'update_summary.json')
            self.assertEqual(result['source_partial_sha256'], summary['initial_partial_sha256'])
            self.assertTrue(summary['bn_buffers_unchanged'])
            self.assertEqual(before, {str(p):digest(p) for p in run.rglob('*') if p.is_file()})
            (out/'secret.wav').write_bytes(b'not audio')
            (out/'last.pt').write_bytes(b'not a model')
            render(out); archive = package(out)
            with zipfile.ZipFile(archive) as z:
                self.assertIn('analysis.html', z.namelist()); self.assertIn('updates.csv',z.namelist())
                self.assertIn('groups.csv',z.namelist())
                self.assertTrue(set(z.namelist()) <= set(REPORT_FILES))
                self.assertFalse(any(n.endswith(('.pt','.wav')) for n in z.namelist()))
            # A failed BN branch still restores the original state before updates.
            failed = root/'failed-bn'; failed.mkdir()
            def fail_bn(model,*args):
                with torch.no_grad(): next(model.head.parameters()).add_(99)
                raise RuntimeError('injected BN failure')
            with patch('v318_model_audit.runner.recalibrate', fail_bn), contextlib.redirect_stderr(io.StringIO()):
                failure = execute(request,failed,model_factory=tiny_model,verify=False)
            self.assertEqual(failure['phases']['bn'],'failed')
            self.assertEqual(failure['phases']['updates'],'passed')
            self.assertEqual((out/'updates.jsonl').read_bytes(), (failed/'updates.jsonl').read_bytes())
            # Reject source tampering before model load.
            (run/'config.json').write_text((run/'config.json').read_text().replace('"epochs": 2','"epochs": 9'))
            with self.assertRaisesRegex(ValueError,'identity mismatch'): checkpoint(run)
            atomic_json(run/'config.json',cfg)
            with (run/'dev_rows.json').open('a') as f: f.write(' ')
            with self.assertRaises(ValueError): checkpoint(run)

    def test_invalid_output_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            with self.assertRaisesRegex(ValueError,'outside the training run'):
                execute(dict(run=str(run)),run,verify=False)

    def test_viewer_ctrl_c_does_not_signal_worker(self):
        from v318_model_audit.__main__ import watch
        with tempfile.TemporaryDirectory() as d, patch('time.sleep',side_effect=KeyboardInterrupt), patch('os.kill') as kill:
            self.assertEqual(watch(Path(d)),0)
            kill.assert_not_called()

    def test_viewer_detects_exited_or_reused_worker_pid(self):
        from v318_model_audit.__main__ import watch
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            out = Path(d); atomic_json(out/'process.json',dict(pid=123,start_ticks='old'))
            with patch('v318_model_audit.__main__.os',SimpleNamespace(name='posix')):
                for identity in (None,'new'):
                    with patch('v318_model_audit.__main__.process_identity',return_value=identity):
                        self.assertEqual(watch(out),1)

    def test_detached_launch_has_no_stdin_and_separate_tracking(self):
        import os
        import subprocess
        from v318_model_audit.__main__ import launch
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); source = root/'source'; source.mkdir()
            (source/'last.pt').write_bytes(b'fixture'); (source/'completed.json').write_text('{}')
            # Stub only OS/process boundaries; no native process runs on Windows.
            fake_os = SimpleNamespace(name='posix',environ=os.environ,replace=os.replace)
            args = SimpleNamespace(run=str(source),workers=4,bn_batches=128,update_steps=100,
                                   device='cuda:0',upload_temp=True,no_watch=True)
            with patch('v318_model_audit.__main__.ROOT',root), patch('v318_model_audit.__main__.LATEST',root/'latest-audit'), \
                 patch('v318_model_audit.__main__.__file__',str(root/'package'/'__main__.py')), \
                 patch('v318_model_audit.__main__.os',fake_os), \
                 patch('v318_model_audit.__main__.process_identity',return_value='started'), \
                 patch('subprocess.Popen',return_value=SimpleNamespace(pid=123)) as popen, \
                 patch('pathlib.Path.read_bytes',return_value=b'fixture code'):
                self.assertEqual(launch(args),0)
            kwargs = popen.call_args.kwargs
            self.assertTrue(kwargs['start_new_session']); self.assertTrue(kwargs['close_fds'])
            self.assertEqual(kwargs['stdin'],subprocess.DEVNULL)
            self.assertEqual(kwargs['stderr'],subprocess.STDOUT)
            self.assertTrue((root/'latest-audit').is_file())
            self.assertFalse((root/'exp'/'.latest_v318_run').exists())

    def test_upload_failure_retains_report_and_retry_needs_finished_audit(self):
        from v318_model_audit.__main__ import finish_report, main
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            out = Path(d)/'audit'; out.mkdir()
            atomic_json(out/'result.json',dict(status='failed',errors=['fixture']))
            with patch('w2v_aasist.launch.upload_archive',side_effect=RuntimeError('network fixture')):
                finish_report(out,True)
            self.assertTrue((out.parent/'audit_report.zip').is_file())
            self.assertEqual(read_json(out/'upload_status.json')['status'],'failed')
            with patch('sys.argv',['audit','--upload-only',str(out)]), self.assertRaisesRegex(ValueError,'finished audit'):
                main()


if __name__ == '__main__': unittest.main()
