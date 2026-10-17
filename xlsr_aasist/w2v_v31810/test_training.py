"""Actual detector lifecycle with deterministic in-memory audio and Dev fixtures.

The separate sampler, DSP, protocol and calibration suites test their real
implementations. This suite focuses on joint integration, atomic boundaries,
optimizer/RNG resume and selected-checkpoint final audit.
"""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayout
from w2v_v3189.test_model import tiny_cfg
from .calibration import CalibrationError
from .common import RECIPE, atomic_json, fingerprint, partial_state, read_json
from .meta import train_step
from .model import Detector, optimizer_for
from .test_meta import assert_state_equal, fixture as meta_fixture
from .train import run_experiment


class FixturePlan:
    def __init__(self, rows, cfg, seed):
        self.steps = 2

    def batches(self, epoch):
        for step in range(self.steps):
            yield [dict(epoch=epoch, step=step, row=i, task_kind=('noise', 'rtc', 'joint')[step % 3])
                   for i in range(16)]

    def calibration_batches(self):
        for i in range(12):
            yield [dict(epoch=1, step=i + 100, row=j, task_kind=('noise', 'rtc', 'joint')[i // 4])
                   for j in range(16)]

    def coverage(self, epoch):
        return dict(epoch=epoch, fixture=True, steps=self.steps, classification_views=32)


class FixtureWaves:
    def __init__(self, rows, cfg):
        self.cache = {}

    def __getitem__(self, ticket):
        key = ticket['epoch'], ticket['step']
        if key not in self.cache:
            _, units = meta_fixture(copies=1)
            for unit in units:
                row = unit[0]
                row['wave'] = np.tile(row['wave'], 10)
                row['task_kind'] = ticket['task_kind']
            self.cache[key] = units
        return self.cache[key][ticket['row']]


class FixtureLoader:
    def __init__(self, dataset, sampler):
        self.dataset, self.sampler = dataset, sampler

    def __iter__(self):
        for tickets in self.sampler.batches:
            yield [self.dataset[ticket] for ticket in tickets]


def fixture_loader(dataset, cfg, *, batch_sampler):
    return FixtureLoader(dataset, batch_sampler)


def tiny_model(cfg):
    return Detector(TinyFrontend(), TinyEncoder(), cfg, TinyLayout)


def settings():
    cfg = tiny_cfg()
    protocol = dict(fixture='locked source-isolated protocol')
    cfg.update(version='3.18.10', recipe=RECIPE, experiment='E2', seed=3181001,
               device='cpu', backend_precision='fp32', epochs=2, warm_epochs=1,
               consistency='none', noisy_views_per_source=1, rtc_activity_policy='mixed',
               label_smoothing=.02, max_grad_norm=1., microbatch=8, frame_budget=9000,
               gradient_audit_interval=100, meta_beta=.5, meta_beta_ramp_epochs=2,
               lr_floor=.1, lr_warmup_updates=200,
               initialization=dict(mode='pretrained', old_detector_loaded=False),
               validation_protocol_hash=fingerprint(protocol))
    return cfg, protocol


def make_run(root, name, cfg):
    run = root / name
    run.mkdir()
    atomic_json(run / 'config.json', cfg)
    return run


class Evaluator:
    def __init__(self, test, run):
        self.test, self.run = test, run
        self.calls = []

    def __call__(self, model, protocol, cfg, *, split, selected_checkpoint=None):
        self.calls.append(split)
        model.eval()
        if split == 'audit':
            saved = torch.load(self.run / 'last.pt', weights_only=True)
            self.test.assertEqual(selected_checkpoint, saved['best_tag'])
            for name, value in partial_state(model).items():
                torch.testing.assert_close(value, saved['best_model'][name], rtol=0, atol=0)
        # Deliberately select epoch 1, not the newest epoch 2 checkpoint.
        score = .92 if not model.joint else .90
        panels = {key: dict(macro_f1=score) for key in ('official_online', 'regular_noisy', 'heldout_noisy')}
        return dict(score=score, panels=panels, groups={})


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def patches(self, stack, calibrator=None):
        stack.enter_context(patch('w2v_v31810.train.Plan', FixturePlan))
        stack.enter_context(patch('w2v_v31810.train.Waves', FixtureWaves))
        stack.enter_context(patch('w2v_v31810.train.loader', fixture_loader))
        stack.enter_context(patch('w2v_v31810.train.close'))
        stack.enter_context(patch('w2v_v31810.train.render'))
        stack.enter_context(redirect_stdout(io.StringIO()))
        def calibrated(model, optimizer, factory, cfg, **kwargs):
            self.assertTrue(model.joint)
            self.assertTrue(all(p.requires_grad for n, p in model.named_parameters()
                                if n.endswith(('.lora_a', '.lora_b'))))
            tasks = list(factory())
            self.assertEqual(len(tasks), 12)
            self.assertEqual(sum(len(task) for task in tasks), 192)
            return dict(status='complete', selected_alpha=.003, fixture=True,
                        task_count=12, task_counts=dict(noise=4, rtc=4, joint=4))
        return stack.enter_context(patch('w2v_v31810.calibration.calibrate',
                                         side_effect=calibrator or calibrated))

    def test_mid_joint_pause_resume_exact_and_final_audit_only_selected_best(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            calibration = self.patches(stack)
            root = Path(directory); cfg, protocol = settings()
            full, resumed = make_run(root, 'full', cfg), make_run(root, 'resumed', cfg)
            ev_full, ev_resume = Evaluator(self, full), Evaluator(self, resumed)
            done = run_experiment(cfg, full, [], [], protocol, model_factory=tiny_model, evaluator=ev_full)
            self.assertEqual(done['best_tag'], 'epoch_1_step_2')
            def request_after_joint(model, optimizer, units, settings, **kwargs):
                result = train_step(model, optimizer, units, settings, **kwargs)
                if kwargs['joint']:
                    atomic_json(resumed / 'stop_requested.json', dict(reason='fixture mid-joint pause'))
                return result
            with patch('w2v_v31810.train.train_step', request_after_joint):
                paused = run_experiment(cfg, resumed, [], [], protocol, model_factory=tiny_model, evaluator=ev_resume)
            self.assertEqual(paused['status'], 'paused')
            partial = torch.load(resumed / 'last.pt', weights_only=True)
            self.assertEqual((partial['epoch'], partial['next_step'], partial['cursor']), (1, 1, 3))
            self.assertEqual(partial['alpha'], .003)
            self.assertEqual(partial['epoch_accumulator']['updates'], 1)
            self.assertEqual(ev_resume.calls, ['select'])
            with (resumed / 'steps.jsonl').open('a') as stream:
                stream.write('{"update":999}\n{"broken":')
            run_experiment(cfg, resumed, [], [], protocol, model_factory=tiny_model, evaluator=ev_resume)
            a = torch.load(full / 'last.pt', weights_only=True)
            b = torch.load(resumed / 'last.pt', weights_only=True)
            for field in ('model', 'best_model', 'optimizer', 'rng', 'grad_scaler'):
                assert_state_equal(self, a[field], b[field])
            self.assertEqual(a['alpha'], b['alpha'])
            self.assertEqual(a['cursor'], b['cursor'])
            self.assertEqual(calibration.call_count, 2)  # once per fresh experiment
            self.assertEqual([json.loads(line)['update'] for line in (resumed / 'steps.jsonl').read_text().splitlines()], [1, 2, 3, 4])
            self.assertEqual(ev_resume.calls, ['select', 'select', 'audit'])
            self.assertEqual(ev_full.calls, ['select', 'select', 'audit'])
            self.assertEqual([entry['phase'] for entry in b['history']], ['head_warmup', 'joint'])
            for left, right in zip(a['history'], b['history']):
                self.assertEqual(left['training']['loss'], right['training']['loss'])
                self.assertEqual(left['training']['views'], 32)
            # A repeated completed invocation must not reopen the audit.
            run_experiment(cfg, resumed, [], [], protocol, model_factory=tiny_model, evaluator=ev_resume)
            self.assertEqual(ev_resume.calls, ['select', 'select', 'audit'])
            self.assertFalse(read_json(resumed / 'locked_audit.json')['used_for_selection'])

    def test_initialization_preflight_never_evaluates_or_calibrates(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            calibration = self.patches(stack)
            cfg, protocol = settings(); run = make_run(Path(directory), 'preflight', cfg)
            evaluator = Evaluator(self, run)
            result = run_experiment(cfg, run, [], [], protocol, model_factory=tiny_model,
                                    evaluator=evaluator, baseline_only=True)
            self.assertEqual(result['status'], 'initialization_verified')
            saved = torch.load(run / 'last.pt', weights_only=True)
            self.assertIsNone(saved['alpha'])
            self.assertIsNone(saved['best_model'])
            self.assertIsNone(saved['best_tag'])
            self.assertEqual(saved['cursor'], 0)
            self.assertEqual(calibration.call_count, 0)
            self.assertEqual(evaluator.calls, [])
            self.assertFalse(read_json(run / 'parameter_inventory.json')['pcl_state'])

    def test_calibration_failure_preserves_completed_warmup_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            def fail(*args, **kwargs):
                raise CalibrationError('No candidate passed', dict(status='failed', fixture=True))
            self.patches(stack, fail)
            cfg, protocol = settings(); run = make_run(Path(directory), 'failure', cfg)
            evaluator = Evaluator(self, run)
            with self.assertRaises(CalibrationError):
                run_experiment(cfg, run, [], [], protocol, model_factory=tiny_model, evaluator=evaluator)
            saved = torch.load(run / 'last.pt', weights_only=True)
            self.assertEqual((saved['epoch'], saved['next_step'], saved['cursor']), (1, 0, 2))
            self.assertIsNone(saved['alpha'])
            self.assertEqual(saved['best_tag'], 'epoch_1_step_2')
            failure = read_json(run / 'inner_calibration_failure.json')
            self.assertEqual(failure['details']['status'], 'failed')
            self.assertEqual(evaluator.calls, ['select'])
            self.assertFalse((run / 'completed.json').exists())

    def test_real_amp_overflow_retries_same_audio_without_consuming_update(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.patches(stack)
            cfg, protocol = settings(); cfg['experiment'] = 'E0'
            run = make_run(Path(directory), 'retry', cfg)
            evaluator = Evaluator(self, run)
            calls, identities = [], []
            def opt_with_cpu_scaler(model, settings):
                opt = optimizer_for(model, settings)
                opt._v31810_scaler = torch.amp.GradScaler('cpu', init_scale=128.)
                return opt
            def inject_once(model, opt, units, settings, **kwargs):
                identities.append(tuple(id(unit[0]['wave']) for unit in units))
                if not calls:
                    hook = model.head.aasist.out_layer.weight.register_hook(lambda g: g * float('inf'))
                    try:
                        result = train_step(model, opt, units, settings, **kwargs)
                    finally:
                        hook.remove()
                else:
                    result = train_step(model, opt, units, settings, **kwargs)
                calls.append(result)
                return result
            with patch('w2v_v31810.train.optimizer_for', opt_with_cpu_scaler), \
                 patch('w2v_v31810.train.train_step', inject_once):
                run_experiment(cfg, run, [], [], protocol, model_factory=tiny_model, evaluator=evaluator)
            self.assertEqual(len(calls), 5)
            self.assertEqual(identities[0], identities[1])
            self.assertTrue(calls[0]['amp']['skipped_update'])
            saved = torch.load(run / 'last.pt', weights_only=True)
            self.assertEqual(saved['cursor'], 4)
            self.assertEqual(saved['history'][0]['training']['amp_retry_attempts'], 1)
            self.assertEqual(sum(entry['training']['views'] for entry in saved['history']), 64)
            first = json.loads((run / 'steps.jsonl').read_text().splitlines()[0])
            self.assertEqual(len(first['amp_retry_attempts']), 1)
            self.assertGreater(first['encoded_views'], first['views'])

    def test_repeated_overflow_fails_before_success_cursor_advances(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.patches(stack)
            cfg, protocol = settings(); run = make_run(Path(directory), 'repeated', cfg)
            failed = dict(loss=None, encoded_views=16,
                          amp=dict(enabled=True, skipped_update=True, scale=1.))
            with patch('w2v_v31810.train.train_step', return_value=failed) as attempted:
                with self.assertRaisesRegex(FloatingPointError, 'Repeated mixed-precision'):
                    run_experiment(cfg, run, [], [], protocol, model_factory=tiny_model,
                                   evaluator=Evaluator(self, run))
            self.assertEqual(attempted.call_count, 4)
            saved = torch.load(run / 'last.pt', weights_only=True)
            self.assertEqual(saved['cursor'], 0)
            self.assertIsNone(saved['best_tag'])
            self.assertEqual(len(read_json(run / 'precision_failure.json')['attempts']), 4)


if __name__ == '__main__':
    unittest.main()
