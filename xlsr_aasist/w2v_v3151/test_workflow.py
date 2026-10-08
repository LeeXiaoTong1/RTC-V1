"""Tiny CPU training/commit/resume/export integration; no server assets needed.

The real detector, TFCL, optimizer, scheduler, train orchestration, transactional
checkpoint and selection/export paths run. Audio I/O and evaluation metrics are
controlled fixtures so two deliberately different winners can be asserted.
"""
import copy
from contextlib import contextmanager, ExitStack, redirect_stdout
import io
from pathlib import Path
import random
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from w2v_v39.common import atomic_json, digest, read_json
from w2v_v39.metrics import GROUPS
from w2v_v3151 import train as training
from w2v_v3151 import evaluate as evaluation
from w2v_v3151.state import load_resume, load_selected, apply_candidate
from w2v_v3151.test_step import examples, model_fixture


class QuietPhase:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


class TinyPlan:
    def __init__(self, rows, batch, seed, cfg):
        self.rows, self.steps = rows, 4

    def coverage(self, epoch):
        return dict(available_sources=4, unique_sources=4, steps=self.steps,
                    excluded_missing_online=0, excluded_source_groups={})


class WorkflowFixture:
    def __init__(self, root):
        self.root = Path(root)
        torch.manual_seed(315101)
        self.base = model_fixture()
        # Exercise the saved RNG, not just a deterministic dropout-free update.
        for module in self.base.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = .1
        self.dev = [dict(id=f'{condition}:{language}:{label}',
            source_id=f'{language}:{label}', group_id=f'{language}:{label}',
            condition=condition, language=language, label=label,
            audio_sha256='fixture', split='dev')
            for condition in ('online', 'seen', 'heldout')
            for language in ('en', 'zh') for label in (0, 1)]
        self.panel = [dict(row, id=f'{part}:{row["id"]}', panel_partition=part)
                      for part in ('tune', 'audit') for row in self.dev[:4]]
        self.recipes = [dict(panel_partition=row['panel_partition'], family='fixture')
                        for row in self.panel]
        self.cfg = dict(device='cpu', amp='none', seed=73151, tfcl_heads=2, tfcl_bins=21,
            source_batch=4, epochs=1, checks_per_epoch=2, matched_fake_recall=.99,
            microbatch=12, frame_budget=1000, checkpointing=False,
            disk_margin_bytes=0, free_reserve_bytes=0, rolling_cache_bytes=0,
            maximum_new_peak_bytes=2*1024**3, objective_ramp_epochs=.25,
            tfcl_time_weight=.15, tfcl_structure_weight=.045,
            tfcl_bridge_mass=.5, tfcl_matched_mass=.5, max_grad_norm=1.,
            lr_warmup_fraction=.25, progress_min_delta=.0001,
            minimum_epochs=1, patience=10, catastrophic_weighted_drop=.1,
            workers=0, code_fingerprints={}, starting_tag='fixture_v312_last',
            parent_run=str(self.root/'parent'), parent_selected_tag='epoch_1_step_2417',
            parent_selector='best_guarded')
        for key in ('base_checkpoint', 'starting_checkpoint', 'parent_checkpoint'):
            path = self.root/(key+'.bin')
            path.write_bytes(key.encode())
            self.cfg[key] = str(path)
            self.cfg[key+'_sha256'] = digest(path)
        self.segments, self.steps_seen, self.closed = [], [], []

    def run_dir(self, name):
        run = self.root/name
        run.mkdir()
        atomic_json(run/'config.json', self.cfg)
        return run

    @staticmethod
    def stage(label):
        match = re.search(r'epoch_1_step_(\d+)', label)
        return 0 if match is None else int(match.group(1)) // 2

    def infer(self, model, rows, cfg, label):
        z = np.zeros((len(rows), 2), dtype=np.float32)
        z[:, 0] = self.stage(label)
        return z, None

    def infer_panel(self, model, rows, recipes, cfg, run, label):
        return self.infer(model, rows, cfg, label)[0]

    @staticmethod
    def measure(rows, logits, target):
        stage = int(logits[0, 0])
        score = (.90, .94, .93)[stage]
        group = dict(class_counts=[10, 10], recall=[.99, .90], auc=.98)
        return dict(complete=True, clean_f1=score, noisy_f1=score, weighted_f1=score,
            groups={name: copy.deepcopy(group) for name in GROUPS},
            matched={name: dict(real_recall=.9) for name in GROUPS})

    @staticmethod
    def panel_metrics(rows, recipes, logits, target):
        stage = int(logits[0, 0])
        f1, auc = ((.80, .90), (.805, .90), (.90, .92))[stage]
        return dict(complete=True, partition=rows[0]['panel_partition'], views=len(rows),
            macro_f1=f1, macro_auc=auc, macro_matched_real=.9,
            groups={language: dict(class_counts=[10, 10], recall=[.99, .90], auc=auc)
                    for language in ('en', 'zh')},
            families=dict(fixture=dict(class_counts=[10, 10], macro_f1=f1, auc=auc)))

    def load_model(self, cfg, device=None, training=True):
        return copy.deepcopy(self.base).to(device or cfg['device']).train(training)

    @staticmethod
    def optimizer(model, cfg, auxiliary):
        return torch.optim.AdamW([
            dict(params=[p for p in model.parameters() if p.requires_grad],
                 lr=1e-4, initial_lr=1e-4, name='tiny_detector'),
            dict(params=list(auxiliary.parameters()),
                 lr=2e-4, initial_lr=2e-4, name='tiny_auxiliary')], weight_decay=.01)

    @staticmethod
    def execution(model, auxiliary, optimizer, plan, cfg, run):
        value = dict(selected=dict(microbatch=12, frame_budget=1000, checkpointing=False))
        atomic_json(Path(run)/'execution_plan.json', value)
        return value

    @contextmanager
    def patched(self, interrupt_step=None):
        owner = self

        @contextmanager
        def bundles(cfg):
            yield dict(rows=[]), dict(rows=owner.dev)

        class Loader:
            def __init__(self, plan, cfg, run):
                self.run = Path(run).name

            def segment(self, epoch, start, stop):
                owner.segments.append((self.run, epoch, start, stop))
                for step in range(start, stop):
                    batch = examples()
                    perturbation = random.random() * .001 + np.random.random() * .001
                    for row in batch:
                        row['features'] += perturbation
                        row['fixture_step'] = step + 1
                    yield batch

            def close(self):
                owner.closed.append(self.run)

        original_step = training.train_step

        def run_step(model, aux, opt, rows, cfg, warm, audit=False):
            step = rows[0]['fixture_step']
            owner.steps_seen.append(step)
            if step == interrupt_step:
                raise RuntimeError('fixture interruption before uncommitted update')
            return original_step(model, aux, opt, rows, cfg, warm, audit=audit)

        replacements = dict(verify_inputs=lambda cfg: None, load_model=self.load_model,
            optimizer_for=self.optimizer, bundles=bundles, SourcePlan=TinyPlan,
            TrainingLoader=Loader, offline_rows=lambda *args: (self.dev[:4], {}),
            build_panel=lambda *args: (self.panel, self.recipes, {'fixture': True}),
            infer=self.infer, infer_panel=self.infer_panel, measure=self.measure,
            panel_metrics=self.panel_metrics, select_execution=self.execution,
            print_metrics=lambda *args, **kwargs: None, announce=lambda *args: None,
            Phase=QuietPhase, train_step=run_step)
        with ExitStack() as stack:
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(training, name, replacement))
            stack.enter_context(redirect_stdout(io.StringIO()))
            yield


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def assert_nested_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(actual, dict):
            self.assertEqual(set(actual), set(expected))
            for key in actual:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_committed_aliases_final_audit_and_export_identify_actual_weights(self):
        with tempfile.TemporaryDirectory(prefix='v3151_workflow_') as folder:
            f = WorkflowFixture(folder)
            run = f.run_dir('complete')
            with f.patched():
                done = training.run_experiment(f.cfg, run)
                self.assertEqual(done['committed_updates'], 4)
                self.assertEqual(done['selections'], dict(best_weighted='epoch_1_step_2',
                                                        best_guarded='epoch_1_step_4'))
                before = digest(run/'completed.json')
                audit = training.final_audit(f.cfg, run)
                self.assertEqual(audit['selected'], 'epoch_1_step_4')
                self.assertFalse(audit['used_for_model_selection'])
                self.assertEqual(before, digest(run/'completed.json'))
                self.assertEqual(training.final_audit(f.cfg, run), audit)
            self.assertEqual(f.closed, ['complete'])
            self.assertFalse((run/'validation_pending.json').exists())
            self.assertEqual(read_json(run/'startup_replay.json')['status'], 'historical_scores_unavailable')
            for kind, expected in (('best_weighted', 'epoch_1_step_2'),
                                   ('best_guarded', 'epoch_1_step_4'), ('last', 'epoch_1_step_4')):
                checkpoint, meta = load_selected(run, kind)
                self.assertEqual(meta['selected'], expected)
                self.assertEqual(meta['parent_selected_tag'], f.cfg['parent_selected_tag'])
                self.assertEqual(read_json(run/(kind+'.json'))['selected'], expected)
                self.assertIsNotNone(checkpoint['candidate'])
            state = load_resume(run/'last.pt', f.cfg)
            self.assertEqual(len(state['candidates']), 2)
            self.assertEqual(len(state['history']), 2)
            self.assertTrue(all(row['committed'] for row in state['history']))

            protocol = Path(folder)/'protocol.txt'
            protocol.write_text('sample_a\nsample_b\n', encoding='utf-8')
            selected_checkpoint, _ = load_selected(run, 'best_guarded')
            observed = []

            def export_infer(model, rows, cfg, label):
                for name, value in selected_checkpoint['candidate']['state'].items():
                    torch.testing.assert_close(dict(model.named_parameters())[name], value, atol=0, rtol=0)
                self.assertFalse(model.training)
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                observed.append(len(rows))
                return np.array([[2., 1.], [0., 2.]], dtype=np.float32), None

            with patch.object(evaluation, 'load_model', f.load_model), \
                 patch.object(evaluation, 'verify_parent', lambda cfg: None), \
                 patch.object(evaluation, 'read_protocol', lambda *args, **kwargs: [dict(id='sample_a'), dict(id='sample_b')]), \
                 patch.object(evaluation, 'infer', export_infer), redirect_stdout(io.StringIO()):
                output = Path(folder)/'submission_best_guarded'
                archive = evaluation.export(run, protocol, folder, output, device='cpu', workers=0)
            self.assertEqual(observed, [2])
            metadata = read_json(output/'submission_meta.json')
            self.assertEqual(metadata['selected'], 'epoch_1_step_4')
            self.assertEqual(metadata['parent_selector'], 'best_guarded')
            self.assertFalse(metadata['baseline_fallback'])
            self.assertEqual(metadata['zip_sha256'], digest(archive))
            with zipfile.ZipFile(archive) as handle:
                self.assertEqual(handle.namelist(), ['scores.txt'])
                lines = handle.read('scores.txt').decode().splitlines()
                self.assertEqual([line.split()[0] for line in lines], ['sample_a', 'sample_b'])

    def test_resume_replays_only_uncommitted_segment_and_exact_rng_adam_state(self):
        with tempfile.TemporaryDirectory(prefix='v3151_resume_') as folder:
            f = WorkflowFixture(folder)
            full, resumed = f.run_dir('full'), f.run_dir('resumed')
            with f.patched():
                training.run_experiment(f.cfg, full)
            with f.patched(interrupt_step=4):
                with self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
                    training.run_experiment(f.cfg, resumed)
            interrupted = load_resume(resumed/'last.pt', f.cfg)
            self.assertEqual(interrupted['cursor'], 2)
            self.assertEqual(interrupted['last_tag'], 'epoch_1_step_2')
            self.assertFalse((resumed/'completed.json').exists())
            start = len(f.steps_seen)
            with f.patched():
                done = training.run_experiment(f.cfg, resumed)
            self.assertEqual(f.steps_seen[start:], [3, 4])
            self.assertEqual(f.segments[-1], ('resumed', 0, 2, 4))
            self.assertEqual(done['committed_updates'], 4)
            left = load_resume(full/'last.pt', f.cfg)
            right = load_resume(resumed/'last.pt', f.cfg)
            for key in ('model', 'auxiliary', 'optimizer', 'rng', 'cursor', 'last_tag', 'selections', 'scores'):
                self.assert_nested_equal(left[key], right[key])
            self.assertEqual(f.closed.count('resumed'), 2)


if __name__ == '__main__':
    unittest.main()
