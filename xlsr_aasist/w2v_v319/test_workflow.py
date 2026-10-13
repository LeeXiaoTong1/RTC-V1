"""Native tiny-model training, exact interrupted resume, and selected export.

The real w2v-BERT/MultiConv detector, parameter-free auxiliary loss, Adam,
schedule and atomic state run. Audio and validation scores are controlled so
the default best checkpoint deliberately differs from LAST.
"""
import copy
from contextlib import contextmanager, ExitStack, redirect_stdout
import io
import json
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
from . import train as training, evaluate as evaluation
from .config import parser
from .state import load_resume, load_selected, identity
from .test_step import examples, model_fixture


class QuietPhase:
    def __init__(self, *args, **kwargs):
        pass
    def update(self, *args, **kwargs):
        pass


class TinyPlan:
    def __init__(self, rows, batch, seed, cfg):
        self.rows, self.steps = rows, 2
    def coverage(self, epoch):
        return dict(available_sources=4, unique_sources=4, steps=self.steps, missing_online_sources=0)


class WorkflowFixture:
    def __init__(self, root):
        self.root = Path(root)
        torch.manual_seed(31901)
        self.base = model_fixture()
        # Make equality depend on checkpoint RNG restoration, not just weights.
        for module in self.base.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = .1
        self.dev = [dict(id=f'{condition}:{language}:{label}', source_id=f'{language}:{label}',
            group_id=f'{language}:{label}', condition=condition, language=language, label=label,
            audio_sha256='fixture', split='dev')
            for condition in ('online', 'seen', 'heldout') for language in ('en', 'zh') for label in (0, 1)]
        self.panel = [dict(row, id=f'{part}:{row["id"]}', panel_partition=part)
                      for part in ('tune', 'audit') for row in self.dev[:4]]
        self.recipes = [dict(panel_partition=row['panel_partition'], family='fixture') for row in self.panel]
        self.cfg = dict(version='3.19', recipe='v319_received_tfcl_en6_v1', device='cpu', amp='none',
            seed=731901, tfcl_heads=2, tfcl_bins=13, alignment_max_bins=13,
            alignment_local_band_fraction=.2, alignment_min_frames=4, alignment_temperature=.07,
            source_batch=4, group_quotas=[1, 1, 1, 1], epochs=2, checks_per_epoch=1,
            matched_fake_recall=.99, microbatch=12, frame_budget=1000, checkpointing=False,
            disk_margin_bytes=0, free_reserve_bytes=0, rolling_cache_bytes=0,
            maximum_new_peak_bytes=2*1024**3, objective_ramp_epochs=0., warm_epochs=0,
            tfcl_time_weight=.15, tfcl_structure_weight=.045, tfcl_bridge_mass=.2,
            tfcl_matched_mass=.8, max_grad_norm=1., online_ce_mass=.3,
            reference_ce_mass=.1, initial_noisy_ce_mass=.6, main_noisy_ce_mass=.6,
            consistency='matched-bridge', encoder_lr=1e-6, head_lr=1e-5, adapter_lr=2e-5,
            output_lr=2e-5, layer_decay=.8,
            lr_warmup_fraction=0., lr_floor=.1, progress_min_delta=.0001,
            minimum_epochs=2, patience=0, catastrophic_weighted_drop=.1,
            workers=0, code_fingerprints={}, starting_tag='fixture_v312_last',
            parent_run=str(self.root/'parent'), parent_selected_tag='epoch_1_step_2417', parent_selector='best_guarded')
        for key in ('base_checkpoint', 'starting_checkpoint', 'parent_checkpoint'):
            path = self.root/(key+'.bin')
            path.write_bytes(key.encode())
            self.cfg[key] = str(path)
            self.cfg[key+'_sha256'] = digest(path)
        self.segments, self.steps_seen, self.closed, self.warm_seen, self.lr_seen = [], [], [], [], []

    def run_dir(self, name):
        run = self.root/name
        run.mkdir()
        atomic_json(run/'config.json', self.cfg)
        return run

    @staticmethod
    def stage(label):
        match = re.search(r'epoch_(\d+)_step_2', label)
        return 0 if match is None else int(match.group(1))

    def infer(self, model, rows, cfg, label):
        z = np.zeros((len(rows), 2), dtype=np.float32)
        z[:, 0] = self.stage(label)
        return z, None

    def infer_panel(self, model, rows, recipes, cfg, run, label):
        return self.infer(model, rows, cfg, label)[0]

    @staticmethod
    def measure(rows, logits, target):
        score = (.90, .94, .93)[int(logits[0, 0])]
        group = dict(class_counts=[10, 10], recall=[.99, .90], auc=.98)
        return dict(complete=True, clean_f1=score, noisy_f1=score, weighted_f1=score,
            groups={name: copy.deepcopy(group) for name in GROUPS},
            matched={name: dict(real_recall=.9) for name in GROUPS})

    @staticmethod
    def panel_metrics(rows, recipes, logits, target):
        f1, auc = ((.80, .90), (.805, .90), (.90, .92))[int(logits[0, 0])]
        return dict(complete=True, partition=rows[0]['panel_partition'], views=len(rows),
            macro_f1=f1, macro_auc=auc, macro_matched_real=.9,
            groups={language: dict(class_counts=[10, 10], recall=[.99, .90], auc=auc) for language in ('en', 'zh')},
            families=dict(fixture=dict(class_counts=[10, 10], macro_f1=f1, auc=auc)))

    def load_model(self, cfg, device=None, training=True):
        return copy.deepcopy(self.base).to(device or cfg['device']).train(training)

    @staticmethod
    def optimizer(model, cfg, auxiliary):
        if list(auxiliary.parameters()):
            raise AssertionError('V3.19 consistency must not adapt a discarded auxiliary module')
        return torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.requires_grad],
            lr=1e-4, initial_lr=1e-4, name='tiny_detector')], weight_decay=.01)

    @staticmethod
    def execution(model, auxiliary, optimizer, plan, cfg, run):
        result = dict(selected=dict(microbatch=12, frame_budget=1000, checkpointing=False))
        atomic_json(Path(run)/'execution_plan.json', result)
        return result

    @contextmanager
    def patched(self, interrupt_step=None):
        owner = self
        @contextmanager
        def bundles(cfg):
            yield dict(rows=[]), dict(rows=owner.dev)
        class Loader:
            def __init__(self, plan, cfg, run):
                self.run, self.steps = Path(run).name, plan.steps
            def segment(self, epoch, start, stop):
                owner.segments.append((self.run, epoch, start, stop))
                for step in range(start, stop):
                    batch = examples()
                    perturbation = random.random()*.001 + np.random.random()*.001
                    for row in batch:
                        row['features'] += perturbation
                        row['fixture_step'] = epoch*self.steps+step+1
                    yield batch
            def close(self):
                owner.closed.append(self.run)
        original_step = training.train_step
        def step(model, auxiliary, optimizer, rows, cfg, warm, audit=False):
            current = rows[0]['fixture_step']
            owner.steps_seen.append(current)
            if current == interrupt_step:
                raise RuntimeError('fixture interruption before uncommitted update')
            owner.warm_seen.append(warm)
            owner.lr_seen.append(optimizer.param_groups[0]['lr'])
            return original_step(model, auxiliary, optimizer, rows, cfg, warm, audit=audit)
        replacements = dict(verify_inputs=lambda cfg: None, load_model=self.load_model,
            optimizer_for=self.optimizer, bundles=bundles, SourcePlan=TinyPlan, TrainingLoader=Loader,
            offline_rows=lambda *args: (self.dev[:4], {}),
            build_panel=lambda *args: (self.panel, self.recipes, {'fixture': True}),
            infer=self.infer, infer_panel=self.infer_panel, measure=self.measure,
            panel_metrics=self.panel_metrics, select_execution=self.execution,
            print_metrics=lambda *args, **kwargs: None, announce=lambda *args: None,
            Phase=QuietPhase, train_step=step)
        with ExitStack() as stack:
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(training, name, replacement))
            stack.enter_context(redirect_stdout(io.StringIO()))
            yield


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def equal_state(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.equal_state(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal_state(a, b)
        else:
            self.assertEqual(left, right)

    def test_default_five_epochs_first_update_has_full_rate_and_no_preheat(self):
        args = parser().parse_args([])
        self.assertEqual(args.epochs, 5)
        self.assertFalse(hasattr(args, 'warm_epochs'))
        p = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([dict(params=[p], lr=1e-4, initial_lr=1e-4)])
        training.schedule(optimizer, 0, 10, dict(lr_floor=.1))
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        training.schedule(optimizer, 9, 10, dict(lr_floor=.1))
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-5)

    def test_training_best_not_last_and_default_export_uses_exact_selected_model(self):
        with tempfile.TemporaryDirectory(prefix='v319_workflow_') as folder:
            f = WorkflowFixture(folder)
            run = f.run_dir('complete')
            with f.patched():
                done = training.run_experiment(f.cfg, run)
                self.assertEqual(done['committed_updates'], 4)
                self.assertEqual(done['selections'], dict(best_weighted='epoch_1_step_2', best_guarded='epoch_2_step_2'))
                before = digest(run/'completed.json')
                audit = training.final_audit(f.cfg, run)
                self.assertFalse(audit['used_for_model_selection'])
                self.assertEqual(before, digest(run/'completed.json'))
                self.assertEqual(training.final_audit(f.cfg, run), audit)
            self.assertEqual(f.closed, ['complete'])
            self.assertEqual(f.warm_seen, [1.]*4)
            self.assertEqual(f.lr_seen[0], 1e-4)
            self.assertAlmostEqual(f.lr_seen[-1], 1e-5)
            self.assertFalse((run/'validation_pending.json').exists())
            state = load_resume(run/'last.pt', f.cfg)
            self.assertEqual(len(state['history']), 2)
            self.assertEqual(state['auxiliary'], {})
            self.assertEqual(len(state['candidates']), 2)
            self.assertTrue(all(r['committed'] and r['phase']=='received_domain_joint' for r in state['history']))
            checkpoint, meta = load_selected(run, 'best_weighted')
            self.assertEqual(meta['selected'], 'epoch_1_step_2')
            self.assertNotEqual(meta['selected'], state['last_tag'])
            self.assertTrue(any(not torch.equal(value, state['model'][name])
                                for name, value in checkpoint['candidate']['state'].items()))
            protocol = Path(folder)/'protocol.txt'
            protocol.write_text('sample_a\nsample_b\n', encoding='utf-8')
            def infer(model, rows, cfg, label):
                for name, value in checkpoint['candidate']['state'].items():
                    torch.testing.assert_close(dict(model.named_parameters())[name], value, atol=0, rtol=0)
                self.assertFalse(model.training)
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                return np.array([[2., 1.], [0., 2.]], dtype=np.float32), None
            output = Path(folder)/'submission_best'
            with patch.object(evaluation, 'verify_parent', lambda cfg: None), \
                 patch.object(evaluation, 'read_protocol', lambda *a, **k: [dict(id='sample_a'), dict(id='sample_b')]), \
                 patch.object(evaluation, 'infer', infer), redirect_stdout(io.StringIO()):
                archive = evaluation.export(run, protocol, folder, output, device='cpu', workers=0, model_factory=f.load_model)
            saved = read_json(output/'submission_meta.json')
            self.assertEqual(saved['selected'], 'epoch_1_step_2')
            self.assertEqual(saved['checkpoint_kind'], 'best_weighted')
            self.assertEqual(saved['parent_selector'], 'best_guarded')
            self.assertEqual(saved['id_order_sha256'], identity(['sample_a', 'sample_b']))
            self.assertEqual(saved['zip_sha256'], digest(archive))
            self.assertFalse(saved['baseline_fallback'])
            with zipfile.ZipFile(archive) as handle:
                self.assertEqual(handle.namelist(), ['scores.txt'])
                lines = handle.read('scores.txt').decode().splitlines()
                self.assertEqual([line.split()[0] for line in lines], ['sample_a', 'sample_b'])
            with self.assertRaisesRegex(ValueError, 'empty'):
                evaluation.export(run, protocol, folder, output, device='cpu', model_factory=f.load_model)

    def test_resume_discards_partial_epoch_restores_exact_rng_adam_and_weights(self):
        with tempfile.TemporaryDirectory(prefix='v319_resume_') as folder:
            f = WorkflowFixture(folder)
            full, resumed = f.run_dir('full'), f.run_dir('resumed')
            with f.patched():
                training.run_experiment(f.cfg, full)
            with f.patched(interrupt_step=4):
                with self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
                    training.run_experiment(f.cfg, resumed)
            partial = load_resume(resumed/'last.pt', f.cfg)
            self.assertEqual(partial['cursor'], 2)
            self.assertEqual(partial['last_tag'], 'epoch_1_step_2')
            self.assertFalse((resumed/'completed.json').exists())
            start = len(f.steps_seen)
            with f.patched():
                done = training.run_experiment(f.cfg, resumed)
            self.assertEqual(f.steps_seen[start:], [3, 4])
            self.assertEqual(f.segments[-1], ('resumed', 1, 0, 2))
            self.assertEqual(done['committed_updates'], 4)
            left, right = load_resume(full/'last.pt', f.cfg), load_resume(resumed/'last.pt', f.cfg)
            for key in ('model', 'auxiliary', 'optimizer', 'rng', 'cursor', 'last_tag', 'selections', 'scores'):
                self.equal_state(left[key], right[key])
            self.assertEqual(f.closed.count('resumed'), 2)
            rows = [json.loads(line) for line in (resumed/'steps.jsonl').read_text().splitlines()]
            self.assertEqual([row['update'] for row in rows], [1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
