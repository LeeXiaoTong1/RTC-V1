"""Small real-optimizer tests for V5.3 commits, resume and final-data boundaries."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import train, workflow
from .common import RECIPE, atomic_json, digest, fingerprint, partial_state
from w2v_v5.processing import policy
from .fusion import AdaptiveLayerFusion
from w2v_v3183r2.fusion import LayerFusion


class TinyDetector(torch.nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.base = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        self.encoder.prefix = torch.nn.Module()
        self.encoder.prefix.lora_a = torch.nn.Parameter(torch.ones(1, 2), requires_grad=False)
        self.encoder.prefix.lora_b = torch.nn.Parameter(torch.ones(2, 1), requires_grad=False)
        cfg = configuration() if cfg is None else cfg
        self.fusion = AdaptiveLayerFusion(LayerFusion(2, 3, 2, False),
            mode=cfg['adaptive_fusion_mode'], width=cfg['adaptive_fusion_width'], seed=cfg['adaptive_fusion_seed'])
        self.head = torch.nn.Linear(2, 1)

    def forward(self, x):
        features = self.fusion((x.square()+.4)[:,None],
            {1: x[:,None], 2: (x*.3+.1)[:,None]},
            condition=torch.stack((x, x.square()+.2), dim=1), lengths=[2]*len(x))
        return self.head(torch.tanh(features[:,0] + self.encoder.base))


class TinyPlan:
    def __init__(self, steps):
        self.steps = steps

    def batches(self, epoch):
        return [[epoch, i] for i in range(self.steps)]

    def coverage(self, epoch):
        return dict(epoch=epoch, steps=self.steps)


class TinyLoader:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        yield from self.sampler.batches


def configuration(stage='screen'):
    return dict(version='5.3', recipe=RECIPE, v5_stage=stage, lora_layers=16,
        retained_lora_layers=60, experiment='E0', robust_arm='baseline',
        seed=123, sampling_seed=123, epochs=2, v5_tail_snapshots=4, v5_tail_start=.75,
        adaptive_fusion_mode='dynamic', adaptive_fusion_width=8, adaptive_fusion_seed=53001,
        adaptive_fusion_condition='frozen_frontend_masked_mean_std_v1',
        checkpoint_interval=2, workers=0, device='cpu', backend_precision='fp32',
        classification_ce_weights=dict(online=.25, clean_rtc=.25, noisy=.5),
        v5_input=policy(), training_recipe_base='5.1', contrastive_weight=0.,
        v53_parent_pcl_weights=dict(noise=.15, platform=.045), pcl_noise_weight=.15, pcl_platform_weight=.045)


def initialize_tiny(model, cfg):
    with torch.no_grad():
        for index, (name, parameter) in enumerate(model.named_parameters()):
            if name == 'encoder.base' or name.startswith('fusion.conditioner.'):
                continue  # Frozen pretrained base is reconstructed by factory.
            parameter.fill_((index + 1) * .04)
    return partial_state(model)


def optimization_step(model, optimizer, ticket, cfg, audit=False):
    if cfg.get('_candidate_diagnostics') is not False:
        raise AssertionError('V5.3 re-enabled unused receiver candidate scoring')
    if cfg['classification_ce_weights'] != dict(online=.25, clean_rtc=.25, noisy=.5):
        raise AssertionError('V5.3 CE allocation changed')
    # Exercise Python, NumPy and Torch RNG preservation plus optimizer momentum.
    x = torch.rand(1, 2) + np.random.random() * .1 + random.random() * .1
    target = torch.tensor([[.2 + .03 * ticket[1]]])
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - target).square().mean()
    loss.backward()
    optimizer.step()
    return dict(loss=float(loss.detach()), classification_loss=float(loss.detach()),
        amp=dict(skipped_update=False))


class StageLifecycleTests(unittest.TestCase):
    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(train, 'optimizer_for',
            side_effect=lambda model, cfg: torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad], lr=.03, momentum=.8)))
        stack.enter_context(patch.object(train, 'make_plan', side_effect=lambda rows, cfg: TinyPlan(8)))
        stack.enter_context(patch.object(train, 'Waves', side_effect=lambda rows, cfg: rows))
        stack.enter_context(patch.object(train, 'loader',
            side_effect=lambda dataset, cfg, batch_sampler: TinyLoader(batch_sampler)))
        stack.enter_context(patch.object(train, 'close'))
        stack.enter_context(patch.object(train, 'schedule'))
        stack.enter_context(patch.object(train, 'adapter_inventory', return_value={'test': True}))
        return stack

    def run_tiny(self, root, cfg=None, step=optimization_step, initializer=initialize_tiny):
        return train.run_stage(cfg or configuration(), Path(root)/'stage', [dict(id='train')],
            model_factory=TinyDetector, step_fn=step, initializer=initializer)

    def assert_models_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertTrue(torch.equal(left[name], right[name]), name)

    def test_pause_resume_preserves_full_detector_optimizer_rng_and_tail_average(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder, self.patches():
            root = Path(folder)
            snapshots = []
            def record(model, optimizer, ticket, cfg, audit=False):
                stats = optimization_step(model, optimizer, ticket, cfg, audit)
                snapshots.append(partial_state(model))
                return stats
            _, reference = self.run_tiny(root/'reference', step=record)
            stop_root = root/'paused'
            calls = []
            def pause(model, optimizer, ticket, cfg, audit=False):
                stats = optimization_step(model, optimizer, ticket, cfg, audit)
                calls.append(ticket)
                if len(calls) == 13:
                    (stop_root/'STOP_REQUESTED').touch()
                return stats
            _, first = self.run_tiny(stop_root, step=pause)
            self.assertEqual(first['status'], 'paused')
            self.assertEqual(first['cursor'], 13)
            self.assertEqual(first['tail_count'], 1)
            self.assertEqual((first['epoch'], first['next_step']), (1, 5))
            (stop_root/'STOP_REQUESTED').unlink()
            with patch.object(train, 'initialize', side_effect=AssertionError('parent reinitialized')):
                _, resumed = self.run_tiny(stop_root, initializer=lambda *args: self.fail('resume initialized parent'))
            self.assertEqual(resumed['status'], 'complete')
            self.assertEqual(resumed['cursor'], 16)
            self.assertEqual(resumed['tail_count'], 4)
            self.assert_models_equal(reference['model'], resumed['model'])
            self.assertNotIn('contrastive_bank', resumed)
            self.assert_models_equal(reference['tail_model'], resumed['tail_model'])
            self.assertTrue(torch.equal(resumed['model']['encoder.prefix.lora_a'],
                snapshots[0]['encoder.prefix.lora_a']))
            expected = sum(s['head.weight'].double() for s in snapshots[-4:]) / 4
            torch.testing.assert_close(resumed['tail_model']['head.weight'], expected.float(), rtol=1e-6, atol=1e-8)
            lines = [json.loads(s) for s in (stop_root/'stage'/'steps.jsonl').read_text().splitlines()]
            self.assertEqual([r['update'] for r in lines], list(range(1, 17)))

    def test_crash_replays_only_uncommitted_updates_and_trims_stale_log(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder, self.patches():
            root = Path(folder)
            _, reference = self.run_tiny(root/'reference')
            seen = []
            def crashing(model, optimizer, ticket, cfg, audit=False):
                seen.append(ticket)
                if len(seen) == 6:
                    raise KeyboardInterrupt('simulated abrupt termination')
                return optimization_step(model, optimizer, ticket, cfg, audit)
            with self.assertRaises(KeyboardInterrupt):
                self.run_tiny(root/'crash', step=crashing)
            saved = torch.load(root/'crash'/'stage'/'last.pt', weights_only=True)
            self.assertEqual(saved['cursor'], 4)
            _, resumed = self.run_tiny(root/'crash')
            self.assert_models_equal(reference['model'], resumed['model'])
            self.assertNotIn('contrastive_bank', resumed)
            self.assert_models_equal(reference['tail_model'], resumed['tail_model'])
            lines = [json.loads(s) for s in (root/'crash'/'stage'/'steps.jsonl').read_text().splitlines()]
            self.assertEqual([r['update'] for r in lines], list(range(1, 17)))

    def test_amp_retry_is_not_a_committed_update_or_extra_tail_snapshot(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder, self.patches():
            calls = []
            def retry(model, optimizer, ticket, cfg, audit=False):
                calls.append(ticket)
                if len(calls) == 1:
                    return dict(loss=None, classification_loss=None, amp=dict(skipped_update=True))
                return optimization_step(model, optimizer, ticket, cfg, audit)
            _, state = self.run_tiny(folder, step=retry)
            self.assertEqual(len(calls), 17)
            self.assertEqual(calls[0], calls[1])
            self.assertEqual(state['cursor'], 16)
            self.assertEqual(state['tail_count'], 4)

    def test_complete_stage_does_not_retrain_and_changed_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder, self.patches():
            _, done = self.run_tiny(folder)
            _, again = self.run_tiny(folder, step=lambda *args, **kwargs: self.fail('completed stage retrained'))
            self.assert_models_equal(done['model'], again['model'])
            with self.assertRaisesRegex(ValueError, 'incompatible resume'):
                self.run_tiny(folder, dict(configuration(), sampling_seed=999))


class FinalBoundaryTests(unittest.TestCase):
    def make_locked_run(self, root):
        cfg = configuration()
        (root/'screen').mkdir(parents=True)
        (root/'screen'/'last.pt').write_bytes(b'screen-checkpoint')
        lock = dict(base_config=fingerprint(cfg), screen_checkpoint=digest(root/'screen'/'last.pt'),
            input=policy(), weights=dict(name='anchor_tail_50', source='tail', parent_weight=.5))
        atomic_json(root/'recipe_lock.json', lock)
        final_cfg = dict(cfg, v5_stage='final',
            v5_recipe_lock_sha256=digest(root/'recipe_lock.json'))
        (root/'final').mkdir()
        atomic_json(root/'final'/'config.json', final_cfg)
        model = TinyDetector()
        initialize_tiny(model, cfg)
        parent = {k: v for k,v in partial_state(model).items() if not k.startswith('fusion.conditioner.')}
        trained = partial_state(model)
        trained['head.weight'] += .2
        state = dict(status='complete', model=trained, tail_model=trained)
        cfg['parent_init'] = dict(run=str(root/'parent'))
        # Freeze the parent binding in both the recipe and final configuration.
        lock['base_config'] = fingerprint(cfg)
        atomic_json(root/'recipe_lock.json', lock)
        final_cfg.update(parent_init=cfg['parent_init'],
            v5_recipe_lock_sha256=digest(root/'recipe_lock.json'))
        atomic_json(root/'final'/'config.json', final_cfg)
        return cfg, final_cfg, model, parent, state

    def test_final_fit_never_evaluates_or_reselects_after_dev_enters_train(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            root = Path(folder)
            cfg, final_cfg, model, parent, state = self.make_locked_run(root)
            rows = [dict(id='official-dev', split='train', v5_origin_split='dev')]
            with patch.object(workflow, 'read_stage', return_value=(final_cfg, rows)), \
                 patch.object(workflow, 'run_stage', return_value=(model, state)) as runner, \
                 patch.object(workflow, 'load_parent', return_value=(None, {'best_model': parent}, None, None)), \
                 patch.object(workflow, 'checked_eval', side_effect=AssertionError('trained Dev was evaluated')), \
                 patch.object(workflow, 'infer', side_effect=AssertionError('trained Dev was inferred')), \
                 patch.object(workflow, 'choose_candidate', side_effect=AssertionError('final weights reselected')), \
                 patch.object(workflow, 'metadata', side_effect=AssertionError('existing final stage rebuilt')):
                self.assertTrue(workflow.final_fit(root, cfg))
            self.assertEqual(runner.call_args.args[2], rows)
            exported = torch.load(root/'final.pt', weights_only=True)
            torch.testing.assert_close(exported['model']['head.weight'], parent['head.weight'] + .1)
            completed = json.loads((root/'completed.json').read_text())
            self.assertIsNone(completed['evaluation'])
            self.assertEqual(completed['recipe_sha256'], digest(root/'recipe_lock.json'))

    def test_paused_final_does_not_export_or_evaluate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            root = Path(folder)
            cfg, final_cfg, model, parent, state = self.make_locked_run(root)
            state['status'] = 'paused'
            with patch.object(workflow, 'read_stage', return_value=(final_cfg, [])), \
                 patch.object(workflow, 'run_stage', return_value=(model, state)), \
                 patch.object(workflow, 'load_parent', side_effect=AssertionError('paused stage exported')), \
                 patch.object(workflow, 'checked_eval', side_effect=AssertionError('paused final evaluated')):
                self.assertFalse(workflow.final_fit(root, cfg))
            self.assertFalse((root/'final.pt').exists())
            self.assertFalse((root/'completed.json').exists())

    def test_recipe_or_screen_checkpoint_drift_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            root = Path(folder)
            cfg, *_ = self.make_locked_run(root)
            with patch.object(workflow, 'run_stage', side_effect=AssertionError('bad recipe trained')):
                with self.assertRaisesRegex(ValueError, 'another run'):
                    workflow.final_fit(root, dict(cfg, epochs=77))
                (root/'screen'/'last.pt').write_bytes(b'modified')
                with self.assertRaisesRegex(ValueError, 'changed after recipe lock'):
                    workflow.final_fit(root, cfg)

    def test_locked_audit_resumes_without_reopening_candidate_selection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            root = Path(folder)
            cfg, final_cfg, model, parent, state = self.make_locked_run(root)
            dev = [dict(id='select-row'), dict(id='audit-row')]
            partition = dict(select=[0], calibration=[1])
            (root/'STOP_REQUESTED').touch()
            with patch.object(workflow, 'checked_eval', side_effect=AssertionError('pause ignored')):
                self.assertFalse(workflow.locked_audit(root, cfg, cfg, model, state, dev, partition))
            (root/'STOP_REQUESTED').unlink()
            with patch.object(workflow, 'load_parent', return_value=(None, {'best_model': parent}, None, None)), \
                 patch('w2v_v53.diagnostics.fixed_train_weights', return_value=[1/3]*3), \
                 patch.object(workflow, 'choose_candidate', side_effect=AssertionError('Audit reopened selection')), \
                 patch.object(workflow, 'checked_eval', return_value=dict(weighted_f1=.1)) as evaluate:
                self.assertTrue(workflow.locked_audit(root, cfg, cfg, model, state, dev, partition))
            self.assertEqual(evaluate.call_args.args[1], [dev[1]])
            report = json.loads((root/'screen_completed.json').read_text())
            self.assertFalse(report['audit_used_for_selection'])
            self.assertEqual(json.loads((root/'recipe_lock.json').read_text())['weights']['name'], 'anchor_tail_50')
            # A very poor Audit remains diagnostic; it cannot change the lock.
            with patch.object(workflow, 'checked_eval', side_effect=AssertionError('Audit repeated')), \
                 patch.object(workflow, 'load_parent', side_effect=AssertionError('completed Audit rebuilt')):
                self.assertTrue(workflow.locked_audit(root, cfg, cfg, model, state, dev, partition))


if __name__ == '__main__':
    unittest.main()
