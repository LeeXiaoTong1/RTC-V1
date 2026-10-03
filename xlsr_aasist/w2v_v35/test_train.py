"""Real tiny HF model: head/joint training, EMA export and interrupted epoch resume."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from w2v_aasist.runtime import atomic_save, sha256
from . import train as training
from . import validation
from .test_losses import model, examples
from .losses import group_weights


class Plan:
    steps = 2
    source_count = 4

    def batches(self, epoch):
        return [[0, 1], [2, 3]]

    def coverage(self):
        return dict(sources_per_epoch=4)


def fixture(root):
    torch.manual_seed(97)
    detector = model()
    checkpoint = root / 'reference.pt'
    atomic_save(checkpoint, dict(schema='rtc_w2v_multiconv_v33', kind='weights', tag='reference',
                                model=detector.state_dict(), **detector.architecture()))
    cfg = dict(version='3.5', device='cpu', amp='none', eval_amp='none', seed=731,
        reference_checkpoint=str(checkpoint), reference_checkpoint_sha256=sha256(checkpoint),
        source_batch_size=2, source_batch=2, source_chunk=1, microbatch=2, frame_budget=160,
        head_epochs=1, joint_epochs=2, trainable_layers=2, checkpointing=True,
        head_lr=1e-4, encoder_lr=1e-6, joint_head_lr=1e-5, layer_decay=.9,
        early_stop_patience=2, ema_decay=.9, ema_device='model', warmup_fraction=.05,
        min_lr_scale=.1, evals_per_epoch=1)
    rows = examples(4)
    conditions = {}
    for condition in ('clean', 'seen', 'heldout'):
        conditions[condition] = []
        for index in range(4):
            row = next(row for row in rows if row['source_id'] == str(index))
            conditions[condition].append(dict(row, id=f'{condition}/{index}', domain='online', band=index))
    weights = group_weights(dict.fromkeys(('en/0', 'en/1', 'zh/0', 'zh/1'), 1))
    plan = Plan()

    def train_loader(plan, cfg, *, training=False, epoch=0, batches=None):
        return [[row for row in rows if int(row['source_id']) in batch] for batch in batches]

    stack = ExitStack()
    stack.enter_context(patch.object(training, 'initialize', side_effect=lambda cfg: copy.deepcopy(detector)))
    stack.enter_context(patch.object(training, 'reference_model', side_effect=lambda cfg: copy.deepcopy(detector)))
    stack.enter_context(patch.object(training, 'build_data', return_value=(plan, conditions, weights,
        dict.fromkeys(weights, 1), {str(checkpoint): sha256(checkpoint)})))
    stack.enter_context(patch.object(training, 'loader', side_effect=train_loader))
    stack.enter_context(patch.object(validation, 'loader', side_effect=lambda rows, cfg: [rows]))
    stack.enter_context(patch.object(training, 'retire_generations', return_value=None))
    stack.enter_context(patch.object(training, 'source_fingerprints', return_value={'fixture': 'fixed-v35'}))
    return cfg, detector, stack


class TrainingTests(unittest.TestCase):
    def test_production_source_fingerprints_are_safe_weights_only_values(self):
        fingerprints = training.source_fingerprints()
        self.assertTrue(all(type(value) is str for value in fingerprints.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'versions.pt'
            atomic_save(path, dict(source_hashes=fingerprints))
            self.assertEqual(training.read_state(path)['source_hashes'], fingerprints)

    def assert_same(self, actual, expected):
        for field in ('epoch', 'phase', 'phase_steps', 'global_steps', 'controller', 'history', 'complete'):
            self.assertEqual(actual[field], expected[field], field)
        self.assertEqual(actual['optimizer']['param_groups'], expected['optimizer']['param_groups'])
        for name in expected['model']:
            torch.testing.assert_close(actual['model'][name], expected['model'][name], atol=0, rtol=0)
        for name in expected['ema']['shadow']:
            torch.testing.assert_close(actual['ema']['shadow'][name], expected['ema']['shadow'][name], atol=0, rtol=0)
        for parameter, state in expected['optimizer']['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(actual['optimizer']['state'][parameter][name], value, atol=0, rtol=0)

    def test_head_joint_reference_ema_and_exact_resume_after_committed_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, detector, patches = fixture(root)
            with patches:
                full = root / 'full'
                training.train(cfg, full)
                expected = training.read_state(full / 'last.pt')
                self.assertTrue(expected['complete'])
                self.assertEqual(expected['global_steps'], 6)
                self.assertEqual(len(expected['history']), 4)
                self.assertEqual(expected['history'][1]['phase'], 'head')
                self.assertEqual(expected['history'][2]['phase'], 'joint')
                self.assertEqual(expected['history'][1]['train']['short_views'], 0)
                self.assertGreater(expected['history'][1]['train']['full_views'], 0)
                winner = training.read_state(full / 'best_model.pt')
                self.assertEqual(winner['kind'], 'weights')
                self.assertEqual(winner['tag'], expected['controller']['best_selected']['tag'])
                if winner['tag'] == 'reference':
                    for name, value in detector.state_dict().items():
                        torch.testing.assert_close(winner['model'][name], value, atol=0, rtol=0)
                candidate = training.read_state(full / 'best_candidate.pt')
                self.assertEqual(candidate['weight_source'], 'ema')
                frozen = 'backbone.feature_projection.layer_norm.weight'
                torch.testing.assert_close(expected['model'][frozen], detector.state_dict()[frozen], atol=0, rtol=0)
                self.assertTrue(any(not torch.equal(value, expected['model'][name])
                    for name, value in detector.state_dict().items() if name.startswith('backbone.encoder.layers.0.')))
                # Adam head step survives the phase transition; encoder starts at joint.
                names = expected['optimizer']['param_groups']
                head_group = next(group for group in names if group['name'] == 'head.decay')
                encoder_group = next(group for group in names if group['name'] == 'encoder.layer_00.decay')
                self.assertEqual(float(expected['optimizer']['state'][head_group['params'][0]]['step']), 6.)
                self.assertEqual(float(expected['optimizer']['state'][encoder_group['params'][0]]['step']), 4.)

                interrupted = root / 'interrupted'
                actual_save = training.atomic_save

                def interrupt(path, state):
                    actual_save(path, state)
                    if Path(path).name == 'last.pt' and state.get('epoch') == 1:
                        raise RuntimeError('simulated interruption after commit')

                with patch.object(training, 'atomic_save', side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        training.train(cfg, interrupted)
                training.train(cfg, interrupted, interrupted / 'last.pt')
                self.assert_same(training.read_state(interrupted / 'last.pt'), expected)
                self.assertFalse(list(interrupted.glob('*.previous')))
                # A completed resume cannot advance any optimizer state.
                training.train(cfg, interrupted, interrupted / 'last.pt')
                self.assert_same(training.read_state(interrupted / 'last.pt'), expected)

    def test_checkpoint_tamper_and_resume_config_drift_fail_before_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _, patches = fixture(root)
            with patches:
                run = root / 'smoke'
                training.train(cfg, run, smoke_steps=1)
                self.assertEqual(training.read_state(run / 'last.pt')['global_steps'], 0)
                self.assertFalse((run / 'completed.json').exists())
                with self.assertRaisesRegex(ValueError, 'Exact V3.5 resume'):
                    training.train({**cfg, 'encoder_lr': 2e-6}, run, run / 'last.pt')
                Path(cfg['reference_checkpoint']).write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'Protected source checkpoint'):
                    training.train(cfg, run, run / 'last.pt')

    def test_uncommitted_promotion_rolls_back_then_replays_same_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _, patches = fixture(root)
            cfg['joint_epochs'] = 1
            with patches:
                real_validate = training.validate

                def increasing(model, records, cfg, device, path):
                    result = real_validate(model, records, cfg, device, path)
                    epoch = 0 if Path(path).name.startswith('reference') else int(Path(path).name.split('_')[1])
                    result.update(clean_f1=.7 + epoch * .02, noisy_f1=.7 + epoch * .02,
                                  weighted_f1=.7 + epoch * .02)
                    return result

                with patch.object(training, 'validate', side_effect=increasing):
                    full = root / 'full'
                    training.train(cfg, full)
                    expected = training.read_state(full / 'last.pt')
                    broken = root / 'broken'
                    actual_save = training.atomic_save

                    def interrupt(path, state):
                        actual_save(path, state)
                        if Path(path).name == 'best_model.pt' and state.get('tag') == 'epoch_1':
                            raise RuntimeError('crash before training-state commit')

                    with patch.object(training, 'atomic_save', side_effect=interrupt):
                        with self.assertRaisesRegex(RuntimeError, 'before training-state commit'):
                            training.train(cfg, broken)
                    self.assertEqual(training.read_state(broken / 'last.pt')['epoch'], 0)
                    self.assertEqual(training.read_state(broken / 'best_model.pt')['tag'], 'epoch_1')
                    self.assertTrue((broken / 'best_model.pt.previous').is_file())
                    with patch.object(training, 'retire_generations') as retire:
                        training.train(cfg, broken, broken / 'last.pt')
                        self.assertEqual(retire.call_args_list[0].kwargs['keep_epochs'], [0, 1])
                    self.assert_same(training.read_state(broken / 'last.pt'), expected)
                    self.assertEqual(training.read_state(broken / 'best_model.pt')['tag'], 'epoch_2')
                    self.assertFalse(list(broken.glob('*.previous')))


if __name__ == '__main__':
    unittest.main()
