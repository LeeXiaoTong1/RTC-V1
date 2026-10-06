"""Real tiny-encoder continuation, transactional resume and LAST export tests.

Synthetic waveforms below are test fixtures, created before V3.13 executes.
Production code still verifies every source hash and never creates audio caches.
"""
from contextlib import contextmanager, redirect_stdout
import copy
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import torch

from w2v_v39.common import atomic_json, digest, read_json
from w2v_v312.data import bundles as original_bundles
from w2v_v312.model import load_model as load_v312_model
from w2v_v312.state import apply_partial as apply_v312_partial
from w2v_v312.test_workflow import fixture, synthetic_expanded_bundles
from w2v_v312.train import run_experiment as run_v312
from .config import configuration, parser, verify_inputs
from .model import load_model
from .state import atomic_save, apply_partial, load_selected
from .train import run_experiment, infer


torch.set_num_threads(1)


@contextmanager
def synthetic_paired_bundles(cfg):
    """Provide 24 actual small source files in each language/label stratum."""
    root = Path(cfg['v37_run']).parent / 'paired_test_waveforms'
    root.mkdir(exist_ok=True)
    with original_bundles(cfg) as (train, dev):
        rows, identity = [], {}
        for copy_index in range(12):
            for row in train['rows']:
                key = (copy_index, row['source_id'])
                if key not in identity:
                    name = f"toy_{copy_index}_{row['source_id']}.wav"
                    path = root / name
                    if not path.is_file():
                        waveform, sample_rate = sf.read(row['audio'], dtype='float32')
                        # Genuine distinct test waveforms, not invented hashes or
                        # duplicated metadata that bypasses source verification.
                        waveform = waveform + np.float32((copy_index + 1) * 1e-4)
                        sf.write(path, waveform, sample_rate, subtype='FLOAT')
                    stat = path.stat()
                    identity[key] = dict(name=name, audio=str(path), audio_sha256=digest(path),
                                         audio_size=stat.st_size, audio_mtime_ns=stat.st_mtime_ns)
                item = identity[key]
                source = f"offline/{row['language']}/{item['name']}"
                recording_id = (f"online/{row['language']}/{item['name']}"
                                if row['condition'] == 'online' else source)
                rows.append(dict(row, id=recording_id, source_id=source,
                                 group_id=item['audio_sha256'], source_sha256=item['audio_sha256'],
                                 **{key: value for key, value in item.items() if key != 'name'}))
        # These historical vectors are deliberately unchanged. V3.13 must derive
        # preservation references from the actual trained LAST, never these.
        yield dict(rows=rows, x=np.tile(train['x'], (12, 1)),
                   logits=np.tile(train['logits'], (12, 1))), dev


def assert_nested_equal(test, first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        test.assertEqual(set(first), set(second))
        for key in first:
            assert_nested_equal(test, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            assert_nested_equal(test, a, b)
    else:
        test.assertEqual(first, second)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='v313_integration_')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        with redirect_stdout(io.StringIO()):
            _, old, _, cls.records = fixture(cls.root)
            cls.source = cls.root / 'v312'
            cls.source.mkdir()
            atomic_json(cls.source / 'config.json', old)
            with patch('w2v_v312.train.bundles', synthetic_expanded_bundles):
                done = run_v312(old, cls.source)
            if not done['baseline_fallback'] or not done['committed_updates']:
                raise AssertionError('Fixture must retain original best alongside a different trained LAST')
            cls.old = old
            cls.source_state = torch.load(cls.source / 'last.pt', map_location='cpu', weights_only=True)
            cls.cfg = configuration(parser().parse_args([
                '--source-run', str(cls.source), '--device', 'cpu', '--workers', '0', '--epochs', '1']))
            cls.cfg.update(trainable_layers=1, adapter_hidden=8, adversary_hidden=8,
                           probe_sources_per_language=4, probe_steps=8, probe_hidden=8,
                           pair_probe_sources_per_group=4, pair_micro_sources=4,
                           microbatch=4, eval_batch=4, feature_batch=4, reference_batch=128,
                           disk_margin_bytes=0, min_gain=2., patience=10,
                           catastrophic_weighted_drop=2.)
            # Complete fixture creation before checking the production disk footprint.
            with synthetic_paired_bundles(cls.cfg) as (train, _):
                cls.train_rows = copy.deepcopy(train['rows'])
            cls.protected = {str(path): digest(path) for path in cls.root.rglob('*') if path.is_file()}

    def _run(self, name, cfg=None):
        run = self.root / name
        run.mkdir()
        atomic_json(run / 'config.json', self.cfg if cfg is None else cfg)
        return run

    def test_source_is_actual_trained_last_and_missing_or_changed_last_never_falls_back(self):
        with redirect_stdout(io.StringIO()):
            cfg = self.cfg
            verify_inputs(cfg)
            self.assertEqual(cfg['starting_kind'], 'trained_v312_last')
            self.assertEqual(cfg['starting_checkpoint_sha256'], digest(self.source / 'last.pt'))
            self.assertEqual(cfg['starting_cursor'], self.source_state['cursor'])
            self.assertEqual(cfg['starting_tag'], self.source_state['history'][-1]['tag'])
            current = load_model(cfg, training=False)
            actual = load_v312_model(self.old, training=False)
            original = load_v312_model(self.old, training=False)
            apply_v312_partial(actual, self.source_state['model'])
            assert_nested_equal(self, current.state_dict(), actual.state_dict())
            self.assertTrue(any(not torch.equal(value, original.state_dict()[name])
                                for name, value in current.state_dict().items()))
            broken = dict(cfg, starting_checkpoint_sha256='0' * 64)
            with self.assertRaises(ValueError):
                load_model(broken, training=False)
            missing = self.root / 'missing_last_source'
            missing.mkdir()
            for name in ('config.json', 'completed.json', 'best.pt'):
                shutil.copy2(self.source / name, missing / name)
            with self.assertRaisesRegex(FileNotFoundError, 'never substitute'):
                configuration(parser().parse_args(['--source-run', str(missing), '--device', 'cpu']))

    def test_actual_finetuning_and_replay_use_last_without_creating_audio_or_frame_caches(self):
        with redirect_stdout(io.StringIO()):
            run = self._run('v313_complete')
            with patch('w2v_v313.train.bundles', synthetic_paired_bundles):
                done = run_experiment(self.cfg, run)
            self.assertEqual(done['selected'], 'starting_last')
            self.assertTrue(done['baseline_fallback'])
            self.assertEqual(done['fallback_target'], 'trained_v312_last')
            self.assertGreater(done['committed_updates'], 0)
            self.assertIsNone(done['stability_stop'])
            startup = read_json(run / 'startup_replay.json')
            self.assertEqual(startup['starting_kind'], 'trained_v312_last')
            self.assertEqual(startup['comparison']['status'], 'passed')
            with np.load(self.source / ('dev_scores_' + self.cfg['starting_tag'] + '.npz')) as source_scores, \
                    np.load(run / 'dev_scores_baseline.npz') as new_scores:
                np.testing.assert_allclose(new_scores['logits'], source_scores['logits'], atol=2e-5, rtol=1e-5)
            saved = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
            changed = {name for name, value in saved['model'].items()
                       if not torch.equal(value, self.source_state['model'][name])}
            self.assertTrue(any(name.startswith('backbone.encoder.layers.1.') for name in changed))
            self.assertTrue(any(name.startswith('head.blocks.') for name in changed))
            self.assertFalse(any(name.startswith('backbone.encoder.layers.0.') for name in saved['model']))
            self.assertFalse(any(name.startswith('head.classifier.2.') for name in saved['model']))
            history = read_json(run / 'training_history.json')
            self.assertTrue(all(entry['committed'] for entry in history))
            self.assertGreater(history[-1]['segment_training']['ranking_candidate_edges'], 0)
            self.assertGreater(history[-1]['last_training_step']['ranking_loss'], 0.)
            inventory = read_json(run / 'retention_inventory.json')
            self.assertEqual(inventory['starting_checkpoint_sha256'], self.cfg['starting_checkpoint_sha256'])
            self.assertEqual(inventory['scalar_values'], 3 * len(self.train_rows))
            self.assertEqual(inventory['new_audio_bytes'], 0)
            self.assertEqual(inventory['new_frame_cache_bytes'], 0)
            self.assertFalse((run / 'features').exists())
            self.assertFalse(list(run.rglob('*.wav')))
            self.assertFalse(list(run.glob('epoch_*.pt')))
            for path in (run / 'train_scalar_references').glob('*.npz'):
                with np.load(path) as values:
                    self.assertEqual(set(values.files), {'logits', 'rms'})
                    self.assertEqual(values['logits'].shape[1], 2)
                    self.assertEqual(values['rms'].ndim, 1)
            for path, expected in self.protected.items():
                self.assertEqual(digest(path), expected)
            selected, _ = load_selected(run)
            self.assertIsNone(selected['model'])
            self._assert_exports(run, saved)

    def _assert_exports(self, run, last):
        from .evaluate import export
        rows = [self.records['dev'][index] for index in (9, 0, 6, 3)]
        protocol = run / 'protocol.txt'
        protocol.write_text('\n'.join(row['id'] for row in rows) + '\n', encoding='utf-8')
        protected = {str(path): digest(path) for path in (run / 'last.pt', run / 'best.pt', self.source / 'last.pt')}
        source_model = load_model(self.cfg, training=False)
        last_model = load_model(self.cfg, training=False)
        apply_partial(last_model, last['model'])
        expected_source, _ = infer(source_model, rows, self.cfg, 'expected source LAST')
        expected_last, _ = infer(last_model, rows, self.cfg, 'expected V3.13 LAST')
        self.assertGreater(float(np.abs(expected_last - expected_source).max()), 0.)
        for kind, expected in (('best', expected_source), ('last', expected_last)):
            output = self.root / (run.name + '_submission_' + kind)
            archive = export(run, protocol, self.root / 'audio', output, 'cpu', 0, checkpoint_kind=kind)
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(zipped.namelist(), ['scores.txt'])
                scored = [line.split() for line in zipped.read('scores.txt').decode().splitlines()]
            self.assertEqual([item[0] for item in scored], [row['id'] for row in rows])
            np.testing.assert_allclose([float(item[1]) for item in scored],
                                       torch.from_numpy(expected).softmax(-1)[:, 0].numpy(), atol=1e-9, rtol=0)
            metadata = read_json(output / 'submission_meta.json')
            self.assertEqual(metadata['checkpoint_kind'], kind)
            self.assertEqual(metadata['starting_checkpoint_sha256'], self.cfg['starting_checkpoint_sha256'])
            if kind == 'best':
                self.assertEqual(metadata['selected'], 'starting_last')
            else:
                self.assertEqual(metadata['selected'], 'last:' + last['history'][-1]['tag'])
                self.assertFalse(metadata['baseline_fallback'])
        self.assertEqual(protected, {path: digest(path) for path in protected})

    def test_atomic_commit_resume_reproduces_exact_weights_optimizer_and_rng(self):
        with redirect_stdout(io.StringIO()):
            uninterrupted = self._run('v313_uninterrupted')
            interrupted = self._run('v313_interrupted')
            with patch('w2v_v313.train.bundles', synthetic_paired_bundles):
                run_experiment(self.cfg, uninterrupted)
            first = True
            def stop_after_commit(path, value, margin):
                nonlocal first
                atomic_save(path, value, margin)
                if Path(path).name == 'last.pt' and first:
                    first = False
                    raise InterruptedError('test interruption immediately after atomic commit')
            with patch('w2v_v313.train.bundles', synthetic_paired_bundles), \
                    patch('w2v_v313.train.atomic_save', side_effect=stop_after_commit):
                with self.assertRaises(InterruptedError):
                    run_experiment(self.cfg, interrupted)
            checkpoint = torch.load(interrupted / 'last.pt', map_location='cpu', weights_only=True)
            self.assertEqual(len(checkpoint['history']), 1)
            reference_files = {str(path): digest(path) for path in (interrupted / 'train_scalar_references').iterdir()}
            def resume_infer(*args, **kwargs):
                label = str(args[3])
                if any(text in label for text in ('source-LAST reference', 'baseline Dev', 'initial language')):
                    raise AssertionError('Committed starting references and diagnostics must be reused')
                return infer(*args, **kwargs)
            with patch('w2v_v313.train.bundles', synthetic_paired_bundles), \
                    patch('w2v_v313.train.infer', side_effect=resume_infer):
                run_experiment(self.cfg, interrupted)
            full = torch.load(uninterrupted / 'last.pt', map_location='cpu', weights_only=True)
            resumed = torch.load(interrupted / 'last.pt', map_location='cpu', weights_only=True)
            self.assertEqual(full['cursor'], resumed['cursor'])
            self.assertEqual([entry['cursor'] for entry in full['history']], [entry['cursor'] for entry in resumed['history']])
            for name in ('model', 'adversary', 'optimizer', 'rng'):
                assert_nested_equal(self, full[name], resumed[name])
            self.assertEqual(reference_files, {path: digest(path) for path in reference_files})
            self.assertFalse((interrupted / 'validation_pending.json').exists())

    def test_default_eight_source_microgroups_mine_two_negatives_and_step_once_per_logical_batch(self):
        from collections import Counter
        from .data import paired_microgroups
        from .model import optimizer_for
        from .train import train_step
        cfg = dict(self.cfg, pair_micro_sources=8)
        updates, batches = [], []
        def counted_optimizer(*args, **kwargs):
            optimizer = optimizer_for(*args, **kwargs)
            original_step = optimizer.step
            def step(*step_args, **step_kwargs):
                updates.append(1)
                return original_step(*step_args, **step_kwargs)
            optimizer.step = step
            return optimizer
        def checked_step(model, adversary, optimizer, examples, config, strength, auxiliary=1.):
            self.assertEqual(len(examples), 32)
            self.assertAlmostEqual(sum(row['ce_weight'] for row in examples), 1.)
            groups = paired_microgroups(examples)
            self.assertEqual(len(groups), 2)
            for group in groups:
                self.assertEqual(len(group), 16)
                sources = {row['pair_occurrence']:(row['language'], row['label']) for row in group}
                self.assertEqual(len(sources), 8)
                self.assertEqual(Counter(sources.values()),
                                 Counter({('en', 0):2, ('en', 1):2, ('zh', 0):2, ('zh', 1):2}))
            count = len(updates)
            value = train_step(model, adversary, optimizer, examples, config, strength, auxiliary)
            self.assertEqual(len(updates), count + 1)
            self.assertEqual(value['ranking_anchors'], 16.)
            self.assertEqual(value['ranking_candidate_edges'], 32.)
            self.assertTrue(np.isfinite(value['gradient_norm']))
            batches.append(len(examples))
            return value
        with redirect_stdout(io.StringIO()):
            run = self._run('v313_default_eight', cfg)
            with patch('w2v_v313.train.bundles', synthetic_paired_bundles), \
                    patch('w2v_v313.train.optimizer_for', side_effect=counted_optimizer), \
                    patch('w2v_v313.train.train_step', side_effect=checked_step):
                done = run_experiment(cfg, run)
            self.assertIsNone(done['stability_stop'])
            self.assertEqual(len(updates), done['committed_updates'])
            self.assertEqual(len(batches), done['committed_updates'])
            saved = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
            changed = {name for name, value in saved['model'].items()
                       if not torch.equal(value, self.source_state['model'][name])}
            self.assertTrue(any(name.startswith('backbone.encoder.layers.1.') for name in changed))
            self.assertTrue(any(name.startswith('head.blocks.') for name in changed))


if __name__ == '__main__':
    unittest.main()
