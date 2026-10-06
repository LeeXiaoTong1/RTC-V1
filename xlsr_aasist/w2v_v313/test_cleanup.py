"""Exercise completed-run compaction with real checkpoint files and score replay."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from w2v_v39.common import atomic_json, digest
from .cleanup import cleanup
from .state import SCHEMA, apply_partial, identity, load_last, load_selected


def checkpoint_fixture(root, selected=True):
    root = Path(root)
    source = root / 'v312'
    source.mkdir()
    base = root / 'original_best.pt'
    torch.save({'weight': torch.arange(6).reshape(2, 3).float()}, base)
    torch.save({'model': {'protected': torch.tensor([93.558])}}, source / 'last.pt')
    run = root / 'v313'
    run.mkdir()
    cfg = dict(version='3.13', base_checkpoint=str(base), base_checkpoint_sha256=digest(base),
               starting_checkpoint=str(source / 'last.pt'),
               starting_checkpoint_sha256=digest(source / 'last.pt'),
               starting_tag='epoch_3_step_2385', starting_kind='trained_v312_last', disk_margin_bytes=0)
    trained = dict(weight=torch.tensor([[.4, -.8, .1], [-.2, .7, .3]]),
                   bias=torch.tensor([.15, -.25]))
    chosen = dict(weight=trained['weight'] * .8, bias=trained['bias'] * .9)
    tag = 'epoch_1_step_2'
    best = dict(schema=SCHEMA, identity=identity(cfg), config=cfg,
                model=chosen if selected else None, selected='epoch_1_step_1' if selected else 'starting_last')
    torch.save(best, run / 'best.pt')
    last = dict(schema=SCHEMA, identity=identity(cfg), model=trained, cursor=2,
                history=[dict(cursor=2, tag=tag, committed=True, language_evidence={'evidence': False})],
                best_model=chosen, selected=best['selected'], best_score=.964,
                optimizer={'state': {0: {'exp_avg': torch.ones(8192), 'exp_avg_sq': torch.ones(8192)}}},
                adversary={'weight': torch.ones(128, 8)}, rng={'torch': torch.get_rng_state()})
    torch.save(last, run / 'last.pt')
    done = dict(version='3.13', status='complete', selected=best['selected'],
                checkpoint_sha256=digest(run / 'best.pt'),
                base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
                starting_checkpoint_sha256=cfg['starting_checkpoint_sha256'],
                baseline_fallback=not selected, committed_updates=2)
    atomic_json(run / 'config.json', cfg)
    atomic_json(run / 'completed.json', done)
    return run, cfg, done


def scores(parameters):
    model = torch.nn.Linear(3, 2)
    apply_partial(model, parameters)
    with torch.no_grad():
        return model(torch.tensor([[1., 0., 0.], [.1, .8, -.2], [-.5, .1, .7]])).softmax(-1)[:, 0]


class CleanupTests(unittest.TestCase):
    def test_preview_is_read_only_then_windows_safe_replace_preserves_both_scores(self):
        with tempfile.TemporaryDirectory(prefix='v313_compact_') as root, redirect_stdout(io.StringIO()):
            run, cfg, _ = checkpoint_fixture(root)
            before = {str(path): digest(path) for path in Path(root).rglob('*') if path.is_file()}
            last_before, _ = load_last(run)
            best_before, _ = load_selected(run)
            last_scores = scores(last_before['model'])
            best_scores = scores(best_before['model'])
            self.assertFalse(torch.equal(last_scores, best_scores))
            old_size = (run / 'last.pt').stat().st_size
            self.assertGreater(cleanup(run), 0)
            self.assertEqual(before, {path: digest(path) for path in before})
            self.assertGreater(cleanup(run, apply=True), 0)
            saved = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
            self.assertTrue(saved['inference_only'])
            self.assertTrue({'optimizer', 'adversary', 'rng', 'best_model'}.isdisjoint(saved))
            self.assertLess((run / 'last.pt').stat().st_size, old_size)
            last_after, metadata = load_last(run)
            best_after, _ = load_selected(run)
            self.assertEqual(metadata['selected'], 'last:epoch_1_step_2')
            torch.testing.assert_close(scores(last_after['model']), last_scores, rtol=0, atol=0)
            torch.testing.assert_close(scores(best_after['model']), best_scores, rtol=0, atol=0)
            for path, original in before.items():
                if Path(path) != run / 'last.pt':
                    self.assertEqual(digest(path), original)
            self.assertEqual(digest(cfg['starting_checkpoint']), cfg['starting_checkpoint_sha256'])
            compact_hash = digest(run / 'last.pt')
            self.assertEqual(cleanup(run, apply=True), 0)
            self.assertEqual(digest(run / 'last.pt'), compact_hash)
            self.assertFalse((run / 'last.pt.tmp').exists())

    def test_starting_last_fallback_remains_fallback_after_compaction(self):
        with tempfile.TemporaryDirectory(prefix='v313_fallback_') as root, redirect_stdout(io.StringIO()):
            run, cfg, _ = checkpoint_fixture(root, selected=False)
            protected = digest(cfg['starting_checkpoint'])
            cleanup(run, apply=True)
            selected, done = load_selected(run)
            self.assertIsNone(selected['model'])
            self.assertEqual(selected['selected'], 'starting_last')
            self.assertTrue(done['baseline_fallback'])
            self.assertIsNotNone(load_last(run)[0]['model'])
            self.assertEqual(digest(cfg['starting_checkpoint']), protected)

    def test_unfinished_or_uncommitted_training_is_refused_without_file_changes(self):
        for fault in ('incomplete', 'uncommitted'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(prefix='v313_refuse_') as root:
                run, _, done = checkpoint_fixture(root)
                if fault == 'incomplete':
                    atomic_json(run / 'completed.json', dict(done, status='training'))
                else:
                    last = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
                    last['history'][-1]['committed'] = False
                    torch.save(last, run / 'last.pt')
                before = {str(path): digest(path) for path in Path(root).rglob('*') if path.is_file()}
                with self.assertRaises(ValueError):
                    cleanup(run, apply=True)
                self.assertEqual(before, {path: digest(path) for path in before})

    def test_protected_source_as_target_is_refused_before_replacement(self):
        with tempfile.TemporaryDirectory(prefix='v313_source_guard_') as root:
            run, cfg, _ = checkpoint_fixture(root)
            altered = copy.deepcopy(cfg)
            altered.update(starting_checkpoint=str(run / 'last.pt'), starting_checkpoint_sha256=digest(run / 'last.pt'))
            original = digest(run / 'last.pt')
            with patch('w2v_v313.cleanup.load_selected', return_value=({'config': altered}, {})), \
                    patch('w2v_v313.cleanup.load_last', return_value=({}, {})):
                with self.assertRaisesRegex(ValueError, 'external or starting checkpoint'):
                    cleanup(run, apply=True)
            self.assertEqual(digest(run / 'last.pt'), original)

    def test_insufficient_atomic_space_keeps_original_resume_and_models(self):
        with tempfile.TemporaryDirectory(prefix='v313_no_space_') as root, redirect_stdout(io.StringIO()):
            run, _, _ = checkpoint_fixture(root)
            before = {str(path): digest(path) for path in Path(root).rglob('*') if path.is_file()}
            with patch('w2v_v313.state.shutil.disk_usage', return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(OSError, 'previous last.pt retained'):
                    cleanup(run, apply=True)
            self.assertEqual(before, {path: digest(path) for path in before})
            self.assertFalse((run / 'last.pt.tmp').exists())

    def test_cli_refuses_an_active_job_before_entering_cleanup(self):
        from .cleanup import main
        with patch('sys.argv', ['cleanup', '--run', 'unused', '--apply']), \
                patch('w2v_aasist.full_workflow.ensure_idle', side_effect=RuntimeError('active training')), \
                patch('w2v_v313.cleanup.cleanup') as operation:
            with self.assertRaisesRegex(RuntimeError, 'active training'):
                main()
            operation.assert_not_called()

    def test_symlink_run_or_external_last_is_rejected_before_loading_or_writing(self):
        with tempfile.TemporaryDirectory(prefix='v313_link_guard_') as root:
            run, _, _ = checkpoint_fixture(root)
            before = digest(run / 'last.pt')
            original = Path.is_symlink
            for target, message in ((run, 'symlink run'), (run / 'last.pt', 'external or starting')):
                with self.subTest(target=target):
                    # Windows may disallow creating links without extra privileges;
                    # exercise the exact link guard without requiring that privilege.
                    resolved_target = target.resolve()
                    with patch.object(Path, 'is_symlink', lambda path: path in (target, resolved_target) or original(path)):
                        with self.assertRaisesRegex(ValueError, message):
                            cleanup(run, apply=True)
                    self.assertEqual(digest(run / 'last.pt'), before)


if __name__ == '__main__':
    unittest.main()
