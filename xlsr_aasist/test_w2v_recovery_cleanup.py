"""Scoped deletion checks, kept outside the training source-hash set."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from recover_w2v_storage import old_checkpoint_plan, prune_old_checkpoints, training_command
from w2v_rebuild.train import parser


class OldCheckpointCleanup(unittest.TestCase):
    def fixture(self, root):
        original = root/'exp'/'old_run'
        for stage in (1, 2, 3):
            directory = original/f'stage{stage}'
            directory.mkdir(parents=True)
            (directory/'last.pt').write_bytes(b'training state')
            (directory/'best_model.pt').write_bytes(b'earlier weights')
            (directory/'config.json').write_text('{}')
            (directory/'metrics.jsonl').write_text('{}\n')
        best = original/'stage3'/'best_model.pt'
        torch.save({'stage': 3, 'kind': 'weights', 'model': {'weight': torch.ones(2)}}, best)
        return original, best

    def test_only_five_known_files_removed_and_final_best_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original, best = self.fixture(root)
            kept_bytes = best.read_bytes()
            new = root/'exp'/'active_run'/'stage3'
            new.mkdir(parents=True)
            (new/'last.pt').write_bytes(b'active training')
            extra = original/'stage3'/'custom_model.pt'
            extra.write_bytes(b'not in whitelist')
            plan = old_checkpoint_plan(best)
            self.assertEqual(len(plan), 5)
            self.assertTrue(all(p.exists() for p, _ in plan))  # preview has no mutations
            removed = prune_old_checkpoints(best, plan)
            self.assertEqual(removed, sum(s[2] for _, s in plan))
            self.assertTrue(all(not p.exists() for p, _ in plan))
            self.assertEqual(best.read_bytes(), kept_bytes)
            self.assertEqual((new/'last.pt').read_bytes(), b'active training')
            self.assertEqual(extra.read_bytes(), b'not in whitelist')
            for stage in (1, 2, 3):
                self.assertTrue((original/f'stage{stage}'/'config.json').is_file())
                self.assertTrue((original/f'stage{stage}'/'metrics.jsonl').is_file())

    def test_changed_target_aborts_before_any_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            original, best = self.fixture(Path(d))
            plan = old_checkpoint_plan(best)
            (original/'stage2'/'last.pt').write_bytes(b'changed while planning')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                prune_old_checkpoints(best, plan)
            self.assertTrue(all(p.exists() for p, _ in plan))

    def test_unreadable_kept_model_prevents_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            _, best = self.fixture(Path(d))
            plan = old_checkpoint_plan(best)
            best.write_bytes(b'broken model')
            with self.assertRaises(Exception):
                prune_old_checkpoints(best, plan)
            self.assertTrue(all(p.exists() for p, _ in plan))

    def test_symlink_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            original, best = self.fixture(Path(d))
            target = (original/'stage1'/'last.pt').resolve()
            is_symlink = Path.is_symlink
            with patch.object(Path, 'is_symlink', lambda p: p == target or is_symlink(p)):
                with self.assertRaisesRegex(ValueError, 'symlink'):
                    old_checkpoint_plan(best)
            self.assertTrue(target.exists())

    def test_keep_cache_flag_preserves_the_configured_cache(self):
        config = dict(stage=3, out='old', ssl_path='ssl', train_data_path='train', dev_data_path='dev',
                      train_protocol='tp', dev_protocol='dp', rtc_pairs='pairs', train_noise_manifest='noise',
                      train_noisy_cache='original', dev_noisy_cache='seen', dev_heldout_cache='held',
                      feature_cache='existing/features', epochs=8, encoder_lr=2e-7, head_lr=5e-6)
        for preflight in (True, False):
            command = training_command(config, 'new/stage3', 'baseline.pt', preflight, True)
            args = parser().parse_args(command[4:])
            self.assertEqual(args.feature_cache, config['feature_cache'])
            self.assertFalse(args.no_feature_cache)
            self.assertEqual(args.encoder_lr, config['encoder_lr'])
            self.assertEqual(args.head_lr, config['head_lr'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
