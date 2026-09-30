import tempfile
from pathlib import Path
import unittest
from cleanup_w2v_checkpoints import plan, remove


class PruneTests(unittest.TestCase):
    def test_only_intermediates_removed_best_candidates_and_cache_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'exp'; run = root/'run'/'stage3'; run.mkdir(parents=True)
            names = ['best_model.pt','best_noisy.pt','candidate_best.pt','last.pt',
                     'epoch_3.pt','epoch_4_dev_loss_0.2.pth','unknown.pt','audio.wav','manifest.jsonl']
            for name in names: (run/name).write_bytes(b'original contents')
            planned, skipped = plan([root], [run/'best_model.pt'])
            self.assertEqual({Path(x['path']).name for x in planned},
                             {'last.pt','epoch_3.pt','epoch_4_dev_loss_0.2.pth'})
            remove(planned)
            self.assertEqual({p.name for p in run.iterdir()}, set(names)-{'last.pt','epoch_3.pt','epoch_4_dev_loss_0.2.pth'})
            for p in run.iterdir(): self.assertEqual(p.read_bytes(), b'original contents')

    def test_run_without_named_best_and_explicit_pins_are_not_deleted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'exp'
            for run in ('with_best','unfinished'):
                (root/run).mkdir(parents=True)
                (root/run/'epoch_1.pt').write_bytes(b'weights')
            # Experiment paths containing 'best' are themselves conservatively protected.
            (root/'with_best'/'best_model.pt').write_bytes(b'weights')
            planned, skipped = plan([root], [])
            self.assertFalse(planned)
            self.assertEqual(skipped, [str((root/'unfinished'/'epoch_1.pt').resolve())])
            run=root/'other';run.mkdir()
            (run/'best_model.pt').write_bytes(b'weights')
            (run/'epoch_2.pt').write_bytes(b'weights')
            planned, _=plan([root],[run/'epoch_2.pt'])
            self.assertFalse(planned)

    def test_all_targets_checked_before_any_deletion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'exp';run=root/'run';run.mkdir(parents=True)
            for name in ('best_model.pt','last.pt','epoch_2.pt'):(run/name).write_bytes(b'weights')
            planned,_=plan([root],[])
            Path(planned[-1]['path']).write_bytes(b'newly updated checkpoint')
            with self.assertRaises(RuntimeError):remove(planned)
            self.assertTrue(all(Path(x['path']).exists() for x in planned))

    def test_external_target_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'exp';root.mkdir()
            external=Path(td)/'last.pt';external.write_bytes(b'weights')
            from cleanup_w2v_checkpoints import identity
            with self.assertRaises(ValueError):
                remove([{'path':str(external),'root':str(root),'identity':identity(external)}])
            self.assertTrue(external.exists())


if __name__ == '__main__':
    unittest.main()
