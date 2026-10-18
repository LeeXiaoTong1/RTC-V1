import json
from pathlib import Path
import tempfile
import unittest

from cleanup_keep_best_last import plan, apply


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.exp = Path(self.temp.name).resolve() / 'exp'
        self.exp.mkdir()
        self.run = self.exp / 'w2v_v3189_test'
        self.run.mkdir()

    def file(self, name, data=b'checkpoint', folder=None):
        path = (folder or self.run) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_preview_and_apply_keep_best_last_inference_and_unknown(self):
        protected = [self.file(name) for name in (
            'best.pt', 'best_guarded.pth', 'best_weighted.ckpt', 'last.pt',
            'checkpoint_last.pt', 'latest.pt', 'inference.pt', 'custom_weights.pt',
            'last.pt.previous', 'last.pt.tmp', 'warmup.pt')]
        candidates = [self.file(name) for name in ('epoch_1.pt', 'step_200.pth', 'checkpoint-12.ckpt')]
        report = plan([self.exp])
        self.assertEqual({Path(row['path']) for row in report['candidates']}, set(candidates))
        self.assertTrue(all(path.exists() for path in candidates))
        self.assertEqual(set(apply(report)), {str(path) for path in candidates})
        self.assertTrue(all(path.read_bytes() == b'checkpoint' for path in protected))

    def test_real_versions_last_contains_best_and_is_never_rewritten(self):
        last = self.file('last.pt', b'opaque best + last + optimizer states')
        self.file('completed.json', json.dumps(dict(best_tag='epoch_6_step_1403', last_tag='epoch_10_step_1403')).encode())
        best = self.file('epoch_6_step_1403.pt')
        latest = self.file('epoch_10_step_1403.pt')
        old = self.file('epoch_5_step_1403.pt')
        report = plan([self.exp])
        self.assertEqual([row['path'] for row in report['candidates']], [str(old)])
        apply(report)
        self.assertEqual(last.read_bytes(), b'opaque best + last + optimizer states')
        self.assertTrue(best.exists() and latest.exists())

    def test_reference_from_other_run_and_selected_json_alias_protect(self):
        self.file('last.pt')
        referenced = self.file('epoch_2.pt')
        selected = self.file('epoch_3.pt')
        other = self.exp / 'w2v_v31810_test'
        self.file('config.json', json.dumps(dict(parent_init=dict(checkpoint='/old/moved/epoch_2.pt'))).encode(), other)
        self.file('best_guarded.json', json.dumps(dict(tag='epoch_3')).encode())
        self.file('epoch_4.pt')
        result = plan([self.exp])
        self.assertEqual([Path(row['path']).name for row in result['candidates']], ['epoch_4.pt'])
        apply(result)
        self.assertTrue(referenced.exists() and selected.exists())

    def test_no_surviving_checkpoint_and_excluded_assets_are_not_deleted(self):
        lonely = self.file('epoch_2.pt')
        asset = self.file('assets/epoch_7.pt')
        self.file('warmup.pt')  # Cannot infer a surviving best/last from this.
        result = plan([self.exp])
        self.assertEqual(result['candidates'], [])
        self.assertEqual(len(result['skipped']), 1)
        apply(result)
        self.assertTrue(lonely.exists() and asset.exists())

    def test_hard_linked_best_is_preserved(self):
        import os
        selected = self.file('epoch_2.pt')
        os.link(selected, self.run / 'best.pt')
        other = self.file('epoch_3.pt')
        result = plan([self.exp])
        self.assertEqual([row['path'] for row in result['candidates']], [str(other)])
        apply(result)
        self.assertTrue(selected.exists())
        self.assertEqual((self.run / 'best.pt').read_bytes(), selected.read_bytes())

    def test_linked_best_target_is_retained(self):
        selected = self.file('epoch_2.pt')
        try:
            (self.run / 'best.pt').symlink_to(selected.name)
        except OSError:
            self.skipTest('OS does not permit symlink creation')
        other = self.file('epoch_3.pt')
        result = plan([self.exp])
        self.assertEqual([row['path'] for row in result['candidates']], [str(other)])
        apply(result)
        self.assertEqual((self.run / 'best.pt').read_bytes(), selected.read_bytes())

    def test_metadata_change_aborts_before_first_delete(self):
        self.file('last.pt')
        candidate = self.file('epoch_1.pt')
        metadata = self.file('config.json', b'{}')
        result = plan([self.exp])
        metadata.write_text('{"checkpoint":"epoch_1.pt"}')
        with self.assertRaisesRegex(RuntimeError, 'Input changed'):
            apply(result)
        self.assertTrue(candidate.exists())

    def test_invalid_json_and_wrong_root_fail_closed(self):
        self.file('last.pt')
        candidate = self.file('epoch_1.pt')
        self.file('best.json', b'broken')
        with self.assertRaisesRegex(RuntimeError, 'Unreadable metadata'):
            plan([self.exp])
        with self.assertRaises(ValueError):
            plan([self.exp.parent])
        self.assertTrue(candidate.exists())


if __name__ == '__main__':
    unittest.main()
