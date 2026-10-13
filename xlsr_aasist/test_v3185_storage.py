"""Only synthetic temporary payloads are deleted by these tests."""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_v3184_storage import fixture
from w2v_v3184.noise import sha256
from v3184_storage import noise_plan
from v3185_storage import (active_jobs, canonical_directory, emit, execute,
                           is_producer, main, retire_verified)


class StorageTests(unittest.TestCase):
    def test_training_preparation_and_supervisors_are_detected(self):
        for args in (["python", "-m", "w2v_v3184_aug5.train"],
                     ["python3", "-m", "w2v_v3185.workflow"],
                     ["python", "-m", "noise_v3184_rnnoise.prepare"],
                     ["bash", "v3184_aug5_supervise.sh", "output.log"],
                     ["bash", "v3185_supervise.sh", "output.log"],
                     ["bash", "run_w2v_v3185.sh"],
                     ["bash", "prepare_noise_v3184_rnnoise.sh"]):
            self.assertTrue(is_producer(args), args)
        for args in ([], ["python", "v3185_storage.py", "--apply"],
                     ["bash", "cleanup_w2v_v3185.sh", "--apply"], ["python", "unrelated.py"]):
            self.assertFalse(is_producer(args), args)

    def test_process_inventory_excludes_self_and_exited_processes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for pid, args in ((101, ['python', '-m', 'w2v_v3184_aug5.train']),
                              (102, ['bash', 'v3184_aug5_supervise.sh']),
                              (103, ['python', 'v3185_storage.py'])):
                entry = root / str(pid); entry.mkdir()
                (entry / 'cmdline').write_bytes('\0'.join(args).encode())
            (root / '104').mkdir()  # Simulate exit before cmdline can be read.
            uid = (root / '101').stat().st_uid
            self.assertEqual([x['pid'] for x in active_jobs(root, uid, 999)], [101, 102])
            self.assertEqual([x['pid'] for x in active_jobs(root, uid, 101)], [102])

    def test_preview_keeps_all_payloads_and_protected_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); bank, sources = fixture(root)
            before = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file()}
            result = execute(root, bank, sources_path=sources, progress=lambda text: None)
            self.assertEqual(len(result['candidates']), 4)
            for name, digest in before.items():
                self.assertEqual(sha256(root / name), digest)

    def test_active_training_refuses_before_planning_or_deleting(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); bank, sources = fixture(root)
            with patch('v3185_storage.idle_launch_lock', side_effect=RuntimeError('Active training')), \
                    patch('v3185_storage.noise_plan') as planning:
                with self.assertRaisesRegex(RuntimeError, 'Active training'):
                    execute(root, bank, apply=True, sources_path=sources, progress=lambda text: None)
                planning.assert_not_called()
            self.assertTrue((bank / 'raw/rnnoise_contributions.tar.gz').is_file())

    def test_active_preparation_refuses_before_planning(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); bank, sources = fixture(root)
            with patch('v3185_storage.preparation_lock', side_effect=RuntimeError('Noise preparation')), \
                    patch('v3185_storage.noise_plan') as planning:
                with self.assertRaisesRegex(RuntimeError, 'Noise preparation'):
                    execute(root, bank, sources_path=sources, progress=lambda text: None)
                planning.assert_not_called()

    def test_apply_only_removes_whitelisted_raw_and_counts_actual_space(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); bank, sources = fixture(root)
            plan = noise_plan(bank, sources)
            before = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file()}
            removed = {str((bank / row['file']).relative_to(root)) for row in plan['candidates']}
            result = retire_verified(plan, guard=lambda: None, progress=lambda text: None)
            self.assertEqual(result['removed_files'], 4)
            self.assertEqual(result['removed_bytes'], plan['reclaimable_bytes'])
            self.assertEqual(result['freed_bytes'], max(0, result['free_bytes_after'] - result['free_bytes_before']))
            for name, digest in before.items():
                if name not in removed:
                    self.assertEqual(sha256(root / name), digest)
            again = retire_verified(noise_plan(bank, sources), guard=lambda: None, progress=lambda text: None)
            self.assertEqual(again['removed_files'], 0)

    def test_job_appearing_after_plan_prevents_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            bank, sources = fixture(Path(d)); plan = noise_plan(bank, sources)
            def blocked(): raise RuntimeError('Active training')
            with self.assertRaisesRegex(RuntimeError, 'Active training'):
                retire_verified(plan, guard=blocked, progress=lambda text: None)
            self.assertTrue((bank / 'raw/rnnoise_contributions.tar.gz').is_file())

    def test_changed_payload_cancels_all_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            bank, sources = fixture(Path(d)); plan = noise_plan(bank, sources)
            payload = bank / 'raw/silero_vad.jit'
            payload.write_bytes(b'x' * payload.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'no files removed'):
                retire_verified(plan, guard=lambda: None, progress=lambda text: None)
            self.assertTrue((bank / 'raw/rnnoise_contributions.tar.gz').is_file())

    def test_linked_root_and_unready_bank_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(Path, 'is_symlink', return_value=True):
                with self.assertRaisesRegex(ValueError, 'link/junction'):
                    canonical_directory(d)
            root = Path(d); (root / 'bank').mkdir()
            with self.assertRaises(FileNotFoundError):
                execute(root, root / 'bank', progress=lambda text: None)

    def test_terminal_output_is_ascii_and_errors_return_nonzero(self):
        with patch('sys.stdout', new_callable=io.StringIO) as stdout:
            emit('PATH=/tmp/\u6d4b\u8bd5')
            stdout.getvalue().encode('ascii')
        with patch('v3185_storage.execute', side_effect=RuntimeError('Active training')), \
                patch('v3185_storage.emit') as output:
            self.assertEqual(main([]), 2)
            output.assert_called_once_with('CLEANUP_REFUSED=Active training')


if __name__ == '__main__':
    unittest.main()
