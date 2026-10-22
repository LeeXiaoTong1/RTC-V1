from pathlib import Path
import tempfile
import unittest
from .resources import audit_training
from .baseline import source_digest, verify_baseline
from w2v_v5.test_integration import inventory


class ResourceTests(unittest.TestCase):
    def test_every_epoch_atomic_group_bound_and_no_partial_group_fallback(self):
        rows,cfg=inventory()
        rows=[dict(r,output_samples=16000) for r in rows]
        cfg.update(stream_sources=16,source_draws_per_stream=[16]*4,group_quotas=(4,)*4,
                   epochs=2,microbatch=16,frame_budget=9000)
        result=audit_training(rows,[],cfg)['hard_positive']
        self.assertGreater(result['scheduled_groups'],0)
        self.assertEqual(result['all_epochs_checked'],2)
        self.assertLessEqual(result['maximum_atomic_views'],13)
        self.assertEqual(result['extra_encoder_views_per_update_at_most'],4)
        with self.assertRaisesRegex(ValueError,'metric group preflight'):
            audit_training(rows,[],dict(cfg,frame_budget=1000))

    def test_baseline_hash_ignores_windows_line_endings_but_not_code_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            file=root/'original.py'
            file.write_bytes(b'x = 1\ny = 2\n')
            expected={'original.py':source_digest(file)}
            file.write_bytes(b'x = 1\r\ny = 2\r\n')
            verify_baseline(root,expected)
            file.write_bytes(b'x = 2\ny = 2\n')
            with self.assertRaisesRegex(ValueError,'Inherited V5.1 code changed'):
                verify_baseline(root,expected)


if __name__=='__main__':
    unittest.main()
