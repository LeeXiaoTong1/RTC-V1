import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from audit_v318_workers import recent_steps, failed_logs, kernel_evidence, cgroup_dirs


class Tests(unittest.TestCase):
    def test_recent_epoch_and_partial_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'steps.jsonl'
            rows = [dict(epoch=1, step=9, compute_seconds=99, wait_seconds=99),
                    dict(epoch=2, step=1, compute_seconds=4, wait_seconds=1),
                    dict(epoch=2, step=2, compute_seconds=6, wait_seconds=1)]
            p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n{"epoch":')
            r = recent_steps(p)
            self.assertEqual((r['epoch'],r['step'],r['count']), (2,2,2))
            self.assertEqual(r['compute_seconds'], 5)
            self.assertAlmostEqual(r['wait_fraction'], 1/6)

    def test_failed_pid_extraction_and_kernel_permissions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'exp').mkdir()
            (root/'exp'/'v318_test.log').write_text('RuntimeError: DataLoader worker (pid(s) 974674, 974675) exited unexpectedly\n')
            failures = failed_logs(root)
            self.assertEqual(failures[0]['worker_pids'], [974674,974675])
            def denied(*a, **kw): return SimpleNamespace(returncode=1, stderr='permission denied')
            self.assertFalse(kernel_evidence(failures,denied)['available'])
            def found(*a, **kw): return SimpleNamespace(returncode=0, stdout='[123] Killed process 974674 (python)\n[456] Killed process 1974675 (other)\n')
            self.assertEqual(kernel_evidence(failures,found)['matched_worker_pids'], [974674])

    def test_cgroup_v1_v2_ancestors_and_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            self.assertEqual(cgroup_dirs('0::/a/b', root), {root,root/'a',root/'a'/'b'})
            self.assertEqual(cgroup_dirs('4:memory:/a', root), {root/'memory',root/'memory'/'a'})
            self.assertFalse(cgroup_dirs('0::/../../escape', root))


if __name__ == '__main__': unittest.main()
