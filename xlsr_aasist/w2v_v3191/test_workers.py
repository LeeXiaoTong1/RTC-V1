from types import SimpleNamespace
import unittest
from .workers import settings


class WorkerBudgetTests(unittest.TestCase):
    def args(self, **values):
        return SimpleNamespace(**dict(dict(workers=None, eval_workers=None,
            prefetch_factor=1, eval_batch=24), **values))

    def host(self, gib=64, cores=32):
        return dict(usable_cpu_cores=cores, available_memory_bytes=int(gib*1024**3),
                    cgroup_cpu_quota=cores)

    def test_candidates_expand_budget_and_auto_reduces_workers(self):
        old=dict(whole_max_frames=3000,stream_sources=24)
        ordinary=settings(self.args(robust_candidate_count=2),old,self.host())
        expanded=settings(self.args(robust_candidate_count=4),old,self.host())
        self.assertGreater(expanded['worker_resource_audit']['estimated_train_task_pcm_bytes'],
                           ordinary['worker_resource_audit']['estimated_train_task_pcm_bytes'])
        self.assertLessEqual(expanded['workers'],ordinary['workers'])
        self.assertLessEqual(expanded['worker_resource_audit']['estimated_pool_bytes'],32*1024**3)

    def test_explicit_cpu_and_memory_overcommit_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'CPU cores'):
            settings(self.args(workers=33),{},self.host())
        with self.assertRaisesRegex(ValueError,'RAM estimate'):
            settings(self.args(workers=8,eval_workers=4),{},self.host(gib=8))

    def test_explicit_large_count_has_no_arbitrary_eight_worker_cap(self):
        result=settings(self.args(workers=12,eval_workers=4),dict(whole_max_frames=100),
                        self.host(gib=128))
        self.assertEqual(result['workers'],12)

    def test_zero_workers_has_no_prefetch_process_and_keeps_main_cache_budget(self):
        result=settings(self.args(workers=0,eval_workers=0),{},self.host())
        self.assertEqual(result['worker_resource_audit']['queued_train_batches'],0)
        self.assertEqual(result['raw_audio_cache_mib'],512)
        self.assertGreater(result['worker_resource_audit']['estimated_pool_bytes'],0)

    def test_larger_prefetch_reduces_automatic_concurrency(self):
        a=settings(self.args(prefetch_factor=1),{},self.host())
        b=settings(self.args(prefetch_factor=3),{},self.host())
        self.assertLessEqual(b['workers'],a['workers'])
        self.assertLessEqual(b['worker_resource_audit']['estimated_pool_bytes'],32*1024**3)


if __name__=='__main__':
    unittest.main()
