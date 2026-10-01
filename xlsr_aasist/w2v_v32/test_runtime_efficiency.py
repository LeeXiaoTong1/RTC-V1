"""Memory-query amortization preserves placement budgets and autograd values."""
import copy
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import torch

from .runtime import HybridActivationStore


_GIB = 1024 ** 3


class _Storage:
    def __init__(self, pointer, size):
        self.pointer, self.size = pointer, size

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.size


class _Tensor:
    """No CUDA allocation is needed to exercise the storage placement policy."""
    device = torch.device('cuda:0')

    def __init__(self, pointer, size=32):
        self.storage = _Storage(pointer, size)

    def numel(self):
        return self.storage.size // 4

    def element_size(self):
        return 4

    def untyped_storage(self):
        return self.storage

    def detach(self):
        return _Tensor(self.storage.pointer, self.storage.size)


class _Model:
    def parameters(self):
        return iter([_Tensor(1, 1000)])

    def buffers(self):
        return iter(())


class MemoryQueryTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.driver = self.patches.enter_context(patch('torch.cuda.mem_get_info', return_value=(10000, 20000)))
        self.reserved = self.patches.enter_context(patch('torch.cuda.memory_reserved', return_value=1000))
        self.allocated = self.patches.enter_context(patch('torch.cuda.memory_allocated', return_value=1000))
        self.patches.enter_context(patch('w2v_v32.runtime.available_host_bytes', return_value=_GIB))

    def make_store(self, gpu_budget=8000, reserve=500):
        store = HybridActivationStore(_Model(), gpu_budget_gib=gpu_budget / _GIB,
                                      reserve_gib=reserve / _GIB, host_budget_gib=.1,
                                      pin_memory=False)
        store._cpu.pack_hook = Mock(side_effect=lambda t: ('spilled', t.detach()))
        return store

    def test_one_driver_snapshot_replaces_hundreds_of_hook_queries(self):
        store = self.make_store()
        for index in range(100):
            store._pack(_Tensor(index + 10))
        self.assertEqual(self.driver.call_count, 1)
        self.assertEqual(store.cuda_memory_queries, 1)
        self.assertEqual(store.allocator_memory_checks, 100)
        self.assertEqual(store.gpu_bytes, 3200)
        self.assertEqual(store.offloaded_tensors, 0)
        store.refresh()
        self.assertEqual(self.driver.call_count, 2)
        self.assertEqual(store.cuda_memory_queries, 2)
        store._pack(_Tensor(1000))
        self.assertEqual(self.driver.call_count, 2)

    def test_live_allocator_growth_enforces_reserve_without_driver_query(self):
        store = self.make_store()
        self.allocated.return_value = 10600  # 11000 capacity - 10600 live = 400 < 500 reserve.
        self.assertEqual(store._pack(_Tensor(10))[0], 'spilled')
        self.assertEqual(store.offloaded_tensors, 1)
        self.assertEqual(store.gpu_bytes, 0)
        self.assertEqual(self.driver.call_count, 1)
        self.allocated.return_value = 10400
        self.assertEqual(store._pack(_Tensor(11))[0], torch.device('cuda:0'))
        self.assertEqual(store.gpu_bytes, 32)
        self.assertEqual(self.driver.call_count, 1)

    def test_refresh_observes_other_process_pressure_before_next_microbatch(self):
        store = self.make_store()
        store._pack(_Tensor(10))
        self.driver.return_value = (300, 20000)
        self.assertEqual(store.refresh(), 300)
        self.assertEqual(store._pack(_Tensor(11))[0], 'spilled')
        self.assertEqual(store.gpu_bytes, 32)
        self.assertEqual(store.bytes, 32)
        self.assertEqual(self.driver.call_count, 2)

    def test_gpu_cap_and_alias_accounting_remain_exact(self):
        store = self.make_store(gpu_budget=64)
        store._pack(_Tensor(10))
        store._pack(_Tensor(10))  # Existing storage adds no new memory charge/query.
        store._pack(_Tensor(1, 1000))  # Resident parameter is exempt.
        self.assertEqual(store.allocator_memory_checks, 1)
        store._pack(_Tensor(11))
        self.assertEqual(store.gpu_bytes, 64)
        self.assertEqual(store._pack(_Tensor(12))[0], 'spilled')
        self.assertEqual(store.gpu_kept_tensors, 3)
        self.assertEqual(store.offloaded_tensors, 1)
        self.assertEqual(self.driver.call_count, 1)

    def test_allocator_cache_growth_does_not_double_count_available_capacity(self):
        store = self.make_store()
        self.reserved.return_value = 10500
        self.allocated.return_value = 10400
        self.assertEqual(store._estimated_cuda_free(), 600)
        # A new driver snapshot sees correspondingly less free VRAM.
        self.driver.return_value = (500, 20000)
        self.assertEqual(store.refresh(), 600)
        self.assertEqual(store._estimated_cuda_free(), 600)

    def test_host_spill_limit_still_fails_before_copy(self):
        store = self.make_store(gpu_budget=0)
        store.limit = 16
        with self.assertRaisesRegex(MemoryError, 'No optimizer update'):
            store._pack(_Tensor(10))
        store._cpu.pack_hook.assert_not_called()

    def test_cpu_refresh_and_hooks_leave_gradients_and_rng_unchanged(self):
        torch.manual_seed(717)
        reference = torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.Tanh(), torch.nn.Linear(11, 2))
        actual = copy.deepcopy(reference)
        x = torch.randn(5, 7)
        expected = reference(x).square().sum()
        expected.backward()
        rng = torch.get_rng_state().clone()
        store = HybridActivationStore(actual)
        with store:
            self.assertEqual(store.refresh(), 0)
            actual(x).square().sum().backward()
        self.driver.assert_not_called()
        self.allocated.assert_not_called()
        self.assertEqual(store.cuda_memory_queries, 0)
        self.assertEqual(store.allocator_memory_checks, 0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for got, wanted in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(got.grad, wanted.grad, rtol=0, atol=0)


class CudaMemoryQueryTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA residency/spill requires CUDA')
    def test_multiple_refreshes_preserve_loss_and_gradients_with_cpu_spill(self):
        torch.manual_seed(227)
        reference = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.Tanh(),
                                        torch.nn.Linear(128, 2)).cuda()
        actual = copy.deepcopy(reference)
        inputs = [torch.randn(8, 64, device='cuda') for _ in range(3)]
        expected = sum(reference(x).square().sum() for x in inputs)
        expected.backward()
        store = HybridActivationStore(actual, gpu_budget_gib=4096 / _GIB,
                                      host_budget_gib=.1, reserve_gib=0)
        with store:
            losses = []
            for x in inputs:
                store.refresh()
                losses.append(actual(x).square().sum())
            got = sum(losses)
            got.backward()
        self.assertEqual(store.cuda_memory_queries, 4)
        self.assertGreater(store.allocator_memory_checks, 4)
        self.assertGreater(store.gpu_bytes, 0)
        self.assertGreater(store.bytes, 0)
        self.assertLessEqual(store.gpu_bytes, store.gpu_limit)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        for (name,a),(_,b) in zip(actual.named_parameters(),reference.named_parameters()):
            with self.subTest(parameter=name):
                self.assertTrue(bool(torch.isfinite(a.grad).all()))
                # CPU spill preserves dtype/values, but restored strides may
                # change CUDA backward reduction order. Storage roundtrips are
                # checked bitwise separately in test_runtime.StorageTests.
                torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-8)


if __name__ == '__main__':
    unittest.main()
