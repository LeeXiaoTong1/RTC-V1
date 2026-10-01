"""Bounded GPU-first saved activations, with exact CPU spill for autograd.

This changes placement only. It neither quantizes saved tensors nor replays an
encoder pass. A reserve protects transient forward/backward work; it is not an
absolute peak-memory guarantee for arbitrarily long individual recordings.
"""
import math
import torch
from w2v_v3.step import available_host_bytes


_GIB = 1024 ** 3


def _nonnegative(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def _keep_on_gpu(size, retained, limit, free, reserve):
    """Pure policy: charge whole unique storage, and retain a free-memory reserve."""
    return size >= 0 and retained + size <= limit and free >= reserve


def _usable_cuda_bytes(device):
    """Driver-free plus this process's reusable allocator cache (no other process RAM)."""
    free, _ = torch.cuda.mem_get_info(device)
    cached = max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
    return free + cached


class HybridActivationStore(torch.autograd.graph.saved_tensors_hooks):
    """Keep saved activations on GPU until a bounded budget; then spill to CPU.

    GPU storage aliases are counted once. A detached representative keeps each
    accounted storage alive until context exit, preventing pointer reuse from
    defeating the budget. Model weights and buffers remain resident and exempt.
    CPU spill is charged conservatively per saved tensor (no alias assumption).

    Counters remain available after exit: ``gpu_bytes`` is retained unique GPU
    storage, ``bytes`` is copied host tensor bytes, ``gpu_limit`` and ``limit``
    are their effective byte budgets. ``gpu_kept_tensors`` / ``offloaded_tensors``
    count hooks; both are zero for an entirely CPU model.

    ``refresh()`` snapshots driver memory before each physical microbatch. Hook
    calls then use allocator counters, rather than querying the CUDA driver for
    every saved tensor. Changes in this process's allocations are reflected at
    every hook. Other processes' allocations are observed at the next refresh;
    as with the original check-then-allocate policy, the reserve cannot protect
    against an unrelated process taking arbitrary GPU memory during a forward.
    """
    def __init__(self, model, gpu_budget_gib=18., host_budget_gib=0., reserve_gib=6.,
                 pin_memory=True):
        gpu_budget_gib = _nonnegative(gpu_budget_gib, 'gpu_budget_gib')
        host_budget_gib = _nonnegative(host_budget_gib, 'host_budget_gib')
        reserve_gib = _nonnegative(reserve_gib, 'reserve_gib')
        resident_tensors = [t for t in (*model.parameters(), *model.buffers()) if t.numel()]
        devices = {t.device for t in resident_tensors if t.device.type == 'cuda'}
        if len(devices) > 1:
            raise ValueError('Hybrid activation storage supports one CUDA device')
        self.device = next(iter(devices), None)
        self.reserve = int(reserve_gib * _GIB)
        self.gpu_limit = 0
        self.cuda_memory_queries = self.allocator_memory_checks = 0
        self._cuda_capacity = 0
        if self.device is not None:
            free = self.refresh()
            self.gpu_limit = min(int(gpu_budget_gib * _GIB), max(0, free - self.reserve))
        available = available_host_bytes()
        if available is None and not host_budget_gib and self.device is not None:
            raise RuntimeError('Cannot determine host RAM; set host_budget_gib explicitly')
        requested = int(host_budget_gib * _GIB) if host_budget_gib else int((available or 0) * .5)
        if available is not None:
            requested = min(requested, max(0, available - min(2 * _GIB, available // 4)))
        self.limit = max(0, requested)
        self.bytes = self.gpu_bytes = 0
        self.gpu_kept_tensors = self.offloaded_tensors = 0
        self._kept_storages = {}
        self._resident = {self._storage_key(t) for t in resident_tensors}
        self._cpu = torch.autograd.graph.save_on_cpu(pin_memory=pin_memory)
        super().__init__(self._pack, self._cpu.unpack_hook)

    @staticmethod
    def _storage_key(tensor):
        storage = tensor.untyped_storage()
        return tensor.device, storage.data_ptr(), storage.nbytes()

    def refresh(self):
        """Refresh external-memory pressure once per physical microbatch.

        Driver-free memory plus our allocator's reserved memory is the total
        capacity currently available to this process. Subtracting live tensor
        allocation at each hook accounts for growing activations without a
        driver query. This does not call synchronize or change allocator state.
        CPU callers may use the same entry point; they make no CUDA calls.
        """
        if self.device is None:
            return 0
        free, _ = torch.cuda.mem_get_info(self.device)
        self.cuda_memory_queries += 1
        reserved = torch.cuda.memory_reserved(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        # Clamp the reusable cache at zero, matching _usable_cuda_bytes.
        usable = free + max(0, reserved - allocated)
        self._cuda_capacity = usable + allocated
        return usable

    def _estimated_cuda_free(self):
        self.allocator_memory_checks += 1
        return max(0, self._cuda_capacity - torch.cuda.memory_allocated(self.device))

    def _pack(self, tensor):
        if tensor.device.type == 'cpu' or not tensor.numel():
            return tensor.device, tensor.detach()
        if self.device is None or tensor.device != self.device:
            raise ValueError('Saved activation belongs to an unconfigured CUDA device')
        key = self._storage_key(tensor)
        if key in self._resident:
            return tensor.device, tensor.detach()
        if key in self._kept_storages:
            self.gpu_kept_tensors += 1
            return tensor.device, tensor.detach()
        size = key[2]
        free = self._estimated_cuda_free()
        if _keep_on_gpu(size, self.gpu_bytes, self.gpu_limit, free, self.reserve):
            saved = tensor.detach()
            self._kept_storages[key] = saved
            self.gpu_bytes += size
            self.gpu_kept_tensors += 1
            return tensor.device, saved
        size = tensor.numel() * tensor.element_size()
        if self.bytes + size > self.limit:
            raise MemoryError(
                f'V3.2 saved activations exceed the {self.limit / _GIB:.2f} GiB host spill '
                'budget. No optimizer update has occurred; reduce both logical source '
                'batch sizes together, or increase the host budget when RAM permits.')
        packed = self._cpu.pack_hook(tensor)
        self.bytes += size
        self.offloaded_tensors += 1
        return packed

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self._kept_storages.clear()

