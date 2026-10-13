"""Choose physical execution using real Train views, with exact probe rollback.

The 16-source / 48-view objective is invariant. Profiling changes only padded
microbatch limits and activation recomputation. No audio or feature disk cache
is created. The recorded winner is reused on resume instead of re-tuning it.
"""
import gc
import math
from pathlib import Path
import time

import torch

from w2v_v39.common import atomic_json, read_json
from w2v_v313.state import capture_rng, restore_rng, to_cpu, identity
from w2v_v315.step import clear_features


GIB = 1024 ** 3


def checkpointing(model, enabled):
    """The inherited runtime skips checkpointing for frozen encoder blocks."""
    model.backbone.encoder.gradient_checkpointing = bool(enabled)


def execution_candidates(cfg):
    """At most six trials; the upper limit never exceeds one logical batch."""
    size, budget, sources = (cfg['microbatch'], cfg['frame_budget'], cfg['source_batch'])
    if (any(isinstance(x, bool) or not isinstance(x, int) for x in (size, budget, sources))
            or size < 3 or budget < 1 or sources < 1):
        raise ValueError('Positive execution limits and physical microbatch >=3 required')
    maximum = min(size, 3 * sources)
    levels = []
    for proposed_size, proposed_budget in ((12, 7200), (24, 14400), (maximum, budget)):
        level = (min(maximum, proposed_size), min(budget, proposed_budget))
        if level not in levels:
            levels.append(level)
    # Checkpointing first provides a safer initial candidate for a large encoder.
    return [dict(microbatch=n, frame_budget=b, checkpointing=enabled)
            for enabled in (True, False) for n, b in levels]


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape
                and left.dtype == right.dtype and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(_equal(left[k], right[k]) for k in left))
    if isinstance(left, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_equal(a, b) for a, b in zip(left, right)))
    return left == right


class ProbeSnapshot:
    """Rollback all parameters, buffers, gradients, Adam moments and RNG.

    Frozen parameters and nonpersistent buffers are included deliberately. A
    profiling failure must not leave a mutated detector, and CPU Adam state
    must never alias this immutable snapshot after optimizer.load_state_dict.
    """
    def __init__(self, model, auxiliary, optimizer):
        self.modules = (model, auxiliary)
        self.optimizer = optimizer
        self.weights = []
        self.buffers = []
        self.gradients = []
        self.requires_grad = []
        self.modes = []
        for module in self.modules:
            self.weights.append({n: p.detach().cpu().clone() for n, p in module.named_parameters()})
            self.buffers.append({n: p.detach().cpu().clone() for n, p in module.named_buffers()})
            self.gradients.append({n: None if p.grad is None else p.grad.detach().cpu().clone()
                                   for n, p in module.named_parameters()})
            self.requires_grad.append({n: p.requires_grad for n, p in module.named_parameters()})
            self.modes.append({n: m.training for n, m in module.named_modules()})
        self.moments = to_cpu(optimizer.state_dict())
        self.rng = capture_rng()
        self.checkpointing = model.backbone.encoder.gradient_checkpointing

    @torch.no_grad()
    def restore(self):
        clear_features(self.modules[0])
        self.optimizer.zero_grad(set_to_none=True)
        for i, module in enumerate(self.modules):
            parameters, buffers = dict(module.named_parameters()), dict(module.named_buffers())
            if parameters.keys() != self.weights[i].keys() or buffers.keys() != self.buffers[i].keys():
                raise RuntimeError('Profiling changed the module parameter/buffer inventory')
            for name, parameter in parameters.items():
                parameter.copy_(self.weights[i][name])
                parameter.requires_grad_(self.requires_grad[i][name])
                gradient = self.gradients[i][name]
                parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
            for name, buffer in buffers.items():
                buffer.copy_(self.buffers[i][name])
            for name, child in module.named_modules():
                child.training = self.modes[i][name]
        # Clone on every load: torch can retain references to CPU step tensors.
        self.optimizer.load_state_dict(to_cpu(self.moments))
        self.modules[0].backbone.encoder.gradient_checkpointing = self.checkpointing
        restore_rng(self.rng)

    def assert_restored(self):
        for i, module in enumerate(self.modules):
            for name, parameter in module.named_parameters():
                if (not _equal(parameter, self.weights[i][name])
                        or parameter.requires_grad != self.requires_grad[i][name]
                        or not _equal(parameter.grad, self.gradients[i][name])):
                    raise AssertionError('Profiling changed parameter/gradient: ' + name)
            for name, buffer in module.named_buffers():
                if not _equal(buffer, self.buffers[i][name]):
                    raise AssertionError('Profiling changed a buffer: ' + name)
            if {n: m.training for n, m in module.named_modules()} != self.modes[i]:
                raise AssertionError('Profiling changed module training flags')
        if not _equal(self.optimizer.state_dict(), self.moments):
            raise AssertionError('Profiling changed optimizer state')
        if not _equal(capture_rng(), self.rng):
            raise AssertionError('Profiling changed random generator state')
        if self.modules[0].backbone.encoder.gradient_checkpointing != self.checkpointing:
            raise AssertionError('Profiling changed activation checkpointing')


def _hardware(device):
    if not str(device).startswith('cuda'):
        return dict(device_name='cpu', device_total_bytes=0)
    info = torch.cuda.get_device_properties(torch.device(device))
    return dict(device_name=info.name, device_total_bytes=info.total_memory)


def memory_budget(free, reserved, total, requested_reserve):
    """Leave >=4 GiB and >=10% of device RAM; respect other GPU occupants."""
    if not all(math.isfinite(float(v)) and v >= 0 for v in (free, reserved, total, requested_reserve)):
        raise ValueError('Invalid GPU memory accounting')
    reserve = max(int(requested_reserve), 4 * GIB, math.ceil(total * .10))
    return max(0, min(total - reserve, free + reserved - reserve)), reserve


def _winner(attempts):
    good = [row for row in attempts if row.get('eligible') and row.get('status') == 'ok'
            and math.isfinite(row.get('seconds_per_update', float('inf')))
            and row['seconds_per_update'] > 0]
    if not good:
        raise RuntimeError('No measured execution leaves the required GPU reserve; reduce physical limits in a NEW run')
    return min(good, key=lambda row: row['seconds_per_update'])


def select_execution(model, auxiliary, optimizer, plan, cfg, run):
    path = Path(run) / 'execution_plan.json'
    signature, hardware = identity(cfg), _hardware(cfg['device'])
    candidates = execution_candidates(cfg)
    if path.exists():
        saved = read_json(path)
        if saved.get('config_identity') != signature:
            raise ValueError('Execution profile belongs to another configuration')
        if any(saved.get(k) != value for k, value in hardware.items()):
            raise ValueError('Resume hardware changed; recorded execution cannot be replaced silently')
        if (saved.get('logical_sources') != cfg['source_batch']
                or saved.get('full_views_per_update') != 3 * cfg['source_batch']
                or saved.get('selected') not in candidates
                or saved.get('rollback_verified') is not True):
            raise ValueError('Recorded physical execution or logical objective differs')
        return saved

    selected = dict(microbatch=min(cfg['microbatch'], 3 * cfg['source_batch']),
                    frame_budget=cfg['frame_budget'], checkpointing=bool(cfg.get('checkpointing', True)))
    saved = dict(config_identity=signature, selected=selected, attempts=[], rollback_verified=False,
                 logical_sources=cfg['source_batch'], full_views_per_update=3 * cfg['source_batch'],
                 candidates=candidates, **hardware)
    if not cfg['autotune'] or not str(cfg['device']).startswith('cuda'):
        saved.update(reason='Explicit physical limits; no CUDA profiling', rollback_verified=True)
        atomic_json(path, saved)
        return saved

    # The data module selects actual median/long/longest Train sources in every
    # group and applies the slowest allowed base augmentation to both views.
    from .data import probe_examples
    from .step import train_step
    snapshot = ProbeSnapshot(model, auxiliary, optimizer)
    try:
        examples = probe_examples(plan, cfg, run)
        for trial in candidates:
            snapshot.restore()
            gc.collect()
            torch.cuda.empty_cache()
            checkpointing(model, trial['checkpointing'])
            torch.cuda.reset_peak_memory_stats()
            result = dict(trial)
            print('V319_PROFILE testing ' + str(trial), flush=True)
            try:
                elapsed = []
                for _ in range(2):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    train_step(model, auxiliary, optimizer, examples, dict(cfg, **trial), warm=1.)
                    torch.cuda.synchronize()
                    elapsed.append(time.perf_counter() - started)
                free, total = torch.cuda.mem_get_info()
                reserved = torch.cuda.memory_reserved()
                peak = torch.cuda.max_memory_reserved()
                usable, reserve = memory_budget(free, reserved, total, cfg.get('gpu_reserve_bytes', 4 * GIB))
                result.update(seconds_per_update=sum(elapsed) / len(elapsed),
                              peak_reserved_bytes=peak, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              available_peak_bytes=usable, required_reserve_bytes=reserve,
                              eligible=peak <= usable, status='ok')
            except torch.cuda.OutOfMemoryError:
                result.update(status='out_of_memory', eligible=False)
            saved['attempts'].append(result)
            print('V319_PROFILE result ' + str(result), flush=True)
    finally:
        snapshot.restore()
        snapshot.assert_restored()
        del snapshot
        gc.collect()
        torch.cuda.empty_cache()
    try:
        winner = _winner(saved['attempts'])
    except RuntimeError:
        atomic_json(Path(run) / 'execution_profile_failed.json', saved)
        raise
    saved.update(selected={k: winner[k] for k in ('microbatch', 'frame_budget', 'checkpointing')},
                 rollback_verified=True, reason='Fastest measured real Train update within GPU reserve')
    atomic_json(path, saved)
    return saved
