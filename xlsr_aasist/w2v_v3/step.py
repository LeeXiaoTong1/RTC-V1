"""One encoder forward per microbatch, exact logical-batch CE and CKA gradients.

CUDA saved activations are offloaded to bounded host memory. Only the encoder's
explicit gradient-checkpointed layers recompute during backward; no second
whole-encoder pass is made for CKA. No optimizer update occurs until the entire
logical batch has a finite loss and finite gradients.
"""
from contextlib import nullcontext
import math
import os
import torch
from torch.nn import functional as F
from w2v_aasist.runtime import amp_context
from .model import diversity_cka, microbatches


def available_host_bytes():
    """Actual available memory, including the process's Linux cgroup limit."""
    available = None
    if os.name == 'nt':
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(n, ctypes.c_ulonglong) for n in ('total', 'available', 'page_total',
                          'page_available', 'virtual_total', 'virtual_available', 'extended')]]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            available = int(status.available)
    else:
        from pathlib import Path
        try:
            fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            available = int(fields['MemAvailable'].split()[0]) * 1024
        except (OSError, KeyError, ValueError):
            pass
        # Constrain the host figure when training inside a container or scheduler.
        for limit_path, used_path in (
            ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
            ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes'),
        ):
            try:
                limit, used = int(Path(limit_path).read_text()), int(Path(used_path).read_text())
                headroom = max(0, limit - used)
                available = headroom if available is None else min(available, headroom)
            except (OSError, ValueError):
                continue
    return available


class ActivationOffload(torch.autograd.graph.saved_tensors_hooks):
    """Bounded save_on_cpu, retaining already-resident model weights on device.

    Copying every saved matrix view would duplicate weights per microbatch.
    Persistent parameter/buffer storage is already resident, so only transient
    saved activations are copied. The limit measures the total staged bytes for
    this logical batch, conservatively without relying on allocator statistics.
    """
    def __init__(self, model, activation_budget_gib=0., pin_memory=True):
        if not math.isfinite(activation_budget_gib) or activation_budget_gib < 0:
            raise ValueError('activation_budget_gib must be finite and nonnegative')
        available = available_host_bytes()
        if available is None and not activation_budget_gib:
            raise RuntimeError('Cannot determine host RAM; set activation_budget_gib explicitly')
        requested = int(activation_budget_gib * 1024**3) if activation_budget_gib else int(available * .5)
        if available is not None:
            reserve = min(2 * 1024**3, available // 4)
            requested = min(requested, max(0, available - reserve))
        if requested <= 0:
            raise MemoryError('No available host activation budget; optimizer has not advanced')
        self.limit, self.bytes = requested, 0
        resident = {(t.device, t.untyped_storage().data_ptr())
                    for t in (*model.parameters(), *model.buffers()) if t.numel()}
        implementation = torch.autograd.graph.save_on_cpu(pin_memory=pin_memory)

        def pack(tensor):
            storage = (tensor.device, tensor.untyped_storage().data_ptr())
            if tensor.device.type == 'cpu' or storage in resident:
                # Hooks must not retain the original autograd Tensor object:
                # its grad_fn can point back to the node storing this payload.
                return tensor.device, tensor.detach()
            size = tensor.numel() * tensor.element_size()
            if self.bytes + size > self.limit:
                raise MemoryError(
                    f'V3 activation offload exceeds {self.limit / 1024**3:.2f} GiB host budget. '
                    'Reduce ordinary_batch and noisy_batch together or explicitly raise '
                    'activation_budget_gib when RAM permits; no optimizer update was performed.')
            self.bytes += size
            return implementation.pack_hook(tensor)

        super().__init__(pack, implementation.unpack_hook)


def _components(examples, weights, noisy_class_weights, device, noisy_weight):
    if not examples or not 0 <= noisy_weight <= 1:
        raise ValueError('Nonempty examples and noisy_weight in [0,1] required')
    labels = torch.tensor([e['label'] for e in examples], dtype=torch.long, device=device)
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('Labels must be fake=0 or real=1')
    noisy = torch.tensor([e['noisy'] for e in examples], dtype=torch.bool, device=device)
    ordinary_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    noisy_weights = torch.as_tensor(noisy_class_weights if noisy_class_weights is not None else weights,
                                    dtype=torch.float32, device=device)
    for w in (ordinary_weights, noisy_weights):
        if w.shape != (2,) or not bool(torch.isfinite(w).all()) or not bool((w > 0).all()):
            raise ValueError('Two finite positive class weights are required per component')
    n, s = int((~noisy).sum()), int(noisy.sum())
    if not n:
        raise ValueError('An ordinary classification component is required')
    coefficients = ordinary_weights[labels] * ((1 - noisy_weight) if s else 1.) / n
    if s:
        coefficients = torch.where(noisy, noisy_weights[labels] * noisy_weight / s, coefficients)
    return labels, noisy, ordinary_weights, noisy_weights, coefficients, n, s


def loss_function(logits, blocks, labels, noisy, weights, noisy_weight=.3, cka_weight=.01,
                  noisy_class_weights=None):
    """Reference logical-batch objective, also useful for gradient-equivalence checks."""
    examples = [{'label': int(y), 'noisy': bool(n)} for y, n in zip(labels, noisy)]
    labels, noisy, ordinary_weights, noisy_weights, coefficients, n, s = _components(
        examples, weights, noisy_class_weights, logits.device, noisy_weight)
    ce = F.cross_entropy(logits.float(), labels, reduction='none')
    ordinary_ce = (ce[~noisy] * ordinary_weights[labels[~noisy]]).sum() / n
    noisy_ce = (ce[noisy] * noisy_weights[labels[noisy]]).sum() / s if s else ce.sum() * 0
    classification = (ce * coefficients).sum()
    cka = diversity_cka(blocks) if cka_weight else blocks.sum() * 0
    return classification + cka_weight * cka, {
        'ce': classification, 'ordinary_ce': ordinary_ce, 'noisy_ce': noisy_ce, 'cka': cka}


def supervised_step(model, examples, optimizer, weights, device, amp='bf16',
                    noisy_weight=.3, cka_weight=.01, grad_clip=1., microbatch=4,
                    frame_budget=1600, noisy_class_weights=None,
                    offload_activations=True, activation_budget_gib=0.):
    """Accumulate CE/CKA exactly without replaying frozen encoder computation.

    With CKA enabled, tiny pooled tensors retain the complete batch graph while
    saved CUDA activations live on CPU. With CKA disabled, each microbatch is
    immediately backpropagated using the same global CE denominators.
    """
    if not math.isfinite(cka_weight) or cka_weight < 0 or not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError('Finite nonnegative CKA weight and positive gradient clip required')
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Exact variable-length microbatching requires no BatchNorm')
    device = torch.device(device)
    labels, noisy, ow, nw, coefficients, n, s = _components(
        examples, weights, noisy_class_weights, device, noisy_weight)
    optimizer.zero_grad(set_to_none=True)
    offload = ActivationOffload(model, activation_budget_gib) if offload_activations and device.type == 'cuda' else None
    zs, hs, order = [], [], []
    stats = {'ordinary_ce': 0., 'noisy_ce': 0.}
    classification = torch.zeros((), device=device)
    with offload if offload is not None else nullcontext():
        for indices, features, mask in microbatches(examples, microbatch, frame_budget):
            with amp_context(device, amp):
                z, h = model(features.to(device), mask.to(device))
            if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(h).all()):
                raise FloatingPointError('Non-finite model output; optimizer has not advanced')
            ce = F.cross_entropy(z.float(), labels[indices], reduction='none')
            local_loss = (ce * coefficients[indices]).sum()
            if not bool(torch.isfinite(local_loss)):
                raise FloatingPointError('Non-finite classification loss; optimizer has not advanced')
            local_noisy, local_labels = noisy[indices], labels[indices]
            stats['ordinary_ce'] += float((ce.detach()[~local_noisy] * ow[local_labels[~local_noisy]]).sum()) / n
            if s:
                stats['noisy_ce'] += float((ce.detach()[local_noisy] * nw[local_labels[local_noisy]]).sum()) / s
            zs.append(z.detach().float())
            order.extend(indices)
            if cka_weight:
                classification = classification + local_loss
                hs.append(h.float())
            else:
                local_loss.backward()
                classification += local_loss.detach()
        cka = diversity_cka(torch.cat(hs)) if cka_weight else classification.new_zeros(())
        loss = classification + cka_weight * cka
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite logical loss; optimizer has not advanced')
        if cka_weight:
            loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    scores = torch.cat(zs)[torch.argsort(torch.tensor(order, device=device))]
    return {**stats, 'ce': float(classification.detach()), 'cka': float(cka.detach()),
            'loss': float(loss.detach()), 'grad_norm': float(norm),
            'activation_offload_gib': offload.bytes / 1024**3 if offload else 0.,
            'activation_budget_gib': offload.limit / 1024**3 if offload else 0.,
            'encoder_forward_microbatches': len(zs)}, scores


@torch.inference_mode()
def predict(model, examples, device, amp='bf16', microbatch=4, frame_budget=1600):
    scores = torch.empty(len(examples), 2)
    for indices, features, mask in microbatches(examples, microbatch, frame_budget):
        with amp_context(device, amp):
            logits, _ = model(features.to(device), mask.to(device))
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Non-finite validation score')
        scores[indices] = logits.detach().float().cpu()
    return scores
