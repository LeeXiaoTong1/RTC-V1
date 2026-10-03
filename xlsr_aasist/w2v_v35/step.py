"""One forward per complete view, bounded source graphs, one optimizer update."""
from contextlib import nullcontext
import math
import torch
from torch.nn import functional as F
from w2v_aasist.runtime import amp_context
from w2v_v32.runtime import HybridActivationStore
from w2v_v32.batching import DeviceBatches, PreparedBatch
from w2v_v33.model import microbatches
from .losses import source_components, objective


def source_rows(examples, indices, microbatch, frame_budget):
    """Reuse worker-padded, packed and pinned chunks without another CPU copy."""
    rows = [examples[index] for index in indices]
    if (not hasattr(examples, 'batches') or examples.size != microbatch
            or examples.frame_budget != frame_budget):
        return rows
    remap = {index: local for local, index in enumerate(indices)}
    selected, covered = [], set()
    for batch_indices, features, mask in examples.batches:
        overlap = set(batch_indices) & remap.keys()
        if not overlap:
            continue
        if len(overlap) != len(batch_indices):
            return rows  # Foreign prepared grouping: regroup safely, without truncation.
        selected.append(([remap[index] for index in batch_indices], features, mask))
        covered.update(batch_indices)
    if covered != set(indices):
        return rows
    return PreparedBatch(rows, selected, microbatch, frame_budget)


def supervised_step(model, examples, optimizer, weights, device, amp='bf16',
                    *, source_denominator=None, source_microbatch=4,
                    offline_weight=.1, online_weight=.3, noisy_weight=.6,
                    grad_clip=1., microbatch=4, frame_budget=1600,
                    offload_activations=True, activation_budget_gib=0.,
                    gpu_activation_gib=18., gpu_reserve_gib=8.):
    if not model.training or any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Training mode without BatchNorm required for source microbatching')
    if isinstance(source_microbatch, bool) or not isinstance(source_microbatch, int) or source_microbatch < 1:
        raise ValueError('Positive source_microbatch required')
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError('Positive finite gradient clipping norm required')
    device = torch.device(device)
    parts = source_components(examples, weights, device, source_denominator,
                              offline_weight, online_weight, noisy_weight)
    optimizer.zero_grad(set_to_none=True)
    scores = torch.empty((len(examples), 2), device=device)
    sums, forward_calls, copied, pinned, peak_host, peak_gpu = {}, 0, 0, 0, 0, 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    source_ids = list(parts['sources'])
    try:
        for offset in range(0, len(source_ids), source_microbatch):
            keys = source_ids[offset:offset + source_microbatch]
            global_indices = [index for key in keys for index in parts['sources'][key]['conditions'].values()]
            rows = examples if len(keys) == len(source_ids) else source_rows(examples, global_indices, microbatch, frame_budget)
            if rows is examples:
                global_indices = list(range(len(examples)))
            storage = HybridActivationStore(model, gpu_budget_gib=gpu_activation_gib,
                host_budget_gib=activation_budget_gib, reserve_gib=gpu_reserve_gib) \
                if device.type == 'cuda' and offload_activations else None
            values = {}
            with storage if storage is not None else nullcontext():
                transfers = DeviceBatches(microbatches(rows, microbatch, frame_budget), device)
                for local_indices, features, mask in transfers:
                    if storage is not None:
                        storage.refresh()
                    with amp_context(device, amp):
                        output = model(features, mask, _validated=True)
                        logits = output[0] if isinstance(output, tuple) else output
                    indices = [global_indices[i] for i in local_indices]
                    ce = F.cross_entropy(logits.float(), parts['labels'][indices], reduction='none')
                    for local, index in enumerate(indices):
                        values[index] = ce[local]
                    scores[indices] = logits.detach().float()
                    forward_calls += 1
                loss, stats = objective(values, parts, keys)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Non-finite classification risk; optimizer has not advanced')
                loss.backward()
                sums['loss'] = sums.get('loss', 0.) + loss.detach()
                for name, value in stats.items():
                    sums[name] = sums.get(name, 0.) + value
            copied += transfers.copy_batches
            pinned += transfers.pinned_batches
            if storage:
                peak_host = max(peak_host, storage.bytes)
                peak_gpu = max(peak_gpu, storage.gpu_bytes)
            # No saved graph is retained until the next source chunk.
            del loss, values, output, logits, ce
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    names = list(sums)
    scalars = torch.stack([sums[k].float() for k in names]).cpu().tolist()
    stats = dict(zip(names, scalars))
    stats.update(ce=stats['loss'], grad_norm=float(norm.detach().cpu()),
        source_count=parts['source_count'], full_views=len(examples), short_views=0,
        encoder_forward_microbatches=forward_calls, gpu_copy_batches=copied,
        gpu_pinned_copy_batches=pinned, activation_offload_gib=peak_host / 1024**3,
        gpu_saved_activations_gib=peak_gpu / 1024**3,
        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0.,
        gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == 'cuda' else 0.)
    return stats, scores.cpu()
