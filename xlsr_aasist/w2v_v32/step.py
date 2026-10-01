"""V3.1's exact source objective, GPU-first storage and fewer host synchronizations."""
from contextlib import nullcontext
import math
import torch
from torch.nn import functional as F
from w2v_aasist.runtime import amp_context
from w2v_v3.model import diversity_cka, microbatches as exact_microbatches
from .model import Detector, microbatches
from w2v_v31.step import view_components, loss_function
from .runtime import HybridActivationStore
from .batching import DeviceBatches


def supervised_step(model, examples, optimizer, weights, device, amp='bf16',
                    noisy_weight=.5, cka_weight=.01, grad_clip=1., microbatch=4,
                    frame_budget=1600, noisy_class_weights=None,
                    offload_activations=True, activation_budget_gib=0.,
                    gpu_activation_gib=18., gpu_reserve_gib=8.):
    if not math.isfinite(cka_weight) or cka_weight < 0 or not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError('Finite nonnegative CKA weight and positive gradient clip required')
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Exact variable-length microbatching requires no BatchNorm')
    device = torch.device(device)
    labels,noisy,ow,nw,coefficients,n,s,full,vw = view_components(
        examples,weights,noisy_class_weights,device,noisy_weight)
    optimizer.zero_grad(set_to_none=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    storage = HybridActivationStore(model,gpu_budget_gib=gpu_activation_gib,
        host_budget_gib=activation_budget_gib,reserve_gib=gpu_reserve_gib) if device.type=='cuda' and offload_activations else None
    logits,full_blocks,order,finite = [],[],[],[]
    statistics = torch.zeros(4,device=device)
    classification = torch.zeros((),device=device)
    with storage if storage is not None else nullcontext():
        prepared_training = isinstance(model,Detector) and model.training
        batching = microbatches if prepared_training else exact_microbatches
        transfers = DeviceBatches(batching(examples,microbatch,frame_budget),device)
        for indices,features,mask in transfers:
            if storage is not None:
                storage.refresh()
            with amp_context(device,amp):
                # microbatches validates native CPU masks before transfer. Avoid
                # repeating value checks that synchronize CUDA for every batch.
                z,h = model(features,mask,_validated=True) if prepared_training else model(features,mask)
            finite.append(torch.isfinite(z).all() & torch.isfinite(h).all())
            ce = F.cross_entropy(z.float(),labels[indices],reduction='none')
            contribution = ce*coefficients[indices]
            local_loss = contribution.sum()
            finite.append(torch.isfinite(local_loss))
            ln,ly,lf,lvw = noisy[indices],labels[indices],full[indices],vw[indices]
            d = ce.detach()
            statistics[0] += (d*ow[ly]*lvw*(~ln)).sum()/n
            if s:
                statistics[1] += (d*nw[ly]*lvw*ln).sum()/s
            statistics[2] += (contribution.detach()*lf).sum()
            statistics[3] += (contribution.detach()*(~lf)).sum()
            logits.append(z.detach().float());order.extend(indices)
            if cka_weight:
                classification = classification+local_loss
                positions = [j for j,i in enumerate(indices) if examples[i]['view']=='full']
                if positions:
                    full_blocks.append(h[positions].float())
            else:
                local_loss.backward()
                classification += local_loss.detach()
        cka = diversity_cka(torch.cat(full_blocks)) if cka_weight else classification.new_zeros(())
        loss = classification+cka_weight*cka
        # One host check per logical batch; all checks still precede optimizer.step.
        finite.append(torch.isfinite(loss))
        if not bool(torch.stack(finite).all()):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Non-finite model output/loss; optimizer has not advanced')
        if cka_weight:
            loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip,error_if_nonfinite=True)
    optimizer.step()
    # One transfer replaces per-example GPU->CPU synchronization in Metrics.update.
    scores = torch.cat(logits)[torch.argsort(torch.tensor(order,device=device))].cpu()
    values = torch.cat((statistics,torch.stack((classification.detach(),cka.detach(),loss.detach(),norm.detach())))).cpu().tolist()
    names = ('ordinary_ce','noisy_ce','full_ce_contribution','short_ce_contribution','ce','cka','loss','grad_norm')
    return {**dict(zip(names,values)), 'ordinary_sources':n,'noisy_sources':s,
        'full_views':n+s,'short_views':len(examples)-n-s,'cka_full_views':n+s if cka_weight else 0,
        'activation_offload_gib':storage.bytes/1024**3 if storage else 0.,
        'activation_budget_gib':storage.limit/1024**3 if storage else 0.,
        'gpu_saved_activations_gib':storage.gpu_bytes/1024**3 if storage else 0.,
        'gpu_activation_limit_gib':storage.gpu_limit/1024**3 if storage else 0.,
        'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0.,
        'gpu_peak_reserved_gib':torch.cuda.max_memory_reserved(device)/1024**3 if device.type=='cuda' else 0.,
        'gpu_memory_driver_queries':storage.cuda_memory_queries if storage else 0,
        'gpu_allocator_checks':storage.allocator_memory_checks if storage else 0,
        'gpu_pinned_copy_batches':transfers.pinned_batches,
        'gpu_copy_batches':transfers.copy_batches,
        'encoder_forward_microbatches':len(logits)},scores
