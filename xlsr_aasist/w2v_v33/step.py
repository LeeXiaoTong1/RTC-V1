"""One encoder pass per view; joint source CE, Offline CKA, and true-source pairing."""
from contextlib import nullcontext
import math
import torch
from torch.nn import functional as F
from w2v_aasist.runtime import amp_context
from w2v_v3.model import diversity_cka
from w2v_v32.runtime import HybridActivationStore
from w2v_v32.batching import DeviceBatches
from .model import Detector, microbatches
from .losses import source_components, pair_objective


def _shared_grad_norm(loss, frames):
    gradients=torch.autograd.grad(loss,frames,retain_graph=True,allow_unused=True)
    values=[g.float().square().sum() for g in gradients if g is not None]
    return torch.stack(values).sum().sqrt() if values else loss.new_zeros(())


def supervised_step(model, examples, optimizer, weights, device, amp='bf16',
                    noisy_weight=.5, cka_weight=.01, grad_clip=1., microbatch=4,
                    frame_budget=1600, noisy_class_weights=None,
                    offload_activations=True, activation_budget_gib=0.,
                    gpu_activation_gib=18., gpu_reserve_gib=8.,
                    pair_weight=.02, aux_max_tokens=256, diagnose_aux_grad=False,
                    pair_temperature=.1,pair_time_prior=.25):
    """Pair weighting is explicit and fixed by the exposure schedule, never auto-scaled."""
    for value,name in ((cka_weight,'cka_weight'),(pair_weight,'pair_weight')):
        if not math.isfinite(value) or value<0:raise ValueError(f'{name} must be finite and nonnegative')
    if not math.isfinite(grad_clip) or grad_clip<=0:raise ValueError('Positive finite gradient clip required')
    if not math.isfinite(pair_temperature) or pair_temperature<=0 or not math.isfinite(pair_time_prior) or pair_time_prior<=0:
        raise ValueError('Positive finite pair_temperature and pair_time_prior required')
    if not isinstance(model,Detector) or not model.training:
        raise ValueError('Use a V3.3 Detector in training mode')
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Variable-length physical microbatches require no BatchNorm')
    device=torch.device(device)
    components=source_components(examples,weights,device,noisy_weight,noisy_class_weights)
    optimizer.zero_grad(set_to_none=True)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    storage=HybridActivationStore(model,gpu_budget_gib=gpu_activation_gib,
        host_budget_gib=activation_budget_gib,reserve_gib=gpu_reserve_gib) if device.type=='cuda' and offload_activations else None
    logits,order,finite=[],[],[]
    offline_blocks={};frame_records={};diagnostic_frames=[]
    statistics=torch.zeros(4,device=device)
    classification=torch.zeros((),device=device)
    offline_ids=set(components['offline_full'])
    with storage if storage is not None else nullcontext():
        transfers=DeviceBatches(microbatches(examples,microbatch,frame_budget),device)
        for indices,features,mask in transfers:
            if storage is not None:storage.refresh()
            with amp_context(device,amp):
                z,h,frames,frame_mask=model.forward_training(features,mask,_validated=True)
            finite.append(torch.isfinite(z).all() & torch.isfinite(h).all() & torch.isfinite(frames).all())
            ce=F.cross_entropy(z.float(),components['labels'][indices],reduction='none')
            contribution=ce*components['coefficients'][indices]
            classification=classification+contribution.sum()
            detached=ce.detach()
            statistics[0]+=(detached*components['ordinary_coefficients'][indices]).sum()
            statistics[1]+=(detached*components['noisy_coefficients'][indices]).sum()
            full=torch.tensor([examples[i]['view']=='full' for i in indices],device=device)
            statistics[2]+=(contribution.detach()*full).sum()
            statistics[3]+=(contribution.detach()*(~full)).sum()
            logits.append(z.detach().float());order.extend(indices)
            if diagnose_aux_grad and pair_weight:diagnostic_frames.append(frames)
            for local,index in enumerate(indices):
                if index in offline_ids:offline_blocks[index]=h[local:local+1].float()
                if examples[index]['view']=='full' and pair_weight:
                    length=examples[index]['features'].shape[1]
                    audible=examples[index].get('audibility_mask')
                    if audible is not None and len(audible)!=length:
                        raise ValueError('Audibility mask must describe native acoustic frames')
                    frame_records[index]=(frames[local,:length].float(),frame_mask[local,:length],audible)
        cka=diversity_cka(torch.cat([offline_blocks[i] for i in components['offline_full']])) if cka_weight else classification*0
        pair,pair_stats=pair_objective(frame_records,components,max_tokens=aux_max_tokens,
            temperature=pair_temperature,time_sigma=pair_time_prior)
        loss=classification+cka_weight*cka+pair_weight*pair
        finite.append(torch.isfinite(loss))
        if not bool(torch.stack(finite).all()):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Non-finite model output/loss; optimizer has not advanced')
        diagnostic={}
        if diagnose_aux_grad and pair_weight and pair_stats['pair_valid_count']:
            ce_norm=_shared_grad_norm(classification,diagnostic_frames)
            pair_norm=_shared_grad_norm(pair,diagnostic_frames)
            diagnostic=dict(ce_shared_feature_grad_norm=ce_norm,
                pair_shared_feature_grad_norm=pair_norm,
                weighted_pair_shared_feature_grad_norm=pair_weight*pair_norm,
                weighted_pair_to_ce_shared_grad_ratio=pair_weight*pair_norm/ce_norm.clamp_min(1e-12))
        loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip,error_if_nonfinite=True)
    optimizer.step()
    scores=torch.cat(logits)[torch.argsort(torch.tensor(order,device=device))].cpu()
    scalar=dict(ordinary_ce=statistics[0],noisy_ce=statistics[1],full_ce_contribution=statistics[2],
        short_ce_contribution=statistics[3],ce=classification.detach(),cka=cka.detach(),
        loss=loss.detach(),grad_norm=norm.detach(),**pair_stats,**diagnostic)
    tensor_keys=[k for k,v in scalar.items() if isinstance(v,torch.Tensor)]
    values=torch.stack([scalar[k].detach().float() for k in tensor_keys]).cpu().tolist()
    scalar.update(zip(tensor_keys,values))
    source_count=components['source_count']
    full_count=sum(e['view']=='full' for e in examples)
    return {**scalar,'source_count':source_count,'ordinary_sources':source_count,
        'noisy_sources':source_count,'full_views':full_count,'short_views':len(examples)-full_count,
        'cka_full_views':source_count if cka_weight else 0,'pair_weight':pair_weight,
        'pair_temperature':pair_temperature,'pair_time_prior':pair_time_prior,
        'activation_offload_gib':storage.bytes/1024**3 if storage else 0.,
        'activation_budget_gib':storage.limit/1024**3 if storage else 0.,
        'gpu_saved_activations_gib':storage.gpu_bytes/1024**3 if storage else 0.,
        'gpu_activation_limit_gib':storage.gpu_limit/1024**3 if storage else 0.,
        'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0.,
        'gpu_peak_reserved_gib':torch.cuda.max_memory_reserved(device)/1024**3 if device.type=='cuda' else 0.,
        'gpu_memory_driver_queries':storage.cuda_memory_queries if storage else 0,
        'gpu_allocator_checks':storage.allocator_memory_checks if storage else 0,
        'gpu_pinned_copy_batches':transfers.pinned_batches,'gpu_copy_batches':transfers.copy_batches,
        'encoder_forward_microbatches':len(logits)},scores
