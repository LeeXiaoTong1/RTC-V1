"""Server-side shape probes; no optimizer step, data choice or metric selection."""
import gc
import time
import torch
from torch.nn import functional as F
from .common import capture_rng, restore_rng
from .gradients import workspace_for
from .pcl import phoneme_mse


def envelope(cfg):
    """Non-dominated full-utterance physical batch shapes allowed by the packer."""
    maximum=min(cfg['whole_max_frames'],cfg.get('resource_audit',{}).get('minimum_required',{}).get('whole_max_frames',cfg['whole_max_frames']))
    shapes=[(n,min(maximum,cfg['frame_budget']//n))
            for n in range(1,cfg['microbatch']+1) if cfg['frame_budget']//n>=11]
    return [(n,t) for n,t in shapes if not any(m>=n and u>=t and (m,u)!=(n,t) for m,u in shapes)]


def _probe(model,optimizer,cfg,shapes):
    params=[p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]
    ws=workspace_for(optimizer,params)
    rng=torch.Generator().manual_seed(4101)
    for count,frames in shapes:
        optimizer.zero_grad(set_to_none=True);ws.reset()
        rows=[dict(mel=torch.randn(frames,160,generator=rng),mel_length=frames) for _ in range(count)]
        sequences=model.encode_rows(rows)
        logits=model.classify_sequences(sequences).float()
        labels=torch.arange(count,device=logits.device)%2
        ce=F.cross_entropy(logits,labels,label_smoothing=cfg.get('label_smoothing',.02))
        both=cfg.get('consistency','dual')!='none'
        gradients=torch.autograd.grad(ce,params,retain_graph=both,allow_unused=True)
        ws.add('online',gradients);del gradients
        if both:
            # A physical batch has only one stream, hence only one PCL kind.
            a,b=sequences[0],sequences[-1]
            samples=(frames-1)*320+560
            pcl,_=phoneme_mse(a,b,[[0,samples,0,samples]],samples,samples)
            gradients=torch.autograd.grad(pcl,params,allow_unused=True)
            ws.add('official',gradients);del gradients,pcl
        ws.coordinate(cfg.get('gradient_coordination','pcgrad')=='pcgrad',1.)
        torch.nn.utils.clip_grad_norm_(params,cfg.get('grad_clip',1.))
        del rows,sequences,logits,ce
    optimizer.zero_grad(set_to_none=True)


def select_checkpoints(model,optimizer,cfg):
    depth=len(model.encoder.layers)
    if cfg.get('checkpoint_policy','fixed')!='auto':
        count=cfg.get('checkpoint_layers',depth);model.set_checkpoint_count(count)
        return dict(policy='fixed',selected_layers=count,measured=False)
    device=next(model.parameters()).device
    if device.type!='cuda':
        model.set_checkpoint_count(depth)
        return dict(policy='auto',selected_layers=depth,measured=False,reason='CUDA measurement unavailable')
    rng=capture_rng();records=[];selected=depth
    shapes=envelope(cfg)
    # Adam states for newly activated parameters may not yet exist after warmup.
    pending=sum(2*p.numel()*p.element_size() for g in optimizer.param_groups for p in g['params']
                if p.requires_grad and not optimizer.state.get(p))
    margin=max(cfg.get('checkpoint_margin_mib',2048)*1024**2,
               .1*torch.cuda.get_device_properties(device).total_memory)
    try:
        for count in sorted(set((depth,max(0,depth-6),max(0,depth-12),max(0,depth-18),0)),reverse=True):
            optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            restore_rng(rng);model.set_checkpoint_count(count)
            free,total=torch.cuda.mem_get_info(device)
            budget=free+torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);start=time.monotonic()
            print(f'V4.1 probing checkpoint layers={count}, shapes={shapes}; no optimizer update',flush=True)
            outcome='ok';error=None
            try:
                _probe(model,optimizer,cfg,shapes);torch.cuda.synchronize(device)
            except torch.cuda.OutOfMemoryError as exc:
                outcome='oom';error=str(exc);exc.__traceback__=None
            peak=torch.cuda.max_memory_reserved(device)
            fits=outcome=='ok' and peak+pending+margin<=budget
            records.append(dict(layers=count,status=outcome,within_margin=fits,
                peak_reserved_mib=peak/1024**2,available_budget_mib=budget/1024**2,
                seconds=time.monotonic()-start,error=error))
            print(f'V4.1 memory probe: checkpoint layers={count}, peak={peak/1024**3:.2f} GiB, margin={margin/1024**3:.2f} GiB, fits={fits}',flush=True)
            if fits:selected=count
            else:break
    finally:
        optimizer.zero_grad(set_to_none=True);restore_rng(rng);model.set_checkpoint_count(selected)
        gc.collect();torch.cuda.empty_cache()
    return dict(policy='auto',selected_layers=selected,measured=True,records=records,
        shapes=shapes,reserve_mib=margin/1024**2,pending_optimizer_mib=pending/1024**2,
        optimizer_updates=0,uses_dev_or_test=False,
        fallback='All layers recomputed on a pre-optimizer training OOM; same task and RNG retried')
