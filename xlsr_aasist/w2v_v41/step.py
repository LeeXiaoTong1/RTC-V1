"""Two balanced CE gradients plus separate official/simulated PCL gradients."""
from collections import Counter
import torch
from torch.nn import functional as F
from w2v_v31810.precision import scaler_for, unscale_and_clip, commit
from .pcl import expand, pack, phoneme_mse, ramp
from .gradients import workspace_for, unpack_stats


def components(model, units, cfg, enabled=True):
    rows=[r for unit in units for r in unit]
    cell=lambda r:(r['role'],r['language'],r['label'])
    counts=Counter(map(cell,rows))
    if set(counts)!={(r,l,y) for r in ('online','noisy') for l in ('en','zh') for y in (0,1)}:
        raise ValueError('V4 requires all eight CE cells')
    if len({r['canonical_source'] for r in rows})!=len(rows) or any(r.get('split')!='train' for r in rows):
        raise ValueError('Classification needs distinct registered Train sources')
    expanded=[expand(r,enabled) for r in rows]
    pairs=Counter((x['kind'],x['rows'][0]['language'],x['rows'][0]['label']) for x in expanded if x['pair'])
    groups=Counter(k[0] for k in pairs)
    for role in ('online','noisy'):
        for batch in pack([x for x in expanded if x['rows'][0]['role']==role],cfg['microbatch'],cfg['frame_budget']):
            values=[];slots=[];main=[]
            for item in batch:
                slots.append(len(values));main.append(len(values));values.extend(item['rows'])
            sequences=model.encode_rows(values) if hasattr(model,'encode_rows') else model.encode([r['wave'] for r in values])
            logits=model.classify_sequences([sequences[i] for i in main]).float()
            labels=torch.tensor([x['rows'][0]['label'] for x in batch],device=logits.device)
            losses=F.cross_entropy(logits,labels,reduction='none',label_smoothing=cfg.get('label_smoothing',.02))
            ce=sum(v/(8*counts[cell(x['rows'][0])]) for v,x in zip(losses,batch))
            auxiliary={};valid=Counter()
            for item,offset in zip(batch,slots):
                if not item['pair']:continue
                a,b=item['pair'];left,right=item['rows'][a],item['rows'][b]
                value,n=phoneme_mse(sequences[offset+a],sequences[offset+b],item['matches'],left['original_samples'],right['original_samples'])
                if not n:raise ValueError('PCL geometry changed after packing')
                row=item['rows'][0];kind=item['kind'];denom=groups[kind]*pairs[kind,row['language'],row['label']]
                auxiliary[kind]=auxiliary.get(kind,0.)+value/denom;valid[kind]+=1
            yield role,ce,auxiliary,dict(encoded=len(values),references=len(values)-len(main),pairs=dict(valid))


class PreStepOOM(RuntimeError):
    """Allocation failed before GradScaler/optimizer was allowed to mutate state."""


def _compute(model,optimizer,units,cfg,weights,epoch,step,steps):
    params=[p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]
    optimizer.zero_grad(set_to_none=True);scaler=scaler_for(optimizer,cfg)
    workspace=workspace_for(optimizer,params);scale=scaler.get_scale()
    strength=ramp(cfg,epoch,step,steps)
    enabled=strength>0 and cfg.get('consistency','dual')!='none'
    totals=params[0].new_zeros(4,dtype=torch.float32)
    indices=dict(online=0,noisy=1,official=2,noise=3)
    encoded=references=0;valid=Counter()
    for role,ce,aux,info in components(model,units,cfg,enabled):
        terms=[(role,ce)]+[(k,v*strength*weights.get(k,0.)) for k,v in aux.items() if weights.get(k,0.)>0]
        for index,(kind,loss) in enumerate(terms):
            gs=torch.autograd.grad(scaler.scale(loss),params,retain_graph=index<len(terms)-1,allow_unused=True)
            workspace.add(kind,gs)
            del gs
        totals[indices[role]].add_(ce.detach())
        for k,v in aux.items():totals[indices[k]].add_(v.detach())
        encoded+=info['encoded'];references+=info['references'];valid.update(info['pairs'])
    coordination=workspace.coordinate(cfg.get('gradient_coordination','pcgrad')=='pcgrad',scale)
    health=model.finish_checks() if hasattr(model,'finish_checks') else totals.new_tensor(True,dtype=torch.bool)
    health=health & torch.isfinite(totals).all()
    # One host read for all per-microbatch losses, health and coordination stats.
    observed=torch.cat((totals,coordination,health[None].float())).detach().cpu().tolist()
    if not observed[-1]:raise FloatingPointError('Nonfinite V4.1 features/logits/loss; no optimizer step')
    online,noisy,official,noise=observed[:4]
    totals=dict(ce=online+noisy,pcl_official=official,pcl_noise=noise)
    streams=dict(online=2*online,noisy=2*noisy);coord=unpack_stats(observed[4:14])
    norms=dict(official=observed[12],noise=observed[13])
    weighted={k:strength*weights.get(k,0.)*totals['pcl_'+k] for k in ('official','noise')}
    return dict(totals,loss=totals['ce']+sum(weighted.values()),pcl_weighted=weighted,pcl_weights=dict(weights),ramp=strength,
        coordination=coord,stream_ce=streams,views=sum(map(len,units)),
        encoded_views=encoded,reference_views=references,pcl_pairs=dict(valid),pcl_gradient_norms=norms,
        pcl_coverage=[r['pcl_coverage'] for u in units for r in u if 'pcl_coverage' in r],
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if str(cfg.get('device','cpu')).startswith('cuda') else None)


def train_step(model,optimizer,units,cfg,weights,epoch,step,steps):
    if hasattr(model,'begin_checks'):model.begin_checks()
    try:
        stats=_compute(model,optimizer,units,cfg,weights,epoch,step,steps)
    except torch.cuda.OutOfMemoryError as exc:
        optimizer.zero_grad(set_to_none=True)
        exc.__traceback__=None
        raise PreStepOOM(str(exc)) from None
    finally:
        if hasattr(model,'finish_checks'):model.finish_checks()
    # OOM during optimizer.step is NOT safe to retry and is never caught above.
    scaler=scaler_for(optimizer,cfg)
    stats['gradient_norm']=unscale_and_clip(optimizer,scaler,cfg.get('grad_clip',1.))
    stats['amp']=commit(optimizer,scaler)
    return stats


from .calibration import calibrate
