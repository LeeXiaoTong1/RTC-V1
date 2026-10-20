"""Two balanced CE gradients plus separate official/simulated PCL gradients."""
from collections import Counter
import torch
from torch.nn import functional as F
from w2v_v31810.precision import scaler_for, unscale_and_clip, commit
from .pcl import expand, pack, phoneme_mse, ramp
from .gradients import coordinate


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
            sequences=model.encode([r['wave'] for r in values])
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


def train_step(model,optimizer,units,cfg,weights,epoch,step,steps):
    params=[p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]
    kinds=('online','noisy','official','noise')
    buffers={k:[torch.zeros_like(p) for p in params] for k in kinds};used=[False]*len(params)
    optimizer.zero_grad(set_to_none=True);scaler=scaler_for(optimizer,cfg)
    strength=ramp(cfg,epoch,step,steps)
    enabled=strength>0 and cfg.get('consistency','dual')!='none'
    totals=dict(ce=0.,pcl_official=0.,pcl_noise=0.);streams=dict(online=0.,noisy=0.)
    encoded=references=0;valid=Counter()
    for role,ce,aux,info in components(model,units,cfg,enabled):
        terms=[(role,ce)]+[(k,v*strength*weights.get(k,0.)) for k,v in aux.items() if weights.get(k,0.)>0]
        for index,(kind,loss) in enumerate(terms):
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite V4 loss')
            gs=torch.autograd.grad(scaler.scale(loss),params,retain_graph=index<len(terms)-1,allow_unused=True)
            for i,(target,g) in enumerate(zip(buffers[kind],gs)):
                if g is not None:target.add_(g.detach());used[i]=True
        value=float(ce.detach());totals['ce']+=value;streams[role]+=2*value
        for k,v in aux.items():totals['pcl_'+k]+=float(v.detach())
        encoded+=info['encoded'];references+=info['references'];valid.update(info['pairs'])
    aux=[a+b for a,b in zip(buffers['official'],buffers['noise'])]
    norms={k:float(torch.stack([g.float().square().sum() for g in buffers[k]]).sum().sqrt()/scaler.get_scale()) for k in ('official','noise')}
    merged,coord=coordinate(buffers['online'],buffers['noisy'],aux,
        enabled=cfg.get('gradient_coordination','pcgrad')=='pcgrad',scale=scaler.get_scale())
    offset=0
    for i,p in enumerate(params):
        n=p.numel();p.grad=merged[offset:offset+n].view_as(p).clone() if used[i] else None;offset+=n
    del buffers,aux,merged
    norm=unscale_and_clip(optimizer,scaler,cfg.get('grad_clip',1.));amp=commit(optimizer,scaler)
    weighted={k:strength*weights.get(k,0.)*totals['pcl_'+k] for k in ('official','noise')}
    return dict(totals,loss=totals['ce']+sum(weighted.values()),pcl_weighted=weighted,pcl_weights=dict(weights),ramp=strength,
        gradient_norm=norm,amp=amp,coordination=coord,stream_ce=streams,views=sum(map(len,units)),
        encoded_views=encoded,reference_views=references,pcl_pairs=dict(valid),pcl_gradient_norms=norms,
        pcl_coverage=[r['pcl_coverage'] for u in units for r in u if 'pcl_coverage' in r],
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if str(cfg.get('device','cpu')).startswith('cuda') else None)


from .calibration import calibrate
