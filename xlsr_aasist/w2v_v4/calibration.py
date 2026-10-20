"""Train-only dual-PCL scale observation on shared LoRA, never model updates."""
import math
import statistics
import torch
from .common import capture_rng,restore_rng


def calibrate(model,batches,cfg):
    from .step import components
    names=[(n,p) for n,p in model.named_parameters() if p.requires_grad and n.endswith(('.lora_a','.lora_b'))]
    if not names:names=[('fusion.'+n,p) for n,p in model.fusion.named_parameters() if p.requires_grad]
    params=[p for _,p in names]
    if not params:raise ValueError('PCL calibration has no shared trainable parameters')
    branches=('official','noise') if cfg.get('consistency','dual')=='dual' else ('official',)
    records=[];rng=capture_rng();modes=[(m,m.training) for m in model.modules()]
    precision=getattr(model.head,'precision',None);scale=4096. if str(cfg.get('device','cpu')).startswith('cuda') else 1.
    try:
        model.eval()
        if precision is not None:model.head.precision='fp32'
        for index,units in enumerate(batches):
            buffers={k:[torch.zeros_like(p) for p in params] for k in ('ce',*branches)}
            terms={k:0 for k in buffers};losses={k:0. for k in buffers}
            for _,ce,aux,_info in components(model,units,cfg,True):
                active=[('ce',ce)]+[(k,aux[k]) for k in branches if k in aux]
                for i,(kind,loss) in enumerate(active):
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite calibration loss')
                    gs=torch.autograd.grad(loss*scale,params,retain_graph=i<len(active)-1,allow_unused=True)
                    if not any(g is not None for g in gs):raise ValueError('Disconnected '+kind+' calibration gradient')
                    for target,g in zip(buffers[kind],gs):
                        if g is not None:target.add_(g.detach()/scale)
                    terms[kind]+=1;losses[kind]+=float(loss.detach())
            norms={k:float(torch.stack([g.float().square().sum() for g in v]).sum().sqrt()) for k,v in buffers.items()}
            if not all(math.isfinite(v) for v in norms.values()):raise FloatingPointError('Nonfinite calibration gradient')
            records.append(dict(task=index+1,gradient_norms=norms,terms=terms,losses=losses))
            print('PCL Train calibration '+str(index+1)+' | '+ ' '.join(k+'='+format(v,'.3e') for k,v in norms.items()),flush=True)
    finally:
        if precision is not None:model.head.precision=precision
        for module,mode in modes:module.training=mode
        restore_rng(rng)
    if not records:raise ValueError('No PCL calibration tasks')
    weights={};achieved={};targets={}
    for kind in branches:
        usable=[r for r in records if r['terms'][kind]>0]
        if not usable:raise ValueError('No valid '+kind+' PCL intervals in Train calibration; inspect masks/cache coverage')
        ce=statistics.median(r['gradient_norms']['ce'] for r in usable)
        value=statistics.median(r['gradient_norms'][kind] for r in usable)
        if ce<=1e-12:raise ValueError('No usable classification calibration gradient')
        target=cfg.get('pcl_gradient_target',.05);cap=cfg.get('pcl_max_weight',1.)
        weights[kind]=min(cap,target*ce/max(value,1e-12))
        achieved[kind]=weights[kind]*value/ce;targets[kind]=target
    print('PCL calibrated weights='+str(weights)+'; measured shared-gradient ratios='+str(achieved),flush=True)
    return dict(weights=weights,records=records,target_ratios=targets,achieved_ratios=achieved,
        parameter_scope=[n for n,_ in names],backend_precision='fp32',gradient_scale=scale,
        limitation='Train calibration only; full-model live PCL norms are logged separately; no guarantee about Adam update direction')
