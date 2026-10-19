"""Observe Train fusion gradients in FP32 without changing the training recipe."""
from collections import defaultdict
import math
import numpy as np
import torch
from w2v_v31810.common import capture_rng, restore_rng


def calibrate(model, tasks, cfg):
    from .step import components
    rng=capture_rng();modes=[(m,m.training) for m in model.modules()]
    head=getattr(model,'head',None);precision=getattr(head,'precision',None)
    parameters=[p for p in model.fusion.parameters() if p.requires_grad]
    if not parameters:raise ValueError('Calibration needs trainable shared fusion parameters')
    norms=defaultdict(list);records=[]
    try:
        model.eval()
        # Training uses GradScaler; an unscaled backward through FP16 here can
        # turn a valid small prior gradient into zero. Only this short observer
        # uses FP32 AASIST; no optimizer update or training precision change.
        if precision is not None:head.precision='fp32'
        for index,units in enumerate(tasks):
            rows=[r for unit in units for r in unit]
            record=dict(task=index+1,xeus_pairs=sum(bool(r.get('xeus_pair')) for r in rows)//2,
                pcl_sources=sum('original_pcl' in r for r in rows),components={})
            for component in ('ce','prior','pcl'):
                if component=='prior' and cfg['mode']=='baseline':continue
                accum=[torch.zeros_like(p) for p in parameters];seen=0;connected=False;total=0.
                for _,loss in components(model,units,cfg,component):
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite calibration loss: '+component)
                    if not loss.requires_grad:raise ValueError('Calibration loss has no autograd graph: '+component)
                    gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
                    for target,g in zip(accum,gradients):
                        if g is not None:target.add_(g.detach());connected=True
                    seen+=1;total+=float(loss.detach())
                if seen and not connected:raise ValueError('Calibration loss disconnected from shared fusion: '+component)
                norm=math.sqrt(sum(float(x.double().square().sum()) for x in accum))
                if not math.isfinite(norm):raise FloatingPointError('Nonfinite calibration gradient: '+component)
                if seen:norms[component].append(norm)
                record['components'][component]=dict(loss=total,gradient_norm=norm,terms=seen,connected=connected)
            records.append(record)
            detail=' '.join(k+'='+format(v['gradient_norm'],'.3e') for k,v in record['components'].items())
            print(f'Auxiliary Train calibration {index+1} | FP32 | pairs={record["xeus_pairs"]} | '+detail,flush=True)
    finally:
        if precision is not None:head.precision=precision
        for module,mode in modes:module.training=mode
        restore_rng(rng)
    if not norms['ce']:raise ValueError('No Train CE calibration tasks')
    medians={k:float(np.median(v)) for k,v in norms.items() if v};ce=medians['ce']
    if not math.isfinite(ce) or ce<=1e-12:raise ValueError('No valid Train CE calibration gradient')
    weights={};status={}
    for name,target,cap in (('prior',.10,.10),('pcl',.05,.01)):
        if name=='prior' and cfg['mode']=='baseline':weights[name]=0.;status[name]='disabled_by_recipe';continue
        if not norms[name]:raise ValueError('No usable '+name+' calibration terms; inspect Train pair coverage')
        value=medians[name]
        # Connected zero gradients are possible for already satisfied pairs,
        # identical PCL features, or cancellation. Never divide by zero, amplify
        # without a cap, or permanently disable an objective for future batches.
        weights[name]=min(cap,target*ce/max(value,1e-12))
        status[name]='bounded_cap_small_gradient' if value<=1e-12 else 'calibrated'
    return dict(weights=weights,median_gradient_norms=medians,tasks=len(records),records=records,
        status=status,fit_data='Train only',gradient_parameters='shared fusion',optimizer_updates=0,
        calibration_backend_precision='fp32',training_backend_precision=precision or cfg.get('backend_precision','fp32'),
        requested_gradient_ratios=dict(prior=.10,pcl=.05),coefficient_caps=dict(prior=.10,pcl=.01),
        revision='fp32_zero_safe_v1')
