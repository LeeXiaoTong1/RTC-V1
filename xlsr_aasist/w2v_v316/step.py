"""48 views per update, paired microbatching and explicit total-loss telemetry."""
import math
import time

import torch
from torch.nn import functional as F

from .data import validate_batch,grouped,auxiliary_mask
from .model import LoRALinear


def train_step(model,auxiliary,optimizer,examples,cfg,warm,audit=False):
    model.train(); auxiliary.train()
    occurrences = validate_batch(examples,cfg['source_batch'])
    ordinary = [[r] for r in examples if r['role']=='ordinary']
    pairs = [[next(r for r in rows if r['role']==role) for role in ('reference','noisy')]
             for rows in occurrences.values()]
    optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(7,device=cfg['device'])
    eligible,forwards,audited = 0,0,False
    audit_result = {}
    for units,paired in ((ordinary,False),(pairs,True)):
        for rows in grouped(units,cfg['microbatch'],cfg['frame_budget']):
            logits,features,valid,lengths = model([r['wave'] for r in rows])
            labels = torch.tensor([r['label'] for r in rows],device=logits.device)
            ce = F.cross_entropy(logits.float(),labels,reduction='none')
            mass = torch.tensor([r['ce_weight'] for r in rows],device=logits.device)
            loss = (ce*mass).sum()
            sums[0] += loss.detach(); sums[3] = torch.maximum(sums[3],ce.detach().max())
            for idx,role in enumerate(('ordinary','reference','noisy')):
                ix = [i for i,r in enumerate(rows) if r['role']==role]
                if ix: sums[4+idx] += (ce[ix]*mass[ix]).sum().detach()
            if paired and warm>0 and (cfg['tfcl_time_weight'] or cfg['tfcl_structure_weight']):
                inputs = []
                for i in range(0,len(rows),2):
                    if not rows[i]['pair_eligible'] or not rows[i+1]['pair_eligible']: continue
                    masks = [auxiliary_mask(rows[j],lengths[j],features.device) for j in (i,i+1)]
                    if min(int(m.sum()) for m in masks)<2: continue
                    inputs.append((features[i,:lengths[i]],features[i+1,:lengths[i+1]],*masks))
                if inputs:
                    # Auxiliary math stays FP32, including CKA and cosine reductions.
                    with torch.autocast(features.device.type,enabled=False):
                        ta,sd = auxiliary(inputs)
                        ta,sd = ta.sum()/cfg['source_batch'],sd.sum()/cfg['source_batch']
                        aux_loss = warm*(cfg['tfcl_time_weight']*ta+cfg['tfcl_structure_weight']*sd)
                    eligible += len(inputs); sums[1] += ta.detach(); sums[2] += sd.detach()
                    if audit and not audited:
                        tail = [m.b for m in model.modules() if isinstance(m,LoRALinear)][-1]
                        gradients = torch.autograd.grad(aux_loss,[model.head.projection.weight,tail],retain_graph=True,allow_unused=True)
                        audit_result = dict(zip(('fusion','lora_last_output_B'),[float(g.norm()) if g is not None else 0. for g in gradients]))
                        audited = True
                    loss = loss+aux_loss
            # Finite gradient check before optimizer.step also rejects nonfinite losses.
            loss.backward(); forwards += 1
            del logits,features,loss
    params = [p for g in optimizer.param_groups for p in g['params']]
    norm = torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'],error_if_nonfinite=True)
    values = sums.cpu().tolist()
    if not all(math.isfinite(v) for v in values):
        raise FloatingPointError('Nonfinite loss; checkpoint and optimizer update preserved')
    optimizer.step()
    ce,ta,sd,peak,*role_ce = values
    weighted_time = warm*cfg['tfcl_time_weight']*ta
    weighted_structure = warm*cfg['tfcl_structure_weight']*sd
    return dict(classification_loss=ce,time_loss=ta,structure_loss=sd,
        weighted_time=weighted_time,weighted_structure=weighted_structure,
        total_loss=ce+weighted_time+weighted_structure,maximum_example_ce=peak,
        role_ce=dict(zip(('ordinary','reference','noisy'),role_ce)),eligible_pairs=eligible,
        gradient_norm=float(norm),physical_forwards=forwards,aux_gradient_audit=audit_result)
