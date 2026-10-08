"""One shared detector forward per triplet; two kinds of same-source TFCL edges."""
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

from w2v_v315.step import clear_features, _grad_norm
from .data import validate_batch
from .objectives import fusion_frames


def triplet_batches(groups,size,frame_budget):
    if isinstance(size,bool) or not isinstance(size,int) or size<3:
        raise ValueError('All three graphs must remain together; microbatch must be >=3')
    if isinstance(frame_budget,bool) or not isinstance(frame_budget,int) or frame_budget<1:
        raise ValueError('Triplet frame budget must be a positive integer')
    for group in groups:
        if len(group)!=3:
            raise ValueError('Each physical source group must contain exactly three views')
        for row in group:
            features,mask=row['features'],row['mask']
            if (features.ndim!=3 or features.shape[0]!=1 or features.shape[2]<1
                    or mask.ndim!=2 or features.shape[:2]!=mask.shape
                    or not features.is_floating_point()):
                raise ValueError('Each view requires floating [1,T,C] features and a matching [1,T] mask')
            if features.shape[1]<1 or not bool((mask==1).all()):
                raise ValueError('Input triplet views must contain only valid native frames')
    ordered=sorted(groups,key=lambda rows:max(r['features'].shape[1] for r in rows))
    pending=[]
    def emit(items):
        rows=[r for group in items for r in group]
        values=[r['features'][0] for r in rows]
        x=torch.nn.utils.rnn.pad_sequence(values,batch_first=True)
        lengths=torch.tensor([len(v) for v in values])
        mask=(torch.arange(x.shape[1])[None]<lengths[:,None]).long()
        return rows,x,mask
    for group in ordered:
        length=max(r['features'].shape[1] for r in group)
        if pending and (3*(len(pending)+1)>size or 3*(len(pending)+1)*length>frame_budget
                        or length/min(r['features'].shape[1] for r in pending[0])>1.5):
            yield emit(pending);pending=[]
        pending.append(group)
    if pending:
        yield emit(pending)


def train_step(model,auxiliary,optimizer,examples,cfg,warm,audit=False):
    model.train();auxiliary.train()
    occurrences=validate_batch(examples,cfg['source_batch'],cfg)
    order={'online':0,'reference':1,'noisy':2}
    groups=[sorted(rows,key=lambda r:order[r['role']]) for rows in occurrences.values()]
    optimizer.zero_grad(set_to_none=True)
    stats=dict(classification_loss=0.,time_loss=0.,structure_loss=0.,
        weighted_time_loss=0.,weighted_structure_loss=0.,total_loss=0.,
        bridge_time_loss=0.,bridge_structure_loss=0.,matched_time_loss=0.,matched_structure_loss=0.,
        bridge_pairs=0,matched_pairs=0,eligible_pairs=0,maximum_example_ce=0.,
        maximum_padded_frames=0,physical_forwards=0)
    context=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if cfg['amp']=='bf16' else nullcontext()
    active=warm>0 and (cfg['tfcl_time_weight']>0 or cfg['tfcl_structure_weight']>0)
    checked=False
    try:
        for rows,x,mask in triplet_batches(groups,cfg['microbatch'],cfg['frame_budget']):
            with fusion_frames(model) as capture,context():
                z,_=model(x.to(cfg['device']),mask.to(cfg['device']))
                if len(capture)!=1:
                    raise ValueError('Expected one pre-MultiConv fusion tensor')
                features=capture[0]
                labels=torch.tensor([r['label'] for r in rows],device=z.device)
                mass=torch.tensor([r['ce_weight'] for r in rows],device=z.device)
                ce=F.cross_entropy(z.float(),labels,reduction='none')
                ce_loss=(ce*mass).sum()
                stats['classification_loss']+=float(ce_loss.detach())
                stats['maximum_example_ce']=max(stats['maximum_example_ce'],float(ce.detach().max()))
                aux_loss=features.sum()*0.
                if active:
                    left=[];right=[];lm=[];rm=[];kinds=[]
                    for j in range(0,len(rows),3):
                        for kind,a,b,eligible in (
                            ('bridge',j,j+1,rows[j].get('online_pair_eligible',False)),
                            ('matched',j+1,j+2,rows[j+2].get('noise_pair_eligible',rows[j+2]['pair_eligible']))):
                            if not eligible:
                                continue
                            am=torch.as_tensor(rows[a]['aux_valid'],device=features.device,dtype=torch.bool)
                            bm=torch.as_tensor(rows[b]['aux_valid'],device=features.device,dtype=torch.bool)
                            left.append(features[a,:len(am)]);right.append(features[b,:len(bm)])
                            lm.append(am);rm.append(bm);kinds.append(kind)
                    if left:
                        temporal,structure,valid=auxiliary.forward_batch(left,right,lm,rm)
                        budgets=torch.tensor([cfg['tfcl_bridge_mass' if k=='bridge' else 'tfcl_matched_mass']
                            /cfg['source_batch'] for k in kinds],device=features.device)
                        ta=(temporal*budgets).sum();sd=(structure*budgets).sum()
                        wt=warm*cfg['tfcl_time_weight']*ta
                        ws=warm*cfg['tfcl_structure_weight']*sd
                        aux_loss=wt+ws
                        stats['time_loss']+=float(ta.detach());stats['structure_loss']+=float(sd.detach())
                        stats['weighted_time_loss']+=float(wt.detach());stats['weighted_structure_loss']+=float(ws.detach())
                        for kind in ('bridge','matched'):
                            chosen=torch.tensor([k==kind for k in kinds],device=features.device)
                            stats[kind+'_pairs']+=int((valid&chosen).sum())
                            stats[kind+'_time_loss']+=float(temporal[chosen].sum().detach())/cfg['source_batch']
                            stats[kind+'_structure_loss']+=float(structure[chosen].sum().detach())/cfg['source_batch']
                    if audit and left and not checked:
                        targets=[model.head.projection.weight]
                        encoder=[p for p in model.backbone.encoder.layers[-1].parameters() if p.requires_grad]
                        if encoder:targets.append(encoder[0])
                        gradients=torch.autograd.grad(aux_loss,targets,retain_graph=True,allow_unused=True)
                        stats['aux_gradient_audit']=dict(zip(('fusion','last_encoder_layer'),
                            [float(g.float().norm()) if g is not None else 0. for g in gradients]))
                        if not all(np.isfinite(v) for v in stats['aux_gradient_audit'].values()):
                            raise FloatingPointError('Nonfinite auxiliary gradient')
                        checked=True
                loss=ce_loss+aux_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite CE/TFCL loss; update was not applied')
            stats['total_loss']+=float(loss.detach())
            loss.backward();clear_features(model)
            stats['maximum_padded_frames']=max(stats['maximum_padded_frames'],x.shape[0]*x.shape[1])
            stats['physical_forwards']+=1
            del features,z,loss,aux_loss
        stats['eligible_pairs']=stats['matched_pairs']
        params=[p for g in optimizer.param_groups for p in g['params']]
        stats['aux_parameter_grad_norm']=_grad_norm(auxiliary.parameters())
        norm=torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'])
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('Nonfinite gradient; update was not applied')
        stats['gradient_norm']=float(norm)
        optimizer.step()
        return stats
    finally:
        clear_features(model)
