"""Independent three-view classification, conditional one-way auxiliary gradients."""
from contextlib import nullcontext
import time
import torch
from torch.nn import functional as F
from w2v_v315.step import clear_features,_grad_norm
from .data import validate_batch
from .objectives import capture_features,importance


def source_batches(groups,size,frame_budget):
    if size<3 or frame_budget<1:raise ValueError('Positive full-source batch budgets required')
    for group in groups:
        for row in group:
            f,m=row['features'],row['mask']
            if f.ndim!=3 or f.shape[0]!=1 or f.shape[:2]!=m.shape or not bool((m==1).all()):raise ValueError('Only native complete feature inputs')
    pending=[]
    def emit(items):
        rows=[r for group in items for r in group];values=[r['features'][0] for r in rows]
        x=torch.nn.utils.rnn.pad_sequence(values,batch_first=True)
        mask=(torch.arange(x.shape[1])[None]<torch.tensor([len(v) for v in values])[:,None]).long()
        return rows,x,mask
    for group in sorted(groups,key=lambda g:max(r['features'].shape[1] for r in g)):
        n=sum(len(g) for g in pending)+len(group);length=max(r['features'].shape[1] for r in group)
        if pending and (n>size or n*length>frame_budget or length/min(r['features'].shape[1] for r in pending[0])>1.5):
            yield emit(pending);pending=[]
        pending.append(group)
    if pending:yield emit(pending)


def train_step(model,auxiliary,optimizer,examples,cfg,warm,audit=False):
    model.train();auxiliary.train();optimizer.zero_grad(set_to_none=True)
    groups=validate_batch(examples,cfg['source_batch'],cfg)
    order={'offline':0,'online':1,'noisy':2}
    groups=[sorted(g,key=lambda r:order[r['role']]) for g in groups.values()]
    stats=dict(classification_loss=0.,time_loss=0.,structure_loss=0.,weighted_time_loss=0.,
        weighted_structure_loss=0.,total_loss=0.,maximum_example_ce=0.,bridge_pairs=0,matched_pairs=0,
        reference_eligible=0,reference_total=0,alignment_coverage_sum=0.,alignment_attempts=0,
        alignment_seconds=0.,saliency_seconds=0.,physical_forwards=0.,reference_groups={},
        maximum_padded_frames=0,aux_gradient_audit={})
    active=warm>0 and cfg['tfcl_mode']!='ce'
    first_audit=True
    try:
        for rows,x,mask in source_batches(groups,cfg['microbatch'],cfg['frame_budget']):
            context=torch.autocast('cuda',dtype=torch.bfloat16) if cfg['amp']=='bf16' else nullcontext()
            with capture_features(model,cfg) as captured,context:
                z,_=model(x.to(cfg['device']),mask.to(cfg['device']))
                if len(captured['content'])!=1 or len(captured['detection'])!=1:raise ValueError('Expected single frozen content and detection capture')
                features=captured['detection'][0];content=captured['content'][0]
                labels=torch.tensor([r['label'] for r in rows],device=z.device)
                ce=F.cross_entropy(z.float(),labels,reduction='none')
                mass=torch.tensor([r['ce_weight'] for r in rows],device=z.device)
                ce_loss=(ce*mass).sum();aux_loss=features[:0].float().sum()
                stats['classification_loss']+=float(ce_loss.detach());stats['maximum_example_ce']=max(stats['maximum_example_ce'],float(ce.detach().max()))
                if active:
                    off=[i for i,r in enumerate(rows) if r['role']=='offline']
                    with torch.no_grad():
                        probability=z.float().softmax(-1)
                        reliable={i:(True if cfg.get('profile_force_reference',False) else bool(z[i].argmax()==labels[i] and probability[i,labels[i]]>=cfg['reference_probability'])) for i in off}
                    gradient=None
                    if cfg['tfcl_mode']=='weighted' and any(reliable.values()):
                        began=time.perf_counter()
                        indices=torch.tensor([i for i in off if reliable[i]],device=z.device)
                        margins=z[indices,labels[indices]].float()-z[indices,1-labels[indices]].float()
                        gradient=torch.autograd.grad(margins.sum(),features,retain_graph=True,create_graph=False)[0].detach()
                        stats['saliency_seconds']+=time.perf_counter()-began
                    temporal=[];structure=[]
                    for i in off:
                        row=rows[i];key=row['language']+'/'+('fake' if row['label']==0 else 'real')
                        counts=stats['reference_groups'].setdefault(key,dict(total=0,reliable=0,online_accepted=0,noisy_accepted=0))
                        counts['total']+=1;counts['reliable']+=int(reliable[i]);stats['reference_total']+=1;stats['reference_eligible']+=int(reliable[i])
                        if cfg['tfcl_mode']!='original' and not reliable[i]:continue
                        n=len(row['aux_valid']);ref=features[i,:n];ref_valid=row['aux_valid']
                        weights=(importance(ref,gradient[i,:n],ref_valid,cfg['importance_uniform_mass'],cfg['importance_cap'])
                            if gradient is not None and cfg['tfcl_mode']=='weighted' else torch.ones_like(ref,dtype=torch.float32))
                        for j,target in enumerate(rows):
                            if target['pair_occurrence']!=row['pair_occurrence'] or target['role']=='offline':continue
                            if target['role']=='noisy' and not target['pair_eligible']:continue
                            length=len(target['aux_valid']);began=time.perf_counter()
                            t,s,meta=auxiliary(ref,features[j,:length],content[i,:n],content[j,:length],
                                ref_valid,target['aux_valid'],weights,cfg,known_mapping=target['role']=='noisy',
                                hop=target.get('sample_hop',320.),reference_hop=row.get('sample_hop',320.))
                            stats['alignment_seconds']+=time.perf_counter()-began;stats['alignment_attempts']+=1
                            stats['alignment_coverage_sum']+=meta['coverage']
                            accepted=not meta.get('rejected',False)
                            counts[target['role']+'_accepted']+=int(accepted)
                            stats['bridge_pairs' if target['role']=='online' else 'matched_pairs']+=int(accepted)
                            temporal.append(t*.5/cfg['source_batch']);structure.append(s*.5/cfg['source_batch'])
                    if temporal:
                        ta=torch.stack(temporal).sum();st=torch.stack(structure).sum()
                        wt=warm*cfg['tfcl_time_weight']*ta
                        ws=cfg.get('structure_strength',warm)*cfg['tfcl_structure_weight']*st
                        aux_loss=wt+ws
                        stats['time_loss']+=float(ta.detach());stats['structure_loss']+=float(st.detach())
                        stats['weighted_time_loss']+=float(wt.detach());stats['weighted_structure_loss']+=float(ws.detach())
                loss=ce_loss+aux_loss
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite CE/TFCL; no update applied')
            if audit and first_audit:
                cg=torch.autograd.grad(ce_loss,features,retain_graph=True,allow_unused=True)[0]
                ag=torch.autograd.grad(aux_loss,features,retain_graph=True,allow_unused=True)[0]
                def norm(g):return float(g.float().norm()) if g is not None else 0.
                stats['aux_gradient_audit']=dict(ce_feature_grad=norm(cg),aux_feature_grad=norm(ag),
                    cosine=float(F.cosine_similarity(cg.float().flatten(),ag.float().flatten(),dim=0)) if cg is not None and ag is not None else 0.)
                first_audit=False
            loss.backward();stats['total_loss']+=float(loss.detach());clear_features(model)
            stats['physical_forwards']+=1;stats['maximum_padded_frames']=max(stats['maximum_padded_frames'],x.shape[0]*x.shape[1])
        params=[p for group in optimizer.param_groups for p in group['params']]
        gradient_norm=torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'])
        if not bool(torch.isfinite(gradient_norm)):raise FloatingPointError('Nonfinite gradient; no update applied')
        stats['gradient_norm']=float(gradient_norm);stats['aux_parameter_grad_norm']=_grad_norm(auxiliary.parameters())
        optimizer.step();return stats
    finally:clear_features(model)
