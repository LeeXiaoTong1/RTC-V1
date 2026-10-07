"""One optimizer update per 16 sources, regardless of physical waveform grouping."""
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

from w2v_v32.model import microbatches
from .data import validate_batch
from .objectives import fusion_frames


def pair_batches(pairs, size, frame_budget):
    """Never split reference/noisy graphs; group nearby lengths without truncation."""
    if size < 2:
        raise ValueError('Paired microbatch requires at least two waveforms')
    ordered = sorted(pairs,key=lambda p:max(r['features'].shape[1] for r in p))
    selected = []
    def emit(items):
        rows = [r for pair in items for r in pair]
        values = [r['features'][0] for r in rows]
        x = torch.nn.utils.rnn.pad_sequence(values,batch_first=True)
        lengths = torch.tensor([len(v) for v in values])
        mask = (torch.arange(x.shape[1])[None] < lengths[:,None]).long()
        return rows,x,mask
    for pair in ordered:
        length = max(r['features'].shape[1] for r in pair)
        if selected and (2*(len(selected)+1)>size or 2*(len(selected)+1)*length>frame_budget
                         or length/min(r['features'].shape[1] for r in selected[0])>1.5):
            yield emit(selected); selected=[]
        selected.append(pair)
    if selected:
        yield emit(selected)


def _grad_norm(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def clear_features(model):
    if hasattr(model.head.classifier,'features'):
        model.head.classifier.features=None
        model.head.classifier.monitor={}


def train_step(model, auxiliary, optimizer, examples, cfg, warm, audit=False):
    model.train(); auxiliary.train()
    occurrences = validate_batch(examples,cfg['source_batch'])
    ordinary = [r for r in examples if r['role']=='ordinary']
    pairs = [sorted([r for r in rows if r['role']!='ordinary'],key=lambda r:r['role']=='noisy')
             for rows in occurrences.values()]
    optimizer.zero_grad(set_to_none=True)
    stats = dict(classification_loss=0.,time_loss=0.,structure_loss=0.,maximum_example_ce=0.,
                 eligible_pairs=0,maximum_padded_frames=0,physical_forwards=0)
    active = warm>0 and (cfg['tfcl_time_weight']>0 or cfg['tfcl_structure_weight']>0)
    context = lambda: torch.autocast('cuda',dtype=torch.bfloat16) if cfg['amp']=='bf16' else nullcontext()

    def classification(rows,z):
        labels = torch.tensor([r['label'] for r in rows],device=z.device)
        ce = F.cross_entropy(z.float(),labels,reduction='none')
        mass = torch.tensor([r['ce_weight'] for r in rows],device=z.device)
        loss = (ce*mass).sum()
        stats['classification_loss'] += float(loss.detach())
        stats['maximum_example_ce'] = max(stats['maximum_example_ce'],float(ce.detach().max()))
        return loss

    def backward(loss):
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite CE/TFCL loss; update not applied')
        loss.backward()
        clear_features(model)

    try:
        for indices,x,mask in microbatches(ordinary,cfg['microbatch'],cfg['frame_budget']):
            with context():
                z,_ = model(x.to(cfg['device']),mask.to(cfg['device']))
            backward(classification([ordinary[i] for i in indices],z))
            stats['maximum_padded_frames'] = max(stats['maximum_padded_frames'],x.shape[0]*x.shape[1])
            stats['physical_forwards'] += 1
            del z
        checked = False
        for rows,x,mask in pair_batches(pairs,cfg['microbatch'],cfg['frame_budget']):
            with fusion_frames(model) as capture, context():
                z,_ = model(x.to(cfg['device']),mask.to(cfg['device']))
                loss = classification(rows,z)
                if len(capture)!=1:
                    raise ValueError('Expected one pre-MultiConv fusion feature tensor')
                features = capture[0]
                aux_loss = features.sum()*0.
                for j in range(0,len(rows),2):
                    if not active or not rows[j]['pair_eligible'] or not rows[j+1]['pair_eligible']:
                        continue
                    masks = [torch.as_tensor(rows[k]['aux_valid'],device=features.device,dtype=torch.bool) for k in (j,j+1)]
                    if min(int(v.sum()) for v in masks)<2:
                        continue
                    ta,sd = auxiliary(features[j,:len(masks[0])],features[j+1,:len(masks[1])],*masks)
                    stats['time_loss'] += float(ta.detach())/cfg['source_batch']
                    stats['structure_loss'] += float(sd.detach())/cfg['source_batch']
                    stats['eligible_pairs'] += 1
                    aux_loss = aux_loss+warm*(cfg['tfcl_time_weight']*ta+cfg['tfcl_structure_weight']*sd)/cfg['source_batch']
                if audit and active and not checked and stats['eligible_pairs']:
                    # Infrequent, explicit evidence that auxiliary gradients reach
                    # BOTH the fusion projection and trainable encoder, not just MHA.
                    targets = [model.head.projection.weight]
                    encoder = [p for p in model.backbone.encoder.layers[-1].parameters() if p.requires_grad]
                    targets.append(encoder[0])
                    gradients = torch.autograd.grad(aux_loss,targets,retain_graph=True,allow_unused=True)
                    stats['aux_gradient_audit'] = dict(zip(('fusion','last_encoder_layer'),
                        [float(g.float().norm()) if g is not None else 0. for g in gradients]))
                    if not all(np.isfinite(v) for v in stats['aux_gradient_audit'].values()):
                        raise FloatingPointError('Nonfinite auxiliary gradient')
                    checked=True
                loss = loss+aux_loss
            backward(loss)
            stats['maximum_padded_frames'] = max(stats['maximum_padded_frames'],x.shape[0]*x.shape[1])
            stats['physical_forwards'] += 1
            del features,z,loss,aux_loss
        params = [p for g in optimizer.param_groups for p in g['params']]
        stats['aux_parameter_grad_norm'] = _grad_norm(auxiliary.parameters())
        norm = torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'])
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('Nonfinite gradient; update not applied')
        stats['gradient_norm'] = float(norm)
        optimizer.step()
        return stats
    finally:
        clear_features(model)
