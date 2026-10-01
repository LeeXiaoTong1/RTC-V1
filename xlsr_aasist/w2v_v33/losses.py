"""Source-balanced classification and parameter-free, masked temporal pairing.

The pair objective is inspired by soft temporal correspondence, not a literal
TFCL reproduction or a frequency-domain objective. Both sides retain label CE.
No teacher, confidence filtering, gradient-ratio rescaling, or new projector is
introduced. A constant representation remains a possible invariant solution;
variance/collapse diagnostics make that visible rather than claiming otherwise.
"""
import math
import torch
from torch.nn import functional as F
from w2v_v3.model import diversity_cka


CONDITIONS = ('offline', 'online', 'noisy_a', 'noisy_b')


def source_components(examples, weights, device, noisy_weight=.5, noisy_class_weights=None):
    """Each canonical source owns one class weight and one normalized CE budget."""
    if not examples or not math.isclose(float(noisy_weight), .5, rel_tol=0, abs_tol=1e-12):
        raise ValueError('V3.3 requires a nonempty batch and fixed 0.5 ordinary/noisy source budget')
    class_weights = torch.as_tensor(weights, dtype=torch.float32, device='cpu')
    if class_weights.shape != (2,) or not bool(torch.isfinite(class_weights).all()) or not bool((class_weights > 0).all()):
        raise ValueError('Two finite positive weights from unique source class counts are required')
    if noisy_class_weights is not None and not torch.allclose(
            torch.as_tensor(noisy_class_weights,dtype=torch.float32,device='cpu'),class_weights,rtol=0,atol=0):
        raise ValueError('V3.3 uses one source class weight; separate noisy weights are unsupported')
    groups = {}
    for index, ex in enumerate(examples):
        source = ex.get('source_id'); condition = ex.get('condition'); view = ex.get('view')
        if not isinstance(source, str) or not source or condition not in CONDITIONS or view not in ('full','short'):
            raise ValueError('Each view needs source_id, official condition, and full/short view')
        if ex.get('label') not in (0,1):
            raise ValueError('Source labels must be fake=0 or real=1')
        weight = float(ex.get('view_weight',float('nan')))
        if not math.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError('View weights must be finite and positive')
        if bool(ex.get('noisy',condition.startswith('noisy'))) != condition.startswith('noisy'):
            raise ValueError('Condition and noisy metadata disagree')
        language=ex.get('language',source.split('/')[1] if len(source.split('/'))>2 else 'unknown')
        if language not in ('en','zh'):language='unknown'
        group = groups.setdefault(source, {'label':ex['label'],'language':language,'conditions':{}})
        if ex['label'] != group['label']:
            raise ValueError('Conditions of one canonical source disagree on label')
        if language != group['language']:
            raise ValueError('Conditions of one canonical source disagree on language')
        group['conditions'].setdefault(condition,[]).append(index)
    coefficients = [0.] * len(examples)
    ordinary_coefficients = [0.] * len(examples)
    noisy_coefficients = [0.] * len(examples)
    offline_full = []
    for group in groups.values():
        conditions = group['conditions']
        if not {'offline','noisy_a','noisy_b'} <= set(conditions):
            raise ValueError('Each canonical source requires Offline and both full noisy versions')
        label_weight = float(class_weights[group['label']])
        ordinary_count = 1 + int('online' in conditions)
        group['full'] = {}
        for condition,indices in conditions.items():
            rows = [examples[i] for i in indices]
            views = [r['view'] for r in rows]
            if len(rows) > 2 or views.count('full') != 1 or len(set(views)) != len(views):
                raise ValueError('Each source condition needs one full and at most one short view')
            if not math.isclose(sum(float(r['view_weight']) for r in rows),1.,abs_tol=1e-7,rel_tol=0):
                raise ValueError('Full/short view weights must sum to one within each condition')
            full = next(i for i in indices if examples[i]['view']=='full')
            group['full'][condition] = full
            if condition=='offline':offline_full.append(full)
            noisy = condition.startswith('noisy')
            conditional = label_weight / (2 if noisy else ordinary_count) / len(groups)
            for index in indices:
                component = conditional * float(examples[index]['view_weight'])
                coefficients[index] = .5 * component
                (noisy_coefficients if noisy else ordinary_coefficients)[index] = component
    return dict(groups=groups,source_count=len(groups),
                labels=torch.tensor([e['label'] for e in examples],dtype=torch.long,device=device),
                weights=class_weights.to(device),
                coefficients=torch.tensor(coefficients,dtype=torch.float32,device=device),
                ordinary_coefficients=torch.tensor(ordinary_coefficients,dtype=torch.float32,device=device),
                noisy_coefficients=torch.tensor(noisy_coefficients,dtype=torch.float32,device=device),
                offline_full=offline_full)


def local_tokens(frames, valid_mask, max_tokens=256, audible_mask=None):
    """Masked local averages on the original time axis; silence is valid by default.

    An explicit audibility mask can remove unusable frames, but this routine
    never infers silence from feature amplitude or treats zeros as padding.
    Token positions retain their relative recording time even across a gap.
    """
    if isinstance(max_tokens,bool) or not isinstance(max_tokens,int) or max_tokens < 2:
        raise ValueError('aux_max_tokens must be an integer >=2')
    if frames.ndim != 2 or valid_mask.ndim != 1 or len(frames)!=len(valid_mask):
        raise ValueError('Expected [T,D] frames and [T] mask')
    valid=valid_mask.bool()
    if audible_mask is not None:
        audible=torch.as_tensor(audible_mask,device=frames.device,dtype=torch.bool)
        if audible.shape != valid.shape:raise ValueError('Audibility mask does not match native frame count')
        valid=valid & audible
    # Drop trailing padding before pooling; keep internal missing-frame gaps on
    # their original timeline. Length is metadata, never model confidence.
    positions=torch.nonzero(valid_mask.bool(),as_tuple=False).flatten()
    if positions.numel()<2:return None
    length=int(positions[-1])+1
    features=frames[:length].float()
    valid=valid[:length]
    count=min(max_tokens,length)
    masses=F.adaptive_avg_pool1d(valid.float()[None,None,:],count).reshape(-1)
    pooled=F.adaptive_avg_pool1d(features.masked_fill(~valid[:,None],0).T[None],count)[0].T
    pooled=pooled/masses.clamp_min(1e-12)[:,None]
    timeline=torch.linspace(0.,1.,length,device=frames.device)
    centers=F.adaptive_avg_pool1d((timeline*valid)[None,None,:],count).reshape(-1)/masses.clamp_min(1e-12)
    present=masses>0
    pooled=pooled[present];centers=centers[present]
    if len(pooled)<2:return None
    return F.normalize(pooled,p=2,dim=-1,eps=1e-8),centers


def aligned_temporal_loss(left, right, *, temperature=.1, time_sigma=.25):
    """Bidirectional soft alignment plus aligned within-recording relations."""
    x,tx=left;y,ty=right
    if not math.isfinite(temperature) or not math.isfinite(time_sigma) or temperature<=0 or time_sigma<=0:
        raise ValueError('Positive finite alignment temperature and time sigma required')
    similarity=x@y.T
    prior=-.5*((tx[:,None]-ty[None,:])/time_sigma).square()
    # Avoid optimizing correspondence confidence instead of the shared features.
    # Values on BOTH sides remain connected to the classification backbone.
    ab=(similarity.detach()/temperature+prior).softmax(-1)
    ba=(similarity.detach().T/temperature+prior.T).softmax(-1)
    aligned_y=F.normalize(ab@y,p=2,dim=-1,eps=1e-8)
    aligned_x=F.normalize(ba@x,p=2,dim=-1,eps=1e-8)
    time=.5*((1-(x*aligned_y).sum(-1)).clamp_min(0).mean()
             +(1-(y*aligned_x).sum(-1)).clamp_min(0).mean())
    # The complete local relation matrices distinguish temporal organization
    # from a single utterance-average embedding. Diagonals are valid (usually 1).
    structure=.5*((x@x.T-aligned_y@aligned_y.T).square().mean()
                  +(y@y.T-aligned_x@aligned_x.T).square().mean())
    variance=.5*(x.var(0,unbiased=False).mean()+y.var(0,unbiased=False).mean())
    return .5*(time+structure),time,structure,variance


def pair_objective(frame_records, components, *, max_tokens=256, temperature=.1, time_sigma=.25):
    """Mean pairs within source, then one class weight, then mean all sources."""
    grouped={}
    for language in ('en','zh','unknown'):
        for label in ('fake','real'):
            prefix=f'pair_{language}_{label}'
            grouped.update({prefix+'_'+key:0 for key in ('sources','planned_pairs','valid_sources',
                'valid_pairs','skipped_pairs','collapsed_pairs')})
    for group in components['groups'].values():
        prefix=f"pair_{group['language']}_{'real' if group['label'] else 'fake'}"
        grouped[prefix+'_sources']+=1
        grouped[prefix+'_planned_pairs']+=len(group['full'])-1
    if not frame_records:
        zero=components['weights'].sum()*0
        return zero,dict(pair_loss=zero,time_loss=zero,structure_loss=zero,pair_variance=zero,
            pair_valid_count=0,pair_source_count=0,pair_skipped_count=0,pair_collapsed_count=0,**grouped)
    zero=next(iter(frame_records.values()))[0].sum()*0
    total=time_total=structure_total=variance_total=zero
    valid_count=source_count=skipped_count=0
    collapse_values=[]
    for group in components['groups'].values():
        prefix=f"pair_{group['language']}_{'real' if group['label'] else 'fake'}"
        full=group['full']; reference=frame_records.get(full['offline'])
        targets=[full[c] for c in ('online','noisy_a','noisy_b') if c in full]
        ref_tokens=local_tokens(*reference[:2],max_tokens,reference[2]) if reference is not None else None
        rows=[]
        for index in targets:
            target=frame_records.get(index)
            tokens=local_tokens(*target[:2],max_tokens,target[2]) if target is not None else None
            if ref_tokens is None or tokens is None:
                skipped_count+=1;grouped[prefix+'_skipped_pairs']+=1;continue
            rows.append(aligned_temporal_loss(ref_tokens,tokens,temperature=temperature,time_sigma=time_sigma))
        if not rows:continue
        source_count+=1;valid_count+=len(rows)
        grouped[prefix+'_valid_sources']+=1;grouped[prefix+'_valid_pairs']+=len(rows)
        means=torch.stack([torch.stack(row) for row in rows]).mean(0)
        factor=components['weights'][group['label']]/components['source_count']
        total=total+factor*means[0];time_total=time_total+factor*means[1]
        structure_total=structure_total+factor*means[2]
        variance_total=variance_total+means[3]*len(rows)
        collapse_values.extend(row[3].detach()<=1e-8 for row in rows)
        grouped[prefix+'_collapsed_pairs']=grouped[prefix+'_collapsed_pairs']+torch.stack(
            [row[3].detach()<=1e-8 for row in rows]).sum()
    variance=variance_total/max(valid_count,1)
    collapsed=torch.stack(collapse_values).sum() if collapse_values else zero.detach()
    return total,dict(pair_loss=total,time_loss=time_total,structure_loss=structure_total,
        pair_variance=variance,pair_valid_count=valid_count,pair_source_count=source_count,
        pair_skipped_count=skipped_count,pair_collapsed_count=collapsed,**grouped)


def reference_objective(logits, blocks, examples, weights, frame_records=None,
                        cka_weight=.01,pair_weight=.02,aux_max_tokens=256):
    components=source_components(examples,weights,logits.device)
    ce=F.cross_entropy(logits.float(),components['labels'],reduction='none')
    classification=(ce*components['coefficients']).sum()
    cka=diversity_cka(blocks[components['offline_full']]) if cka_weight else classification*0
    pair,diagnostics=pair_objective(frame_records or {},components,max_tokens=aux_max_tokens)
    loss=classification+cka_weight*cka+pair_weight*pair
    return loss,dict(ce=classification,cka=cka,
        ordinary_ce=(ce*components['ordinary_coefficients']).sum(),
        noisy_ce=(ce*components['noisy_coefficients']).sum(),**diagnostics)
