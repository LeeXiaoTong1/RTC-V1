"""Fixed-pair measurements of representation changes and detector behavior.

These permitted training sources are NOT an unseen validation set. No metric
here selects weights, thresholds, recipes, or training coefficients.
"""
from collections import Counter
from contextlib import contextmanager
import numpy as np
import torch
from .common import capture_rng, restore_rng, atomic_json, fingerprint
from .repair import aligned_reference, source_feature_loss


@contextmanager
def evaluation(model):
    rng,modes = capture_rng(),[(m,m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module,mode in modes:
            module.training = mode
        restore_rng(rng)


def buckets(record):
    seconds = record.get('seconds',0.)
    snr = record.get('snr_db')
    snr = 'unknown_or_echo' if snr is None else '<0' if snr < 0 else '0-10' if snr < 10 else '10-20' if snr < 20 else '>=20'
    duration = '<1' if seconds < 1 else '1-3' if seconds < 3 else '3-6' if seconds < 6 else '>=6'
    return (record['group'], 'origin/'+record['origin'], 'condition/'+record.get('condition','unknown'),
            'snr/'+snr, 'seconds/'+duration, record['group']+'/origin/'+record['origin'])


def outcome(clean, noisy, label):
    c,n = int(clean.argmax()) == label,int(noisy.argmax()) == label
    return 'both_correct' if c and n else 'noisy_only_wrong' if c else 'clean_only_wrong' if n else 'both_wrong'


def pair_measure(model, clean, noisy, info, label, *, intervention=False):
    """Full sequences, identical head mode, and original-prefix pairing masks."""
    c,n = clean.float(),noisy.float()
    target,mask = aligned_reference(c,torch.as_tensor(info['clean_valid'],device=c.device),len(n),info['shift_frames'])
    mask &= torch.as_tensor(info['noisy_valid'],device=n.device)
    ready = bool(model.repair.scale_ready)
    scale = model.repair.frame_scale if ready else torch.ones_like(model.repair.frame_scale)
    modified = [model.repair(x[None],[len(x)]) for x in (c,n)]
    fused = [x['fused'][0] for x in modified] if model.cfg['repair_arm'] != 'baseline' else [c,n]
    logits = model.classify_processed(fused)
    bypass = model.classify_processed([c,n])
    # Two raw values distinguish the target's drift from actual repair error.
    raw,_ = source_feature_loss(n[None],target[None],mask[None],scale,model.cfg['repair_min_frames'])
    repaired,_ = source_feature_loss(modified[1]['repaired'],target[None],mask[None],scale,model.cfg['repair_min_frames'])
    applied = fused[1]-n
    gate = modified[1]['gate'].flatten()
    margin = logits[:,label]-logits[:,1-label]
    bypass_margin = bypass[:,label]-bypass[:,1-label]
    values = torch.cat((margin,bypass_margin,raw,repaired,
        torch.stack((applied.norm()/n.norm().clamp_min(1e-8),gate.mean(),gate.min(),gate.max(),
                     (logits.softmax(-1)[:,0]-bypass.softmax(-1)[:,0]).abs().max())))).cpu().tolist()
    keys = ('clean_margin','noisy_margin','bypass_clean_margin','bypass_noisy_margin','raw_feature_loss',
            'repaired_feature_loss','correction_ratio','gate_mean','gate_min','gate_max','max_bypass_probability_change')
    result = dict(zip(keys,values),valid_frames=int(mask.sum()),frames=len(n),
        outcome=outcome(logits[0],logits[1],label),bypass_outcome=outcome(bypass[0],bypass[1],label),
        noisy_correct=int(logits[1].argmax())==label,bypass_noisy_correct=int(bypass[1].argmax())==label)
    squared = ((n-target)/scale).square().mean(-1)
    chunks = torch.tensor_split(torch.arange(len(n),device=n.device),8)
    result['temporal_mse'] = [float(squared[part][mask[part]].mean()) if mask[part].any() else None for part in chunks]
    result['channel_mse'] = ((n[mask]-target[mask])/scale).square().mean(0).cpu().tolist() if mask.any() else None
    if intervention:
        # Fixed middle quarter; never select whichever interval improves the
        # prediction most. Privileged hidden intervention is diagnostic only.
        region = torch.arange(len(n),device=n.device)
        patch_mask = mask & (region >= len(n)//4) & (region < len(n)//2)
        replaced = torch.where(patch_mask[:,None],target,n)
        replaced_logits = model.classify_sequences([replaced])[0]
        result['intervention_frames'] = int(patch_mask.sum())
        result['intervention_margin_change'] = float(replaced_logits[label]-replaced_logits[1-label]-margin[1])
    return result,logits.cpu(),mask,target


def encode_bounded(model, waves, cfg):
    """Preparation/probes obey the same padding budget as training."""
    result,chunk,maximum = [],[],0
    for wave in waves:
        frames = max(1,(len(wave)-400)//320+1)
        if frames > cfg['frame_budget']:
            raise ValueError('A diagnostic view exceeds --frame-budget; do not silently crop')
        if chunk and (max(maximum,frames)*(len(chunk)+1) > cfg['frame_budget'] or len(chunk) >= cfg['microbatch']):
            result.extend(model.encode(chunk));chunk=[];maximum=0
        chunk.append(wave);maximum=max(maximum,frames)
    if chunk:
        result.extend(model.encode(chunk))
    return result


def aggregate(records):
    groups = {}
    for record in records:
        for key in buckets(record):
            cell = groups.setdefault(key,Counter())
            cell.update(pairs=1,valid=int(record.get('valid_frames',0)>=4),
                frames=record['frames'],valid_frames=record.get('valid_frames',0))
            for field in ('clean_margin','noisy_margin','raw_feature_loss','repaired_feature_loss',
                          'correction_ratio','reference_drift','gate_mean','max_bypass_probability_change'):
                if field in record:
                    cell[field+'_sum'] += record[field]
            if 'outcome' in record:
                cell[record['outcome']] += 1
            if 'bypass_noisy_correct' in record:
                cell['repair_corrected'] += int(record['noisy_correct'] and not record['bypass_noisy_correct'])
                cell['repair_new_error'] += int(not record['noisy_correct'] and record['bypass_noisy_correct'])
            if 'initial_noisy_correct' in record:
                cell['since_initial_corrected'] += int(record['noisy_correct'] and not record['initial_noisy_correct'])
                cell['since_initial_new_error'] += int(not record['noisy_correct'] and record['initial_noisy_correct'])
    return {key:dict(value) for key,value in groups.items()}


def probe(model, cfg, run, tag, *, cached_initial=False):
    """Re-encode only the fixed 64-pair monitor at phase/end boundaries."""
    path = run/'fixed_pairs.pt'
    if not path.is_file():
        raise ValueError('Fixed pair reference missing; do not claim functional repair audit')
    saved = torch.load(path,map_location='cpu',weights_only=True)
    if saved['config'] != fingerprint(cfg):
        raise ValueError('Fixed pair monitor belongs to another recipe')
    records,encoded_views = [],0
    with evaluation(model):
        for item in saved['pairs']:
            if cached_initial:
                device = model.repair.frame_scale.device
                c,n = (item[key].to(device) for key in ('clean_feature','noisy_feature'))
            else:
                context=item.get('encode_context',[item['clean_wave'],item['noisy_wave']])
                values=encode_bounded(model,context,cfg)
                c,n=(values[i] for i in item.get('encode_indices',[0,1]))
                encoded_views+=len(context)
            info = {k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in item['info'].items()}
            row,logits,mask,target = pair_measure(model,c,n,info,item['label'],intervention=cached_initial)
            original = item['clean_feature'].to(c.device)
            valid = item['scale_valid'].to(c.device)
            drift = ((c.float()[valid]-original[valid])/model.repair.frame_scale).square().mean() if valid.any() else c.new_zeros(())
            row.update(item['identity'],reference_drift=float(drift),
                initial_noisy_correct=int(item['initial_logits'][1].argmax())==item['label'],
                initial_noisy_margin=float(item['initial_logits'][1,item['label']]-item['initial_logits'][1,1-item['label']]),
                noisy_fake_probability=float(logits[1].softmax(-1)[0]))
            records.append(row)
    result = dict(schema='v54_fixed_pair_probe_v1',tag=tag,pairs=len(records),groups=aggregate(records),records=records,
        encoder_views=encoded_views,
        reference_geometry='Same saved encoding companions and frame budget as the initial feature reference',
        scope='Fixed permitted training sources/recipes; no model selection, no claim of unseen generalization')
    atomic_json(run/(tag+'.json'),result)
    return result
