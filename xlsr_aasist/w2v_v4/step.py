"""Two CE streams, unchanged XEUS/PCL objectives, one optimizer transaction."""
from collections import Counter, defaultdict
import torch
from torch.nn import functional as F
from w2v_v31813.step import packs, components, calibrate
from w2v_v31813.objectives import paired_margin, ramp
from w2v_v31810.precision import scaler_for, unscale_and_clip, commit
from .gradients import coordinate


def train_step(model, optimizer, units, cfg, weights, epoch, step, steps):
    rows = [r for unit in units for r in unit]
    cell = lambda r: (r['role'], r['language'], r['label'])
    counts = Counter(map(cell, rows))
    if set(counts) != {(role, lang, label) for role in ('online','noisy') for lang in ('en','zh') for label in (0,1)}:
        raise ValueError('V4 requires all eight balanced CE cells')
    if len({r['canonical_source'] for r in rows}) != len(rows) or any(r.get('split') != 'train' for r in rows):
        raise ValueError('V4 classification requires distinct registered Train sources')
    params = [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
    buffers = {role: [torch.zeros_like(p) for p in params] for role in ('online','noisy','aux')}
    used = [False]*len(params)
    optimizer.zero_grad(set_to_none=True)
    scaler = scaler_for(optimizer, cfg)
    strength = ramp(epoch, step, steps)
    totals = dict(ce=0., prior=0., pcl=0.)
    stream_losses = {}
    pair_counts = Counter((r['role'],r['language']) for r in rows if r.get('xeus_pair') and r['label']==1)

    def accumulate(loss, role, retain=False):
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite V4 loss')
        gradients = torch.autograd.grad(scaler.scale(loss), params, retain_graph=retain, allow_unused=True)
        for i,(target, gradient) in enumerate(zip(buffers[role], gradients)):
            if gradient is not None:
                used[i] = True
                target.add_(gradient.detach())

    for role in ('online','noisy'):
        stream_losses[role] = 0.
        for batch in packs([r for r in rows if r['role']==role], cfg):
            logits = model([r['wave'] for r in batch]).float()
            labels = torch.tensor([r['label'] for r in batch], device=logits.device)
            losses = F.cross_entropy(logits, labels, reduction='none', label_smoothing=cfg.get('label_smoothing',.02))
            ce = sum(loss/(8*counts[cell(r)]) for loss,r in zip(losses,batch))
            pairs = defaultdict(dict)
            for i,r in enumerate(batch):
                if r.get('xeus_pair'): pairs[r['xeus_pair']][r['label']] = (i,r)
            prior = logits.sum()*0
            for pair in pairs.values():
                i,real = pair[1]; j,_ = pair[0]
                prior = prior + paired_margin(logits[i],logits[j],cfg['prior_margin'])/(4*pair_counts[role,real['language']])
            active = bool(pairs and strength*weights['prior'] > 0)
            accumulate(ce, role, active)
            if active: accumulate(prior*strength*weights['prior'], 'aux')
            value = float(ce.detach()); totals['ce'] += value; stream_losses[role] += 2*value
            totals['prior'] += float(prior.detach())
    if strength*weights['pcl'] > 0:
        for _,loss in components(model, units, cfg, 'pcl'):
            accumulate(loss*strength*weights['pcl'], 'aux')
            totals['pcl'] += float(loss.detach())
    merged, coordination = coordinate(buffers['online'], buffers['noisy'], buffers['aux'],
        enabled=cfg.get('gradient_coordination','pcgrad')=='pcgrad', scale=scaler.get_scale())
    offset = 0
    for i,parameter in enumerate(params):
        count = parameter.numel()
        parameter.grad = merged[offset:offset+count].view_as(parameter).clone() if used[i] else None
        offset += count
    del buffers, merged
    norm = unscale_and_clip(optimizer, scaler, cfg.get('grad_clip',1.))
    amp = commit(optimizer, scaler)
    return dict(totals, loss=totals['ce']+strength*(weights['prior']*totals['prior']+weights['pcl']*totals['pcl']),
        ramp=strength, gradient_norm=norm, amp=amp, coordination=coordination,
        stream_ce=stream_losses, views=len(rows), pcl_sources=sum('original_pcl' in r for r in rows),
        xeus_pairs=sum(bool(r.get('xeus_pair')) for r in rows)//2,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if str(cfg.get('device','cpu')).startswith('cuda') else None)
