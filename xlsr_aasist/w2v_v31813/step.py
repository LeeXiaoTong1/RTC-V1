"""Eight-cell CE, coupled XEUS margins, and official-original phoneme MSE."""
from collections import Counter,defaultdict
import math
import numpy as np
import torch
from torch.nn import functional as F
from w2v_v31810.precision import scaler_for,unscale_and_clip,commit
from w2v_v31810.common import capture_rng,restore_rng
from .objectives import paired_margin,phoneme_mse,ramp,frame_count


def cell(row):return row['role'],row['language'],row['label']


def packs(rows,cfg):
    groups=defaultdict(list);singles=[]
    for r in rows:
        if r.get('xeus_pair'):groups[r['xeus_pair']].append(r)
        else:singles.append([r])
    atoms=[]
    for values in groups.values():
        if len(values)!=2 or {r['label'] for r in values}!={0,1} or len({(r['role'],r['language']) for r in values})!=1 or len({r['canonical_source'] for r in values})!=2:
            raise ValueError('XEUS loss requires both opposite-label, different-source partners in the same graph')
        atoms.append(values)
    atoms+=singles
    atoms.sort(key=lambda a:max(len(r['wave']) for r in a))
    current=[]
    for atom in atoms:
        merged=current+atom
        if len(atom)>cfg['microbatch'] or max(frame_count(len(r['wave'])) for r in atom)*len(atom)>cfg['frame_budget']:
            raise ValueError('An atomic XEUS pair exceeds the frame budget; increase frame budget or use a smaller supported maximum duration')
        if current and (len(merged)>cfg['microbatch'] or max(frame_count(len(r['wave'])) for r in merged)*len(merged)>cfg['frame_budget']):
            yield current;current=[]
        current+=atom
    if current:yield current


def components(model,units,cfg,component='all'):
    rows=[r for unit in units for r in unit];counts=Counter(map(cell,rows))
    if len(counts)!=8:raise ValueError('Every logical batch must contain all eight CE cells')
    if len({r['canonical_source'] for r in rows})!=len(rows):raise ValueError('Source repeated in a logical batch')
    pair_counts=Counter((r['role'],r['language']) for r in rows if r.get('xeus_pair') and r['label']==1)
    refs=[r for r in rows if 'original_pcl' in r];phone_counts=Counter((r['language'],r['label']) for r in refs)
    if component in ('all','ce','prior'):
        for batch in packs(rows,cfg):
            logits=model([r['wave'] for r in batch]).float()
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite classification logits')
            labels=torch.tensor([r['label'] for r in batch],device=logits.device)
            raw=F.cross_entropy(logits,labels,reduction='none',label_smoothing=cfg.get('label_smoothing',.02))
            ce=sum(loss/(8*counts[cell(r)]) for r,loss in zip(batch,raw))
            pairs=defaultdict(dict)
            for i,r in enumerate(batch):
                if r.get('xeus_pair'):pairs[r['xeus_pair']][r['label']]=(i,r)
            prior=logits.sum()*0
            for pair in pairs.values():
                i,real=pair[1];j,_=pair[0]
                prior=prior+paired_margin(logits[i],logits[j],cfg['prior_margin'])/(4*pair_counts[real['role'],real['language']])
            # Fixed four-cell denominator leaves missing neighbor cells at zero;
            # scarce matches cannot gain weight by shrinking the denominator.
            if component=='ce':yield 'ce',ce
            elif component=='prior':
                if pairs:yield 'prior',prior
            else:yield 'classification',(ce,prior)
    if component in ('all','pcl'):
        for r in refs:
            p=r['original_pcl'];a,b=p['refs']
            if max(frame_count(len(a['wave'])),frame_count(len(b['wave'])))*2>cfg['frame_budget']:
                raise ValueError('Original PCL reference pair exceeds frame budget')
            h=model.encode([a['wave'],b['wave']])
            loss,n=phoneme_mse(h[0],h[1],p['matches'],a['original_samples'],b['original_samples'])
            if not n:raise ValueError('Prepared PCL references have no genuine supported frames')
            yield 'pcl',loss/(4*phone_counts[r['language'],r['label']])


def train_step(model,optimizer,units,cfg,weights,epoch,step,steps):
    optimizer.zero_grad(set_to_none=True);scaler=scaler_for(optimizer,cfg)
    strength=ramp(epoch,step,steps);totals=dict(ce=0.,prior=0.,pcl=0.)
    for kind,value in components(model,units,cfg):
        if kind=='classification':
            ce,prior=value;loss=ce+strength*weights['prior']*prior
            totals['ce']+=float(ce.detach());totals['prior']+=float(prior.detach())
        else:
            totals['pcl']+=float(value.detach());loss=strength*weights['pcl']*value
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite V3.18.13 objective')
        scaler.scale(loss).backward()
    norm=unscale_and_clip(optimizer,scaler,cfg.get('grad_clip',1.));amp=commit(optimizer,scaler)
    return dict(totals,loss=totals['ce']+strength*(weights['prior']*totals['prior']+weights['pcl']*totals['pcl']),
                ramp=strength,gradient_norm=norm,amp=amp,views=sum(map(len,units)),
                pcl_sources=sum('original_pcl' in r for u in units for r in u),
                xeus_pairs=sum(bool(r.get('xeus_pair')) for u in units for r in u)//2)


def calibrate(model,tasks,cfg):
    """Train only. Gradient scale on the shared fusion, with no optimizer step."""
    rng=capture_rng();modes=[(m,m.training) for m in model.modules()];model.eval()
    parameters=[p for p in model.fusion.parameters() if p.requires_grad]
    norms=defaultdict(list)
    try:
        for index,units in enumerate(tasks):
            for component in ('ce','prior','pcl'):
                if component=='prior' and cfg['mode']=='baseline':continue
                accum=[torch.zeros_like(p) for p in parameters];seen=False
                for _,loss in components(model,units,cfg,component):
                    gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
                    for target,g in zip(accum,gradients):
                        if g is not None:target.add_(g.detach())
                    seen=True
                if seen:
                    norm=math.sqrt(sum(float(x.double().square().sum()) for x in accum))
                    if not math.isfinite(norm):raise FloatingPointError('Nonfinite auxiliary calibration gradient')
                    norms[component].append(norm)
            print(f'Auxiliary Train calibration {index+1}',flush=True)
    finally:
        for m,mode in modes:m.training=mode
        restore_rng(rng)
    ce=float(np.median(norms['ce']));weights={};medians={k:float(np.median(v)) for k,v in norms.items()}
    if not math.isfinite(ce) or ce<=1e-12:raise ValueError('No valid Train CE calibration gradient')
    for name,target,cap in (('prior',.10,.10),('pcl',.05,.01)):
        if name=='prior' and cfg['mode']=='baseline':weights[name]=0.;continue
        value=medians.get(name,0.)
        if value<=1e-12:raise ValueError('No usable '+name+' calibration gradient; inspect Train pair coverage')
        weights[name]=min(cap,target*ce/value)
    return dict(weights=weights,median_gradient_norms=medians,tasks=len(norms['ce']),
                fit_data='Train only',gradient_parameters='shared fusion',optimizer_updates=0,
                requested_gradient_ratios=dict(prior=.10,pcl=.05),coefficient_caps=dict(prior=.10,pcl=.01))
