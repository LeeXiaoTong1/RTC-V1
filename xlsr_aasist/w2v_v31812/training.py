"""Cached-feature CE experiments. Parent/classifier receive no gradients."""
from collections import defaultdict
from pathlib import Path
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from w2v_v31810.common import atomic_save, atomic_json, read_json, append_json
from w2v_v3189.sampling import quota_schedule
from w2v_v31810.precision import unscale_and_clip, commit
from .model import Residual, CachedDetector
from .metrics import report, guard
from . import SCHEMA

CELLS = tuple((role,lang,label) for role in ('online','noisy') for lang,label in (('en',0),('en',1),('zh',0),('zh',1)))


def cells(rows):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['role'],r['language'],r['label']].append(i)
    if set(groups)!=set(CELLS):raise ValueError('All eight Train cells are required')
    return groups


def balanced_ce(logits, labels, group_index, smoothing=0.):
    losses=F.cross_entropy(logits,labels,reduction='none',label_smoothing=smoothing)
    if set(group_index.tolist())!=set(range(8)):raise ValueError('Every update requires all eight CE cells')
    return torch.stack([losses[group_index==i].mean() for i in range(8)]).mean()


def normalization(data):
    x=np.concatenate((data['global'],data['local']),1).astype(np.float64)
    groups=cells(data['rows'])
    mean=np.mean([x[ii].mean(0) for ii in groups.values()],0)
    variance=np.mean([((x[ii]-mean)**2).mean(0) for ii in groups.values()],0)
    return torch.tensor(mean,dtype=torch.float32),torch.tensor(np.sqrt(variance).clip(1e-4),dtype=torch.float32)


class Probe(nn.Module):
    def __init__(self, kind, mean, scale):
        super().__init__();self.kind=kind
        self.register_buffer('mean',mean);self.register_buffer('scale',scale)
        self.linear=nn.Linear(256 if kind=='global' else 384,2)
        nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)

    def forward(self,g,l):
        x=torch.cat((g,l),-1);x=(x-self.mean)/self.scale
        return self.linear(x[:,:256] if self.kind=='global' else x[:,256:])


def schedule(rows, budgets, batch, seed, epoch):
    """Parent source budgets; cached variants rotate, sources cycle without replacement."""
    groups=cells(rows);rng=np.random.default_rng(seed+epoch*1009)
    queues={}
    for cell,indices in groups.items():
        sources=defaultdict(list)
        for i in indices:sources[rows[i]['canonical_source']].append(i)
        names=sorted(sources);needed=budgets[CELLS.index(cell)%4];items=[];cycle=0
        while len(items)<needed:
            for j in rng.permutation(len(names)):
                variants=sources[names[j]]
                items.append(variants[(epoch+cycle)%len(variants)])
            cycle+=1
        queues[cell]=items[:needed]
    offset=[0]*4
    for quotas in quota_schedule(budgets,batch//2):
        result=[]
        for c in CELLS:
            j=CELLS.index(c)%4;result.extend(queues[c][offset[j]:offset[j]+quotas[j]])
        offset=[a+b for a,b in zip(offset,quotas)]
        yield result


@torch.no_grad()
def predict(model,data,device):
    model.eval();output=[]
    for i in range(0,len(data['rows']),1024):
        g=torch.from_numpy(data['global'][i:i+1024]).to(device)
        l=torch.from_numpy(data['local'][i:i+1024]).to(device)
        output.append(model(g,l).float().cpu().numpy())
    return np.concatenate(output)


def fit(kind, train, validation, classifier, cfg, directory, seed, identity, stop=None):
    """Only Train-inner chooses epoch. External Dev is never read here."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);device=torch.device(cfg['train_device'])
    mean,scale=normalization(train)
    if kind=='residual':
        model=CachedDetector(classifier,Residual(mean,scale),cfg['backend_precision'])
    else:model=Probe(kind,mean,scale)
    model=model.to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=kind=='residual' and device.type=='cuda' and cfg['backend_precision']=='fp16',init_scale=256.)
    path=directory/'last.pt';begin=0;best=None;stale=0;history=[]
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['schema']!=SCHEMA or state['identity']!=identity or state['kind']!=kind or state['seed']!=seed:
            raise ValueError('Residual/probe resume identity changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        begin=state['epoch'];best=state['best'];stale=state['stale'];history=state['history']
    baseline=report(validation['rows'],validation['logits'])
    labels=torch.tensor([r['label'] for r in train['rows']],device=device)
    group=torch.tensor([CELLS.index((r['role'],r['language'],r['label'])) for r in train['rows']],device=device)
    # Small pooled features fit comfortably beside the parent; no waveforms on GPU.
    g=torch.from_numpy(train['global']).to(device);l=torch.from_numpy(train['local']).to(device)
    for epoch in range(begin,cfg['epochs']):
        if stale>=cfg['patience']:break
        if stop and stop():raise InterruptedError('Stop requested; resume from saved residual epoch')
        # Replaying an interrupted epoch reproduces its ordering and dropout.
        torch.manual_seed(seed+epoch*1009);model.train();total=0.;steps=0;skipped=0
        expected_steps=len(quota_schedule(cfg['source_draws_per_stream'],cfg['batch_size']//2))
        for indices in schedule(train['rows'],cfg['source_draws_per_stream'],cfg['batch_size'],seed,epoch):
            if stop and stop():raise InterruptedError('Stop requested; resume from saved residual epoch')
            ii=torch.tensor(indices,device=device);optimizer.zero_grad(set_to_none=True)
            z=model(g[ii],l[ii]);loss=balanced_ce(z,labels[ii],group[ii],cfg['label_smoothing'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite residual loss')
            scaler.scale(loss).backward()
            unscale_and_clip(optimizer,scaler,1.)
            skipped+=int(commit(optimizer,scaler)['skipped_update'])
            total+=float(loss.detach());steps+=1
            if steps%100==0:
                print(f'{kind} seed={seed} | epoch {epoch+1}/{cfg["epochs"]} | {steps}/{expected_steps} | loss={total/steps:.5f}',flush=True)
        if skipped==steps:raise FloatingPointError('Every residual update overflowed')
        z=predict(model,validation,device);metrics=report(validation['rows'],z)
        protected=guard(baseline,metrics,scope='inner');eligible=protected['passed'] if kind=='residual' else True
        # An eligible checkpoint always ranks above a non-eligible diagnostic one.
        rank=(int(eligible),metrics['score'])
        improved=best is None or rank>tuple(best['rank'])
        if improved:
            best=dict(epoch=epoch+1,rank=list(rank),metrics=metrics,guard=protected,
                      model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            stale=0
        else:stale+=1
        event=dict(epoch=epoch+1,loss=total/max(1,steps),metrics=metrics,guard=protected,best=improved,steps=steps,skipped_updates=skipped)
        history.append(event)
        atomic_save(path,dict(schema=SCHEMA,identity=identity,kind=kind,seed=seed,epoch=epoch+1,
            model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),best=best,stale=stale,history=history))
        atomic_json(directory/'history.json',history)
        en=metrics['groups']['official_online/en']
        print(f'{kind} seed={seed} | epoch {epoch+1}/{cfg["epochs"]} | loss={event["loss"]:.5f} | '
              f'Inner EN-real recall={100*en["real_recall"]:.2f} | '
              f'Regular={100*metrics["panels"]["regular_noisy"]["macro_f1"]:.2f} | '
              f'Heldout={100*metrics["panels"]["heldout_noisy"]["macro_f1"]:.2f} | '
              f'Inner EN-fake recall={100*en["fake_recall"]:.2f} | '
              f'inner_guard={protected["passed"]} | mode={kind}',flush=True)
    if best is None:raise ValueError('No completed residual/probe epoch')
    model.load_state_dict(best['model']);model.eval()
    atomic_json(directory/'selected.json',dict(kind=kind,seed=seed,epoch=best['epoch'],
        metrics=best['metrics'],guard=best['guard'],selection_scope='Train-inner only',identity=identity))
    return model,best
