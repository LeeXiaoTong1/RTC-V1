"""Positive global affine calibration; reserved Dev sources only, after selection."""
import numpy as np
import torch
from torch.nn import functional as F
from .common import fingerprint


def fit(rows,logits,indices):
    z=np.asarray(logits,dtype=np.float64)
    cells={}
    for i in indices:cells.setdefault((rows[i]['condition'],rows[i]['language'],rows[i]['label']),[]).append(i)
    if len(cells)!=12:raise ValueError('Calibration needs all condition/language/class cells')
    weights=np.zeros(len(rows),np.float64)
    for (condition,language,label),ix in cells.items():
        mass={'online':.3,'seen':.35,'heldout':.35}[condition]/4
        weights[ix]=mass/len(ix)
    keep=np.asarray(indices,dtype=np.int64)
    margin=torch.from_numpy(z[keep,0]-z[keep,1]);target=torch.tensor([rows[i]['label']==0 for i in indices],dtype=torch.float64)
    weight=torch.from_numpy(weights[keep]);log_scale=torch.zeros((),dtype=torch.float64,requires_grad=True)
    bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.Adam([log_scale,bias],lr=.025)
    for _ in range(1000):
        optimizer.zero_grad(set_to_none=True)
        objective=(F.binary_cross_entropy_with_logits(log_scale.exp()*margin+bias,target,reduction='none')*weight).sum()
        objective=objective+.001*(log_scale.square()+bias.square())
        objective.backward();optimizer.step()
        with torch.no_grad():log_scale.clamp_(-3,3);bias.clamp_(-10,10)
    scale=float(log_scale.detach().exp());intercept=float(bias.detach())
    if not np.isfinite([scale,intercept]).all() or scale<=0:raise ValueError('Invalid calibration')
    return dict(scale=scale,bias=intercept,count=len(indices),indices_sha256=fingerprint(indices),
                scope='reserved Dev sources excluded from model selection; global positive affine; no language routing',
                classification_threshold=.5,ranking_preserved=True)


def apply(logits,calibration):
    z=np.asarray(logits,dtype=np.float64)
    margin=(z[:,0]-z[:,1])*calibration['scale']+calibration['bias']
    return np.stack((margin/2,-margin/2),axis=1)
