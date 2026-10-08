"""Batched two-sided TFCL on complete valid trajectories, CKA per source.

Magnitude-normalized auxiliary inputs prevent reducing feature scale to evade
the constraint. Classification sees the unchanged, unnormalized fusion features.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TFCL(nn.Module):
    def __init__(self,channels=128,heads=8,bins=201):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels,heads,batch_first=True,dropout=0.)
        self.time_projection = nn.Linear(bins,bins)
        self.bins = bins
        nn.init.eye_(self.time_projection.weight); nn.init.zeros_(self.time_projection.bias)

    def forward(self,pairs):
        # Callers exclude invalid pairs before attention (all-masked attention is NaN).
        if not pairs: raise ValueError('No valid TFCL pairs')
        left,right = [],[]
        for a,b,am,bm in pairs:
            a,b = a[am],b[bm]
            if min(len(a),len(b))<2: raise ValueError('Need >=2 valid frames')
            left.append(F.normalize(a.float(),dim=-1,eps=1e-6))
            right.append(F.normalize(b.float(),dim=-1,eps=1e-6))
        def padded(values):
            x = nn.utils.rnn.pad_sequence(values,batch_first=True)
            valid = torch.arange(x.shape[1],device=x.device)[None]<torch.tensor([len(v) for v in values],device=x.device)[:,None]
            return x,valid
        a,am = padded(left); b,bm = padded(right)
        ab,_ = self.attention(a,b,b,key_padding_mask=~bm,need_weights=False)
        ba,_ = self.attention(b,a,a,key_padding_mask=~am,need_weights=False)
        ta = ((1-F.cosine_similarity(ab.float(),a,dim=-1))*am).sum(1)/am.sum(1)
        tb = ((1-F.cosine_similarity(ba.float(),b,dim=-1))*bm).sum(1)/bm.sum(1)
        def structure(values):
            x = torch.stack([F.adaptive_avg_pool1d(v.T[None],self.bins)[0] for v in values])
            x = self.time_projection(x).float()
            x = x-x.mean(1,keepdim=True)
            return x@x.transpose(1,2)
        ga,gb = structure(left),structure(right)
        denom = ga.flatten(1).norm(dim=1)*gb.flatten(1).norm(dim=1)
        cka = (ga*gb).sum((1,2))/denom.clamp_min(1e-8)
        return .5*(ta+tb),1-cka.clamp(-1.,1.)
