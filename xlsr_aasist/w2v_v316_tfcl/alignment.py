"""Detached content matching: monotone edit alignment with explicit unmatched states.

This is an engineering adaptation of the Drop-DTW principle, not its exact
recurrence or a claim to recover phonetic ground truth. Only the matching grid
is bounded; detection and losses retain the native valid feature frames.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F


def monotone_pairs(cost, drop_cost):
    """Global sequence alignment: match, drop reference, or drop processed bin.

    Independent diagonal vector operations avoid a Python loop per cell. There
    is no forced endpoint match. Unrelated sequences may return no matches.
    """
    cost=np.asarray(cost,dtype=np.float64)
    if cost.ndim!=2 or not np.isfinite(cost).all() or drop_cost<=0:
        raise ValueError('Finite pairwise cost and positive drop cost required')
    n,m=cost.shape
    dp=np.full((n+1,m+1),np.inf);dp[:,0]=np.arange(n+1)*drop_cost;dp[0,:]=np.arange(m+1)*drop_cost
    action=np.zeros((n+1,m+1),np.uint8)
    for diagonal in range(2,n+m+1):
        i=np.arange(max(1,diagonal-m),min(n,diagonal-1)+1);j=diagonal-i
        values=np.stack((dp[i-1,j-1]+cost[i-1,j-1],dp[i-1,j]+drop_cost,dp[i,j-1]+drop_cost))
        best=values.argmin(0);dp[i,j]=values[best,np.arange(len(i))];action[i,j]=best
    pairs=[];i,j=n,m
    while i and j:
        choice=action[i,j]
        if choice==0:pairs.append((i-1,j-1));i-=1;j-=1
        elif choice==1:i-=1
        else:j-=1
    return pairs[::-1]


def _empty(device):
    return dict(reference=torch.empty(0,device=device,dtype=torch.long),
                processed=torch.empty(0,device=device,dtype=torch.long),
                confidence=torch.empty(0,device=device),coverage=0.,coarse_matches=0)


def _normalized(x,mask):
    # Remove per-recording stationary offsets before content cosine matching.
    valid=x[mask]
    if not len(valid):return torch.zeros_like(x)
    return F.normalize((x-valid.mean(0,keepdim=True)).masked_fill(~mask[:,None],0),dim=-1,eps=1e-6)


@torch.no_grad()
def align(reference,processed,reference_valid,processed_valid,cfg,known_mapping=False,hop=1.,reference_hop=1.):
    """Return integer native-frame correspondences, detached from all parameters.

    Simulated audio keeps input sample time (fresh codecs compensate their own
    delay); content refines at most two frames around that recorded map. Real
    Online uses skip-capable monotone alignment on <=192 contiguous time bins.
    Padding/missing bins are never removed from the time axis before matching.
    """
    with torch.autocast(device_type=reference.device.type,enabled=False):
        a,b=reference.detach().float(),processed.detach().float()
        am=torch.as_tensor(reference_valid,device=a.device,dtype=torch.bool)
        bm=torch.as_tensor(processed_valid,device=a.device,dtype=torch.bool)
        if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or len(am)!=len(a) or len(bm)!=len(b):
            raise ValueError('Content trajectories and masks differ')
        if not torch.isfinite(a[am]).all() or not torch.isfinite(b[bm]).all():
            raise FloatingPointError('Nonfinite frozen content features')
        if min(int(am.sum()),int(bm.sum()))<2:return _empty(a.device)
        a=_normalized(a,am);b=_normalized(b,bm)
        threshold=cfg.get('alignment_min_cosine',.6)
        if known_mapping:
            js=torch.arange(len(b),device=a.device)
            expected=((js+.5)*hop/reference_hop-.5).round().long()
            radius=cfg.get('alignment_local_radius',2)
            candidates=expected[:,None]+torch.arange(-radius,radius+1,device=a.device)
            valid=(candidates>=0)&(candidates<len(a))
            bounded=candidates.clamp(0,len(a)-1);valid&=am[bounded]&bm[:,None]
            similarity=(a[bounded]*b[:,None]).sum(-1).masked_fill(~valid,-2.)
            confidence,choice=similarity.max(1);ii=bounded.gather(1,choice[:,None])[:,0]
            keep=(confidence>=threshold)&(expected>=0)&(expected<len(a))
            # Local refinement must not reverse the ordering. Ties are allowed.
            previous=torch.cummax(ii.masked_fill(~keep,-1),0).values
            keep&=ii>=torch.cat((previous.new_full((1,),-1),previous[:-1]))
            ii,jj,confidence=ii[keep],js[keep],confidence[keep]
            coarse_count=len(ii)
        else:
            stride=max(1,math.ceil(max(len(a),len(b))/cfg.get('alignment_max_bins',192)))
            def pool(x,valid):
                count=math.ceil(len(x)/stride);pad=count*stride-len(x)
                weights=F.pad(valid,(0,pad)).reshape(count,stride)
                values=F.pad(x,(0,0,0,pad)).reshape(count,stride,-1)
                mean=(values*weights[:,:,None]).sum(1)/weights.sum(1)[:,None].clamp_min(1)
                return F.normalize(mean,dim=-1),weights.sum(1)>0
            ac,av=pool(a,am);bc,bv=pool(b,bm);sim=ac@bc.T
            permitted=av[:,None]&bv[None,:]
            cost=(1-sim).masked_fill(~permitted,10.)
            # One device transfer, not a synchronization for every matched bin.
            cpu_cost=cost.cpu().numpy();cpu_sim=1-cpu_cost
            path=monotone_pairs(cpu_cost,cfg.get('alignment_drop_cost',.18))
            refs=[];targets=[];coarse_count=0
            for ai,bj in path:
                quality=float(cpu_sim[ai,bj])
                # Repeated content far from the chosen match is ambiguous.
                alternatives=cpu_sim[ai].copy();alternatives[max(0,bj-2):bj+3]=-2.
                margin=quality-float(alternatives.max()) if len(alternatives)>5 else 1.
                if quality<threshold or margin<cfg.get('alignment_unique_margin',.02):continue
                start,end=bj*stride,min((bj+1)*stride,len(b))
                targets.extend(range(start,end))
                refs.extend(min(len(a)-1,ai*stride+j-start) for j in range(start,end))
                coarse_count+=1
            if not refs:return _empty(a.device)
            expected=torch.tensor(refs,device=a.device);jj=torch.tensor(targets,device=a.device)
            radius=max(cfg.get('alignment_local_radius',2),stride//2)
            candidates=expected[:,None]+torch.arange(-radius,radius+1,device=a.device)
            valid=(candidates>=0)&(candidates<len(a));bounded=candidates.clamp(0,len(a)-1)
            valid&=am[bounded]&bm[jj,None]
            similarity=(a[bounded]*b[jj,None]).sum(-1).masked_fill(~valid,-2.)
            confidence,choice=similarity.max(1);ii=bounded.gather(1,choice[:,None])[:,0]
            keep=confidence>=threshold
            previous=torch.cummax(ii.masked_fill(~keep,-1),0).values
            keep&=ii>=torch.cat((previous.new_full((1,),-1),previous[:-1]))
            ii,jj,confidence=ii[keep],jj[keep],confidence[keep]
        return dict(reference=ii,processed=jj,confidence=confidence.detach().clamp(0,1),
                    coverage=len(jj)/max(1,int(bm.sum())),coarse_matches=coarse_count)
