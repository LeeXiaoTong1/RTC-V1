"""Fake=0, Real=1. Pair ordering supplements, never replaces, calibrated CE."""
import math
import torch
from torch.nn import functional as F


def paired_margin(real_logits, fake_logits, margin=.5):
    if margin <= 0 or not math.isfinite(margin):
        raise ValueError('A finite positive logit margin is required')
    dr = real_logits[..., 0].float()-real_logits[..., 1].float()
    df = fake_logits[..., 0].float()-fake_logits[..., 1].float()
    # Unlike two independent margin-CE terms, changing the partner changes the
    # gradient. CE remains responsible for the absolute zero decision boundary.
    return F.softplus(2*margin+dr-df)


def frame_count(samples):
    return max(0, (int(samples)-400)//320+1)


def frame_span(start, end, samples):
    if not all(math.isfinite(float(v)) for v in (start,end)):
        raise ValueError('Nonfinite phoneme boundary')
    if not 0 <= start < end <= samples:
        return 0, 0
    n=frame_count(samples)
    return max(0,min(n,math.ceil((start-200)/320))), max(0,min(n,math.ceil((end-200)/320)))


def phoneme_mse(left, right, matches, left_samples, right_samples):
    if left.ndim!=2 or right.ndim!=2 or left.shape[1]!=right.shape[1]:
        raise ValueError('PCL requires two same-width sequences')
    left=left[:frame_count(left_samples)].float();right=right[:frame_count(right_samples)].float()
    spans=[]
    for a,b,c,d in matches:
        x,y=frame_span(a,b,left_samples);z,w=frame_span(c,d,right_samples)
        if y>x and w>z and y<=len(left) and w<=len(right):spans.append((x,y,z,w))
    if not spans:return (left.sum()+right.sum())*0.,0
    ix=torch.tensor(spans,device=left.device,dtype=torch.long)
    pooled=[]
    for features,a,b in ((left,ix[:,0],ix[:,1]),(right,ix[:,2],ix[:,3])):
        integral=F.pad(features.cumsum(0),(0,0,1,0))
        pooled.append((integral[b]-integral[a])/(b-a)[:,None])
    return (pooled[0]-pooled[1]).square().mean(),len(spans)


def ramp(epoch, step, steps):
    """First epoch CE only; next two epochs smoothly introduce both auxiliaries."""
    return min(1.,max(0.,((epoch-1)*steps+step+1)/(2*steps))) if epoch>=1 else 0.
