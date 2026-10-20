"""Dual phoneme MSE on the native stacked-fbank grid.

Each source is an atomic classification/reference unit. References never enter
the classifier; neither side is detached. Official and simulated received branches share one detector.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F


def frame_count(samples):
    """Complete original support: two 400-sample Mel frames, 160 apart."""
    return max(0, 1 + (int(samples) - 560) // 320)


def frame_span(start, end, frames):
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError('Invalid original phoneme interval')
    return (max(0, min(frames, math.ceil((start-280.)/320.))),
            max(0, min(frames, math.ceil((end-280.)/320.))))


def valid_matches(matches, left_samples, right_samples):
    result = []
    for match in matches:
        if len(match) != 4:
            raise ValueError('Phoneme match requires Offline and Online sample intervals')
        a, b, c, d = map(float, match)
        if not (0 <= a < b <= left_samples and 0 <= c < d <= right_samples):
            continue
        x, y = frame_span(a, b, frame_count(left_samples))
        z, w = frame_span(c, d, frame_count(right_samples))
        if y > x and w > z:
            result.append([a, b, c, d])
    return result


def phoneme_mse(left, right, matches, left_samples, right_samples):
    """Mean over matched phone instances/channels, retaining both gradients."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError('PCL requires two same-width frame sequences')
    left = left[:frame_count(left_samples)].float()
    right = right[:frame_count(right_samples)].float()
    spans = []
    for a, b, c, d in valid_matches(matches, left_samples, right_samples):
        x, y = frame_span(a, b, len(left)); z, w = frame_span(c, d, len(right))
        if y > x and w > z:
            spans.append((x, y, z, w))
    if not spans:
        return (left.sum()+right.sum())*0., 0
    # Prefix sums pool every matched phone in a few tensor operations; avoid
    # hundreds of tiny GPU reductions/launches for each reference recording.
    indices = torch.tensor(spans, device=left.device, dtype=torch.long)
    means = []
    for features, start, end in ((left, indices[:,0], indices[:,1]),
                                 (right, indices[:,2], indices[:,3])):
        integral = F.pad(features.cumsum(0), (0,0,1,0))
        means.append((integral[end]-integral[start])/(end-start)[:,None])
    return (means[0]-means[1]).square().mean(), len(spans)


def ramp(cfg, epoch, step, steps):
    warm=cfg.get('warm_epochs',1)
    return min(1.,max(0.,((epoch-warm)*steps+step+1)/cfg.get('pcl_ramp_updates',300))) if epoch>=warm else 0.


def expand(row, enabled):
    result=dict(rows=[row], pair=None, matches=[], kind=None)
    p=row.get('pcl') if enabled else None
    if not p or not p['matches']:return result
    if p['kind'] not in ('official','noise'):raise ValueError('Unknown PCL branch')
    refs=[p['reference']]+([] if p['reuse_received'] else [p['received']])
    for ref in refs:
        if ref.get('split')!='train' or any(ref.get(k)!=row.get(k) for k in ('canonical_source','language','label')):
            raise ValueError('PCL requires same-source, same-language, same-label Train references')
    if p['reuse_received'] and p['kind']=='official':
        if not np.array_equal(row['wave'],p['received']['wave']):raise ValueError('Augmented Online cannot stand in for an original')
    received=row if p['reuse_received'] else p['received']
    matches=valid_matches(p['matches'],p['reference']['original_samples'],received['original_samples'])
    if not matches:return result
    result.update(rows=[row]+refs,pair=(1,0 if p['reuse_received'] else 2),matches=matches,kind=p['kind'])
    return result


def pack(items, size, budget):
    from .features import frame_count as frames
    bins=[]
    for item in sorted(items,key=lambda x:-max(frames(len(r['wave'])) for r in x['rows'])):
        count=len(item['rows']);longest=max(frames(len(r['wave'])) for r in item['rows'])
        if count>size or count*longest>budget:
            raise ValueError('Full classification/PCL source exceeds microbatch or frame budget; no cropping')
        choices=[(size-b['count']-count,i) for i,b in enumerate(bins)
                 if b['count']+count<=size and (b['count']+count)*max(b['longest'],longest)<=budget]
        if choices:entry=bins[min(choices)[1]]
        else:entry=dict(items=[],count=0,longest=0);bins.append(entry)
        entry['items'].append(item);entry['count']+=count;entry['longest']=max(entry['longest'],longest)
    for entry in bins:yield entry['items']
