"""One-way, task-weighted native-feature consistency with no learned escape path."""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from .alignment import align


@contextmanager
def capture_features(model,cfg):
    boundary=len(model.backbone.encoder.layers)-cfg['trainable_layers']-1
    if boundary<0:raise ValueError('A genuinely frozen content prefix is required')
    layer=model.backbone.encoder.layers[boundary]
    if any(p.requires_grad for p in layer.parameters()):raise ValueError('Alignment prefix must stay frozen')
    result=dict(content=[],detection=[])
    a=layer.register_forward_hook(lambda _m,_a,value:result['content'].append((value[0] if isinstance(value,tuple) else value).detach()))
    b=model.head.blocks[0].register_forward_pre_hook(lambda _m,args:result['detection'].append(args[0]))
    try:yield result
    finally:a.remove();b.remove()


def importance(features,gradient,valid,uniform_mass=.25,cap=4.):
    """Detached |feature * d(correct-class margin)/d(feature)|, not causal evidence."""
    if not 0<uniform_mass<=1 or cap<1:raise ValueError('Invalid bounded importance budget')
    with torch.no_grad():
        valid=torch.as_tensor(valid,device=features.device,dtype=torch.bool)
        raw=(features.detach().float()*gradient.detach().float()).abs()
        raw=raw.masked_fill(~valid[:,None],0.)
        mean=raw.sum()/max(1,int(valid.sum())*raw.shape[1])
        scaled=raw/mean.clamp_min(1e-12)
        weight=uniform_mass+(1-uniform_mass)*scaled.clamp(max=cap)
        return weight.masked_fill(~valid[:,None],0.)


def local_cka(reference,processed,channel_weight):
    """Channels are observations; compare their centered Gram relationships.

    CKA is per corresponding local interval, not pooled over separate entire
    recordings. No learnable projection. The detached Offline side is the target.
    """
    a,b=reference.T.float(),processed.T.float()
    w=channel_weight.detach().float().sqrt()[:,None]
    a=a*w;b=b*w
    a=a-a.mean(0,keepdim=True);b=b-b.mean(0,keepdim=True)
    ga,gb=a@a.T,b@b.T
    denom=torch.linalg.vector_norm(ga)*torch.linalg.vector_norm(gb)
    return 1-(ga*gb).sum().div(denom.clamp_min(1e-8)).clamp(-1,1)


class TFCL(nn.Module):
    """Parameter-free; original mode is an explicit controlled comparator only."""
    def __init__(self,channels=128,heads=8,bins=201,mode='weighted'):
        super().__init__();self.mode=mode
        if mode not in ('ce','original','uniform','weighted'):raise ValueError('Unknown TFCL mode')
        if mode=='original':
            from w2v_v3151.objectives import TFCL as Original
            self.original=Original(channels,heads,bins)

    def forward(self,ref,processed,content_ref,content_processed,ref_valid,processed_valid,weights,cfg,known_mapping=False,hop=1.,reference_hop=1.):
        if self.mode=='original':
            t,s=self.original(ref,processed,ref_valid,processed_valid)
            return t,s,dict(coverage=1.,frames=min(len(ref),len(processed)),windows=1,mode='original_unconstrained')
        mapping=align(content_ref,content_processed,ref_valid,processed_valid,cfg,known_mapping,hop,reference_hop)
        zero=processed[:0].float().sum();ii,jj=mapping['reference'],mapping['processed']
        meta=dict(coverage=mapping['coverage'],frames=len(jj),windows=0)
        if len(jj)<cfg.get('alignment_min_frames',4) or mapping['coverage']<cfg.get('alignment_min_coverage',.1):
            return zero,zero,dict(meta,rejected=True)
        with torch.autocast(device_type=ref.device.type,enabled=False):
            a,b=ref.detach().float()[ii],processed.float()[jj]
            weight=weights.detach().float()[ii] if self.mode=='weighted' else torch.ones_like(a)
            # Stop-gap + native positions keep padding/known missing spans out.
            reliability=mapping['confidence']
            channel=weight.sqrt()
            distance=1-F.cosine_similarity(a*channel,b*channel,dim=-1,eps=1e-6)
            time_weight=weight.mean(1)*reliability
            temporal=((distance-cfg.get('time_tolerance',.02)).clamp_min(0)*time_weight).sum()/time_weight.sum().clamp_min(1e-8)
            # Cap to the retained fraction: sparse reliable regions never acquire
            # the whole recording's budget merely because other regions vanished.
            temporal=temporal*mapping['coverage']
            length=cfg.get('structure_window_frames',50);minimum=cfg.get('structure_min_frames',8)
            breaks=torch.where((ii[1:]-ii[:-1]>2)|(jj[1:]-jj[:-1]>1)|(ii[1:]<ii[:-1]))[0]+1
            boundaries=[0]+breaks.cpu().tolist()+[len(ii)]
            terms=[];supports=[]
            for start,end in zip(boundaries,boundaries[1:]):
                for lo in range(start,end,length):
                    hi=min(end,lo+length)
                    if hi-lo<minimum:continue
                    value=local_cka(a[lo:hi],b[lo:hi],weight[lo:hi].mean(0))
                    terms.append((value-cfg.get('structure_tolerance',.02)).clamp_min(0))
                    supports.append((hi-lo)*reliability[lo:hi].mean())
            structure=zero
            if terms:
                support=torch.stack(supports)
                structure=(torch.stack(terms)*support).sum()/support.sum().clamp_min(1e-8)
                structure=structure*(sum(int(min(end,lo+length)-lo) for start,end in zip(boundaries,boundaries[1:]) for lo in range(start,end,length) if min(end,lo+length)-lo>=minimum)/max(1,int(torch.as_tensor(processed_valid).sum())))
            return temporal,structure,dict(meta,windows=len(terms),rejected=False)
