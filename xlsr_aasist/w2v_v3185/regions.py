"""Overlap-corrected attention statistics pooling of AASIST window evidence."""
from collections import defaultdict,OrderedDict
import torch
from torch import nn
from w2v_v318.regions import RegionHead,windows,coverage_mass


def moments(evidence,weight):
    # Centered FP32 variance avoids cancellation. Smooth sqrt has a finite
    # derivative at zero and returns exactly zero for a single/constant window.
    evidence=evidence.float();weight=weight.float()
    mean=(weight[:,None]*evidence).sum(0)
    variance=(weight[:,None]*(evidence-mean).square()).sum(0).clamp_min(0)
    deviation=torch.sqrt(variance+1e-5)-(1e-5)**.5
    return mean,deviation


class StatisticsHead(RegionHead):
    def __init__(self,old):
        super().__init__(old.aasist,old.local,old.window,old.hop,old.window_batch)
        self.contribution=old.contribution
        self.attention_delta=nn.Sequential(nn.Linear(480,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.attention_delta[-1].weight)
        nn.init.zeros_(self.attention_delta[-1].bias)
        previous=self.aasist.out_layer
        if previous.in_features!=160 or previous.out_features!=2:
            raise ValueError('Unexpected AASIST classifier geometry')
        classifier=nn.Linear(320,2,device=previous.weight.device,dtype=previous.weight.dtype)
        with torch.no_grad():
            classifier.weight[:,:160].copy_(previous.weight)
            classifier.weight[:,160:].zero_();classifier.bias.copy_(previous.bias)
        self.aasist.out_layer=classifier
        # Geometry alone, never waveforms/features or anything with gradients.
        # Per-model LRU, omitted from state_dict; bounded by entries AND bytes.
        self._geometry=OrderedDict();self._geometry_bytes=0

    def _apply(self,fn,recurse=True):
        self._geometry.clear();self._geometry_bytes=0
        return super()._apply(fn,recurse)

    def window_plan(self,length,device):
        key=(length,self.window,self.hop,torch.device(device))
        if key in self._geometry:
            self._geometry.move_to_end(key)
            return self._geometry[key]
        ranges=tuple(windows(length,self.window,self.hop))
        # Coverage depends only on integer frame bounds. Computing/validating
        # it on CPU avoids many tiny CUDA kernels and a GPU->CPU bool sync
        # for EVERY waveform. Keep the FP32 mass on the target device for reuse.
        # A cache warmed by inference_mode must still be usable in training.
        with torch.inference_mode(False):
            mass=coverage_mass(length,ranges).to(device=device,non_blocking=True)
        size=mass.numel()*mass.element_size()
        if size<=256*1024:
            while self._geometry and (len(self._geometry)>=128 or self._geometry_bytes+size>256*1024):
                _,(_,old)=self._geometry.popitem(last=False)
                self._geometry_bytes-=old.numel()*old.element_size()
            self._geometry[key]=(ranges,mass);self._geometry_bytes+=size
        return ranges,mass

    def weights(self,evidence,mass):
        mean,std=moments(evidence,mass)
        context=torch.cat((evidence,mean.expand_as(evidence),std.expand_as(evidence)),dim=-1)
        prior=.5+torch.sigmoid(self.contribution(evidence).squeeze(-1))
        delta=self.attention_delta(context).squeeze(-1)
        # Reproduces the inherited mean weights at initialization; after that
        # the learned delta has no fixed amplitude cap. Coverage is a prior,
        # not a second multiplication after softmax.
        return torch.softmax(mass.log()+prior.log()+delta,dim=0)

    def forward(self,sequences,return_weights=False):
        groups=defaultdict(list);plans=[];collected={}
        for i,seq in enumerate(sequences):
            ranges,mass=self.window_plan(len(seq),seq.device)
            plans.append((ranges,mass))
            for j,(a,b) in enumerate(ranges):groups[b-a].append((i,j,seq[a:b]))
        for _,records in sorted(groups.items()):
            for start in range(0,len(records),self.window_batch):
                chunk=records[start:start+self.window_batch]
                features=self.aasist.evidence(torch.stack([r[2] for r in chunk]).float())
                for r,feature in zip(chunk,features):collected[r[0],r[1]]=feature
        aggregates=[];weights=[]
        for i,(ranges,mass) in enumerate(plans):
            evidence=torch.stack([collected[i,j] for j in range(len(ranges))])
            weight=self.weights(evidence,mass)
            aggregates.append(torch.cat(moments(evidence,weight)));weights.append(weight)
        logits=self.aasist.classify(torch.stack(aggregates))
        return (logits,weights) if return_weights else logits
