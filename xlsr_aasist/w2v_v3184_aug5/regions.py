"""Overlap-corrected attention statistics pooling of AASIST window evidence."""
from collections import defaultdict
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
            ranges=windows(len(seq),self.window,self.hop)
            plans.append(ranges)
            for j,(a,b) in enumerate(ranges):groups[b-a].append((i,j,seq[a:b]))
        for _,records in sorted(groups.items()):
            for start in range(0,len(records),self.window_batch):
                chunk=records[start:start+self.window_batch]
                features=self.aasist.evidence(torch.stack([r[2] for r in chunk]).float())
                for r,feature in zip(chunk,features):collected[r[0],r[1]]=feature
        aggregates=[];weights=[]
        for i,(seq,ranges) in enumerate(zip(sequences,plans)):
            evidence=torch.stack([collected[i,j] for j in range(len(ranges))])
            weight=self.weights(evidence,coverage_mass(len(seq),ranges,evidence.device))
            aggregates.append(torch.cat(moments(evidence,weight)));weights.append(weight)
        logits=self.aasist.classify(torch.stack(aggregates))
        return (logits,weights) if return_weights else logits
