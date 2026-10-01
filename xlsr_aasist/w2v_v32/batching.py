"""Prepare unchanged acoustic inputs in workers and retain pinned transfer buffers."""
import torch
from .model import microbatches


class PreparedBatch:
    """Metadata in original order, exact same length-sorted physical batches.

    This intentionally is not a list/Sequence subclass: DataLoader must call
    pin_memory() on the final padded tensors rather than pinning source slices
    which would be copied into pageable memory later.
    """
    def __init__(self, examples, batches, size, frame_budget):
        self.examples,self.batches=examples,batches
        self.size,self.frame_budget=size,frame_budget

    def __len__(self): return len(self.examples)
    def __iter__(self): return iter(self.examples)
    def __getitem__(self,index): return self.examples[index]

    def pin_memory(self):
        self.batches=[(ids,f.pin_memory(),m.pin_memory()) for ids,f,m in self.batches]
        return self


class PreparedCollator:
    def __init__(self, ssl_path, size=4, frame_budget=1600, feature_batch=8):
        self.ssl_path=str(ssl_path)
        self.size,self.frame_budget,self.feature_batch=size,frame_budget,feature_batch
        self.extractor=None

    def __call__(self, rows):
        if self.extractor is None:
            from transformers import AutoFeatureExtractor
            self.extractor=AutoFeatureExtractor.from_pretrained(self.ssl_path,local_files_only=True)
            if self.extractor.sampling_rate!=16000:
                raise ValueError('Expected official 16 kHz feature extractor')
        expanded=[view for views in rows for view in views]
        examples=[None]*len(expanded)
        # Official normalization still happens separately for each waveform.
        # Only Python/API padding and tensor conversion are amortized.
        ordered=sorted(range(len(expanded)),key=lambda i:len(expanded[i]['wave']))
        for offset in range(0,len(ordered),self.feature_batch):
            indices=ordered[offset:offset+self.feature_batch]
            result=self.extractor([expanded[i]['wave'] for i in indices],sampling_rate=16000,
                padding=True,pad_to_multiple_of=2,return_attention_mask=True,return_tensors='pt')
            for j,index in enumerate(indices):
                f=result['input_features'][j:j+1]
                m=result['attention_mask'][j:j+1].long()
                length=int(m.sum())
                if length<2 or f.shape[-1]!=160 or not bool(m[:,:length].bool().all()):
                    raise ValueError('Invalid official filterbank output')
                f,m=f[:,:length].contiguous().float(),m[:,:length].contiguous()
                if not bool(torch.isfinite(f).all()):
                    raise FloatingPointError('Non-finite acoustic features')
                examples[index]={**{k:v for k,v in expanded[index].items() if k!='wave'},'features':f,'mask':m}
        batches=list(microbatches(examples,self.size,self.frame_budget))
        return PreparedBatch(examples,batches,self.size,self.frame_budget)


class DeviceBatches:
    """Bounded one-batch lookahead H2D; never prefetch another encoder pass."""
    def __init__(self,batches,device):
        self.batches,self.device=batches,torch.device(device)
        self.pinned_batches=0
        self.copy_batches=0

    def __iter__(self):
        if self.device.type!='cuda':
            for ids,f,m in self.batches: yield ids,f.to(self.device),m.to(self.device)
            return
        copy_stream=torch.cuda.Stream(device=self.device)
        source=iter(self.batches)

        def prepare():
            try: ids,f,m=next(source)
            except StopIteration: return None
            self.pinned_batches+=int(f.is_pinned() and m.is_pinned())
            with torch.cuda.stream(copy_stream):
                f=f.to(self.device,non_blocking=True)
                m=m.to(self.device,non_blocking=True)
                ready=torch.cuda.Event();ready.record(copy_stream)
            self.copy_batches+=1
            return ids,f,m,ready

        pending=prepare()
        while pending is not None:
            ids,f,m,ready=pending
            current=torch.cuda.current_stream(self.device)
            current.wait_event(ready)
            f.record_stream(current);m.record_stream(current)
            pending=prepare()
            yield ids,f,m
