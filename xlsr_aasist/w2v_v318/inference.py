"""All fixed Dev rows, complete utterances, no gradients or silent row limits."""
import numpy as np
from .common import observation
from .data import Waves,loader,close,microbatches


def infer(model,rows,cfg,tickets=None,split='train'):
    dataset=Waves(rows,cfg,split)
    options={'batch_size':cfg['eval_batch']}
    if tickets is not None:options={'batch_sampler':[tickets[i:i+cfg['eval_batch']] for i in range(0,len(tickets),cfg['eval_batch'])]}
    batches=loader(dataset,cfg,**options);all_rows=[];all_logits=[]
    try:
        with observation(model):
            for units in batches:
                for micro in microbatches(units,cfg['microbatch'],cfg['frame_budget']):
                    examples=[r for u in micro for r in u]
                    z=model([r['wave'] for r in examples]).float().cpu().numpy()
                    all_logits.append(z)
                    all_rows.extend({k:v for k,v in r.items() if k not in ('wave','augmentation')} for r in examples)
    finally:close(batches)
    z=np.concatenate(all_logits)
    if tickets is None and ([r['row_index'] for r in all_rows]!=list(range(len(rows))) or len(z)!=len(rows)):
        raise ValueError('Inference row coverage/order differs')
    return z,all_rows
