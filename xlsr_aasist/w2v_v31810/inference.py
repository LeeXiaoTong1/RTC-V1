"""Full recordings, common short-input policy, stable output ordering."""
import numpy as np
from .common import observation
from .data import Waves, loader, close, microbatches


def infer(model,rows,cfg,tickets=None,split='train',progress=None):
    cfg=dict(cfg,workers=cfg.get('eval_workers',cfg['workers']))
    dataset=Waves(rows,cfg,split)
    options={'batch_size':cfg['eval_batch']}
    if tickets is not None:
        options={'batch_sampler':[tickets[i:i+cfg['eval_batch']] for i in range(0,len(tickets),cfg['eval_batch'])]}
    batches=loader(dataset,cfg,**options);all_rows=[];all_logits=[]
    try:
        with observation(model):
            for units in batches:
                expected=[r for unit in units for r in unit]
                positions={id(row):i for i,row in enumerate(expected)}
                predictions=[None]*len(expected)
                for micro in microbatches(units,cfg['microbatch'],cfg['frame_budget']):
                    examples=[r for unit in micro for r in unit]
                    z=model([r['wave'] for r in examples]).float().cpu().numpy()
                    for row,value in zip(examples,z):
                        predictions[positions[id(row)]]=value
                if any(value is None for value in predictions):
                    raise ValueError('Inference lost a row during length packing')
                all_logits.extend(predictions)
                all_rows.extend({k:v for k,v in row.items() if k not in ('wave','augmentation','pcl_refs','pcl_pairs')} for row in expected)
                if progress:progress(len(all_logits),len(tickets) if tickets is not None else len(rows))
    finally:
        close(batches)
    if not all_logits:
        raise ValueError('Empty inference dataset')
    z=np.stack(all_logits)
    if tickets is None and ([r['row_index'] for r in all_rows]!=list(range(len(rows))) or len(z)!=len(rows)):
        raise ValueError('Inference row coverage/order differs')
    return z,all_rows
