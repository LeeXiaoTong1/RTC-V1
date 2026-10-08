"""One scoring pass provides every metric; no saved per-frame features."""
import numpy as np
import torch
from live_progress import Phase
from .data import WaveRows,make_loader,rows_list,grouped


def infer(model,rows,cfg,title):
    model.eval(); model.set_phase(True,False)
    output=np.empty((len(rows),2),dtype=np.float32)
    loader=make_loader(WaveRows(rows),cfg,batch_size=cfg['eval_batch'],shuffle=False,collate_fn=rows_list)
    phase=Phase(title,len(rows)); count=0
    try:
        with torch.inference_mode():
            for batch in loader:
                for group in grouped([[r] for r in batch],cfg['microbatch'],cfg['frame_budget']):
                    logits,*_=model([r['wave'] for r in group])
                    output[[r['row_index'] for r in group]]=logits.float().cpu().numpy()
                    count+=len(group); phase.update(count)
    finally:
        iterator=getattr(loader,'_iterator',None)
        if iterator is not None: iterator._shutdown_workers()
    if not np.isfinite(output).all(): raise FloatingPointError('Nonfinite evaluation scores')
    return output
