"""CPU-only real-data check of the production sampler, augmentation and workers."""
import argparse
from collections import Counter
from itertools import islice
import json
import os
from pathlib import Path
import time

# Set before NumPy/Torch imports, including in spawned children.
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONFAULTHANDLER']='1'


def check(run,workers=2,start_step=0,steps=8,epoch=1):
    import numpy as np
    import torch
    from w2v_v318.common import read_json
    from w2v_v318.data import Waves,loader,close
    from w2v_v318.sampling import Plan,validate_units
    torch.set_num_threads(1)
    run=Path(run).resolve()
    cfg=read_json(run/'config.json');cfg=dict(cfg,workers=workers)
    rows=read_json(run/'train_rows.json')
    plan=Plan(rows,cfg['stream_sources'],cfg['seed'])
    tickets=list(islice(plan.batches(epoch-1),start_step,start_step+steps))
    if len(tickets)!=steps:
        raise ValueError(f'Requested steps exceed epoch length: {plan.steps}')
    batches=loader(Waves(rows,cfg),cfg,batch_sampler=tickets)
    families=Counter();codecs=Counter();views=0;started=time.monotonic()
    print(f'WORKER_CHECK_START workers={workers} epoch={epoch} '
          f'steps={start_step+1}-{start_step+steps} model_loaded=False',flush=True)
    try:
        for offset,units in enumerate(batches,1):
            validate_units(units,cfg['stream_sources'])
            for unit in units:
                for row in unit:
                    wave=row['wave']
                    if wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():
                        raise ValueError('Invalid waveform: '+row['audio'])
                    views+=1
                    rec=row.get('augmentation',{}).get('recipe')
                    if rec:
                        families[rec['family']]+=1;codecs[rec['codec']]+=1
            print(f'WORKER_CHECK_STEP {offset}/{steps} views={views} '
                  f'elapsed_seconds={time.monotonic()-started:.1f}',flush=True)
    finally:
        close(batches)
    result=dict(passed=True,workers=workers,steps=steps,views=views,
                families=dict(families),codecs=dict(codecs),seconds=round(time.monotonic()-started,2))
    print('WORKER_CHECK_RESULT='+json.dumps(result,ensure_ascii=True),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--start-step',type=int,default=0,help='Zero-based start step')
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--epoch',type=int,default=1,help='One-based epoch')
    args=p.parse_args()
    if args.workers<0 or args.start_step<0 or args.steps<1 or args.epoch<1:
        p.error('Invalid workers/step/epoch values')
    from w2v_aasist.full_workflow import ensure_idle
    ensure_idle()
    check(args.run,args.workers,args.start_step,args.steps,args.epoch)


if __name__=='__main__':main()
