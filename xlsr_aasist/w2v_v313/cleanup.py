"""Compact only a completed V3.13 resume file, preserving BOTH last and best."""
import argparse
from pathlib import Path
import torch

from w2v_v39.common import ROOT, verify_files, read_json
from .state import atomic_save, load_selected, load_last


def cleanup(run, apply=False):
    requested=Path(run).expanduser()
    if requested.is_symlink():raise ValueError('Refusing symlink run')
    run=requested.resolve()
    best,_=load_selected(run); last,_=load_last(run)
    cfg=best['config']
    protected={cfg['base_checkpoint']:cfg['base_checkpoint_sha256'],
               cfg['starting_checkpoint']:cfg['starting_checkpoint_sha256']}
    verify_files(protected)
    path=run/'last.pt'
    protected_paths={Path(p).expanduser().resolve() for p in protected}
    if path.is_symlink() or path.resolve().parent!=run or path.resolve() in protected_paths:
        raise ValueError('Cannot compact an external or starting checkpoint')
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if state.get('inference_only'):
        print('V313_CLEANUP_ALREADY_COMPACT=True',flush=True);return 0
    slim={key:value for key,value in state.items() if key not in ('optimizer','adversary','rng','best_model')}
    slim['inference_only']=True
    from .state import tensor_bytes
    estimate=max(0,path.stat().st_size-int(1.1*tensor_bytes(slim)))
    print(f'V313_CLEANUP apply={apply}; estimated_reclaim_GiB={estimate/1024**3:.2f}; both best and last predictions retained',flush=True)
    if apply:
        # Copy tensors out of the file mapping before atomic replacement (Windows safe).
        slim['model']={key:value.clone() for key,value in slim['model'].items()}
        del state,last
        atomic_save(path,slim,cfg['disk_margin_bytes'])
        verify_files(protected)
        load_selected(run);load_last(run)
    return estimate


def main():
    from w2v_aasist.full_workflow import ensure_idle
    from w2v_aasist.launch import run_lock
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--apply',action='store_true')
    args=p.parse_args();ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle();cleanup(args.run,args.apply)


if __name__=='__main__':main()
