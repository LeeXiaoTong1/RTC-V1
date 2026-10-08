"""Compact only a completed owned V3.15.1 run; preserve every selected detector."""
import argparse
from pathlib import Path
import shutil

import torch

from w2v_v39.common import ROOT, atomic_json, digest, read_json, verify_files
from w2v_v315.cache import FORMAT
from .state import KINDS, identity, load_selected, atomic_save, tensor_bytes


def cleanup(run, apply=False, remove_cache=False):
    requested=Path(run).expanduser()
    if requested.is_symlink():raise ValueError('Refusing linked run')
    run=requested.resolve()
    _,done=load_selected(run)
    cfg=read_json(run/'config.json')
    protected=dict(cfg['source_fingerprints'],**cfg['data_fingerprints'])
    protected.update({cfg['base_checkpoint']:cfg['base_checkpoint_sha256'],cfg['starting_checkpoint']:cfg['starting_checkpoint_sha256']})
    verify_files(protected)
    paths={Path(p).resolve() for p in protected}
    for name in ('last.pt','inference.pt','rolling_pairs'):
        p=run/name
        if p.is_symlink() or p.resolve().parent!=run or p.resolve() in paths:
            raise ValueError('Cleanup target is not an owned output')
    cache=run/'rolling_pairs'; cache_bytes=0
    if remove_cache and cache.exists():
        owner=read_json(cache/'owner.json')
        if (owner.get('identity')!=identity(cfg) or owner.get('format')!=FORMAT
                or owner.get('cap_bytes')!=cfg['rolling_cache_bytes']):
            raise ValueError('Cannot delete another run cache')
        for child in cache.iterdir():
            if child.is_symlink() or not child.is_file() or child.resolve().parent!=cache.resolve():
                raise ValueError('Cache contains external or linked content')
            if child.name in ('owner.json','.cache.lock'):continue
            stem=child.stem
            if child.suffix not in ('.npz','.tmp') or len(stem)!=64 or any(c not in '0123456789abcdef' for c in stem):
                raise ValueError('Cache contains unexpected files')
            cache_bytes+=child.stat().st_size
    state=torch.load(run/done['state_file'],map_location='cpu',weights_only=True)
    slim={k:v for k,v in state.items() if k not in ('optimizer','rng','auxiliary')}
    slim['inference_only']=True
    estimate=max(0,(run/done['state_file']).stat().st_size-int(1.1*tensor_bytes(slim)))+cache_bytes
    print(f'V3151_CLEANUP apply={apply}; estimated_reclaim_GiB={estimate/1024**3:.2f}; all three export choices and source checkpoints retained',flush=True)
    if apply:
        if done['state_file']!='inference.pt':
            atomic_save(run/'inference.pt',slim,cfg['disk_margin_bytes'])
            done.update(state_file='inference.pt',checkpoint_sha256=digest(run/'inference.pt'),inference_only=True)
            atomic_json(run/'completed.json',done)
        for kind in KINDS:
            load_selected(run,kind)
            alias=read_json(run/(kind+'.json'));alias['checkpoint']='inference.pt'
            atomic_json(run/(kind+'.json'),alias)
        (run/'last.pt').unlink(missing_ok=True)
        if remove_cache and cache.exists():
            # Target and every child were validated by RollingCache; no external
            # links, checkpoints or official data are accepted in this directory.
            for child in cache.iterdir():
                if child.is_symlink() or child.resolve().parent!=cache.resolve():
                    raise ValueError('Cache changed during cleanup')
            shutil.rmtree(cache)
        verify_files(protected)
    return estimate


def main():
    from w2v_aasist.full_workflow import ensure_idle
    from w2v_aasist.launch import run_lock
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--apply',action='store_true')
    p.add_argument('--remove-cache',action='store_true');args=p.parse_args()
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle();cleanup(args.run,args.apply,args.remove_cache)


if __name__=='__main__':main()
