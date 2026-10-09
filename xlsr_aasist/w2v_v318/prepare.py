"""Download one official SSL checkpoint (default 3B), resumed and SHA256-pinned."""
import argparse
from pathlib import Path
import shutil
import subprocess
from .common import atomic_json,read_json,digest
from .assets import DEFAULT_ARCH,MODELS,spec,validate_assets


def prepare(directory=None,checkpoint=None,arch=DEFAULT_ARCH):
    item=spec(arch);expected=item['sha256'];url=item['source']
    directory=directory or 'pretrained/'+item['name']
    directory=Path(directory).expanduser().resolve();directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'assets.json'
    if manifest.exists():
        info=read_json(manifest)
        validate_assets(info,arch)
        if checkpoint and Path(checkpoint).expanduser().resolve()!=Path(info['checkpoint']).resolve():
            raise ValueError('Existing assets reference another checkpoint; choose a separate directory')
        if digest(info['checkpoint'])!=expected:raise ValueError('Recorded checkpoint changed')
        print('OMNI'+arch.upper()+'_ASSETS='+str(manifest));return info
    path=Path(checkpoint).expanduser().resolve() if checkpoint else directory/(item['name']+'.pt')
    if not path.exists():
        if checkpoint:raise FileNotFoundError(path)
        partial=path.with_suffix('.pt.part')
        required=max(0,item['download_bytes']-(partial.stat().st_size if partial.exists() else 0))+12*1024**3
        if shutil.disk_usage(directory).free<required:raise OSError(f'Need {required/1024**3:.2f} GiB free: remaining {arch.upper()} download plus 12 GiB reserve/checkpoints')
        subprocess.run(['curl','--fail','--location','--retry','4','--connect-timeout','30',
            '--continue-at','-','--output',str(partial),url],check=True)
        if digest(partial)!=expected:raise ValueError('Official Omni W2V '+arch.upper()+' hash differs; partial preserved')
        partial.replace(path)
    if digest(path)!=expected:raise ValueError('Expected official W2V '+arch.upper()+' weights')
    info={k:item[k] for k in ('schema','repo','source','arch','sha256','encoder_dim','encoder_layers')}
    info.update(checkpoint=str(path),bytes=path.stat().st_size)
    atomic_json(manifest,info);print('OMNI'+arch.upper()+'_ASSETS='+str(manifest));return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--omni-size',choices=tuple(MODELS),default=DEFAULT_ARCH)
    p.add_argument('--directory');p.add_argument('--checkpoint')
    args=p.parse_args();prepare(args.directory,args.checkpoint,args.omni_size)


if __name__=='__main__':main()
