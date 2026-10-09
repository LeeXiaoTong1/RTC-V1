"""Download ONE official SSL 1B checkpoint with resumable curl and pinned SHA256."""
import argparse
from pathlib import Path
import shutil
import subprocess
from .common import atomic_json,read_json,digest

SHA256='a3194cc50f7ad02a19c92bf75b2970ce94f026b60ab712bd4b720f1219896587'
URL='https://dl.fbaipublicfiles.com/mms/omniASR-W2V-1B.pt'


def prepare(directory,checkpoint=None):
    directory=Path(directory).expanduser().resolve();directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'assets.json'
    if manifest.exists():
        info=read_json(manifest)
        if info.get('schema')!='rtc_omni_w2v1b_assets_v1' or info['sha256']!=SHA256 or digest(info['checkpoint'])!=SHA256:
            raise ValueError('Recorded 1B checkpoint changed')
        print('OMNI1B_ASSETS='+str(manifest));return info
    path=Path(checkpoint).expanduser().resolve() if checkpoint else directory/'omniASR-W2V-1B.pt'
    if not path.exists():
        if checkpoint:raise FileNotFoundError(path)
        partial=path.with_suffix('.pt.part')
        required=max(0,4*1024**3-(partial.stat().st_size if partial.exists() else 0))+12*1024**3
        if shutil.disk_usage(directory).free<required:raise OSError('Need remaining 1B download space plus 12 GiB for reserve/checkpoints')
        subprocess.run(['curl','--fail','--location','--retry','4','--connect-timeout','30',
            '--continue-at','-','--output',str(partial),URL],check=True)
        if digest(partial)!=SHA256:raise ValueError('Official Omni W2V 1B hash differs; partial preserved')
        partial.replace(path)
    if digest(path)!=SHA256:raise ValueError('Expected official W2V 1B, not CTC/LLM/7B weights')
    info=dict(schema='rtc_omni_w2v1b_assets_v1',checkpoint=str(path),sha256=SHA256,bytes=path.stat().st_size,
              repo='facebook/omniASR-W2V-1B',source=URL,arch='1b',encoder_dim=1280,encoder_layers=48)
    atomic_json(manifest,info);print('OMNI1B_ASSETS='+str(manifest));return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',default='pretrained/omniASR-W2V-1B');p.add_argument('--checkpoint')
    args=p.parse_args();prepare(args.directory,args.checkpoint)


if __name__=='__main__':main()
