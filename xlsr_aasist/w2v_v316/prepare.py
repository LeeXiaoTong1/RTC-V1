"""Fetch exactly one official SSL weight file; never clone all OmniASR variants."""
import argparse
from pathlib import Path
import shutil
import subprocess

from w2v_v39.common import atomic_json,digest,read_json


OFFICIAL_SHA256='f2506df31ba78b1d603fbc90f7626b8a63ec52eb1a94303cada106259cc42118'
OFFICIAL_URL='https://dl.fbaipublicfiles.com/mms/omniASR-W2V-7B.pt'


def prepare(directory,checkpoint=None,provider='meta'):
    directory=Path(directory).expanduser().resolve(); directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'assets.json'
    if manifest.exists():
        value=read_json(manifest)
        if value.get('schema')!='rtc_omni_w2v7b_assets_v1' or digest(value['checkpoint'])!=value['sha256']:
            raise ValueError('Recorded Omni weights changed')
        print('OMNI_ASSETS='+str(manifest)); return value
    repo='facebook/omniASR-W2V-7B'
    if checkpoint:
        path=Path(checkpoint).expanduser().resolve()
        provenance=dict(source='user-supplied official W2V 7B checkpoint; architecture verified at preflight',repo=repo)
    elif provider=='meta':
        # Official package 0.2.0 asset URL. Avoids a second HF cache and works
        # without Hugging Face authentication. curl resumes the single partial.
        path=directory/'omniASR-W2V-7B.pt'
        partial=path.with_suffix('.pt.part')
        if not path.exists():
            remaining=max(0,26*1024**3-(partial.stat().st_size if partial.exists() else 0))
            if shutil.disk_usage(directory).free<remaining+14*1024**3:
                raise OSError('Need remaining 7B download space plus 14 GiB for partial saves/reserve')
            subprocess.run(['curl','--fail','--location','--retry','4','--connect-timeout','30',
                '--continue-at','-','--output',str(partial),OFFICIAL_URL],check=True)
            if digest(partial)!=OFFICIAL_SHA256: raise ValueError('Official W2V 7B SHA256 differs; partial preserved for inspection')
            partial.replace(path)
        provenance=dict(repo=repo,source=OFFICIAL_URL,expected_sha256=OFFICIAL_SHA256)
    else:
        if (directory/'omniASR-W2V-7B.pt.part').exists():
            raise RuntimeError('A Meta partial download exists. Resume with --provider meta before switching provider; refusing duplicate 25 GiB downloads.')
        from huggingface_hub import HfApi,hf_hub_download
        info=HfApi().model_info(repo,files_metadata=True)
        candidates=[s for s in info.siblings if s.rfilename.endswith(('.pt','.pth'))]
        if len(candidates)!=1:
            raise ValueError('Expected one official W2V checkpoint; specify --checkpoint for an existing file')
        asset=candidates[0]; size=asset.size
        if not size or size<10*1024**3: raise ValueError('Unexpected 7B checkpoint size')
        existing=directory/asset.rfilename
        required=(0 if existing.exists() else size)+14*1024**3
        if shutil.disk_usage(directory).free<required:
            raise OSError(f'Need {required/1024**3:.1f} GiB free (one weight file + checkpoint/reserve); clean old derived files first')
        path=Path(hf_hub_download(repo,asset.rfilename,revision=info.sha,local_dir=directory))
        provenance=dict(repo=repo,revision=info.sha,filename=asset.rfilename,size=size)
        lfs=asset.lfs
        expected=getattr(lfs,'sha256',None) if lfs is not None else None
        if expected and digest(path)!=expected: raise ValueError('Official LFS SHA256 mismatch')
    actual=digest(path)
    if actual!=OFFICIAL_SHA256: raise ValueError('Expected the official SSL W2V 7B file, not CTC/LLM or converted weights')
    value=dict(schema='rtc_omni_w2v7b_assets_v1',checkpoint=str(path.resolve()),sha256=actual,
        bytes=path.stat().st_size,precision_on_disk='official weights, no converted duplicate',**provenance)
    atomic_json(manifest,value); print('OMNI_ASSETS='+str(manifest),flush=True)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',default='pretrained/omniASR-W2V-7B'); p.add_argument('--checkpoint')
    p.add_argument('--provider',choices=('meta','hf'),default='meta')
    a=p.parse_args(); prepare(a.directory,a.checkpoint,a.provider)
