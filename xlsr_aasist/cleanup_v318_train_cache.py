"""Clear two reviewed generated Train payload stores; retain all metadata/models.

Original audio, noise assets, all Dev stores and checkpoint files are untouched.
Old V3.3 training will require regenerating the deleted Train WAVs.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from audit_w2v_storage import current_inputs, load
from w2v_v316.cleanup import idle_lock, safe, sha

V33 = Path('data/rtc_noisy_v33/train')
ROLLING = Path('exp/w2v_v315_20261008_075905_5663/rolling_pairs')


def identity(path):
    s=path.stat()
    return [s.st_size,s.st_mtime_ns,s.st_dev,s.st_ino]


def inventory(root, protected):
    root=Path(root).resolve()
    candidates=[]; markers={}
    for relative in (V33,ROLLING):
        folder=safe(root/relative,root)
        if not folder.exists(): continue
        if any(p==folder or folder in p.parents or p in folder.parents for p in protected):
            raise ValueError('Current V3.18 input intersects cache: '+str(folder))
        if relative==V33:
            marker=safe(folder/'config.json',root); cfg=load(marker)
            if cfg.get('format')!='rtc_v33_full_condition_cache_v1' or cfg.get('role')!='train':
                raise ValueError('Not the reviewed generated V3.3 Train cache')
            pattern=r'[0-9a-f]{24}_noisy_[ab]\.wav'
            payload=folder/'audio'
        else:
            marker=safe(folder/'owner.json',root); cfg=load(marker)
            run_cfg=folder.parent/'config.json'; original=load(run_cfg)
            wanted=hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()
            if (cfg.get('format')!='rtc_v315_bounded_pairs_v1' or cfg.get('identity')!=wanted
                    or cfg.get('cap_bytes')!=original.get('rolling_cache_bytes')):
                raise ValueError('Rolling cache ownership mismatch')
            markers[str(run_cfg)]=sha(run_cfg)
            pattern=r'[0-9a-f]{64}\.npz'
            payload=folder
        markers[str(marker)]=sha(marker)
        if not payload.exists(): continue
        safe(payload,root)
        print('Checking generated payloads only: '+str(payload),flush=True)
        for current,dirs,names in os.walk(payload,followlinks=False):
            for name in dirs: safe(Path(current)/name,root)
            for name in names:
                path=safe(Path(current)/name,root)
                if not re.fullmatch(pattern,name): continue
                if not path.is_file(): raise ValueError('Payload is not a file: '+str(path))
                candidates.append({'path':str(path),'identity':identity(path)})
    return {'root':str(root),'candidates':candidates,'marker_hashes':markers,
            'total_bytes':sum(p['identity'][0] for p in candidates)}


def apply(plan):
    root=Path(plan['root']).resolve()
    for path,h in plan['marker_hashes'].items():
        if sha(path)!=h: raise ValueError('Ownership changed; cancelled')
    for item in plan['candidates']:
        path=safe(item['path'],root)
        a=root/V33/'audio'; b=root/ROLLING
        allowed=(a in path.parents and re.fullmatch(r'[0-9a-f]{24}_noisy_[ab]\.wav',path.name)) or (
            path.parent==b and re.fullmatch(r'[0-9a-f]{64}\.npz',path.name))
        if not allowed or identity(path)!=item['identity']:
            raise ValueError('Payload changed or escaped the two-store allowlist')
    for item in plan['candidates']: Path(item['path']).unlink()
    return len(plan['candidates'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path(__file__).resolve().parent))
    p.add_argument('--data-run',default='exp/w2v_v316_tfcl_20261009_020301_4d95')
    p.add_argument('--apply',action='store_true');args=p.parse_args()
    root=Path(args.root).resolve()
    def work():
        protected,errors=current_inputs(root,args.data_run,lambda s:print(s,flush=True))
        if errors: raise ValueError('Input check incomplete; no deletion: '+repr(errors))
        result=inventory(root,protected)
        report=root/'exp'/('cleanup_train_payloads_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
        with report.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
        print(f'PLAN={report}\nPAYLOAD_GiB={result["total_bytes"]/1024**3:.3f}',flush=True)
        if args.apply:
            count=apply(result)
            result['files_deleted']=count
            report.write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(f'FILES_DELETED={count}\nFREE_GiB={shutil.disk_usage(root).free/1024**3:.2f}',flush=True)
    if args.apply:
        with idle_lock(root):work()
    else:work()


if __name__=='__main__':main()
