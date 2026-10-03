"""Recover scalar Dev results and retire only replay-unneeded V3.5 noise caches."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil

import torch
from w2v_aasist.launch import ROOT, run_lock
from w2v_aasist.data import read_protocol
from w2v_aasist.runtime import Metrics, atomic_json, sha256
from w2v_v3.train import read_state
from w2v_v35 import SCHEMA
from w2v_v35.cache import _root, FORMAT, retire_generations
from w2v_v35.maintenance import active_jobs
from w2v_v35.storage import GIB, state_peak_bytes, check_space
from w2v_v35.train import _check_resume, source_fingerprints, verify_protected


def expected_scores(cfg):
    original=read_protocol(cfg['dev_protocol'],cfg['dev_data_path'])
    manifest=json.loads((_root(cfg,'dev')/'epoch_000'/'manifest.json').read_text(encoding='utf-8'))
    groups={'online':[r for r in original if r['domain']=='online'],
            'seen':manifest['seen'],'heldout':manifest['heldout']}
    result={}
    for condition,rows in groups.items():
        for row in rows:
            key=(condition,row['id'])
            if key in result:raise ValueError('Duplicate expected Dev score')
            result[key]=row
    return result


def scores_to_metrics(path, expected):
    """Exactly the saved-logit metric definition, with completeness checks."""
    meters=defaultdict(Metrics);seen=set()
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);condition=row['condition'];key=(condition,row['source'])
            if key in seen or key not in expected:raise ValueError('Duplicate or unexpected Dev score')
            source=expected[key]
            if any(row[k]!=source[k] for k in ('label','language')) or row.get('band',-1)!=source.get('band',-1):
                raise ValueError('Saved Dev identity/label/band differs')
            z=torch.tensor([row['logits']],dtype=torch.float32)
            if z.shape!=(1,2) or not bool(torch.isfinite(z).all()):raise ValueError('Invalid logits')
            probability=float(z.softmax(1)[0,0])
            if not math.isfinite(row['p_fake']) or abs(probability-row['p_fake'])>1e-6:
                raise ValueError('Saved probability differs from logits')
            label=[row['label']]
            meters[condition].update(z,label)
            meters[condition+'/'+row['language']].update(z,label)
            if condition!='online':meters[f'{condition}/band{row["band"]}'].update(z,label)
            seen.add(key)
    if seen!=set(expected):raise ValueError(f'Incomplete saved Dev scores: {len(seen)}/{len(expected)}')
    groups={name:meter.result() for name,meter in meters.items()}
    for condition in ('online','seen','heldout'):
        if condition not in groups or min(groups[condition]['class_counts'])==0:
            raise ValueError('Dev condition lacks a class')
    clean,seen_f1,heldout=(groups[k]['macro_f1'] for k in ('online','seen','heldout'))
    noisy=.5*(seen_f1+heldout)
    return dict(groups=groups,clean_f1=clean,seen_f1=seen_f1,heldout_f1=heldout,
                noisy_f1=noisy,weighted_f1=.3*clean+.7*noisy)


def list_retired(cfg, next_epoch):
    root=_root(cfg,'train')
    owner=json.loads((root/'owner.json').read_text(encoding='utf-8'))
    if owner!={'format':FORMAT,'role':'train','root':str(root)}:raise ValueError('Unowned cache')
    paths=[]
    for folder in sorted(root.glob('epoch_*')):
        if folder.is_symlink() or folder.resolve().parent!=root:raise ValueError('Unsafe generation')
        marker=json.loads((folder/'generation.json').read_text(encoding='utf-8'))
        epoch=marker.get('epoch')
        if (type(epoch) is not int or folder.name!=f'epoch_{epoch:03d}' or marker.get('format')!=FORMAT
                or marker.get('role')!='train' or marker.get('root')!=str(root)):
            raise ValueError('Unrecognized generation')
        if epoch>next_epoch:raise ValueError('Future generation exists; inspect experiment ownership first')
        if epoch<next_epoch:paths.append(folder)
    return paths


def rescue_pending_weights(run,state):
    rescued=[]
    for filename in ('best_model.pt','best_candidate.pt'):
        path=run/filename
        if not path.is_file():continue
        candidate=read_state(path)
        if candidate.get('schema')!=SCHEMA or candidate.get('kind')!='weights' or candidate.get('config')!=state['config']:
            raise ValueError('Winner checkpoint identity differs')
        if candidate.get('epoch',0)<=state['epoch']:continue
        epoch=candidate['epoch']
        if epoch!=state['epoch']+1 or candidate.get('tag')!=f'epoch_{epoch}':
            raise ValueError('Unexpected uncommitted winner')
        output=run/f'best_recovered_epoch_{epoch}_{filename}'
        if output.exists():
            if not os.path.samefile(path,output):raise ValueError('Recovery destination already differs')
        else:
            # Same-volume hard link: retain trained EMA without copying GBs.
            # Never fall back to a space-consuming copy on a full disk.
            os.link(path,output)
        rescued.append(str(output))
    return rescued


def recover(run,apply=False):
    run=Path(run).expanduser().resolve()
    run.relative_to((ROOT/'exp').resolve())
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    state=read_state(run/'last.pt')
    if cfg.get('version')!='3.5' or state.get('schema')!=SCHEMA or state.get('kind')!='training':
        raise ValueError('Expected a V3.5 training checkpoint')
    if state['config']!=cfg:raise ValueError('Run config differs from last.pt')
    for path,digest in state['data_fingerprints'].items():
        if sha256(path)!=digest:raise ValueError('Saved data metadata changed: '+path)
    _check_resume(state,cfg,state['data_fingerprints'],source_fingerprints())
    verify_protected(cfg)
    if state.get('complete'):raise ValueError('Run is already complete; no replay cleanup needed')
    next_epoch=state['epoch']+1
    print(f'LAST_COMMITTED_EPOCH={state["epoch"]}; resume replays epoch {next_epoch}.',flush=True)
    score_file=run/f'epoch_{next_epoch}_scores.jsonl'
    report=dict(last_committed_epoch=state['epoch'],resume_epoch=next_epoch,
                last_checkpoint_preserved=True,checkpoint_committed=False)
    if score_file.is_file():
        dev=scores_to_metrics(score_file,expected_scores(cfg))
        report['recovered_dev']=dev
        print(f'\nRecovered epoch {next_epoch} Dev (validation finished; full training state was not committed):')
        print(' '.join(f'{key}={100*dev[key]:.3f}' for key in ('clean_f1','seen_f1','heldout_f1','noisy_f1','weighted_f1')))
        for group in ('online/en','seen/en','heldout/en'):
            if group in dev['groups']:
                recall=dev['groups'][group]['recall']
                print(f'{group} fake={100*recall[0]:.3f}% real={100*recall[1]:.3f}%')
        if apply:atomic_json(run/f'epoch_{next_epoch}_recovered_metrics.json',report)
    retired=list_retired(cfg,next_epoch)
    for path in retired:print('OBSOLETE_DERIVED_CACHE='+str(path))
    if not apply:
        print('PREVIEW_ONLY=True; --apply preserves pending EMA weights and retires only these derived caches.')
        return report
    report['rescued_weights']=rescue_pending_weights(run,state)
    report['removed_cache_generations']=retire_generations(cfg,keep_epochs=[next_epoch])
    atomic_json(run/'storage_recovery.json',report)
    print(f'FREE_GiB={shutil.disk_usage(run).free/GIB:.2f}; original weights, last.pt and fixed Dev retained.',flush=True)
    check_space(cfg,run,next_epoch,state_peak_bytes(state))
    print('V35_STORAGE_RECOVERY_READY=True',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if os.name!='posix':raise RuntimeError('Run recovery on the Linux training server')
    run=args.run or Path((ROOT/'exp'/'.latest_v35_run').read_text().strip())
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        active=active_jobs(ROOT)
        if active:raise RuntimeError('Training/evaluation is active; nothing changed: '+json.dumps(active))
        recover(run,args.apply)


if __name__=='__main__':main()
