"""Stop only this checkout's selected training workflow and its own descendants."""
import argparse
import os
from pathlib import Path
import signal
import time
from w2v_aasist.launch import ROOT


def inspect(pid):
    try:
        proc=Path('/proc')/str(pid)
        if proc.stat().st_uid != os.getuid(): return None
        args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,args=args,ppid=int(fields[1]),start=fields[19],
                    cwd=(proc/'cwd').resolve(strict=True),state=fields[0])
    except (OSError,ValueError,IndexError):
        return None


def owned_roots(processes, root, version):
    result=[]
    for item in processes:
        args=item['args']
        modules={f'w2v_{version}.workflow',f'w2v_{version}.train'}
        if item['cwd']!=root or item['state']=='Z': continue
        if any(a=='-m' and args[i+1] in modules for i,a in enumerate(args[:-1])):
            result.append(item)
    return result


def descendants(processes, selected):
    ids={p['pid'] for p in selected}
    while True:
        expanded=ids|{p['pid'] for p in processes if p['ppid'] in ids}
        if expanded==ids: break
        ids=expanded
    return [p for p in processes if p['pid'] in ids]


def same_process(item):
    current=inspect(item['pid'])
    return current is not None and current['start']==item['start'] and current['state']!='Z'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',choices=('v31','v32'),required=True)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args()
    if os.name!='posix' or not Path('/proc').is_dir():
        raise RuntimeError('Use this stop command on the Linux training server')
    processes=[item for entry in Path('/proc').iterdir() if entry.name.isdigit()
               for item in [inspect(int(entry.name))] if item is not None]
    roots=owned_roots(processes,ROOT.resolve(),args.version)
    targets=descendants(processes,roots)
    if any(t['pid']==os.getpid() for t in targets):
        raise RuntimeError('Refusing to signal the stop command itself')
    if not targets:
        print('No matching active training processes in this checkout.');return
    for t in targets: print(f"PID={t['pid']} "+' '.join(t['args']),flush=True)
    if not args.apply:
        print('Preview only; add --apply to stop these processes.');return
    # Stop the workflow before its trainer, so termination cannot trigger report
    # packaging while files are being changed. Only verified descendants follow.
    ordered=sorted(targets,key=lambda x:0 if f'w2v_{args.version}.workflow' in x['args'] else 1)
    for t in ordered:
        if same_process(t):
            try: os.kill(t['pid'],signal.SIGTERM)
            except ProcessLookupError: pass
    deadline=time.monotonic()+15
    while time.monotonic()<deadline and any(same_process(t) for t in targets): time.sleep(.25)
    pending=[t['pid'] for t in targets if same_process(t)]
    if pending: raise RuntimeError(f'Processes still exiting: {pending}; no forced kill was issued')
    print('TRAINING_STOPPED=True; saved checkpoints retained. Unsaved steps are not checkpointed.')


if __name__=='__main__': main()
