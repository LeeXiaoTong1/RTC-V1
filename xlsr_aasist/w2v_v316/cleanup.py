"""Conservative, dependency-aware cleanup. Standard library only; no tensor loads.

Never touches V3.15+ runs, official audio, fixed Dev, pretrained weights, best
files, or files referenced by checkpoint/selection manifests. Unknown ownership
is a reason to retain a file, not a reason to delete it.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def version(run):
    path=run/'config.json'
    if path.is_file():
        text=str(read(path).get('version',''))
        match=re.fullmatch(r'(\d+)\.(\d+)',text)
        if match: return tuple(map(int,match.groups()))
    match=re.match(r'w2v_v(\d)(\d*)_',run.name)
    if match: return int(match[1]),int(match[2] or 0)
    return None


def safe(path,root):
    path=Path(path).absolute(); root=Path(root).resolve()
    if path==root or root not in path.parents or root not in path.resolve().parents:
        raise ValueError('Cleanup path escaped project: '+str(path))
    for ancestor in (path,*path.parents):
        if ancestor==root: break
        if ancestor.is_symlink() or (hasattr(ancestor,'is_junction') and ancestor.is_junction()):
            raise ValueError('Refusing cleanup through link/junction: '+str(ancestor))
    return path


def strings(value):
    if isinstance(value,str): yield value
    elif isinstance(value,dict):
        for k,v in value.items():
            yield str(k); yield from strings(v)
    elif isinstance(value,list):
        for v in value: yield from strings(v)


MAX_METADATA_BYTES = 256*1024**2


class _JSONReader:
    """Bounded JSON values for the large file array in legacy retirement audits."""
    def __init__(self, stream, chunk_size=64*1024, max_value=8*1024**2):
        self.stream, self.chunk_size, self.max_value = stream, chunk_size, max_value
        self.buffer, self.pos, self.eof = '', 0, False
        self.decoder = json.JSONDecoder()

    def fill(self):
        self.buffer = self.buffer[self.pos:]
        self.pos = 0
        chunk = self.stream.read(self.chunk_size)
        self.buffer += chunk
        self.eof = not chunk

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos] in ' \t\r\n':
                self.pos += 1
            if self.pos < len(self.buffer): return self.buffer[self.pos]
            if self.eof: return ''
            self.fill()

    def expect(self, char):
        if self.peek() != char: raise ValueError('Expected JSON delimiter '+repr(char))
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            if len(self.buffer)-self.pos > self.max_value:
                raise ValueError('Retirement audit contains an oversized individual JSON value')
            try:
                obj, end = self.decoder.raw_decode(self.buffer, self.pos)
            except ValueError:
                if self.eof: raise ValueError('Invalid or truncated retirement audit JSON')
            else:
                # Wait for a delimiter, including when numbers/escapes straddle reads.
                if end < len(self.buffer) and self.buffer[end] in ' \t\r\n,]}:':
                    self.pos = end
                    return obj
                if self.eof and end == len(self.buffer):
                    self.pos = end
                    return obj
                if self.eof: raise ValueError('Invalid retirement audit JSON value suffix')
            self.fill()


def retirement_strings(path, chunk_size=64*1024):
    """Scan every audit reference, including protected paths AFTER the files array.

    The producer (w2v_aasist.cache_retirement) writes one huge top-level files
    array. Decode each record separately; never skip an audit by filename or
    assume its deletion completed. Other fields remain bounded JSON values.
    """
    with Path(path).open(encoding='utf-8') as stream:
        reader = _JSONReader(stream, chunk_size)
        reader.expect('{')
        if reader.peek() != '}':
            while True:
                key = reader.value()
                if not isinstance(key, str): raise ValueError('Retirement audit key must be a string')
                yield key
                reader.expect(':')
                if key == 'files':
                    reader.expect('[')
                    if reader.peek() != ']':
                        while True:
                            yield from strings(reader.value())
                            if reader.peek() == ']': break
                            reader.expect(',')
                    reader.expect(']')
                else:
                    yield from strings(reader.value())
                if reader.peek() == '}': break
                reader.expect(',')
        reader.expect('}')
        if reader.peek(): raise ValueError('Trailing data in retirement audit JSON')


def metadata_strings(path, exp):
    # Only this documented top-level audit format has a streaming decoder.
    # Unknown oversized metadata still stops cleanup rather than hiding references.
    if path.parent == exp and re.fullmatch(r'cache_retirement_\d{8}_\d{6}_\d{6}\.json', path.name):
        if path.stat().st_size > MAX_METADATA_BYTES:
            print('Streaming large cache-retirement audit: '+str(path), flush=True)
        yield from retirement_strings(path)
    else:
        if path.stat().st_size > MAX_METADATA_BYTES:
            raise ValueError('Oversize metadata must be reviewed before cleanup: '+str(path))
        yield from strings(read(path))


def inventory(root):
    root=Path(root).resolve(); exp=root/'exp'
    if not exp.is_dir(): raise ValueError('Expected project root containing exp/')
    protected, metadata, reasons, referrers = set(),{},{},{}
    runs=[p for p in exp.iterdir() if p.is_dir() and not p.is_symlink()]
    # JSON references are sufficient for all supported historical checkpoints;
    # opaque tensor-only checkpoints without a recognized named best are retained.
    for path in exp.rglob('*.json'):
        safe(path,root)
        if path.name.startswith('cleanup_v316_'): continue
        before=sha(path)
        try:
            for value in metadata_strings(path, exp):
                if '\x00' in value or len(value)>4096: continue
                if not value.lower().endswith(('.pt','.pth','.ckpt','.wav','.flac','.npy')): continue
                candidate=Path(value)
                choices=[candidate] if candidate.is_absolute() else [root/candidate,path.parent/candidate]
                for choice in choices:
                    if choice.is_file():
                        protected.add(str(choice.resolve())); reasons[str(choice.resolve())]='referenced by '+str(path.relative_to(root))
                        referrers.setdefault(str(choice.resolve()),set()).add(str(path.resolve()))
        except (OSError,ValueError) as exc:
            raise ValueError('Unreadable metadata; dependency scan cannot be trusted: '+str(path)+'; '+str(exc)) from exc
        if sha(path)!=before: raise ValueError('Metadata changed during dependency scan: '+str(path))
        metadata[str(path)]=before
    for run in runs:
        v=version(run)
        for path in run.rglob('*'):
            if not path.is_file(): continue
            if v is None or v>=(3,15) or 'best' in path.name.lower() or path.name=='inference.pt':
                protected.add(str(path.resolve()))
                reasons[str(path.resolve())]='V3.15+, unknown run, best, or inference state'
    candidates=[]; retained=[]
    def add(path,reason):
        path=safe(path,root)
        if str(path.resolve()) in protected:
            retained.append(dict(path=str(path),reason=reasons.get(str(path.resolve()),'referenced'))); return
        stat=path.stat()
        candidates.append(dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
            inode=stat.st_ino,device=stat.st_dev,reason=reason))
    for run in runs:
        v=version(run)
        if v is None or v>=(3,15): continue
        # A surviving separately named best is necessary before discarding old intermediates.
        has_best=any(p.is_file() and not p.is_symlink() for p in run.rglob('*best*.pt')) or (run/'inference.pt').is_file()
        if has_best:
            for path in run.rglob('*'):
                if path.is_file() and re.fullmatch(r'(?:last|latest|epoch[_-]?\d+|checkpoint[_-]?\d+|step[_-]?\d+)(?:[^/]*)\.(?:pt|pth|ckpt)',path.name,re.I) and 'best' not in path.name.lower():
                    add(path,'pre-V3.15 non-best intermediate; separate best survives')
        for split in ('train','dev'):
            folder=run/'features'/split
            if not all((folder/n).is_file() for n in ('owner.json','complete.json','rows.json')): continue
            owner,complete=read(folder/'owner.json'),read(folder/'complete.json')
            if (owner.get('format')!='rtc_v36_frozen_vectors_v1' or complete.get('format')!='rtc_v36_frozen_vectors_v1'
                    or owner.get('mode')!='eval-fp32-exact-length-full-wave-final-linear-input'
                    or complete.get('owner_sha256')!=sha(folder/'owner.json')
                    or complete.get('files',{}).get('rows.json')!=sha(folder/'rows.json')): continue
            for name in ('x.npy','logits.npy'):
                path=folder/name
                # Own feature completion hashes are cache checks, not model dependencies.
                own=complete.get('files',{}).get(name)
                if path.is_file() and own and sha(path)==own:
                    # References from OTHER metadata remain protective. Bare own name
                    # appears in complete.json and is removed only for this producer.
                    target=str(path.resolve())
                    if referrers.get(target)=={str((folder/'complete.json').resolve())}:
                        protected.discard(target)
                    add(path,'regenerable frozen feature tensor; owner/rows/manifests retained')
    # Large V3.5 training-only audio generations have explicit ownership records.
    # Never scan/delete other audio trees or any Dev generation.
    train_cache=root/'data'/'rtc_v35'/'train'
    if (train_cache/'owner.json').is_file():
        owner=read(train_cache/'owner.json')
        expected=dict(format='rtc_v35_epoch_full_views_v1',role='train',root=str(train_cache.resolve()))
        if owner==expected:
            for marker_path in (train_cache/'owner.json',train_cache/'recipe.json'):
                safe(marker_path,root); metadata[str(marker_path)]=sha(marker_path)
            for folder in train_cache.glob('epoch_*'):
                if not (folder/'generation.json').is_file(): continue
                marker=read(folder/'generation.json')
                if (marker.get('format')!=expected['format'] or marker.get('role')!='train'
                        or marker.get('root')!=expected['root'] or folder.name!=f'epoch_{marker.get("epoch",-1):03d}'
                        or marker.get('recipe_sha256')!=sha(train_cache/'recipe.json')): continue
                metadata[str(folder/'generation.json')]=sha(folder/'generation.json')
                # Whole-generation safety: any referenced audio pins its generation.
                files=[p for p in folder.rglob('*') if p.is_file()]
                if any(str(p.resolve()) in protected for p in files): continue
                for path in files:
                    if path.suffix in ('.wav','.npy','.npz') or path.name.endswith('.wav.json'):
                        add(path,'owned obsolete V3.5 Train augmentation, not original audio or fixed Dev')
    candidates=list({c['path']:c for c in candidates}.values())
    return dict(schema='rtc_v316_cleanup_v1',root=str(root),candidates=candidates,retained=retained,
        total_bytes=sum(v['bytes'] for v in candidates),metadata_fingerprints=metadata,
        protected_weights={p:sha(p) for p in sorted(protected) if Path(p).suffix in ('.pt','.pth','.ckpt')},
        note='V3.15+ untouched; referenced last.pt may be essential for best export; original/fixed-Dev/pretrained data retained')


def apply(plan):
    root=Path(plan['root']).resolve()
    for path,value in plan['metadata_fingerprints'].items():
        if sha(path)!=value: raise ValueError('Metadata changed after scan; cleanup cancelled')
    for item in plan['candidates']:
        path=safe(item['path'],root); stat=path.stat()
        if (stat.st_size,stat.st_mtime_ns,stat.st_ino,stat.st_dev)!=(item['bytes'],item['mtime_ns'],item['inode'],item['device']):
            raise ValueError('Cleanup candidate changed after scan: '+str(path))
    removed=[]
    for item in plan['candidates']:
        path=safe(item['path'],root); path.unlink(); removed.append(item)
    for path,value in plan['protected_weights'].items():
        if sha(path)!=value: raise RuntimeError('Protected checkpoint changed during cleanup')
    return dict(removed=removed,freed_bytes=sum(v['bytes'] for v in removed),protected_weights_verified=True,
        free_bytes=shutil.disk_usage(root).free)


@contextmanager
def idle_lock(root):
    if os.name!='posix': raise RuntimeError('Apply on the Linux training server; Windows supports plan/tests only')
    import fcntl
    lock=Path(root)/'exp'/'.aasist-launch.lock'
    with lock.open('a+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for entry in Path('/proc').glob('[0-9]*'):
            try:
                if int(entry.name)==os.getpid() or entry.stat().st_uid!=os.getuid(): continue
                args=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
                if args and Path(args[0]).name.startswith('python') and any(
                    any(t in a.lower() for t in ('w2v_','rtc_noisy','main_train')) for a in args[1:]):
                    raise RuntimeError('Training/evaluation/cache writer still active; finish it before cleanup: PID '+entry.name)
            except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        try: yield
        finally: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--apply',action='store_true'); args=p.parse_args()
    root=Path(args.root).expanduser().resolve()
    def work():
        plan=inventory(root)
        report=root/'exp'/('cleanup_v316_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
        report.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'CLEANUP_PLAN={report}\nRECLAIMABLE_GiB={plan["total_bytes"]/1024**3:.3f}',flush=True)
        if args.apply:
            result=apply(plan); plan['result']=result
            report.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(f'FREED_GiB={result["freed_bytes"]/1024**3:.3f}\nBEST_DEPENDENCIES_VERIFIED=True',flush=True)
    if args.apply:
        with idle_lock(root): work()
    else: work()


if __name__=='__main__': main()
