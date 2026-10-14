"""Find a completed V3.18.5 parent without a mandatory latest-run pointer.

Discovery reads small JSON metadata and checks required files. The existing
parent loader still verifies checkpoint hashes, saved manifests and tensors.
No Torch/GPU import, disk-wide search, checkpoint mutation, or version fallback.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
RECIPE='v3185_asp_v1'
REQUIRED=('config.json','completed.json','last.pt','train_rows.json','dev_rows.json',
          'dev_partition.json','train_probe_tickets.json','independent_panel.json')


def _json(path):
    with Path(path).open(encoding='utf-8') as stream:return json.load(stream)


def inspect_candidate(run):
    run=Path(run).expanduser().resolve()
    result=dict(path=str(run),eligible=False,version=None,best_tag=None,reason='')
    try:
        cfg=_json(run/'config.json')
        result['version']=cfg.get('version')
        if cfg.get('version')!='3.18.5':raise ValueError('Expected V3.18.5, found '+str(cfg.get('version')))
        if (cfg.get('recipe')!=RECIPE or cfg.get('variant')!='C3' or cfg.get('risk_coefficient')!=.25
                or cfg.get('pooling')!='asp' or cfg.get('view_factors')!=[5,5,1,3]):
            raise ValueError('Requires V3.18.5 C3, ASP, and fixed factors 5/5/1/3')
        missing=[name for name in REQUIRED if not (run/name).is_file()]
        if missing:raise ValueError('Missing required files: '+', '.join(missing))
        done=_json(run/'completed.json')
        result['best_tag']=done.get('best_tag')
        if (done.get('version')!='3.18.5' or done.get('recipe')!=RECIPE
                or done.get('status')!='complete' or not isinstance(done.get('best_tag'),str)
                or not done['best_tag']):
            raise ValueError('Requires completed V3.18.5 with a recorded best epoch')
        if (run/'last.pt').stat().st_size==0:raise ValueError('Empty parent checkpoint')
        result['eligible']=True
    except (OSError,ValueError,TypeError,AttributeError) as exc:
        result['reason']=str(exc)
    return result


def default_roots(root=ROOT):
    root=Path(root).resolve();base=root.parent.parent
    return [root/'exp',base/'RTC'/'xlsr_aasist'/'exp',
            base/'RTC-w2v-improved'/'xlsr_aasist'/'exp']


def scan(roots):
    """Only direct experiment directories under explicitly bounded roots."""
    found={};checked=[]
    for entry in roots:
        path=Path(entry).expanduser().resolve()
        if str(path) in checked:continue
        checked.append(str(path))
        if not path.is_dir():continue
        candidates=[path] if (path/'config.json').is_file() else sorted(path.iterdir())
        for run in candidates:
            if not run.is_dir() or not (run/'config.json').is_file():continue
            canonical=str(run.resolve())
            if canonical not in found:found[canonical]=inspect_candidate(run)
    return list(found.values()),checked


def _details(rows):
    return '\n'.join('  '+r['path']+' | version='+str(r['version'])+' | '+
        ('best='+str(r['best_tag']) if r['eligible'] else r['reason']) for r in rows)


def resolve_parent(value=None,*,root=ROOT,search_roots=None):
    # An empty shell substitution is treated exactly like an omitted option.
    if value is not None and str(value).strip():
        item=inspect_candidate(str(value).strip())
        if not item['eligible']:raise ValueError('Invalid --parent-run: '+item['path']+'; '+item['reason'])
        return Path(item['path'])
    root=Path(root).resolve();pointer=root/'exp/.latest_v3185_run';pointer_note=None
    if pointer.is_file():
        try:
            recorded=pointer.read_text(encoding='utf-8').strip()
            if recorded:
                path=Path(recorded).expanduser()
                if not path.is_absolute():path=root/path
                item=inspect_candidate(path)
                if item['eligible']:return Path(item['path'])
                pointer_note='Ignored stale latest-run pointer: '+item['reason']
            else:pointer_note='Ignored empty latest-run pointer.'
        except (OSError,UnicodeError) as exc:pointer_note='Could not read latest-run pointer: '+str(exc)
    rows,checked=scan(default_roots(root) if search_roots is None else search_roots)
    candidates=[r for r in rows if r['eligible']]
    if len(candidates)==1:
        if pointer_note:print(pointer_note,file=sys.stderr)
        return Path(candidates[0]['path'])
    if len(candidates)>1:
        raise ValueError('Multiple completed V3.18.5 parents found. Choose the intended run with --parent-run PATH:\n'+_details(candidates))
    relevant=[r for r in rows if str(r['version']).startswith('3.18')]
    message='No compatible completed V3.18.5 parent found. Searched:\n  '+'\n  '.join(checked)
    if pointer_note:message+='\n'+pointer_note
    if relevant:message+='\nRecorded runs (not substituted as parents):\n'+_details(relevant[:12])
    message+='\nUse --search-root EXP_DIR to locate moved runs, or --parent-run PATH when known.'
    raise FileNotFoundError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run')
    parser.add_argument('--search-root',action='append',help='Additional bounded experiment directory; may repeat')
    parser.add_argument('--list',action='store_true',help='List recorded versions/statuses without selecting a parent')
    args=parser.parse_args();roots=default_roots(ROOT)+[Path(p) for p in (args.search_root or [])]
    if args.list:
        rows,checked=scan(roots)
        print(_details(rows) if rows else 'No run config.json found under:\n  '+'\n  '.join(checked))
        return 0
    try:parent=resolve_parent(args.parent_run,root=ROOT,search_roots=roots)
    except (OSError,ValueError) as exc:
        print(str(exc),file=sys.stderr);return 2
    # stdout is exactly one path, suitable for a checked shell substitution.
    print(parent);return 0


if __name__=='__main__':raise SystemExit(main())
