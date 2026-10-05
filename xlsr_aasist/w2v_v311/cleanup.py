"""Reclaim this completed V3.11 run's optimizer state; never delete dependencies."""
import argparse
from pathlib import Path

from w2v_v39.common import verify_files
from .state import load_selected


def cleanup(run, apply=False):
    requested = Path(run).expanduser()
    if requested.is_symlink():
        raise ValueError('Refusing a symlink run directory')
    run = requested.resolve()
    checkpoint, _ = load_selected(run)
    cfg = checkpoint['config']
    verify_files({cfg['base_checkpoint']:cfg['base_checkpoint_sha256']})
    # Resolving each exact target also protects Windows junction/symlink escapes.
    targets = [run/name for name in ('last.pt','last.pt.tmp','best.pt.tmp')]
    for path in targets:
        if path.is_symlink() or path.resolve().parent != run:
            raise ValueError('Cleanup target escapes its completed run')
    existing = [p for p in targets if p.is_file()]
    size = sum(p.stat().st_size for p in existing)
    print(f'V311_CLEANUP apply={apply} reclaim_GiB={size/1024**3:.2f}',flush=True)
    for path in existing:
        print(str(path),flush=True)
        if apply:
            path.unlink()
    print('Retained: best.pt, referenced original best, diagnostics, input audio and feature caches.',flush=True)
    return size


def main():
    from w2v_aasist.full_workflow import ensure_idle
    from w2v_aasist.launch import run_lock
    from w2v_v39.common import ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--apply',action='store_true')
    args = p.parse_args()
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        cleanup(args.run,args.apply)


if __name__ == '__main__':
    main()
