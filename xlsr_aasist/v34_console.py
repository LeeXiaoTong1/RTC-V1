"""Reuse the established read-only viewer while displaying V3.4 job identity."""
import argparse
import math
from pathlib import Path
from v33_console import watch

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--once', action='store_true')
    p.add_argument('--interval', type=float, default=1.)
    args = p.parse_args()
    if not math.isfinite(args.interval) or args.interval < .1:
        p.error('interval must be at least 0.1 seconds')
    log = args.log
    if log is None:
        log = Path((ROOT/'exp'/'.latest_v34_log').read_text(encoding='utf-8').strip())
    watch(log, Path(str(log)+'.progress.json'), run=args.run_dir,
          once=args.once, interval=args.interval, version='V3.4')


if __name__ == '__main__':
    main()
