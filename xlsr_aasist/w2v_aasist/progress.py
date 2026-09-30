"""TTY bars when interactive; bounded, readable lines in background logs."""
import sys
import time
from tqdm import tqdm


def progress(items, *, total, label, every=100):
    started = time.monotonic()
    tty = sys.stderr.isatty()
    iterator = tqdm(items, total=total, desc=label, disable=not tty, mininterval=10.)
    if not tty:
        print(f'{label}: 0/{total}', flush=True)
    for i, item in enumerate(iterator, 1):
        yield item
        if not tty and (i == 1 or i % every == 0 or i == total):
            elapsed = time.monotonic() - started
            eta = elapsed / i * (total-i)
            print(f'{label}: {i}/{total} elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m', flush=True)


def training_bar(items, total, label):
    return tqdm(items, total=total, desc=label, disable=not sys.stderr.isatty(), mininterval=10.)
