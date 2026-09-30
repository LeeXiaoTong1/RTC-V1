"""TTY bars when interactive; bounded, readable lines in background logs."""
import sys
import time
from tqdm import tqdm
from live_progress import Phase, phase


def progress(items, *, total, label, every=100):
    started = time.monotonic()
    live = Phase(label, total)
    tty = sys.stderr.isatty()
    iterator = tqdm(items, total=total, desc=label, disable=not tty, mininterval=10.)
    if not tty:
        print(f'{label}: 0/{total}', flush=True)
    for i, item in enumerate(iterator, 1):
        yield item
        live.update(i)
        if not tty and (i == 1 or i % every == 0 or i == total):
            elapsed = time.monotonic() - started
            eta = elapsed / i * (total-i)
            print(f'{label}: {i}/{total} elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m', flush=True)


class training_bar:
    def __init__(self, items, total, label):
        self.bar = tqdm(items, total=total, desc=label, disable=not sys.stderr.isatty(), mininterval=10.)
        self.disable = self.bar.disable
        self.live = Phase(label, total)
        self.loss = None

    def set_postfix(self, *, loss, refresh=False):
        self.loss = float(loss)
        self.bar.set_postfix(loss=loss, refresh=refresh)

    def __iter__(self):
        try:
            for i, item in enumerate(self.bar, 1):
                yield item
                self.live.update(i, self.loss)
        finally:
            self.bar.close()
