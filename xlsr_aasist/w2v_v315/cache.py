"""A process-safe rolling cache. Payloads plus in-flight files never exceed cap."""
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np

from w2v_v39.common import atomic_json, read_json

FORMAT = 'rtc_v315_bounded_pairs_v1'


@contextmanager
def locked(path):
    # OS locks are released even if a worker is terminated. No stale PID locks.
    with Path(path).open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            if not stream.read(1):
                stream.write(b'0'); stream.flush()
            end = time.monotonic()+120
            while True:
                try:
                    stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > end:
                        raise TimeoutError('Rolling cache lock did not become available')
                    time.sleep(.02)
            try:
                yield
            finally:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


class RollingCache:
    def __init__(self, path, identity, cap):
        self.path, self.identity, self.cap = Path(path), identity, int(cap)
        if self.cap < 0 or self.path.is_symlink():
            raise ValueError('Invalid rolling cache ownership/cap')
        self.path.mkdir(parents=True, exist_ok=True)
        self.path = self.path.resolve()
        if (self.path/'.cache.lock').is_symlink():
            raise ValueError('Refusing a linked cache lock')
        with locked(self.path/'.cache.lock'):
            owner = dict(format=FORMAT, identity=identity, cap_bytes=self.cap)
            if (self.path/'owner.json').exists():
                if read_json(self.path/'owner.json') != owner:
                    raise ValueError('Rolling cache belongs to a different run/configuration')
            elif set(p.name for p in self.path.iterdir()) - {'.cache.lock'}:
                raise ValueError('Refusing a nonempty unowned rolling cache')
            else:
                atomic_json(self.path/'owner.json', owner)
            self._files()

    def _files(self):
        files = []
        for p in self.path.iterdir():
            if p.is_symlink() or not p.is_file() or p.resolve().parent != self.path:
                raise ValueError('Rolling cache contains linked or external content')
            if p.name in ('owner.json', '.cache.lock'):
                continue
            stem = p.name.removesuffix('.npz').removesuffix('.tmp')
            if len(stem) != 64 or any(c not in '0123456789abcdef' for c in stem):
                raise ValueError('Unexpected rolling cache file: '+p.name)
            if p.suffix == '.tmp':
                p.unlink()  # An interrupted write; nobody writes outside this lock.
            elif p.suffix == '.npz':
                files.append(p)
            else:
                raise ValueError('Unknown rolling cache payload')
        return files

    def _target(self, key):
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Cache keys must be SHA256 hex digests')
        return self.path/(key+'.npz')

    def get(self, key):
        if not self.cap:
            return None
        target = self._target(key)
        with locked(self.path/'.cache.lock'):
            if not target.exists():
                return None
            if target.is_symlink():
                raise ValueError('Linked cache payload')
            try:
                with np.load(target, allow_pickle=False) as z:
                    a, b = z['reference'].copy(), z['noisy'].copy()
                    meta = json.loads(str(z['meta']))
                if a.shape != b.shape or a.ndim != 1 or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('Invalid cached waveform')
                os.utime(target, None)
                return a, b, meta
            except (ValueError, OSError, KeyError, EOFError, zipfile.BadZipFile):
                target.unlink(missing_ok=True)
                return None  # Deterministically regenerate owned corrupt derived data.

    def put(self, key, a, b, meta):
        if not self.cap:
            return
        target = self._target(key)
        buf = io.BytesIO()
        np.savez(buf, reference=a, noisy=b, meta=np.asarray(json.dumps(meta, sort_keys=True, allow_nan=False)))
        payload = buf.getvalue()
        if len(payload) > self.cap:
            return
        with locked(self.path/'.cache.lock'):
            files = self._files()
            if target.exists():
                return
            used = sum(p.stat().st_size for p in files)
            for p in sorted(files, key=lambda p:p.stat().st_mtime_ns):
                if used+len(payload) <= self.cap:
                    break
                used -= p.stat().st_size
                p.unlink()
            tmp = self.path/(key+'.tmp')
            try:
                with tmp.open('wb') as stream:
                    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)

    def used_bytes(self):
        with locked(self.path/'.cache.lock'):
            return sum(p.stat().st_size for p in self._files())
