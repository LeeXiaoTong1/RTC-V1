"""Resumable, bounded HTTPS ranges with a pinned whole-file checksum."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re
import time
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def fetch(row, destination, jobs=4, opener=urllib.request.urlopen, chunk_bytes=32 * 1024**2):
    """Keep finished ranges across interruption; publish only a verified archive.

    Call under the preparation directory's OS lock. Range receipts are local
    cache integrity checks; the final pinned SHA256 authenticates the complete
    asset. No extraction is attempted before that check.
    """
    if not 1 <= jobs <= 8 or chunk_bytes < 1:
        raise ValueError('Invalid download worker count or chunk size')
    size = int(row['bytes'])
    if not 0 < size <= 8 * 1024**3 or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
        raise ValueError('Invalid pinned asset')
    if not row['url'].startswith('https://'):
        raise ValueError('Only HTTPS assets are accepted')
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        if dest.stat().st_size != size or sha256(dest) != row['sha256']:
            raise ValueError('Corrupt completed cache; move it aside: ' + str(dest))
        return dest
    parts = dest.with_name(dest.name + '.parts')
    parts.mkdir(exist_ok=True)
    # Bind reusable ranges to this exact source and chunk layout.
    import json
    identity = dict(row, chunk_bytes=chunk_bytes)
    identity_path = parts / 'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Download identity changed; choose a new output directory')
    else:
        temporary_identity = parts / 'identity.json.tmp'
        temporary_identity.write_text(json.dumps(identity, sort_keys=True), encoding='utf-8')
        temporary_identity.replace(identity_path)

    def one(start):
        end = min(start + chunk_bytes, size) - 1
        path = parts / str(start)
        receipt = parts / (str(start) + '.sha256')
        if path.is_file() and receipt.is_file() and path.stat().st_size == end - start + 1:
            if sha256(path) == receipt.read_text().strip():
                return path
        tmp = path.with_suffix('.part')
        error = None
        for attempt in range(4):
            try:
                request = urllib.request.Request(row['url'], headers={
                    'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
                    'User-Agent': 'RTC-SDD-noise-preparation/3.18.4'})
                with opener(request, timeout=45) as response:
                    status = getattr(response, 'status', 200)
                    if status == 206:
                        if response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Invalid HTTP Content-Range')
                    elif not (status == 200 and start == 0 and end == size - 1):
                        raise ValueError('Server did not honor the bounded range')
                    count = 0
                    with tmp.open('wb') as stream:
                        while True:
                            data = response.read(256 * 1024)
                            if not data:
                                break
                            count += len(data)
                            if count > end - start + 1:
                                raise ValueError('HTTP response exceeds requested range')
                            stream.write(data)
                    if count != end - start + 1:
                        raise ValueError('Incomplete HTTP range')
                tmp.replace(path)
                receipt.write_text(sha256(path) + '\n', encoding='ascii')
                return path
            except Exception as exc:
                error = exc
                if attempt < 3:
                    time.sleep(min(attempt + 1, 2))
        raise RuntimeError(f'Download range {start}-{end} failed: {error}') from error

    starts = list(range(0, size, chunk_bytes))
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(one, start) for start in starts]
        try:
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 4 == 0 or done == len(starts):
                    print(f'[Download] {dest.name}: {done}/{len(starts)} ranges ready', flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    assembled = dest.with_name(dest.name + '.assembling')
    h = hashlib.sha256()
    with assembled.open('wb') as stream:
        for start in starts:
            with (parts / str(start)).open('rb') as source:
                for data in iter(lambda: source.read(1024 * 1024), b''):
                    h.update(data)
                    stream.write(data)
    if assembled.stat().st_size != size or h.hexdigest() != row['sha256']:
        # Per-range local receipts cannot tell which range came from a changed
        # or broken upstream file. Drop only our known chunks so retry can heal.
        assembled.unlink()
        for start in starts:
            (parts / str(start)).unlink(missing_ok=True)
            (parts / (str(start) + '.sha256')).unlink(missing_ok=True)
        raise ValueError('Complete download checksum mismatch; invalid ranges removed; retry the command')
    assembled.replace(dest)
    # Only delete this helper's known range files after successful verification.
    for start in starts:
        (parts / str(start)).unlink()
        (parts / (str(start) + '.sha256')).unlink()
    return dest
