"""Resumable, checksummed official-Train-only frozen teacher vectors (1 KiB/row)."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from w2v_aasist.data import read_wave
from w2v_aasist.launch import run_lock
from w2v_aasist.progress import progress
from w2v_aasist.runtime import atomic_json, sha256
from w2v_v36.features import _digest, _json, _rows
from .language import DIM, MODE, LanguageEncoder, ensure_language_assets, language_identity

FORMAT = 'rtc_v37_frozen_language_vectors_v1'


def _train_rows(records):
    # Use precisely the V3.6 metadata canonicalization to allow exact row binding.
    rows = _rows(records)
    if any(row.get('split') != 'train' for row in rows):
        raise ValueError('Frozen LID teacher extraction is restricted to official Train rows')
    if any(row.get('view', 'full') != 'full' for row in rows):
        raise ValueError('Only full-wave Train views may enter the frozen LID cache')
    return rows


def _verify_audio(row, digest=False):
    path = Path(row['audio']); stat = path.stat()
    if stat.st_size != row['audio_size'] or stat.st_mtime_ns != row['audio_mtime_ns']:
        raise ValueError('Train audio changed during frozen LID extraction: ' + str(path))
    if digest and sha256(path) != row['audio_sha256']:
        raise ValueError('Train audio SHA256 changed during frozen LID extraction: ' + str(path))


def _chunk_digest(lid, start, stop):
    return hashlib.sha256(np.ascontiguousarray(lid[start:stop]).tobytes()).hexdigest()


def _flush(path, array):
    array.flush()
    with Path(path).open('r+b') as stream:
        os.fsync(stream.fileno())


def _validate_vectors(lid, n):
    if lid.shape != (n, DIM) or lid.dtype != np.float32:
        raise ValueError('Frozen LID vector dimensions/dtype differ')


def _validate_state(lid, state, owner, rows_hash):
    n = owner['shape'][0]; cursor = state.get('cursor')
    if (state.get('identity_digest') != owner['identity_digest'] or state.get('rows_sha256') != rows_hash
            or type(cursor) is not int or not 0 <= cursor <= n):
        raise ValueError('Frozen LID cursor identity/row checksum/count differs')
    previous = 0
    for chunk in state.get('chunks', []):
        start, stop = chunk['start'], chunk['stop']
        if type(start) is not int or type(stop) is not int or start != previous or not start < stop <= cursor:
            raise ValueError('Frozen LID committed chunks are incomplete')
        if _chunk_digest(lid, start, stop) != chunk['sha256']:
            raise ValueError('Committed frozen LID cache bytes changed')
        vectors = np.asarray(lid[start:stop])
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=2e-5):
            raise ValueError('Frozen LID cache contains nonfinite/nonunit embeddings')
        previous = stop
    if previous != cursor:
        raise ValueError('Frozen LID cursor is not fully committed')


def _bound(owner):
    return {k: owner[k] for k in ('identity', 'records_digest', 'mode', 'encoder')}


def load_language_cache(out, identity=None, records=None):
    """Only committed complete vectors can be used to fit the compact student."""
    out = Path(out).expanduser().resolve()
    owner, manifest = _json(out/'owner.json'), _json(out/'complete.json')
    if owner.get('format') != FORMAT or manifest.get('format') != FORMAT or owner.get('mode') != MODE:
        raise ValueError('Unknown frozen LID cache format/mode')
    if identity is not None and owner.get('identity') != identity:
        raise ValueError('Frozen LID cache belongs to another Train/base identity')
    if owner.get('identity_digest') != _digest(_bound(owner)) or manifest.get('owner_sha256') != sha256(out/'owner.json'):
        raise ValueError('Frozen LID cache owner changed')
    if set(manifest.get('files', {})) != {'lid.npy', 'rows.json', 'cursor.json'}:
        raise ValueError('Incomplete frozen LID file checksums')
    for name, digest in manifest['files'].items():
        if sha256(out/name) != digest:
            raise ValueError('Completed frozen LID file changed: ' + name)
    rows, state = _json(out/'rows.json'), _json(out/'cursor.json')
    if (_digest(rows) != owner['records_digest'] or len(rows) != owner['shape'][0]
            or state['cursor'] != len(rows) or manifest.get('cursor') != len(rows)
            or any(row.get('split') != 'train' for row in rows)):
        raise ValueError('Frozen LID row order/count/split differs')
    if records is not None and rows != _train_rows(records):
        raise ValueError('Frozen LID rows do not match V3.6 Train row identity/order')
    lid = np.load(out/'lid.npy', mmap_mode='r', allow_pickle=False)
    try:
        _validate_vectors(lid, len(rows))
        _validate_state(lid, state, owner, sha256(out/'rows.json'))
    except BaseException:
        lid._mmap.close()
        raise
    return dict(lid=lid, rows=rows, manifest=dict(manifest, **{k: owner[k] for k in
                ('shape', 'identity', 'mode', 'records_digest', 'encoder')}))


def extract_language_cache(records, cfg, out, identity, encoder=None):
    """No new audio or frame-feature files; interrupted uncommitted rows replay.

    Records come from the existing V3.6 validated official Train inventory.
    Every consumed waveform is SHA256/size checked against that inventory. A
    completed cache is reused without constructing the teacher or reading audio.
    """
    out = Path(out).expanduser().resolve(); rows = _train_rows(records); n = len(rows)
    commit_rows = int(cfg.get('language_commit_rows', 256))
    if commit_rows < 1:
        raise ValueError('Frozen LID commit interval must be positive')
    if isinstance(identity, dict) and identity.get('split', 'train') != 'train':
        raise ValueError('Frozen LID cache identity must name Train')
    if encoder is None:
        assets = cfg.get('language_assets') or ensure_language_assets(cfg)
        encoder_id = language_identity(cfg, assets)
    else:
        encoder_id = encoder.identity
    bound = dict(identity=identity, records_digest=_digest(rows), mode=MODE, encoder=encoder_id)
    owner = dict(format=FORMAT, **bound, identity_digest=_digest(bound), shape=[n, DIM], dtype='float32')
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out/'.extract.lock'):
        if (out/'owner.json').is_file():
            if _json(out/'owner.json') != owner:
                raise ValueError('Refusing changed frozen LID Train/base/encoder/row identity')
            if (out/'complete.json').is_file():
                return load_language_cache(out, identity, rows)
            state = _json(out/'cursor.json')
            lid = np.load(out/'lid.npy', mmap_mode='r+', allow_pickle=False)
            try:
                if _json(out/'rows.json') != rows:
                    raise ValueError('Frozen LID row metadata changed')
                _validate_vectors(lid, n)
                _validate_state(lid, state, owner, sha256(out/'rows.json'))
            except BaseException:
                lid._mmap.close()
                raise
        else:
            if any(path.name != '.extract.lock' for path in out.iterdir()):
                raise ValueError('Refusing an unowned nonempty frozen LID cache directory')
            required = n*DIM*4 + 2*len(json.dumps(rows).encode()) + int(cfg.get('language_free_margin_bytes', 256*1024**2))
            free = shutil.disk_usage(out).free
            if free < required:
                raise OSError(f'Frozen LID cache needs {required/1024**3:.3f} GiB free; no existing data deleted')
            print(f'V37_LID_STORAGE rows={n} dim={DIM} required_GiB={required/1024**3:.3f} free_GiB={free/1024**3:.3f}', flush=True)
            atomic_json(out/'rows.json', rows)
            lid = np.lib.format.open_memmap(out/'lid.npy', mode='w+', dtype=np.float32, shape=(n, DIM))
            _flush(out/'lid.npy', lid)
            state = dict(identity_digest=owner['identity_digest'], rows_sha256=sha256(out/'rows.json'), cursor=0, chunks=[])
            atomic_json(out/'cursor.json', state)
            atomic_json(out/'owner.json', owner)
        try:
            if encoder is None and state['cursor'] < n:
                encoder = LanguageEncoder(dict(cfg, language_assets=assets))
            print(f'V37_LID_RESUME completed={state["cursor"]}/{n} mode={MODE}', flush=True)
            indices = range(state['cursor'], n)
            for index in progress(indices, total=len(indices), label='V3.7 frozen Train LID', every=100):
                row = rows[index]
                _verify_audio(row, digest=True)
                wave = read_wave(row['audio'])
                _verify_audio(row)
                if row.get('output_samples') is not None and len(wave) != row['output_samples']:
                    raise ValueError('Full Train audio sample count changed')
                value = np.asarray(encoder.encode_waveforms([wave]))
                if (value.shape != (1, DIM) or value.dtype != np.float32 or not np.isfinite(value).all()
                        or not np.allclose(np.linalg.norm(value, axis=1), 1., atol=2e-5)):
                    raise ValueError('Malformed/nonfinite/nonunit frozen LID embedding')
                lid[index] = value[0]
                cursor = index + 1
                if cursor-state['cursor'] >= commit_rows or cursor == n:
                    _flush(out/'lid.npy', lid)
                    state['chunks'].append(dict(start=state['cursor'], stop=cursor,
                        sha256=_chunk_digest(lid, state['cursor'], cursor)))
                    state['cursor'] = cursor
                    atomic_json(out/'cursor.json', state)
            if state['cursor'] != n:
                raise ValueError('Frozen LID extraction ended before full coverage')
            complete = dict(format=FORMAT, owner_sha256=sha256(out/'owner.json'), cursor=n,
                files={name: sha256(out/name) for name in ('lid.npy', 'rows.json', 'cursor.json')})
            atomic_json(out/'complete.json', complete)
        finally:
            lid._mmap.close()
    return load_language_cache(out, identity, rows)
