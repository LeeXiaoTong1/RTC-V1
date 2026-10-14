"""Train-only phoneme metadata, conservative token matching, and exact time maps.

The phoneme recognizer is CTC. Its nonblank states are acoustic support, NOT
ground-truth phone boundaries. Preparation uses CTC Viterbi alignment followed
by bounded midpoint interpolation; both support and approximate intervals are
recorded. Runtime never runs the recognizer and never caches detector features.
"""
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = 'v3186-phonemes-v1'
MODEL_ID = 'facebook/wav2vec2-xlsr-53-espeak-cv-ft'
MODEL_REVISION = '2c733782da5604684829819a5eb744c193fe9398'
SAMPLE_RATE = 16000
ALGORITHM = dict(name='ctc-viterbi-bounded-midpoint-v1', max_half_gap_samples=1920,
                 max_segment_samples=8000, min_token_posterior=.05,
                 chunk_samples=320000, context_samples=16000,
                 timing='approximate; CTC support is not a full phone boundary')


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def digest_json(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    try:
        with tmp.open('w', encoding='utf-8', newline='\n') as f:
            json.dump(value, f, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)
            f.write('\n'); f.flush()
            import os
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        if tmp.exists(): tmp.unlink()


def lineage(row):
    """Portable identity: relocation may change paths, never bytes or split."""
    domain = row.get('condition', row.get('domain'))
    if row.get('split') != 'train' or domain not in ('offline', 'online'):
        raise ValueError('Phoneme metadata accepts original Train Offline/Online only')
    if row.get('language') not in ('en', 'zh') or row.get('label') not in (0, 1):
        raise ValueError('Invalid phoneme source language/label')
    sha = row.get('audio_sha256', '')
    group = row.get('group_id', '')
    if (len(sha) != 64 or len(group) != 64 or
            any(c not in '0123456789abcdef' for c in sha + group)):
        raise ValueError('Verified audio/source SHA256 required for phoneme preparation')
    samples = row.get('output_samples')
    if type(samples) is not int or samples < 400 or not row.get('full_length'):
        raise ValueError('Require full original waveform with at least 400 samples')
    if not isinstance(row.get('id'), str) or not row['id'] or not row.get('source_id'):
        raise ValueError('Missing source identity')
    return dict(id=row['id'], source_id=row['source_id'], group_id=group,
                audio_sha256=sha, domain=domain, language=row['language'],
                label=row['label'], num_samples=samples, sample_rate=SAMPLE_RATE,
                split='train')


def inventory(rows):
    result = [lineage(r) for r in rows]
    if not result or len({r['id'] for r in result}) != len(result):
        raise ValueError('Empty/duplicate phoneme source inventory')
    offline = {r['id']: r for r in result if r['domain'] == 'offline'}
    for row in result:
        source = offline.get(row['source_id'])
        if source is None or any(source[k] != row[k] for k in ('group_id', 'label', 'language')):
            raise ValueError('Phoneme inventory lacks verified same-source lineage: ' + row['id'])
        if source['audio_sha256'] != row['group_id']:
            raise ValueError('Source group must be the original Offline SHA256')
    return sorted(result, key=lambda r: r['id'])


def record_name(row):
    return 'records/' + digest_json(lineage(row)) + '.json'


def _safe_record(root, relative):
    p = (root / relative).resolve()
    try: p.relative_to(root)
    except ValueError: raise ValueError('Phoneme record escapes its cache directory') from None
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Expected relative phoneme record path')
    return p


def validate_record(record, row):
    expected = lineage(row)
    if (record.get('schema') != SCHEMA or record.get('identity') != expected or
            record.get('model_id') != MODEL_ID or record.get('model_revision') != MODEL_REVISION or
            record.get('algorithm') != ALGORITHM):
        raise ValueError('Phoneme record identity/algorithm changed: ' + row['id'])
    prev_end = 0; instances = set()
    for s in record.get('segments', []):
        a, b = s.get('start_sample'), s.get('end_sample')
        posterior = s.get('posterior')
        if (type(a) is not int or type(b) is not int or not 0 <= a < b <= expected['num_samples'] or
                a < prev_end or not isinstance(s.get('token'), str) or not s['token'] or
                type(s.get('instance')) is not int or s['instance'] in instances or
                not isinstance(posterior, (int, float)) or not math.isfinite(posterior) or not 0 <= posterior <= 1):
            raise ValueError('Invalid phoneme spans: ' + row['id'])
        prev_end = b; instances.add(s['instance'])
    if not isinstance(record.get('segments'), list): raise ValueError('Missing phoneme segments')
    return record


def bind_manifest(manifest_path, rows, *, verify_files=True):
    """Bind compact cache identities without rereading every original waveform.

    Preparation verifies waveform SHA256. Training's existing waveform loader
    checks saved size/mtime. Individual metadata hashes are verified on first
    access by PhonemeCache, including after every worker restart.
    """
    path = Path(manifest_path).expanduser().resolve(); m = read_json(path)
    expected = inventory(rows)
    if (m.get('schema') != SCHEMA or m.get('complete') is not True or
            m.get('model_id') != MODEL_ID or m.get('model_revision') != MODEL_REVISION or
            m.get('algorithm') != ALGORITHM or m.get('inventory_sha256') != digest_json(expected) or
            m.get('inventory') != expected):
        raise ValueError('Phoneme manifest differs from the original Train inventory')
    entries = m.get('records', {})
    if set(entries) != {r['id'] for r in expected}: raise ValueError('Incomplete phoneme manifest')
    for row in rows:
        entry = entries[row['id']]
        if entry.get('path') != record_name(row) or len(entry.get('sha256', '')) != 64:
            raise ValueError('Phoneme record identity changed')
        if verify_files and not _safe_record(path.parent, entry['path']).is_file():
            raise FileNotFoundError('Missing phoneme metadata for ' + row['id'])
    return dict(path=str(path), sha256=digest_file(path), inventory_sha256=m['inventory_sha256'],
                model_id=MODEL_ID, model_revision=MODEL_REVISION, records=len(entries),
                schema=SCHEMA, algorithm=ALGORITHM)


class PhonemeCache:
    """Small per-worker JSON LRU. No waveform/feature/model state is retained."""
    def __init__(self, manifest_path, rows, max_records=256, *, binding=None):
        # Root configuration already stat-checked every metadata file. Workers
        # can bind the exact saved manifest hash without repeating tens of
        # thousands of filesystem stats. Individual bytes still hash on get().
        self.binding = bind_manifest(manifest_path, rows, verify_files=binding is None)
        if binding is not None and self.binding != binding:
            raise ValueError('Saved phoneme provenance differs from the worker manifest')
        self.root = Path(self.binding['path']).parent
        self.entries = read_json(self.binding['path'])['records']
        self.rows = {r['id']: r for r in rows}; self.cache = OrderedDict()
        self.max_records = max(1, int(max_records))

    def get(self, row):
        key = row['id']
        if key not in self.rows or lineage(row) != lineage(self.rows[key]):
            raise ValueError('Unknown phoneme source')
        if key not in self.cache:
            entry = self.entries[key]; path = _safe_record(self.root, entry['path'])
            if digest_file(path) != entry['sha256']: raise ValueError('Phoneme metadata checksum changed: ' + key)
            self.cache[key] = validate_record(read_json(path), row)
            if len(self.cache) > self.max_records: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


def map_segments(segments, time_map, output_samples=None):
    """Apply exact affine ORIGINAL -> CURRENT 16 kHz sample coordinates.

    output = input * scale + offset. Only completely retained phone intervals
    survive cropping; their instance IDs never change. Ceil/floor exclude
    uncertain fractional edge samples instead of stretching a clipped phone.
    """
    scale, offset = float(time_map['scale']), float(time_map['offset'])
    total = int(time_map['output_samples'] if output_samples is None else output_samples)
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(offset) or total < 1:
        raise ValueError('Invalid phoneme time map')
    result = []
    for s in segments:
        a = s['start_sample'] * scale + offset; b = s['end_sample'] * scale + offset
        if a < -1e-7 or b > total + 1e-7: continue
        start, end = max(0, math.ceil(a - 1e-7)), min(total, math.floor(b + 1e-7))
        if end <= start: continue
        value = dict(s, start_sample=start, end_sample=end)
        for field in ('support_start_sample', 'support_end_sample'):
            if field in s: value[field] = int(round(s[field] * scale + offset))
        result.append(value)
    return result


def match_segments(a, b):
    """Only token pairs uniquely determined across ALL optimal LCS alignments.

    A token appearing once does not suffice: [a,b] vs [b,a] has two ambiguous
    optimal alignments. Each retained pair is the sole candidate at its LCS
    rank. Repeated phones are retained when context uniquely fixes instances.
    Returns indices into the input lists, not original instance IDs.
    """
    x, y = [s['token'] for s in a], [s['token'] for s in b]
    n, m = len(x), len(y)
    if n * m > 4_000_000: raise ValueError('Unusually long phoneme sequence; inspect preparation')
    pre = np.zeros((n + 1, m + 1), dtype=np.int32)
    suf = np.zeros_like(pre)
    for i in range(n):
        for j in range(m):
            pre[i + 1, j + 1] = pre[i, j] + 1 if x[i] == y[j] else max(pre[i, j + 1], pre[i + 1, j])
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            suf[i, j] = suf[i + 1, j + 1] + 1 if x[i] == y[j] else max(suf[i + 1, j], suf[i, j + 1])
    length = int(pre[n, m]); candidates = [[] for _ in range(length)]
    for i in range(n):
        for j in range(m):
            if x[i] == y[j] and pre[i, j] + 1 + suf[i + 1, j + 1] == length:
                candidates[int(pre[i, j])].append((i, j))
    return [items[0] for items in candidates if len(items) == 1]


def greedy_tokens(log_probs, blank_id, ignored_ids=()):
    paths = np.asarray(log_probs).argmax(-1)
    ignored = set(ignored_ids) | {blank_id}; result = []; previous = None
    for token in paths:
        token = int(token)
        if token != previous and token not in ignored: result.append(token)
        previous = token
    return result


def ctc_align(log_probs, tokens, blank_id):
    """CTC Viterbi alignment to a supplied token sequence, O(T*U), CPU NumPy.

    Used with the recognizer's decoded sequence. Forced alignment preserves
    distinct repeated-token instances and records nonblank support explicitly.
    It does not magically make the CTC timings true phoneme boundaries.
    """
    scores = np.asarray(log_probs, dtype=np.float64)
    if scores.ndim != 2 or not len(scores) or not np.isfinite(scores).all(): raise ValueError('Invalid CTC posterior matrix')
    frames, vocab = scores.shape
    if not 0 <= blank_id < vocab or any(not 0 <= t < vocab or t == blank_id for t in tokens):
        raise ValueError('Invalid CTC token sequence')
    if not tokens: return []
    states = np.full(2 * len(tokens) + 1, blank_id, dtype=np.int64); states[1::2] = tokens
    if frames * len(states) > 50_000_000: raise ValueError('CTC alignment too large; use bounded preparation chunks')
    previous = np.full(len(states), -np.inf); previous[0] = scores[0, blank_id]
    if frames: previous[1] = scores[0, tokens[0]]
    back = np.zeros((frames, len(states)), dtype=np.uint8)
    skip_ok = (np.arange(len(states)) >= 2) & (states != blank_id) & (states != np.roll(states, 2))
    for t in range(1, frames):
        one = np.concatenate(([-np.inf], previous[:-1]))
        two = np.concatenate(([-np.inf, -np.inf], previous[:-2])); two[~skip_ok] = -np.inf
        choices = np.stack((previous, one, two)); transitions = choices.argmax(0)
        previous = choices[transitions, np.arange(len(states))] + scores[t, states]
        back[t] = transitions
    state = len(states) - 1 if previous[-1] > previous[-2] else len(states) - 2
    if not np.isfinite(previous[state]): raise ValueError('Token sequence has no valid CTC alignment')
    assignment = np.empty(frames, dtype=np.int32)
    for t in range(frames - 1, -1, -1):
        assignment[t] = state
        if t: state -= int(back[t, state])
    result = []
    for i, token in enumerate(tokens):
        used = np.flatnonzero(assignment == 2 * i + 1)
        if not len(used): raise ValueError('CTC token received no acoustic support')
        result.append(dict(token_id=int(token), instance=i, start_frame=int(used[0]),
                           end_frame=int(used[-1] + 1), posterior=float(np.exp(scores[used, token]).mean())))
    return result


def segment_support(support, id_to_token, num_samples, stride=320, receptive_field=400):
    """Approximate phone intervals with bounded interpolation across CTC blanks.

    Long blank gaps remain gaps. Initial/final silence is not assigned to the
    first/last token. Extremely diffuse/low-posterior token spans are dropped.
    """
    if not support: return []
    centers = [(s['start_frame'] + s['end_frame'] - 1) * stride / 2 + receptive_field / 2 for s in support]
    result = []; gap = ALGORITHM['max_half_gap_samples']
    for i, s in enumerate(support):
        a = s['start_frame'] * stride; b = (s['end_frame'] - 1) * stride + receptive_field
        left = (centers[i - 1] + centers[i]) / 2 if i else a
        right = (centers[i] + centers[i + 1]) / 2 if i + 1 < len(support) else b
        left = max(left, a - gap, 0); right = min(right, b + gap, num_samples)
        start, end = int(math.ceil(left)), int(math.floor(right))
        if end <= start or end - start > ALGORITHM['max_segment_samples'] or s['posterior'] < ALGORITHM['min_token_posterior']:
            continue
        result.append(dict(token=str(id_to_token[s['token_id']]), instance=s['instance'],
                           start_sample=start, end_sample=end, posterior=s['posterior'],
                           support_start_sample=a, support_end_sample=min(b, num_samples),
                           alignment='ctc-viterbi-bounded-midpoint-approximate'))
    return result
