"""Canonical original-recording relations; never infer a domain from a filename."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

REGISTRY_SCHEMA = 'rtc_v31810_pcm_source_registry_v1'
_IDENTITY_FIELDS = ('id', 'audio', 'source_id', 'group_id', 'source_sha256',
                    'audio_sha256', 'decoded_pcm_sha256', 'canonical_source',
                    'language', 'label', 'split', 'condition')
_LEGACY_MESSAGE = ('This V3.18.10 run has no verified PCM source registry and cannot be resumed '
                   'with the corrected sampler. Start a NEW run with --data-run pointing to '
                   'the original V3.18.4 through V3.18.9 data run; do not use --resume. '
                   'Existing checkpoints remain unchanged.')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def identity_records(rows):
    """Bind identity, actual bytes/PCM and manifest ordering without audio copies."""
    return [{key: row.get(key) for key in _IDENTITY_FIELDS} for row in rows]


def build_registry(train, dev, pair_records, pairing_audit=None):
    registry = dict(schema=REGISTRY_SCHEMA, train=identity_records(train),
                    dev=identity_records(dev), official_pair_records=pair_records)
    if pairing_audit is not None:
        registry['pairing_audit'] = pairing_audit
    registry['registry_hash'] = _digest(registry)
    for row in train:
        row['source_registry_hash'] = registry['registry_hash']
        row['source_registry_schema'] = REGISTRY_SCHEMA
    return registry


def registry_config(registry):
    return dict(schema=REGISTRY_SCHEMA, registry_hash=registry['registry_hash'],
                train_identity_hash=_digest(registry['train']))


def require_registry_config(cfg):
    value = cfg.get('source_registry', {})
    if (value.get('schema') != REGISTRY_SCHEMA or not value.get('registry_hash')
            or not value.get('train_identity_hash')):
        raise ValueError(_LEGACY_MESSAGE)
    return value


def registered_sources(rows, cfg):
    """Consume the startup registry verbatim; never reconstruct source groups."""
    bound = require_registry_config(cfg)
    if _digest(identity_records(rows)) != bound['train_identity_hash']:
        raise ValueError('Train row identity/order differs from the saved PCM source registry')
    groups = defaultdict(list)
    for row in rows:
        if (row.get('source_registry_schema') != REGISTRY_SCHEMA or
                row.get('source_registry_hash') != bound['registry_hash']):
            raise ValueError('Train row is not bound to the verified PCM source registry')
        for field in ('canonical_source', 'audio_sha256', 'decoded_pcm_sha256'):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Missing actual source identity: ' + field)
        if row.get('split') != 'train':
            raise ValueError('Training registry accepts Train only')
        groups[row['canonical_source']].append(row)
    for members in groups.values():
        if len({(r['language'], r['label']) for r in members}) != 1:
            raise ValueError('Registered source has conflicting language/class labels')
        if 'online' in {r['condition'] for r in members} and 'offline' not in {r['condition'] for r in members}:
            raise ValueError('Registered Online lacks a verified Offline relationship')
    offline_only = sum({r['condition'] for r in members} == {'offline'} for members in groups.values())
    return [r['canonical_source'] for r in rows], dict(
        original_recordings=len(groups), rows=len(rows), registry_hash=bound['registry_hash'],
        offline_only_sources=offline_only, paired_sources=len(groups)-offline_only,
        schema=REGISTRY_SCHEMA, actual_pcm_verified=True,
        guarantee='Startup Train/Dev shared source registry; no independent sampler regrouping')


def verify_registry(train, registry, cfg, protocol):
    bound = require_registry_config(cfg)
    if registry.get('schema') != REGISTRY_SCHEMA:
        raise ValueError(_LEGACY_MESSAGE)
    expected = _digest({k: v for k, v in registry.items() if k != 'registry_hash'})
    if expected != registry.get('registry_hash') or expected != bound['registry_hash']:
        raise ValueError('Saved PCM source registry hash changed')
    if protocol.get('source_registry_hash') != expected:
        raise ValueError('Validation and training reference different PCM source registries')
    if identity_records(train) != registry['train']:
        raise ValueError('Train rows differ from the persisted PCM source registry')
    registered_sources(train, cfg)


def read_official_pairs(rows, pair_path=None):
    """An empty Online field documents an Offline source with no retained Online.

    It is not a pairing edge and must not discard that source. A nonempty pair
    must resolve both IDs if either endpoint belongs to the active scope. CSV missing columns
    (None rather than an explicit empty cell) remain a malformed input error.
    """
    pairs = set()
    def add(off, on, origin):
        if not isinstance(off, str) or (on is not None and not isinstance(on, str)):
            raise ValueError('Invalid official pair identifiers: ' + origin)
        off, on = off.strip(), (on or '').strip()
        if not off:
            raise ValueError('Official pair lacks a nonempty Offline ID: ' + origin)
        pairs.add((off, on))
    for row in rows:
        if row.get('offline_id') or row.get('online_id'):
            add(row.get('offline_id', ''), row.get('online_id'), 'embedded '+str(row.get('id', 'row')))
    if pair_path:
        with Path(pair_path).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'offline_id', 'online_id'} <= set(reader.fieldnames or []):
                raise ValueError('Pairing file lacks offline_id/online_id: ' + str(pair_path))
            for number, item in enumerate(reader, 2):
                if item['offline_id'] is None or item['online_id'] is None or None in item:
                    raise ValueError(f'Malformed official pair CSV row at {pair_path}:{number}')
                add(item['offline_id'], item['online_id'], f'{pair_path}:{number}')
    return sorted(pairs)


def canonical_sources(rows, pair_path=None, *, require_online_pairs=False):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)
    seen = {}; ids = defaultdict(set); basenames = defaultdict(set)
    for i, row in enumerate(rows):
        if not row.get('source_id') or not row.get('group_id'):
            raise ValueError('Canonical source_id and group_id are required; filenames cannot establish identity')
        for field in ('source_id', 'group_id', 'source_sha256', 'audio_sha256',
                      'decoded_pcm_sha256', 'canonical_source'):
            value = row.get(field)
            if value:
                key = field, str(value)
                if key in seen: union(i, seen[key])
                else: seen[key] = i
        for field in ('id', 'source_id'):
            value = str(row[field]); ids[value].add(i)
            basenames[Path(value).name].add(i)
    # Explicit pair metadata may use protocol ids or unique basenames. The
    # basename is only a lookup key for a documented pair, never an inferred pair.
    matched = 0; offline_without_online = 0; outside_scope = 0
    for off, on in read_official_pairs(rows, pair_path):
        aa = ids.get(str(off)) or basenames.get(Path(str(off)).name, set())
        bb = (ids.get(str(on)) or basenames.get(Path(str(on)).name, set())) if on else set()
        if not aa and not bb:
            outside_scope += 1
            continue
        if not on:
            if len({find(i) for i in aa}) != 1:
                raise ValueError('Ambiguous official pair endpoint: '+str((off, on)))
            offline_without_online += 1
            continue
        if not aa or not bb:
            raise ValueError('Official pair is missing one endpoint: '+str((off, on)))
        for candidates in (aa, bb):
            if len({find(i) for i in candidates}) != 1:
                raise ValueError('Ambiguous official pair endpoint: '+str((off, on)))
        union(min(aa), min(bb)); matched += 1
    members = defaultdict(list)
    for i in range(len(rows)): members[find(i)].append(i)
    keys = [None] * len(rows)
    for indices in members.values():
        if len({(rows[i]['language'], rows[i]['label']) for i in indices}) != 1:
            raise ValueError('Canonical original recording has conflicting language/class labels')
        if len({rows[i].get('split') for i in indices}) != 1:
            raise ValueError('Canonical original recording crosses data splits')
        conditions = {rows[i].get('condition') for i in indices}
        if require_online_pairs and 'online' in conditions and 'offline' not in conditions:
            raise ValueError('Official Online lacks a verified Offline relationship: '+str(rows[indices[0]]['id']))
        key = min(str(rows[i]['source_id']) for i in indices)
        for i in indices: keys[i] = key
    audit = dict(original_recordings=len(members), rows=len(rows), official_pair_records=matched,
        offline_without_online_records=offline_without_online, outside_scope_records=outside_scope,
        speaker_ids_known=sum(bool(r.get('speaker_id')) for r in rows),
        generator_ids_known=sum(bool(r.get('generator_id')) for r in rows),
        guarantee='Original recording isolation; unknown speaker/generator identities remain unknown')
    return keys, audit
