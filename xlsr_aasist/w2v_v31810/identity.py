"""Canonical original-recording relations; never infer a domain from a filename."""
from collections import defaultdict
import csv
from pathlib import Path


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
        for field in ('source_id', 'group_id', 'source_sha256', 'audio_sha256'):
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
    pairs = []
    if pair_path:
        with Path(pair_path).open(encoding='utf-8-sig', newline='') as stream:
            for item in csv.DictReader(stream): pairs.append((item['offline_id'], item['online_id']))
    for row in rows:
        if row.get('offline_id') and row.get('online_id'):
            pairs.append((row['offline_id'], row['online_id']))
    matched = 0
    for off, on in pairs:
        aa = ids.get(str(off)) or basenames.get(Path(str(off)).name, set())
        bb = ids.get(str(on)) or basenames.get(Path(str(on)).name, set())
        if not aa and not bb: continue
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
        speaker_ids_known=sum(bool(r.get('speaker_id')) for r in rows),
        generator_ids_known=sum(bool(r.get('generator_id')) for r in rows),
        guarantee='Original recording isolation; unknown speaker/generator identities remain unknown')
    return keys, audit
