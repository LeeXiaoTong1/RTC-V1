"""Locked source/noise-isolated model selection, without target adaptation.

Only the select scalar chooses a checkpoint.  Audit is a one-time measurement
of the already selected main model; historical Dev is not an external test.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

GROUPS = (('en', 0), ('en', 1), ('zh', 0), ('zh', 1))
SCENES = ('office', 'coffee', 'rain', 'footsteps', 'keyboard', 'echo')
FAMILIES = ('webrtc', 'ffmpeg', 'light')
PANEL_WEIGHTS = {'official_online': .30, 'regular_noisy': .35, 'heldout_noisy': .35}
SCHEMA = 'rtc_v31810_locked_validation_v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class _Union:
    def __init__(self):
        self.parents = {}

    def find(self, value):
        self.parents.setdefault(value, value)
        if self.parents[value] != value:
            self.parents[value] = self.find(self.parents[value])
        return self.parents[value]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


class _AudioIdentity:
    """One startup pass: verify bytes and exact decoded PCM, not filenames."""
    def __init__(self):
        self.cache = {}
        self.input_files = {}

    def file(self, path):
        path = str(Path(path).expanduser().resolve())
        if path not in self.input_files:
            digest = hashlib.sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            self.input_files[path] = digest.hexdigest()
        return self.input_files[path]

    def audio(self, path, expected=None):
        import soundfile as sf
        path = str(Path(path).expanduser().resolve())
        if path not in self.cache:
            byte_hash = self.file(path)
            pcm = hashlib.sha256()
            with sf.SoundFile(path) as stream:
                if stream.frames < 1:
                    raise ValueError('Validation provenance contains empty audio: ' + path)
                pcm.update(f'{stream.samplerate}:{stream.channels}:{stream.frames}:'.encode())
                for block in stream.blocks(blocksize=262144, dtype='float32', always_2d=True):
                    if not np.isfinite(block).all():
                        raise ValueError('Nonfinite audio in validation provenance: ' + path)
                    pcm.update(np.asarray(block, dtype='<f4').tobytes())
            stat = Path(path).stat()
            self.cache[path] = dict(path=path, sha256=byte_hash, decoded_pcm_sha256=pcm.hexdigest(),
                                    size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        result = self.cache[path]
        if expected and expected != result['sha256']:
            raise ValueError('Audio hash differs from its prepared manifest: ' + path)
        return result


def _speech_partition(cfg, train, dev, official, identity):
    graph = _Union()
    values = []
    tokens = {}
    # Old simulated Dev files are deliberately not opened or reused as Offline.
    # Their source metadata connects variants to actual official recordings.
    for split, rows, raw in (('train', train, True), ('dev', dev, False), ('dev', official, True)):
        for row in rows:
            item = dict(row, _split=split)
            node = f'row:{len(values)}'
            graph.find(node)
            aliases = [('id', row['id'])]
            for field in ('source_id', 'group_id', 'source_sha256'):
                if row.get(field):
                    namespace = 'id' if field == 'source_id' else 'sha256' if field=='source_sha256' else field
                    aliases.append((namespace, str(row[field])))
            if raw:
                info = identity.audio(row['audio'], row.get('audio_sha256'))
                aliases.extend((('sha256', info['sha256']), ('pcm', info['decoded_pcm_sha256'])))
                item.update(audio_sha256=info['sha256'], decoded_pcm_sha256=info['decoded_pcm_sha256'],
                            audio_size=info['size'], audio_mtime_ns=info['mtime_ns'])
            for alias in aliases:
                if alias in tokens:
                    graph.join(node, tokens[alias])
                else:
                    tokens[alias] = node
            values.append(item)
    # Official pair IDs can contain an extra dataset prefix. Only exact IDs or
    # a unique explicitly listed basename may resolve; no name-pattern guessing.
    ids = defaultdict(set)
    stems = defaultdict(set)
    for index, row in enumerate(values):
        # Resolve pairs against real official recordings. Older noisy metadata
        # may use a different dataset path prefix and is not a second candidate.
        if index >= len(train)+len(dev):
            for key in ('id', 'source_id'):
                if row.get(key):
                    ids[row[key]].add(f'row:{index}')
                    stems[Path(row[key]).name].add(f'row:{index}')
    def resolve(value):
        matches = ids.get(value) or stems.get(Path(value).name, set())
        roots = {graph.find(node) for node in matches}
        if len(roots) > 1:
            raise ValueError('Ambiguous official Dev pair ID: ' + value)
        return next(iter(roots), None)
    pair_path = Path(cfg['dev_pairs'])
    identity.file(pair_path)
    with pair_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'offline_id', 'online_id'} <= set(reader.fieldnames or []):
            raise ValueError('Dev pairing file lacks offline_id/online_id')
        for pair in reader:
            left, right = resolve(pair['offline_id']), resolve(pair['online_id'])
            if left is not None and right is not None:
                graph.join(left, right)
            elif (left is None) != (right is None):
                raise ValueError('Official Dev pair has a missing member: ' + str(pair))
    components = defaultdict(list)
    for index, row in enumerate(values):
        components[graph.find(f'row:{index}')].append(row)
    source_keys = {}
    for root, members in components.items():
        if {r['_split'] for r in members} == {'train', 'dev'}:
            raise ValueError('Train/Dev speech source or exact audio content overlaps')
        if len({(r['language'], r['label']) for r in members}) != 1:
            raise ValueError('Canonical speech source has conflicting language/class metadata')
        source_keys[root] = _digest(sorted({(r['_split'], r['id'], r.get('audio_sha256', '')) for r in members}))
    start = len(train) + len(dev)
    prepared = []
    for index in range(start, len(values)):
        row = dict(values[index]); row.pop('_split')
        source = source_keys[graph.find(f'row:{index}')]
        row.update(canonical_source=source, source_id=row['id'], group_id=source, split='dev',
                   condition=row['domain'], full_length=True)
        prepared.append(row)
    strata = defaultdict(set)
    for row in prepared:
        strata[row['language'], row['label']].add(row['canonical_source'])
    fraction = float(cfg.get('validation_audit_fraction', .2))
    if not 0 < fraction < 1:
        raise ValueError('Audit source fraction must be between zero and one')
    audit = set()
    for group in GROUPS:
        ordered = sorted(strata[group], key=lambda key: _digest((cfg['seed'], 'locked-dev', key)))
        if len(ordered) < 2:
            raise ValueError('Need at least two independent Dev sources in ' + str(group))
        count = max(1, min(len(ordered)-1, round(len(ordered)*fraction)))
        audit.update(ordered[:count])
    result = {'select': [], 'audit': []}
    for row in prepared:
        result['audit' if row['canonical_source'] in audit else 'select'].append(row)
    for split, rows in result.items():
        for domain in ('online', 'offline'):
            groups = {(r['language'], r['label']) for r in rows if r['domain'] == domain}
            if groups != set(GROUPS):
                raise ValueError(f'{split} lacks genuine official {domain} speech in every language/class; cannot synthesize missing Offline')
    return result, dict(raw_audio_hashing='SHA256 bytes plus exact decoded float32 PCM; no speaker identity claim',
                       source_counts={s: len({r['canonical_source'] for r in rows}) for s, rows in result.items()},
                       original_official_rows=len(official), historical_cached_noisy_used=False,
                       train_sources=sum(any(r['_split']=='train' for r in members) for members in components.values()),
                       unknown_identity='No inferred speaker, corpus, or generator domain labels')


def split_noise_records(cfg, identity=None):
    """Keep all slices and composed materials of an original source together."""
    identity = identity or _AudioIdentity()
    records = cfg.get('scene_noise_records', {})
    if not records.get('train') or not records.get('dev'):
        raise ValueError('Prepared source-attributed Train/Dev scene noise is required')
    graph = _Union(); seen = {}; values = []
    for split in ('train', 'dev'):
        for raw in records[split]:
            if not raw.get('original_sources') or not raw.get('source_group'):
                raise ValueError('Noise requires original recording/slice provenance')
            info = identity.audio(raw['path'], raw.get('sha256'))
            node = f'noise:{len(values)}'; graph.find(node)
            aliases = [('source', s) for s in raw['original_sources']]
            aliases += [('group', raw['source_group']), ('pcm', info['decoded_pcm_sha256']), ('sha', info['sha256'])]
            for alias in aliases:
                if alias in seen: graph.join(node, seen[alias])
                else: seen[alias] = node
            values.append(dict(raw, **info, _split=split))
    components = defaultdict(list)
    for index, row in enumerate(values): components[graph.find(f'noise:{index}')].append(row)
    for rows in components.values():
        if {r['_split'] for r in rows} == {'train', 'dev'}:
            raise ValueError('Train/Dev noise original source or actual decoded content overlaps')
    dev = {key: rows for key, rows in components.items() if rows[0]['_split'] == 'dev'}
    categories = {c: {k for k, rows in dev.items() if any(r['category'] == c for r in rows)} for c in SCENES[:-1]}
    for category, keys in categories.items():
        if len(keys) < 2:
            raise ValueError(f'Noise category {category} has only {len(keys)} independent Dev source component(s); use the complete RNNoise bank, not slices of one recording')
    # Find one disjoint select/audit anchor per category. Mixed-source materials
    # can connect categories; backtracking preserves all connected components.
    order = sorted(categories, key=lambda c: (len(categories[c]), c))
    def anchors(position, assigned):
        if position == len(order): return assigned
        keys = sorted(categories[order[position]], key=lambda k: _digest((cfg['seed'], 'noise-split', k)))
        if {assigned.get(k) for k in keys} >= {'select', 'audit'}:
            return anchors(position+1, assigned)
        for a in keys:
            if assigned.get(a) not in (None, 'audit'): continue
            for b in keys:
                if a == b or assigned.get(b) not in (None, 'select'): continue
                answer = anchors(position+1, dict(assigned, **{a: 'audit', b: 'select'}))
                if answer is not None: return answer
        return None
    assigned = anchors(0, {})
    if assigned is None:
        raise ValueError('Overlapping noise provenance cannot provide all categories in disjoint select/audit banks')
    fraction = float(cfg.get('validation_audit_fraction', .2))
    for category in order:
        keys = sorted(categories[category], key=lambda k: _digest((cfg['seed'], 'noise-remainder', k)))
        target = max(1, min(len(keys)-1, round(len(keys)*fraction)))
        for key in keys:
            if key not in assigned:
                assigned[key] = 'audit' if sum(assigned.get(k)=='audit' for k in keys) < target else 'select'
    output = {'select': [], 'audit': []}
    for key, rows in dev.items():
        for row in rows:
            item = dict(row); item.pop('_split')
            item['original_component'] = _digest(sorted({s for r in rows for s in r['original_sources']}))
            output[assigned[key]].append(item)
    summary = {s: {c: len({r['original_component'] for r in rows if r['category']==c})
                   for c in SCENES[:-1]} for s, rows in output.items()}
    return output, summary


def _panels(rows, split):
    output = []; tickets = []; panel_indices = {name: [] for name in PANEL_WEIGHTS}
    def add(row, panel, **options):
        index = len(output)
        output.append(dict(row, validation_panel=panel, validation_split=split))
        panel_indices[panel].append(index)
        tickets.append(dict(index=index, epoch=0, step=0, probe=True,
                            occurrence='v31810-fixed:'+split+':'+row['canonical_source'], **options))
    for row in rows:
        if row['domain'] == 'online': add(row, 'official_online', kind='online')
    for group in GROUPS:
        # Exactly one genuine Offline recording per canonical source. Selecting
        # by fixed identity is independent of any prediction or score.
        sources = {}
        for row in sorted(rows, key=lambda r: r['id']):
            if row['domain']=='offline' and (row['language'], row['label'])==group:
                sources.setdefault(row['canonical_source'], row)
        for n, source in enumerate(sorted(sources)):
            row = sources[source]; scene = SCENES[n % len(SCENES)]
            # Three families appear immediately; across 18 sources every scene
            # receives every family, without assigning conditions by class.
            family = FAMILIES[(n + n // len(SCENES)) % len(FAMILIES)]
            add(row, 'regular_noisy', kind='offline', scene=scene, family=family)
            add(row, 'heldout_noisy', kind='offline', scene=scene,
                family='anlmdn' if split=='select' else 'g711_mulaw')
    return dict(rows=output, tickets=tickets, panels=panel_indices,
                condition_coverage={name: dict(scenes=sorted({tickets[i].get('scene') for i in indices
                                                              if tickets[i].get('scene')}),
                                                families=sorted({tickets[i].get('family') for i in indices
                                                                 if tickets[i].get('family')}))
                                    for name, indices in panel_indices.items()},
                speech_sources=sorted({r['canonical_source'] for r in rows}))


def build_protocol(cfg, train, dev, source_run):
    from w2v_aasist.data import read_protocol
    identity = _AudioIdentity()
    official = read_protocol(cfg['official_dev_protocol'], cfg['official_dev_root'])
    identity.file(cfg['official_dev_protocol'])
    speech, audit = _speech_partition(cfg, train, dev, official, identity)
    noise, noise_audit = split_noise_records(cfg, identity)
    if cfg.get('scene_noise_manifest'): identity.file(cfg['scene_noise_manifest'])
    result = dict(schema=SCHEMA, seed=cfg['seed'], threshold=.5, calibration='none',
                  selection_weights=dict(PANEL_WEIGHTS), source_run=str(source_run),
                  evaluation='Unadapted main detector only; no gradients or target-time updates',
                  historical_dev_warning='This Dev was used during earlier development; its newly locked audit is not a previously unseen external benchmark',
                  audit_rule='Evaluate only the selected completed checkpoint after training; never choose epoch, alpha, beta or thresholds using audit results',
                  speech_provenance=audit, noise_provenance=noise_audit,
                  noise_partition_policy='Connected original sources, composed-source provenance and actual decoded PCM; no split of slices')
    for split in ('select', 'audit'):
        result[split] = dict(_panels(speech[split], split), scene_noise_records=noise[split])
    result['input_files'] = dict(sorted(identity.input_files.items()))
    result['protocol_hash'] = _digest(result)
    return result


def binary_metrics(rows, logits):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray([r['label'] for r in rows], dtype=np.int64)
    if z.shape != (len(rows), 2) or not np.isfinite(z).all() or set(y.tolist()) != {0, 1}:
        raise ValueError('Validation needs finite two-class logits and both real/fake labels')
    # p(fake)>=0.5; class order is fake=0, real=1. No fitted affine or threshold.
    prediction = np.where(z[:, 0] >= z[:, 1], 0, 1)
    f1 = []; recalls = []
    for label in (0, 1):
        true = y == label; predicted = prediction == label
        tp = int((true & predicted).sum()); fp = int((~true & predicted).sum()); fn = int((true & ~predicted).sum())
        f1.append(2*tp / max(1, 2*tp+fp+fn)); recalls.append(tp/int(true.sum()))
    return dict(count=len(rows), macro_f1=float(np.mean(f1)), fake_f1=f1[0], real_f1=f1[1],
                recall=recalls, fake_recall=recalls[0], real_recall=recalls[1], real_false_positive=1-recalls[1],
                fake_false_negative=1-recalls[0], threshold=.5,
                real_count=int((y==1).sum()), fake_count=int((y==0).sum()))


def summarize_protocol(protocol, logits, split='select'):
    if split not in ('select', 'audit'): raise ValueError('Unknown validation partition')
    part = protocol[split]; rows = part['rows']; z = np.asarray(logits)
    if z.shape != (len(rows), 2): raise ValueError('Validation output coverage differs from fixed protocol')
    panels = {}; groups = {}
    for panel, indices in part['panels'].items():
        panel_rows = [rows[i] for i in indices]
        panels[panel] = binary_metrics(panel_rows, z[indices])
        for language in ('en', 'zh'):
            selected = [i for i in indices if rows[i]['language']==language]
            groups[panel+'/'+language] = binary_metrics([rows[i] for i in selected], z[selected])
    score = sum(PANEL_WEIGHTS[p]*panels[p]['macro_f1'] for p in PANEL_WEIGHTS) if split=='select' else None
    return dict(split=split, score=score, panels=panels, groups=groups, threshold=.5,
                calibration='none', main_model_only=True, protocol_hash=protocol['protocol_hash'],
                selection_eligible=split=='select')


def evaluate_protocol(model, protocol, cfg, split='select', *, infer_fn=None, selected_checkpoint=None):
    if split=='audit' and not selected_checkpoint:
        raise ValueError('Locked audit requires the already selected checkpoint identity')
    if getattr(model, '_v31810_virtual_active', False):
        raise ValueError('Validation cannot evaluate temporary adapted parameters')
    if infer_fn is None:
        from .validation_data import infer_fixed
        infer_fn = infer_fixed
    part = protocol[split]
    local_cfg = dict(cfg)
    local_cfg['scene_noise_records'] = {'dev': part['scene_noise_records']}
    z, observed = infer_fn(model, part['rows'], local_cfg, part['tickets'], 'dev')
    expected = [(r['id'], r['validation_panel'], r['canonical_source']) for r in part['rows']]
    actual = [(r['id'], r['validation_panel'], r['canonical_source']) for r in observed]
    if actual != expected: raise ValueError('Validation row order/coverage changed')
    result = summarize_protocol(protocol, z, split)
    if selected_checkpoint: result['selected_checkpoint'] = str(selected_checkpoint)
    return result
