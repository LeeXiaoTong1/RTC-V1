"""Full-utterance official Train/Dev, plus existing role-validated noisy caches."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .runtime import sha256
from .composition import CUT, MODES, compose

LABELS = {'fake': 0, 'spoof': 0, 'real': 1, 'bonafide': 1, 'bona-fide': 1}


def check_processing(cfg, rows, role):
    """Validate newer processing caches even with an older V2 cache reader."""
    processing = cfg.get('processing')
    if processing:
        profile = processing.get('profile')
        families = {'diverse': ['ffmpeg', 'webrtc'], 'unseen': ['anlmdn']}
        expected = {'schema': 'rtc_diverse_processing_v1', 'profile': profile,
                    'families': families.get(profile), 'room_probability': .5,
                    'intermittent_probability': .5,
                    'room': 'causal sparse reflections plus decaying synthetic impulse response; same length',
                    'snr_reference': 'active-frame SNR before room and RTC processing'}
        if profile not in families or processing != expected or (profile == 'unseen') != (role == 'dev_heldout'):
            raise ValueError('Unknown or role-incompatible processing recipe')
        if profile == 'diverse' and cfg.get('webrtc_version') != '0.1.3':
            raise ValueError('Unexpected WebRTC cache implementation')
    for row in rows:
        if processing and row.get('processing', {}).get('family') not in processing['families']:
            raise ValueError('Missing or forbidden processing family')
        if not processing and 'processing' in row:
            raise ValueError('Diverse row disguised as legacy cache')


def safe_audio_path(root, name):
    relative = PurePosixPath(name.replace('\\', '/'))
    if relative.is_absolute() or '..' in relative.parts or ':' in name:
        raise ValueError('Audio ID must be a safe relative path: ' + name)
    root = Path(root).resolve()
    path = (root / str(relative)).resolve()
    path.relative_to(root)
    return path


def read_protocol(path, root, labeled=True):
    records, seen = [], set()
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if labeled:
            if len(parts) == 2:
                name, label = parts
            elif len(parts) >= 5:
                name, label = parts[1], parts[4]
            else:
                raise ValueError(f'Missing Train/Dev label at {path}:{line_no}')
            if label.lower() not in LABELS:
                raise ValueError('Unknown label: ' + label)
            label = LABELS[label.lower()]
        else:
            if len(parts) != 1:
                raise ValueError('Submission inference requires an unlabeled ID-only protocol')
            name, label = parts[0], -1
        name = name.replace('\\', '/')
        if name in seen:
            raise ValueError('Duplicate protocol ID: ' + name)
        seen.add(name)
        audio = safe_audio_path(root, name)
        if not audio.is_file():
            raise FileNotFoundError(audio)
        components = set(PurePosixPath(name).parts)
        records.append({'id': name, 'audio': str(audio), 'label': label,
                        'language': 'en' if 'en' in components else 'zh' if 'zh' in components else 'unknown',
                        'domain': 'online' if 'online' in components else 'offline' if 'offline' in components else 'unknown',
                        'noisy': False, 'band': -1})
    if not records or labeled and {r['label'] for r in records} != {0, 1}:
        raise ValueError('Empty protocol or missing fake/real class')
    return records


def cache_index(folder, role, protocol, official_records):
    folder = Path(folder).resolve()
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if cfg.get('format') == 'rtc_noisy_full2_cache_v1':
        from .full_cache import read_index
        return read_index(folder, role, protocol, official_records)
    expected_split = 'train' if role == 'train' else 'dev'
    if (cfg.get('format') != 'rtc_noisy_pair_cache_v1' or cfg.get('role') != role
            or cfg.get('split') != expected_split or cfg.get('protocol_sha256') != sha256(protocol)
            or cfg.get('limit') or cfg.get('sr') != 16000 or cfg.get('cut') != 64600
            or cfg.get('snr_bands') != [[5., 10.], [10., 15.], [15., 20.], [20., 25.]]):
        raise ValueError(f'Cache role/protocol/format/smoke mismatch: {folder}')
    official = {r['id']: r for r in official_records if r['domain'] == 'offline'}
    rows, seen, raw_rows = [], set(), []
    with (folder / 'manifest.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            source, band = row['source'], row['band']
            if (source not in official or type(band) is not int or band not in range(4)
                    or type(row['label']) is not int or row['label'] != official[source]['label']
                    or (source, band) in seen):
                raise ValueError('Invalid/duplicate noisy source, class or band')
            if not cfg['snr_bands'][band][0] <= row['snr_db'] <= cfg['snr_bands'][band][1]:
                raise ValueError('Noisy SNR differs from its band')
            path = safe_audio_path(folder, row['audio'])
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append({**official[source], 'audio': str(path), 'source_audio': official[source]['audio'],
                         'source_sha256': row['source_sha256'], 'noisy': True,
                         'band': band, 'bank': str(folder), 'processing_family':
                         row.get('processing', {}).get('family', 'ffmpeg')})
            raw_rows.append(row)
            seen.add((source, band))
            if len(rows) % 20000 == 0:
                print(f'Checking {role} cache: {len(rows)} entries', flush=True)
    if len(seen) != 4 * len(official) or cfg.get('offline_count') != len(official):
        raise ValueError('Noisy cache is incomplete')
    # Retain the established role, simulator split and heldout-family checks.
    from rtc_noisy_v2.cache import check_metadata
    check_metadata(cfg, raw_rows, role)
    check_processing(cfg, raw_rows, role)
    return rows, (raw_rows, cfg)


def read_wave(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    wave, sr = sf.read(path, dtype='float32', always_2d=True)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Empty/non-finite waveform: ' + str(path))
    wave = wave.mean(1)
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        wave = resample_poly(wave, 16000 // divisor, sr // divisor).astype(np.float32)
    return wave


def stable_seed(seed, epoch, index):
    raw = f'{seed}:{epoch}:{index}'.encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=4).digest(), 'little')


class AudioDataset(Dataset):
    def __init__(self, records, *, training=False, epoch=0, seed=1234, max_seconds=0., rawboost=0, raw_config=None, legacy_prefix=False):
        self.records = records
        self.training, self.epoch, self.seed = training, epoch, seed
        self.max_samples = round(max_seconds * 16000)
        self.rawboost, self.raw_config = rawboost, raw_config or {}
        if max_seconds:
            raise ValueError('This recipe preserves full utterances; max_seconds must be zero')
        self.verified_sources = set()
        self.legacy_prefix = legacy_prefix
        if training and legacy_prefix:
            raise ValueError('Legacy prefix is only for evaluating archived checkpoints')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        ticket = None
        if isinstance(index, (tuple, list)):
            index, partner, mode, ticket = index
            if not self.training:
                raise ValueError('Compositions are Train-only')
        row = self.records[index]
        wave = read_wave(row['audio'])
        if row.get('full_length') and len(wave) != row['output_samples']:
            raise ValueError('Full noisy cache length changed')
        if row['noisy'] and not row.get('full_length') and len(wave) != 64600:
            raise ValueError('Existing noisy cache must have exactly 64600 samples')
        if self.legacy_prefix and not row['noisy']:
            wave = np.tile(wave, math.ceil(CUT / len(wave)))[:CUT]
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, (index, ticket)))
        composition = 'single' if row['noisy'] else 'ordinary'
        if ticket is not None:
            other = self.records[partner]
            if (not row['noisy'] or not other['noisy'] or row['id'] != other['id']
                    or row['label'] != other['label'] or row['source_sha256'] != other['source_sha256']):
                raise ValueError('Only cached views of the SAME labeled source can be composed')
            second = read_wave(other['audio']) if mode == 'switch' else None
            original = None
            if mode == 'prefix_tail':
                source = row['source_audio']
                if source not in self.verified_sources:
                    if sha256(source) != row['source_sha256']:
                        raise ValueError('Original source no longer matches the noisy cache')
                    self.verified_sources.add(source)
                original = read_wave(source)
            if row.get('full_length'):
                if mode != 'full' or index != partner:
                    raise ValueError('Full noisy recipe uses one complete version per ticket')
                composition = 'full_v' + str(row['version'])
            else:
                wave, composition = compose(wave, second, original, mode, rng)
        if self.max_samples and len(wave) > self.max_samples:
            start = int(rng.integers(len(wave) - self.max_samples + 1)) if self.training else 0
            wave = wave[start:start + self.max_samples]
        if self.training and self.rawboost and not row['noisy']:
            from utils.data_utils import process_rawboost_feature
            py, old_np = random.getstate(), np.random.get_state()
            try:
                draw = int(rng.integers(2**32))
                random.seed(draw)
                np.random.seed(draw)
                wave = process_rawboost_feature(wave, 16000, SimpleNamespace(**self.raw_config), self.rawboost)
            finally:
                random.setstate(py)
                np.random.set_state(old_np)
        # AASIST requires >=12 frames; keep all real content and extend only <0.4-s clips.
        if len(wave) < 6400:
            wave = np.tile(wave, math.ceil(6400 / len(wave)))[:6400]
        wave = np.asarray(wave, dtype=np.float32)
        if not np.isfinite(wave).all():
            raise FloatingPointError('Augmentation produced non-finite audio')
        return {**row, 'wave': wave, 'composition': composition,
                'audio_seconds': len(wave) / 16000}


class FeatureCollator:
    def __init__(self, ssl_path):
        self.ssl_path, self.extractor = str(ssl_path), None

    def __call__(self, rows):
        if self.extractor is None:
            from transformers import AutoFeatureExtractor
            self.extractor = AutoFeatureExtractor.from_pretrained(self.ssl_path, local_files_only=True)
            if self.extractor.sampling_rate != 16000:
                raise ValueError('Expected official 16 kHz feature extractor')
        examples = []
        # Per-utterance CMVN and masks; no batch-dependent audio normalization.
        for row in rows:
            features = self.extractor(row['wave'], sampling_rate=16000, padding=True,
                                      pad_to_multiple_of=2, return_attention_mask=True, return_tensors='pt')
            f, m = features['input_features'], features['attention_mask'].long()
            length = int(m.sum())
            if length < 2 or f.shape[-1] != 160 or not bool(m[:, :length].bool().all()):
                raise ValueError('Invalid official filterbank output')
            f, m = f[:, :length].contiguous().float(), m[:, :length].contiguous()
            if not bool(torch.isfinite(f).all()):
                raise FloatingPointError('Non-finite acoustic features')
            examples.append({**{k: v for k, v in row.items() if k != 'wave'}, 'features': f, 'mask': m})
        return examples


def worker_init(_):
    torch.set_num_threads(1)


def loader(records, cfg, *, training=False, epoch=0, batches=None):
    dataset = AudioDataset(records, training=training, epoch=epoch, seed=cfg['seed'],
                           max_seconds=cfg['max_seconds'], rawboost=cfg['rawboost'] if training else 0,
                           raw_config=cfg['raw_config'], legacy_prefix=cfg.get('legacy_prefix', False))
    kwargs = dict(dataset=dataset, collate_fn=FeatureCollator(cfg['ssl_path']),
                  num_workers=cfg['workers'], pin_memory=False,
                  generator=torch.Generator().manual_seed(cfg['seed'] + epoch))
    if cfg['workers']:
        kwargs.update(multiprocessing_context='spawn', worker_init_fn=worker_init, prefetch_factor=2)
    if batches is not None:
        kwargs['batch_sampler'] = batches
    else:
        kwargs.update(batch_size=cfg.get('eval_batch', 8), shuffle=False)
    return DataLoader(**kwargs)


class EpochPlan:
    """Ordinary full traversal, balanced noisy source queues, cyclic band coverage."""
    def __init__(self, ordinary, banks, ordinary_batch=16, noisy_batch=4, seed=1234):
        if ordinary_batch < 4 or noisy_batch < 0 or noisy_batch % 2:
            raise ValueError('Ordinary batch >=4; noisy batch must be even')
        self.ordinary, self.banks = ordinary, banks
        self.n, self.s, self.seed = ordinary_batch, noisy_batch, seed
        self.indices = defaultdict(list)
        self.groups = {0: [], 1: []}
        self.records = list(ordinary)
        self.full = bool(banks and banks[0] and banks[0][0].get('full_length'))
        if self.full and (len(banks) != 1 or not all(r.get('full_length') for r in banks[0])):
            raise ValueError('Use one complete two-view Train bank; no legacy/full cache mixing')
        for bank, rows in enumerate(banks):
            for row in rows:
                self.indices[(row['id'], bank)].append(len(self.records))
                self.records.append(row)
        if banks:
            for row in banks[0]:
                if (row.get('version') == 0 if self.full else row['band'] == 0):
                    self.groups[row['label']].append(row['id'])
            if not all(self.groups.values()):
                raise ValueError('Both noisy source classes are required')
        if noisy_batch and not banks:
            raise ValueError('No noisy training bank')
        self.steps = math.ceil(len(ordinary) / ordinary_batch)
        if len(ordinary) % ordinary_batch in (1, 2, 3) and self.steps > 1:
            self.steps -= 1

    def batches(self, epoch):
        rng = random.Random(self.seed + epoch * 1009)
        order = list(range(len(self.ordinary)))
        rng.shuffle(order)
        # Each class's persistent virtual permutation avoids restarting at rare sources every epoch.
        def source_at(label, ticket):
            sources = self.groups[label]
            cycle, offset = divmod(ticket, len(sources))
            key = (label, cycle)
            if key not in queues:
                q = list(sources)
                random.Random(self.seed + 13007 * cycle + label).shuffle(q)
                queues[key] = q
            return queues[key][offset], cycle
        queues = {}
        output = []
        for step in range(self.steps):
            stop = len(order) if step == self.steps - 1 else (step + 1) * self.n
            batch = order[step * self.n:stop]
            for label in (0, 1):
                for j in range(self.s // 2):
                    ticket = ((epoch - 1) * self.steps + step) * (self.s // 2) + j
                    source, visit = source_at(label, ticket)
                    offset = stable_seed(self.seed, 0, source)
                    bank = (offset + visit // 4) % len(self.banks)
                    band = (offset + visit) % 4
                    choices = self.indices[(source, bank)]
                    if self.full:
                        version = (offset + visit) % 2
                        index = next(i for i in choices if self.records[i]['version'] == version)
                        batch.append((index, index, 'full', ticket))
                        continue
                    index = next(i for i in choices if self.records[i]['band'] == band)
                    partner_band = (band + 1 + stable_seed(self.seed, epoch, ticket) % 3) % 4
                    partner = next(i for i in choices if self.records[i]['band'] == partner_band)
                    # Exactly the same mode schedule for fake and real; no class-specific artifact.
                    mode = MODES[(ticket + self.seed) % len(MODES)]
                    batch.append((index, partner, mode, ticket))
            output.append(batch)
        return output


def build_data(cfg):
    print('Reading official Train and Dev protocols.', flush=True)
    train = read_protocol(cfg['train_protocol'], cfg['train_data_path'])
    dev = read_protocol(cfg['dev_protocol'], cfg['dev_data_path'])
    if Path(cfg['train_data_path']).resolve() == Path(cfg['dev_data_path']).resolve():
        raise ValueError('Train and Dev must have distinct audio roots')
    loaded, banks = [], []
    for path in cfg['train_caches']:
        print('Checking existing Train cache: ' + path, flush=True)
        rows, metadata = cache_index(path, 'train', cfg['train_protocol'], train)
        banks.append(rows)
        loaded.append(metadata)
    validation, metadata = {'clean': dev}, {}
    for name, role, key in [('seen', 'dev_seen', 'dev_noisy_cache'), ('heldout', 'dev_heldout', 'dev_heldout_cache')]:
        print('Checking fixed Dev cache: ' + cfg[key], flush=True)
        validation[name], metadata[name] = cache_index(cfg[key], role, cfg['dev_protocol'], dev)
    from rtc_noisy_v2.cache import check_suite
    check_suite(loaded, metadata['seen'], metadata['heldout'])
    held = metadata['heldout'][1].get('processing')
    if held:
        if not metadata['seen'][1].get('processing'):
            raise ValueError('Unseen processing requires matched diverse Dev-seen')
        for _, config in loaded + [metadata['seen']]:
            if set(held['families']) & set(config.get('processing', {}).get('families', ['ffmpeg'])):
                raise ValueError('Heldout processing family leaked into training/seen')
    counts = torch.bincount(torch.tensor([r['label'] for r in train]), minlength=2)
    weights = counts.sum().float() / (2 * counts.float())
    plan = EpochPlan(train, banks, cfg['ordinary_batch'], cfg['noisy_batch'], cfg['seed'])
    files = [cfg['train_protocol'], cfg['dev_protocol'], str(Path(cfg['ssl_path']) / 'preprocessor_config.json')]
    for path in cfg['train_caches'] + [cfg['dev_noisy_cache'], cfg['dev_heldout_cache']]:
        files.extend(str(Path(path) / name) for name in ('config.json', 'manifest.jsonl'))
        if (Path(path)/'complete.json').is_file():
            files.append(str(Path(path)/'complete.json'))
    fingerprints = {str(Path(p).resolve()): sha256(p) for p in files}
    return plan, validation, weights, counts, fingerprints
