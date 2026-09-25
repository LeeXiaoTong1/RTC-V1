"""Reuse V2 waveform augmentation, verified pairing and view rotation unchanged.

Feature extraction runs in CPU DataLoader workers, BEFORE any transfer to CUDA.
Only the non-learned waveform-to-logmel operation uses NumPy; encoder graphs stay live.
"""
import hashlib
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from .core import class_weights, sha256
from .sampling import BalancedOrdinarySampler


class SeededDataset(Dataset):
    """Stateless per-(epoch,index) augmentation, including after epoch-boundary resume."""
    def __init__(self, inner, seed):
        self.inner, self.seed = inner, int(seed)
        self.epoch = torch.zeros(1, dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        ticket = index
        if isinstance(ticket, tuple):
            index, draw = ticket
            key = f'{self.seed}:{int(self.epoch[0])}:{index}:draw:{draw}'.encode()
        else:
            key = f'{self.seed}:{int(self.epoch[0])}:{index}'.encode()
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), 'little')
        py, npstate, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            return self.inner[index]
        finally:
            random.setstate(py)
            np.random.set_state(npstate)
            torch.set_rng_state(cpu)


class FeatureCollator:
    def __init__(self, directory, kind='ordinary', cache_root=None):
        self.directory, self.kind, self.extractor = str(directory), kind, None
        self.cache_root, self.cache = cache_root, None

    def __call__(self, rows):
        if self.extractor is None:
            from transformers import AutoFeatureExtractor
            self.extractor = AutoFeatureExtractor.from_pretrained(self.directory, local_files_only=True)
            if self.extractor.sampling_rate != 16000:
                raise ValueError('Pretrained extractor requires 16 kHz')
        if self.cache_root and self.cache is None:
            from .feature_cache import FeatureCache
            self.cache = FeatureCache(self.cache_root, self.directory, self.extractor)
        if self.kind in ('pair', 'noisy_pair'):
            waves = [r[0] for r in rows] + [r[1] for r in rows]
            labels = [r[2] for r in rows] * 2
            extra = {'pairs': len(rows)}
        else:
            waves, labels = [r[0] for r in rows], [r[1] for r in rows]
            extra = {'ids': [r[2] for r in rows]} if self.kind == 'ordinary' else {'bands': [r[2] for r in rows]}
        if any(w.device.type != 'cpu' or w.shape != (64600,) or not torch.isfinite(w).all() for w in waves):
            raise ValueError('Expected finite CPU mono 64600-sample V2 waveforms')
        features, mask = self.cache.transform(waves, self.extract) if self.cache else self.extract(waves)
        return {'features': features, 'mask': mask, 'labels': torch.tensor(labels, dtype=torch.long), **extra}

    def extract(self, waves):
        batch = self.extractor([w.float().numpy() for w in waves], sampling_rate=16000,
                               padding=True, pad_to_multiple_of=2,
                               return_attention_mask=True, return_tensors='pt')
        features, mask = batch['input_features'], batch['attention_mask'].long()
        lengths = mask.sum(1)
        if not bool((lengths == lengths[0]).all()):
            raise ValueError('Fixed PCM crop must produce equal valid feature lengths')
        valid = int(lengths[0])
        if valid < 12 or not bool(mask[:, :valid].bool().all()) or bool(mask[:, valid:].bool().any()):
            raise ValueError('Non-contiguous feature mask')
        features, mask = features[:, :valid].contiguous().float(), mask[:, :valid].contiguous()
        if features.shape[2] != 160 or not torch.isfinite(features).all():
            raise ValueError('Invalid official 160-dimensional acoustic features')
        return features, mask


def worker_init(_):
    torch.set_num_threads(1)


class DataBundle:
    def __init__(self, args):
        # Lazy imports keep unit tests independent of fairseq/librosa/HF installation.
        from utils.data_utils import build_dataset_from_protocol
        from utils.rtc_data import BalancedPairBatchSampler, RTCPairDataset
        from utils.rtc_pairs import load_pairs
        from rtc_noisy.data import NoisyDevDataset
        from rtc_noisy_v2.cache import load_v2_cache, check_suite, RotatingNoisyDataset
        from rtc_noisy_v2.sampling import RotatingViewBatchSampler
        self.args, self.rotation, self.real_sampler, self.ordinary_sampler = args, None, None, None
        self.generators = []
        os.environ['RTC_B_NOISE_MANIFEST'] = str(Path(args.train_noise_manifest).resolve())
        pairs = load_pairs(args.rtc_pairs, args.train_protocol, args.train_data_path)
        bank_paths = [args.train_noisy_cache] + args.extra_train_noisy_cache
        banks = [load_v2_cache(p, 'train', args.train_protocol, args.train_data_path) for p in bank_paths]
        seen = load_v2_cache(args.dev_noisy_cache, 'dev_seen', args.dev_protocol, args.dev_data_path)
        held = load_v2_cache(args.dev_heldout_cache, 'dev_heldout', args.dev_protocol, args.dev_data_path)
        check_suite(banks, seen, held)
        mh = sha256(args.train_noise_manifest)
        if any(cfg['noise']['manifest_sha256'] != mh for _, cfg in banks):
            raise ValueError('Ordinary noise and Train caches must use the same training manifest')
        ordinary, ids, labels = build_dataset_from_protocol(args.train_protocol, args.train_data_path,
                                                           mode='train', args=args, algo=args.algo)
        clean, dev_ids, _ = build_dataset_from_protocol(args.dev_protocol, args.dev_data_path, mode='dev', args=args)
        self.weights, self.counts = class_weights(ids, labels)
        if ordinary.env_noise is None:
            raise ValueError('V2 ordinary MUSAN augmentation is disabled; check RTC_B_* variables')
        self.ordinary = SeededDataset(ordinary, args.seed + 50)
        n = {1: 40, 2: 32, 3: 24}[args.stage]
        self.n, self.r, self.s = n, (4 if args.stage > 1 else 0), (4 if args.stage == 3 else 0)
        self.steps = math.ceil(len(ordinary)/n)
        self.order_generator = torch.Generator()
        if args.ordinary_sampling == 'balanced':
            self.weights = torch.ones(2)  # Do not double-correct class imbalance.
            self.ordinary_sampler = BalancedOrdinarySampler([labels[i] for i in ids], n, self.steps, args.seed + 7)
            self.train = [self.loader(self.ordinary, 'ordinary', batch_sampler=self.ordinary_sampler)]
        else:
            sampler = RandomSampler(self.ordinary, generator=self.order_generator)
            self.train = [self.loader(self.ordinary, 'ordinary', n, sampler=sampler)]
        if self.r:
            self.real_sampler = BalancedPairBatchSampler(pairs, 4, self.steps, args.seed + 1)
            self.train.append(self.loader(RTCPairDataset(pairs, args.train_data_path), 'pair',
                                          batch_sampler=self.real_sampler))
        if self.s:
            noisy = RotatingNoisyDataset([rows for rows, _ in banks], args.train_data_path)
            source = BalancedPairBatchSampler(noisy.sources, 4, self.steps, args.seed + 2)
            self.rotation = RotatingViewBatchSampler(source, noisy.sources, args.seed + 3, len(banks), args.noisy_bank_policy)
            self.train.append(self.loader(noisy, 'noisy_pair', batch_sampler=self.rotation))
        self.dev = {'clean': self.loader(clean, 'ordinary', args.eval_batch, evaluation=True),
                    'seen': self.loader(NoisyDevDataset(seen[0]), 'noisy_dev', args.eval_batch, evaluation=True),
                    'heldout': self.loader(NoisyDevDataset(held[0]), 'noisy_dev', args.eval_batch, evaluation=True)}
        sources = [args.train_protocol, args.dev_protocol, args.rtc_pairs, args.train_noise_manifest,
                   str(Path(args.ssl_path)/'preprocessor_config.json'), str(Path(args.ssl_path)/'config.json')]
        sources += [str(Path(p)/name) for p in bank_paths + [args.dev_noisy_cache, args.dev_heldout_cache]
                    for name in ('config.json', 'manifest.jsonl')]
        self.fingerprints = {str(Path(p).resolve()): sha256(p) for p in sources}
        self.base_fingerprints = {str(Path(p).resolve()): self.fingerprints[str(Path(p).resolve())] for p in sources[:6]}
        print(f'Train counts [fake,real]={self.counts.tolist()}, ordinary={args.ordinary_sampling}, CE={self.weights.tolist()}')
        print(f'Logical batch: {n} ordinary + {2*self.r} real-pair + {2*self.s} noisy-pair = 40; steps={self.steps}')
        print('Pair branches: equal CE. Validation: fixed clean Online + noisy seen + noisy heldout, in all stages.')

    def loader(self, dataset, kind, batch_size=None, sampler=None, batch_sampler=None, evaluation=False):
        gen = torch.Generator()
        self.generators.append(gen)
        workers = min(self.args.num_workers, 2 if kind in ('pair', 'noisy_pair') else 4)
        cache = self.args.feature_cache if evaluation or kind in ('pair', 'noisy_pair') else None
        kwargs = {'dataset': dataset, 'num_workers': workers, 'pin_memory': self.args.device.startswith('cuda'),
                  'generator': gen, 'collate_fn': FeatureCollator(self.args.ssl_path, kind, cache)}
        if workers:
            kwargs.update(persistent_workers=True, multiprocessing_context='spawn', worker_init_fn=worker_init,
                          prefetch_factor=2)
        if batch_sampler is not None:
            kwargs['batch_sampler'] = batch_sampler
        else:
            kwargs.update(batch_size=batch_size, sampler=sampler, shuffle=False)
        return DataLoader(**kwargs)

    def begin(self, epoch):
        self.ordinary.epoch[0] = epoch
        self.order_generator.manual_seed(self.args.seed + 100*epoch)
        if self.ordinary_sampler is not None:
            self.ordinary_sampler.set_epoch(epoch)
        for i, g in enumerate(self.generators):
            g.manual_seed(self.args.seed + 1000*epoch + i)
        if self.real_sampler is not None:
            self.real_sampler.set_epoch(epoch)
        if self.rotation is not None:
            self.rotation.set_epoch(epoch)

    def end(self, steps):
        if self.rotation is not None:
            self.rotation.commit_epoch(steps)

    def sampler_state(self):
        return self.rotation.state_dict() if self.rotation is not None else None

    def load_sampler_state(self, state):
        if self.rotation is not None:
            self.rotation.load_state_dict(state)
        elif state is not None:
            raise ValueError('Unexpected sampler state for this stage')


def combine_batches(batches):
    out = {}
    for k in ('features', 'mask', 'labels'):
        values = [b[k] for b in batches]
        if all(v.is_pinned() for v in values):
            result = torch.empty((sum(len(v) for v in values), *values[0].shape[1:]),
                                 dtype=values[0].dtype, pin_memory=True)
            torch.cat(values, out=result)
            out[k] = result
        else:
            out[k] = torch.cat(values)
    n = len(batches[0]['labels'])
    r = batches[1]['pairs'] if len(batches) > 1 else 0
    s = batches[2]['pairs'] if len(batches) > 2 else 0
    return out, (n, r, s)
