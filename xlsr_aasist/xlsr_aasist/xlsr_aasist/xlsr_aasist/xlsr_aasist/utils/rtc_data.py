"""Unaugmented, utterance-level RTC pairs and deterministic balanced sampling."""

import random
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class RTCPairDataset(Dataset):
    """Use records from load_pairs; the main V1 loader retains its augmentation.

    Both waveforms use V1's first-64600-samples crop/repeat convention. This is
    utterance-level pairing, not sample-level or phoneme-level alignment.
    RawBoost and environment augmentation are intentionally not imported here.
    """

    def __init__(self, pairs, data_root, cut=64600):
        self.pairs = pairs
        self.data_root = Path(data_root)
        self.cut = int(cut)
        if self.cut <= 0:
            raise ValueError("cut must be positive")

    def __len__(self):
        return len(self.pairs)

    def _read(self, utt_id):
        audio, _ = librosa.load(str(self.data_root / utt_id), sr=16000, mono=True)
        if not len(audio):
            raise ValueError(f"Empty waveform: {utt_id}")
        if not np.isfinite(audio).all():
            raise ValueError(f"Non-finite waveform: {utt_id}")
        if len(audio) < self.cut:
            audio = np.tile(audio, (self.cut + len(audio) - 1) // len(audio))
        return torch.from_numpy(np.asarray(audio[:self.cut], dtype=np.float32).copy())

    def __getitem__(self, index):
        pair = self.pairs[index]
        return self._read(pair["offline"]), self._read(pair["online"]), pair["label"]


class BalancedPairBatchSampler(Sampler):
    """Each step has equal real/fake pairs; smaller classes cycle as necessary."""

    def __init__(self, pairs, pairs_per_batch=8, steps_per_epoch=1, seed=1234):
        if isinstance(pairs_per_batch, bool) or int(pairs_per_batch) != pairs_per_batch:
            raise ValueError("pairs_per_batch must be an even integer")
        self.pairs_per_batch = int(pairs_per_batch)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed, self.epoch = int(seed), 0
        if self.pairs_per_batch < 2 or self.pairs_per_batch % 2:
            raise ValueError("pairs_per_batch must be a positive even number (at least 2)")
        if self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive")
        self.indices = {label: [] for label in (0, 1)}
        for index, pair in enumerate(pairs):
            if type(pair["label"]) is not int or pair["label"] not in self.indices:
                raise ValueError(f"Invalid pair label: {pair['label']!r}")
            self.indices[pair["label"]].append(index)
        if not all(self.indices.values()):
            raise ValueError("RTC contrast training requires both real and fake pairs")

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        queues = {0: [], 1: []}
        half = self.pairs_per_batch // 2
        for _ in range(self.steps_per_epoch):
            batch = []
            for label in (0, 1):
                chosen = []
                # When the class is too small, duplicates are unavoidable.
                # Otherwise skip already selected indices across queue wraps.
                exclude_repeats = len(self.indices[label]) >= half
                while len(chosen) < half:
                    if not queues[label]:
                        queues[label] = self.indices[label].copy()
                        rng.shuffle(queues[label])
                    candidate = queues[label].pop()
                    if not exclude_repeats or candidate not in chosen:
                        chosen.append(candidate)
                batch.extend(chosen)
            rng.shuffle(batch)
            yield batch
