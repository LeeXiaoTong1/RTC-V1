# RTC_B_ENV_NOISE_V1
from .env_noise import NoiseAugment
import os
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav


LABEL_TO_ID = {
    "spoof": 0,
    "fake": 0,
    "bonafide": 1,
    "bona-fide": 1,
    "real": 1,
}


def set_random_seed(random_seed, args=None):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    cudnn_deterministic = True if args is None else args.cudnn_deterministic_toggle
    cudnn_benchmark = False if args is None else args.cudnn_benchmark_toggle
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


def normalize_label(label):
    key = label.strip().lower()
    if key not in LABEL_TO_ID:
        raise ValueError(f"Unsupported label: {label}")
    return LABEL_TO_ID[key]


def class_weights_from_labels(labels):
    """Return inverse-frequency CE weights [fake, real] from a protocol label map.

    Formula: w_c = N / (2 * N_c). Only the relative ratio matters for mean CE.
    Both classes must be present; silently inventing a weight would hide a data bug.
    """
    if not labels:
        raise ValueError("Training labels are required to compute class weights")
    values = torch.tensor(list(labels.values()), dtype=torch.long)
    counts = torch.bincount(values, minlength=2)
    if counts.numel() != 2 or (counts <= 0).any():
        raise ValueError(f"Both fake and real classes are required, got counts={counts.tolist()}")
    total = counts.sum().float()
    weights = total / (2.0 * counts.float())
    return weights, counts


def read_protocol(protocol_path, require_label=None):
    file_list = []
    labels = {}

    with open(protocol_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 1:
                if require_label is True:
                    raise ValueError(f"Missing label at {protocol_path}:{line_no}")
                file_list.append(parts[0])
                continue
            if len(parts) == 2:
                utt_id, label = parts
            elif len(parts) >= 5:
                utt_id, label = parts[1], parts[4]
            else:
                raise ValueError(f"Invalid protocol line at {protocol_path}:{line_no}: {line.rstrip()}")

            file_list.append(utt_id)
            labels[utt_id] = normalize_label(label)

    has_labels = bool(labels)
    if require_label is False and has_labels:
        raise ValueError(f"Expected an unlabeled protocol file: {protocol_path}")
    return file_list, labels if has_labels else None


def pad_audio(audio, max_len=64600):
    audio_len = audio.shape[0]
    if audio_len >= max_len:
        return audio[:max_len]
    num_repeats = int(max_len / audio_len) + 1
    return np.tile(audio, (1, num_repeats))[:, :max_len][0]


class SpoofAudioDataset(Dataset):
    def __init__(self, file_list, base_dir, labels=None, args=None, algo=0, use_rawboost=False):
        self.file_list = file_list
        self.base_dir = Path(base_dir)
        self.labels = labels
        self.args = args
        self.algo = algo
        self.use_rawboost = use_rawboost
        self.cut = 64600
        self.env_noise = NoiseAugment.from_env() if self.use_rawboost else None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        utt_id = self.file_list[index]
        wav_path = self.base_dir / utt_id
        audio, sample_rate = librosa.load(str(wav_path), sr=16000)
        if self.use_rawboost:
            audio = process_rawboost_feature(audio, sample_rate, self.args, self.algo)
        if self.env_noise is not None:
            audio = self.env_noise(audio[:self.cut], sample_rate)
        audio = Tensor(pad_audio(audio, self.cut))

        if self.labels is None:
            return audio, utt_id
        return audio, self.labels[utt_id], utt_id


def build_dataset_from_protocol(protocol_path, base_dir, mode, args=None, algo=0):
    require_label = mode in {"train", "dev"}
    file_list, labels = read_protocol(protocol_path, require_label=require_label)
    dataset = SpoofAudioDataset(
        file_list=file_list,
        base_dir=base_dir,
        labels=labels,
        args=args,
        algo=algo,
        use_rawboost=(mode == "train"),
    )
    return dataset, file_list, labels


def process_rawboost_feature(feature, sr, args, algo):
    if algo == 1:
        return LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW,
            args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr,
        )
    if algo == 2:
        return ISD_additive_noise(feature, args.P, args.g_sd)
    if algo == 3:
        return SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG,
            args.maxG, sr,
        )
    if algo == 4:
        feature = process_rawboost_feature(feature, sr, args, 1)
        feature = process_rawboost_feature(feature, sr, args, 2)
        return process_rawboost_feature(feature, sr, args, 3)
    if algo == 5:
        feature = process_rawboost_feature(feature, sr, args, 1)
        return process_rawboost_feature(feature, sr, args, 2)
    if algo == 6:
        feature = process_rawboost_feature(feature, sr, args, 1)
        return process_rawboost_feature(feature, sr, args, 3)
    if algo == 7:
        feature = process_rawboost_feature(feature, sr, args, 2)
        return process_rawboost_feature(feature, sr, args, 3)
    if algo == 8:
        feature_1 = process_rawboost_feature(feature, sr, args, 1)
        feature_2 = process_rawboost_feature(feature, sr, args, 2)
        return normWav(feature_1 + feature_2, 0)
    return feature
