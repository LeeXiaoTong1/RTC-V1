import hashlib
import json
from pathlib import Path

import librosa
import numpy as np

from utils.env_noise import NoiseAugment
from utils.rtc_pairs import read_train_protocol

SR = 16000
CUT = 64600
SNR_BANDS = ((5., 10.), (10., 15.), (15., 20.), (20., 25.))
CACHE_FORMAT = "rtc_noisy_pair_cache_v1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed, *parts):
    raw = json.dumps([seed, *parts], ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def audio_domain(utt_id):
    found = set(Path(utt_id).parts) & {"offline", "online"}
    if len(found) != 1:
        raise ValueError(f"Cannot identify Offline/Online: {utt_id}")
    return next(iter(found))


def read_labels(protocol, audio_root):
    labels = read_train_protocol(protocol, audio_root)
    for utt in labels:
        audio_domain(utt)
    return labels


def read_wave(path):
    waveform, _ = librosa.load(str(path), sr=SR, mono=True)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid audio: {path}")
    return np.asarray(waveform, dtype=np.float32)


def atomic_json(path, value):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def noise_catalog(manifest, split):
    """Use the existing split by COMPLETE noise recording, not by crop."""
    augmenter = NoiseAugment(manifest, probability=1., snr_min=5., snr_max=25.)
    recording_ids, hashes = set(), set()
    for row in augmenter.records:
        if row.get("split") != split:
            raise ValueError(f"Noise manifest must contain only split={split}")
        parts = {part.lower() for part in Path(row["path"]).parts}
        if parts & {"speech", "music"}:
            raise ValueError("This implementation uses only the MUSAN noise subset")
        recording = row.get("original_recording")
        if not isinstance(recording, str) or not recording.strip():
            raise ValueError("Each noise entry needs original_recording to prevent split-by-crop leakage")
        digest = sha256(row["path"])
        if row.get("sha256") and row["sha256"] != digest:
            raise ValueError(f"Noise recording differs from manifest SHA256: {row['path']}")
        recording_ids.add(recording)
        hashes.add(digest)
    return augmenter, {"manifest_sha256": sha256(manifest),
                       "recording_ids": sorted(recording_ids), "file_sha256": sorted(hashes)}


def assert_noise_disjoint(train_config, dev_config):
    for key in ("recording_ids", "file_sha256"):
        if set(train_config["noise"][key]) & set(dev_config["noise"][key]):
            raise ValueError(f"Train and Dev noise recordings overlap ({key})")
