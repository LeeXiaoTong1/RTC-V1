"""Fixed cache validation and same-source Offline/noisy-RTC pairs."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from utils.data_utils import pad_audio
from rtc_noisy.common import CACHE_FORMAT, CUT, SR, SNR_BANDS, audio_domain, read_labels, read_wave, sha256


def load_cache(folder, split, protocol, audio_root):
    folder = Path(folder).resolve()
    config = json.loads((folder / "config.json").read_text())
    if (config.get("format") != CACHE_FORMAT or config["split"] != split
            or config["protocol_sha256"] != sha256(protocol)
            or config["snr_bands"] != [list(x) for x in SNR_BANDS]
            or config["cut"] != CUT or config["sr"] != SR):
        raise ValueError("Cache format, split, protocol, SNR bands or audio format differs")
    if config["limit"]:
        raise ValueError("A --limit smoke cache cannot be used for training or model selection")
    labels = read_labels(protocol, audio_root)
    offline = {utt for utt in labels if audio_domain(utt) == "offline"}
    rows, seen = [], set()
    with (folder / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            source, band = row["source"], row["band"]
            if source not in offline or type(row["label"]) is not int or row["label"] != labels[source]:
                raise ValueError("Cached source or truth label differs from the official protocol")
            if type(band) is not int or band not in range(len(SNR_BANDS)) or (source, band) in seen:
                raise ValueError("Duplicate or invalid cached SNR band")
            low, high = SNR_BANDS[band]
            if not low <= row["snr_db"] <= high:
                raise ValueError("Cached SNR does not match its band")
            path = (folder / row["audio"]).resolve()
            path.relative_to(folder)
            info = sf.info(path)
            if info.frames != CUT or info.samplerate != SR or info.channels != 1:
                raise ValueError(f"Invalid cached waveform: {path}")
            row["audio"] = str(path)
            seen.add((source, band))
            rows.append(row)
    expected = {(utt, band) for utt in offline for band in range(len(SNR_BANDS))}
    if seen != expected or config["offline_count"] != len(offline):
        raise ValueError(f"Incomplete cache: {len(seen)} variants, expected {len(expected)}")
    return rows, config


def read_cached(path):
    waveform, sr = sf.read(path, dtype="float32")
    if sr != SR or waveform.shape != (CUT,) or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid cached audio: {path}")
    return torch.from_numpy(waveform)


class NoisyPairDataset(Dataset):
    """ALL Offline sources are eligible, including those without Online audio."""
    def __init__(self, rows, audio_root):
        self.audio_root = Path(audio_root)
        grouped = {}
        for row in rows:
            grouped.setdefault(row["source"], []).append(row)
        self.sources = [{"offline": source, "label": variants[0]["label"],
                         "variants": sorted(variants, key=lambda x: x["band"])}
                        for source, variants in sorted(grouped.items())]

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        source = self.sources[index]
        variants = source["variants"]
        row = variants[int(np.random.randint(len(variants)))]
        # New paired branch has no RawBoost or second additive-noise call.
        original = read_wave(self.audio_root / source["offline"])
        original = torch.from_numpy(np.asarray(pad_audio(original, CUT), np.float32).copy())
        return original, read_cached(row["audio"]), source["label"], row["band"]


class NoisyDevDataset(Dataset):
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda row: (row["source"], row["band"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return read_cached(row["audio"]), row["label"], row["band"]
