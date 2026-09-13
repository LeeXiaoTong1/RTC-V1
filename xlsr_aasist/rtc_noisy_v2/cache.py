"""Cache roles, generation banks and validation of the fixed RTC split."""
import json
from pathlib import Path

from .plan import PLAN_ID, SCHEMA, digest_json, plan_definition, setting_key, settings_for


def check_metadata(config, rows, role):
    """No legacy unrestricted cache may be silently renamed as held out."""
    if (config.get("optimization_schema") != SCHEMA or config.get("role") != role
            or config.get("plan_id") != PLAN_ID or config.get("plan") != plan_definition()
            or config.get("allowed_settings") != [list(x) for x in settings_for(role)]):
        raise ValueError("V2 cache role/plan mismatch; regenerate with prepare_rtc_noisy_v2.py")
    if type(config.get("generation")) is not int or config["generation"] < 0:
        raise ValueError("Invalid cache generation")
    if role != "train" and config["generation"] != 0:
        raise ValueError("Dev must use fixed generation 0")
    allowed = set(settings_for(role))
    for row in rows:
        if setting_key(row["rtc"]) not in allowed:
            raise ValueError(f"Forbidden RTC combination in {role}: {row['rtc']}")
        if row.get("role") != role or row.get("generation") != config["generation"]:
            raise ValueError("Cached row role/generation differs from config")
        if not isinstance(row.get("mix_id"), str) or len(row["mix_id"]) != 64:
            raise ValueError("Missing matched-noise recipe identifier")


def load_v2_cache(folder, role, protocol, audio_root):
    from rtc_noisy.data import load_cache
    split = "train" if role == "train" else "dev"
    rows, config = load_cache(folder, split, protocol, audio_root)
    check_metadata(config, rows, role)
    return rows, config


def check_suite(train_banks, seen, heldout):
    seen_rows, seen_cfg = seen
    held_rows, held_cfg = heldout
    if seen_cfg["ffmpeg_version"] != held_cfg["ffmpeg_version"]:
        raise ValueError("Use the same FFmpeg version for both Dev conditions")
    identities = set()
    for rows, config in train_banks:
        for dev_config in (seen_cfg, held_cfg):
            for field in ("recording_ids", "file_sha256"):
                if set(config["noise"][field]) & set(dev_config["noise"][field]):
                    raise ValueError(f"Train/Dev noise leakage: {field}")
        if config["ffmpeg_version"] != seen_cfg["ffmpeg_version"]:
            raise ValueError("Train/Dev FFmpeg versions differ")
        identity = (config["seed"], config["generation"])
        if identity in identities:
            raise ValueError("Duplicate training cache generation")
        identities.add(identity)
    def mapping(rows):
        return {(r["source"], r["band"]): (r["source_sha256"], r["label"], r["mix_id"])
                for r in rows}
    if mapping(seen_rows) != mapping(held_rows):
        raise ValueError("Dev seen/heldout must use the SAME source, noise mix and SNR per band")
    if seen_cfg["noise"] != held_cfg["noise"]:
        raise ValueError("Dev noise catalogs differ")
    if not train_banks:
        raise ValueError("At least one train bank is required")
    reference = {(r["source"], r["band"]): (r["label"], r["source_sha256"])
                 for r in train_banks[0][0]}
    for rows, _ in train_banks[1:]:
        if {(r["source"], r["band"]): (r["label"], r["source_sha256"]) for r in rows} != reference:
            raise ValueError("All training banks must contain the same source audio and labels")


def build_rotating_dataset(bank_rows, audio_root):
    # Class is module-level for DataLoader spawn, while dependencies stay lazy.
    return RotatingNoisyDataset(bank_rows, audio_root)


class RotatingNoisyDataset:
    """Stateless map-style dataset. The sampler supplies the exact view ticket."""
    def __init__(self, bank_rows, audio_root):
        self.audio_root = Path(audio_root)
        self.banks = []
        for rows in bank_rows:
            grouped = {}
            for row in rows:
                grouped.setdefault(row["source"], {})[row["band"]] = row
            self.banks.append(grouped)
        if not self.banks:
            raise ValueError("Missing training cache")
        self.sources = [{"offline": source, "label": bands[0]["label"]}
                        for source, bands in sorted(self.banks[0].items())]
        for bank in self.banks:
            if set(bank) != {r["offline"] for r in self.sources}:
                raise ValueError("Source IDs differ between banks")
            for source in self.sources:
                rows = bank[source["offline"]]
                if set(rows) != {0, 1, 2, 3} or any(r["label"] != source["label"] for r in rows.values()):
                    raise ValueError("Every source needs exactly four correctly labelled views")

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, ticket):
        import numpy as np
        import torch
        from rtc_noisy.common import CUT, read_wave
        from rtc_noisy.data import read_cached
        from utils.data_utils import pad_audio
        if not isinstance(ticket, (tuple, list)) or len(ticket) != 3:
            raise ValueError("Expected sampler ticket (source, bank, band)")
        index, bank, band = ticket
        if (any(type(v) is not int for v in ticket) or not 0 <= index < len(self)
                or not 0 <= bank < len(self.banks) or band not in range(4)):
            raise ValueError("Invalid view ticket")
        source = self.sources[index]
        row = self.banks[bank][source["offline"]][band]
        original = np.asarray(pad_audio(read_wave(self.audio_root / source["offline"]), CUT), np.float32)
        return torch.from_numpy(original.copy()), read_cached(row["audio"]), source["label"], band, bank


def inspect_initial_history(checkpoint):
    """Be explicit about what local metadata can and cannot establish."""
    config_file = Path(checkpoint).resolve().parent.parent / "config.json"
    if not config_file.is_file():
        return {"status": "unknown", "note": "No initialization training config; held-out claim applies only to this run"}
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if any(k in config for k in ("train_noisy_cache", "train_cache_config", "train_cache_configs")):
        return {"status": "previous_simulation", "config": str(config_file),
                "note": "Initialization already used simulated audio; do NOT claim globally unseen combinations"}
    if "rtc_pairs" in config and "train_protocol_sha256" in config:
        return {"status": "rtc_pair_before_simulation", "config": str(config_file),
                "note": "Recorded initialization is the earlier real-RTC pair stage; earlier undocumented history is not verified"}
    return {"status": "unknown", "config": str(config_file),
            "note": "Cannot establish earlier simulator exposure from this config"}
