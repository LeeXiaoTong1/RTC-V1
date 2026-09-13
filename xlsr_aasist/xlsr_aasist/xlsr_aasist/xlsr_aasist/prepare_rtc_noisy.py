"""Cache four noise-before-RTC views per Offline utterance. Safe to resume."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from utils.data_utils import pad_audio
from rtc_noisy.common import (CACHE_FORMAT, CUT, SR, SNR_BANDS, atomic_json,
                              audio_domain, noise_catalog, read_labels, read_wave, sha256, stable_seed)
from rtc_noisy.simulator import LocalRTC, RTCSettings


def prepare(args):
    dataset = Path(args.dataset_root).resolve()
    audio_root = dataset / "wav" / args.split
    protocol = dataset / f"{args.split}_label.txt"
    labels = read_labels(protocol, audio_root)
    ids = [utt for utt in labels if audio_domain(utt) == "offline"]
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        raise ValueError("No Offline utterances")
    augment, catalog = noise_catalog(args.noise_manifest, args.split)
    rtc = LocalRTC(args.ffmpeg)
    # Exercise all selected presets before starting a large cache.
    probe = (.05 * np.random.RandomState(3).randn(64600)).astype(np.float32)
    for nr, gain, bitrate in zip((6, 12, 18), (2, 4, 8), (16000, 24000, 32000)):
        rtc(probe, RTCSettings(nr, gain, bitrate))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {"format": CACHE_FORMAT, "split": args.split, "seed": args.seed,
              "protocol_sha256": sha256(protocol), "noise": catalog, "cut": CUT, "sr": SR,
              "snr_bands": [list(band) for band in SNR_BANDS], "limit": args.limit,
              "offline_count": len(ids), "ffmpeg_version": rtc.version,
              "engine": "FFmpeg afftdn + dynaudnorm + libopus (NOT WebRTC)",
              "presets": {"noise_reduction": [6, 12, 18], "max_gain": [2, 4, 8],
                          "bitrate": [16000, 24000, 32000], "opus_vbr": True},
              "order": "full utterance -> additive noise -> local RTC -> codec trimming -> first crop/repeat"}
    config_file = output / "config.json"
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            raise ValueError("Cache config differs; choose a NEW output directory")
    elif any(output.iterdir()):
        raise ValueError("Nonempty cache directory without config.json")
    else:
        atomic_json(config_file, config)
    count = len(ids) * len(SNR_BANDS)
    print(f"Offline={len(ids)}; variants={count}; FLOAT WAV about {count*CUT*4/1e9:.1f} GB", flush=True)

    def process(utt):
        # Read the complete source, not a 4-second chunk. This keeps processor
        # state and gain estimation valid before the final baseline crop.
        source = audio_root / utt
        source_hash = sha256(source)
        waveform = read_wave(source)
        key = hashlib.sha256(utt.encode()).hexdigest()[:24]
        folder = output / "audio" / key[:2]
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for band, (low, high) in enumerate(SNR_BANDS):
            record_path = folder / f"{key}_b{band}.json"
            audio_path = folder / f"{key}_b{band}.wav"
            if record_path.exists() and audio_path.exists():
                row = json.loads(record_path.read_text())
                info = sf.info(audio_path)
                if (row["source"] == utt and row["source_sha256"] == source_hash
                        and row["band"] == band and info.frames == CUT
                        and info.channels == 1 and info.samplerate == SR):
                    rows.append(row)
                    continue
            rng = np.random.RandomState(stable_seed(args.seed, args.split, utt, band))
            snr = float(rng.uniform(low, high))
            noisy, noise_info = augment(waveform, SR, rng=rng, snr_db=snr, return_info=True)
            if not noise_info.get("applied"):
                raise ValueError(f"Cannot create a noisy example from silent speech: {utt}")
            settings = RTCSettings.sample(rng)
            processed = rtc(noisy, settings)
            cached = np.asarray(pad_audio(processed, CUT), np.float32)
            tmp_audio = folder / f"{key}_b{band}.tmp.wav"
            sf.write(tmp_audio, cached, SR, subtype="FLOAT")
            os.replace(tmp_audio, audio_path)
            row = {"source": utt, "source_sha256": source_hash, "label": labels[utt], "band": band,
                   "snr_db": snr, "noise": noise_info, "rtc": settings.as_dict(),
                   "source_samples": len(waveform), "output_samples_before_crop": len(processed),
                   "audio": str(audio_path.relative_to(output))}
            atomic_json(record_path, row)
            rows.append(row)
        return rows

    tmp = output / "manifest.jsonl.tmp"
    with ThreadPoolExecutor(max_workers=args.workers) as pool, tmp.open("w", encoding="utf-8") as stream:
        for rows in tqdm(pool.map(process, ids), total=len(ids), desc=f"Caching {args.split}", unit="source"):
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output / "manifest.jsonl")
    print(f"Cache complete: {output / 'manifest.jsonl'}")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", default="./dataset")
    p.add_argument("--split", choices=["train", "dev"], required=True)
    p.add_argument("--noise_manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--limit", type=int, default=0, help="Small smoke cache ONLY; use a separate directory")
    args = p.parse_args()
    if args.workers < 1 or args.limit < 0:
        p.error("workers must be positive and limit nonnegative")
    prepare(args)


if __name__ == "__main__":
    main()
