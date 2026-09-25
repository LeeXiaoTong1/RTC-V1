"""Four views per source; restricted training combinations and matched Dev pairs."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from rtc_noisy.common import (CACHE_FORMAT, CUT, SR, atomic_json, audio_domain,
                              noise_catalog, read_labels, read_wave, sha256)
from rtc_noisy.simulator import LocalRTC, RTCSettings
from rtc_noisy.diverse import DiverseRTC, PROFILES, profile_definition
from rtc_noisy_v2.plan import (BANDS, PLAN_ID, SCHEMA, digest_json, plan_definition,
                               settings_for, stable_seed)
from utils.data_utils import pad_audio


def prepare(args):
    role, split = args.role, "train" if args.role == "train" else "dev"
    profile = getattr(args, 'processing_profile', 'legacy')
    if profile == 'unseen' and role != 'dev_heldout' or profile == 'diverse' and role == 'dev_heldout':
        raise ValueError('diverse is Train/Dev-seen only; unseen is Dev-heldout only')
    if role != "train" and args.generation != 0:
        raise ValueError("Dev caches must remain fixed at generation=0")
    root = Path(args.dataset_root)
    audio_root, protocol = root / "wav" / split, root / f"{split}_label.txt"
    labels = read_labels(protocol, audio_root)
    ids = [utt for utt in labels if audio_domain(utt) == "offline"]
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        raise ValueError("No Offline sources")
    augment, catalog = noise_catalog(args.noise_manifest, split)
    rtc = LocalRTC(args.ffmpeg) if profile == 'legacy' else DiverseRTC(args.ffmpeg, profile)
    if profile != 'legacy':
        augment.intermittent_probability = profile_definition(profile)['intermittent_probability']
    allowed = settings_for(role)
    # Strict codec/filter-length check remains in LocalRTC. No manual padding fix.
    probe = .05 * np.random.RandomState(3).normal(size=70321).astype(np.float32)
    for cfg in allowed:
        if profile == 'legacy':
            rtc(probe, RTCSettings(*cfg))
        else:
            for family in PROFILES[profile]:
                rtc(probe, RTCSettings(*cfg), np.random.RandomState(7), family=family)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {"format": CACHE_FORMAT, "optimization_schema": SCHEMA,
              "role": role, "split": split, "seed": args.seed, "generation": args.generation,
              "plan_id": PLAN_ID, "plan": plan_definition(),
              "allowed_settings": [list(x) for x in allowed],
              "protocol_sha256": sha256(protocol), "noise": catalog,
              "cut": CUT, "sr": SR, "snr_bands": [list(x) for x in BANDS],
              "limit": args.limit, "offline_count": len(ids), "ffmpeg_version": rtc.version,
              "engine": "FFmpeg afftdn + dynaudnorm + libopus (NOT WebRTC)",
              "order": "full utterance -> noise -> local RTC -> first crop/repeat",
              "mix_rule": "same seed/split/source/band/generation; Dev role excluded from noise RNG"}
    if profile != 'legacy':
        config.update(processing=profile_definition(profile), webrtc_version=rtc.webrtc_version,
                      processing_code_sha256=sha256(Path(__file__).parent/'rtc_noisy'/'diverse.py'),
                      noise_code_sha256=sha256(Path(__file__).parent/'utils'/'env_noise.py'),
                      engine='Local '+profile+' processing + Opus; no platform upload',
                      order='full utterance -> optional intermittent noise -> optional synthetic room -> local DSP/Opus -> crop')
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Existing cache config differs. Use a new directory; never overwrite an active cache")
    elif any(output.iterdir()):
        raise ValueError("Nonempty cache directory without config")
    else:
        atomic_json(config_path, config)
    count = len(ids) * 4
    print(f"Role={role}; generation={args.generation}; Offline={len(ids)}; variants={count}; "
          f"audio about {count*CUT*4/1e9:.1f} GB; combinations={len(allowed)}", flush=True)

    def process(utt):
        source = audio_root / utt
        source_hash, waveform = sha256(source), read_wave(source)
        key = hashlib.sha256(utt.encode()).hexdigest()[:24]
        folder = output / "audio" / key[:2]
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for band, (low, high) in enumerate(BANDS):
            record_path, audio_path = folder / f"{key}_b{band}.json", folder / f"{key}_b{band}.wav"
            if record_path.exists() and audio_path.exists():
                row, info = json.loads(record_path.read_text()), sf.info(audio_path)
                if (row["source_sha256"] == source_hash and row["role"] == role
                        and row["generation"] == args.generation and info.frames == CUT
                        and info.channels == 1 and info.samplerate == SR):
                    rows.append(row)
                    continue
            mix_seed = stable_seed(args.seed, "mix", split, utt, band, args.generation) % (2**32)
            rng = np.random.RandomState(mix_seed)
            snr = float(rng.uniform(low, high))
            noisy, info = augment(waveform, SR, rng=rng, snr_db=snr, return_info=True)
            if not info.get("applied"):
                raise ValueError(f"Cannot create a noisy sample from silent speech: {utt}")
            settings_rng = np.random.RandomState(stable_seed(args.seed, "settings", split, utt, band,
                                                            args.generation) % (2**32))
            settings = RTCSettings(*allowed[int(settings_rng.randint(len(allowed)))])
            processing = None
            if profile == 'legacy':
                processed = rtc(noisy, settings)
            else:
                processed, processing = rtc(noisy, settings, settings_rng)
            cached = np.asarray(pad_audio(processed, CUT), np.float32)
            tmp_audio = folder / f"{key}_b{band}.tmp.wav"
            sf.write(tmp_audio, cached, SR, subtype="FLOAT")
            os.replace(tmp_audio, audio_path)
            # Hash exact mixed samples, so matched Dev comparisons cannot silently
            # differ in their noise, source, or SNR even if manifests were edited.
            mix_id = hashlib.sha256(np.asarray(noisy, dtype="<f4").tobytes()).hexdigest()
            row = {"source": utt, "source_sha256": source_hash, "label": labels[utt],
                   "band": band, "snr_db": snr, "noise": info, "mix_id": mix_id,
                   "role": role, "generation": args.generation, "rtc": settings.as_dict(),
                   "source_samples": len(waveform), "output_samples_before_crop": len(processed),
                   "audio": str(audio_path.relative_to(output))}
            if processing is not None:
                row['processing'] = processing
            atomic_json(record_path, row)
            rows.append(row)
        return rows

    temporary = output / "manifest.jsonl.tmp"
    with ThreadPoolExecutor(max_workers=args.workers) as pool, temporary.open("w", encoding="utf-8") as stream:
        for rows in tqdm(pool.map(process, ids), total=len(ids), desc=f"Caching {role}", unit="source"):
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output / "manifest.jsonl")
    print(f"Complete: {output}")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", default="./dataset")
    p.add_argument("--role", choices=("train", "dev_seen", "dev_heldout"), required=True)
    p.add_argument("--noise_manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--generation", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument('--processing_profile', choices=['legacy', 'diverse', 'unseen'], default='legacy')
    p.add_argument("--limit", type=int, default=0, help="Smoke check only; separate directory required")
    args = p.parse_args()
    if args.workers < 1 or args.limit < 0 or args.generation < 0:
        p.error("workers >=1, limit/generation >=0 required")
    prepare(args)


if __name__ == "__main__":
    main()
