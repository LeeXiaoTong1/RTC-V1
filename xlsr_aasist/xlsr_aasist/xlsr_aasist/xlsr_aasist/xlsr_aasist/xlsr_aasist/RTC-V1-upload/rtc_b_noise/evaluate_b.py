"""Evaluate a checkpoint on labeled Dev and deterministic additive-noise Dev.

No Progress/Eval analysis. This is a B diagnostic, without RTC simulation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from env_noise import NoiseAugment


def binary_f1(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    scores = []
    for k in (0, 1):
        tp = matrix[k, k]
        denominator = matrix[k, :].sum() + matrix[:, k].sum()
        scores.append(float(2 * tp / denominator) if denominator else 0.0)
    return {"macro_f1": 100 * float(np.mean(scores)),
            "spoof_f1": 100 * scores[0], "real_f1": 100 * scores[1],
            "confusion_true_rows_pred_columns": matrix.tolist(), "trials": len(labels)}


def utterance_seed(seed, utterance):
    return int.from_bytes(hashlib.sha256(f"{seed}:{utterance}".encode()).digest()[:4], "little")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dev-data", type=Path, required=True)
    p.add_argument("--dev-protocol", type=Path, required=True)
    p.add_argument("--noise-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--snrs", default="10,20,30")
    p.add_argument("--seed", type=int, default=20260908)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Output exists; use a new filename")
    sys.path.insert(0, str(a.repo.expanduser().resolve()))
    import librosa
    import torch
    from torch.utils.data import DataLoader, Dataset
    from model.model import Model
    from utils.data_utils import read_protocol, pad_audio

    ids, labels = read_protocol(str(a.dev_protocol), require_label=True)
    if not ids or set(labels.values()) != {0, 1}:
        raise SystemExit("Need a labeled Dev protocol containing both real and spoof")
    augmentation = NoiseAugment(a.noise_manifest, probability=1.0)
    if {r.get("split") for r in augmentation.records} != {"dev"}:
        raise SystemExit("Validation requires dev.jsonl, not training noises")
    snrs = [float(value) for value in a.snrs.split(",")]
    if not snrs or not np.isfinite(snrs).all():
        raise SystemExit("Invalid --snrs")

    class DevDataset(Dataset):
        def __init__(self, snr):
            self.snr = snr

        def __len__(self):
            return len(ids)

        def __getitem__(self, index):
            key = ids[index]
            path = a.dev_data / key
            audio, sr = librosa.load(str(path), sr=16000)
            if self.snr is not None:
                # Identical file and offset at different SNRs and across checkpoints.
                rng = np.random.RandomState(utterance_seed(a.seed, key))
                audio = augmentation(audio[:64600], sr, rng=rng, snr_db=self.snr)
            return torch.as_tensor(pad_audio(audio, 64600), dtype=torch.float32), labels[key]

    torch.manual_seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = Model(SimpleNamespace(), device).to(device)
    # The official baseline checkpoints are state_dicts.
    state = torch.load(str(a.checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    def evaluate(snr):
        truth, pred = [], []
        loader = DataLoader(DevDataset(snr), batch_size=a.batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda")
        with torch.inference_mode():
            for waveform, target in loader:
                score = model(waveform.to(device)).softmax(dim=1)[:, 0]
                prediction = torch.where(score >= 0.5, 0, 1)
                truth.extend(target.tolist())
                pred.extend(prediction.cpu().tolist())
        return truth, pred, binary_f1(truth, pred)

    _, _, clean = evaluate(None)
    print("Original Dev:", clean, flush=True)
    all_truth, all_pred, per_snr = [], [], {}
    for snr in snrs:
        truth, pred, metrics = evaluate(snr)
        all_truth.extend(truth)
        all_pred.extend(pred)
        per_snr[str(snr)] = metrics
        print(f"Dev + noise, SNR={snr} dB:", metrics, flush=True)
    noisy = binary_f1(all_truth, all_pred)
    report = {"checkpoint": str(a.checkpoint.resolve()), "threshold": 0.5,
              "dev_protocol": str(a.dev_protocol.resolve()),
              "dev_protocol_sha256": hashlib.sha256(a.dev_protocol.read_bytes()).hexdigest(),
              "noise_manifest_sha256": augmentation.manifest_sha256,
              "seed": a.seed, "clean": clean, "by_snr_db": per_snr,
              "pooled_noisy": noisy,
              "proxy_weighted": 0.3 * clean["macro_f1"] + 0.7 * noisy["macro_f1"],
              "scope": "All utterances in the supplied Dev protocol. No RTC simulation; not the official noisy test distribution."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", a.out)


if __name__ == "__main__":
    main()
