"""Baseline single-window inference; optionally package scores.txt for submission."""
import argparse
from pathlib import Path
import zipfile

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_utils import SpoofAudioDataset, read_protocol, set_random_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--eval_data_path", required=True)
    p.add_argument("--protocol_path", required=True)
    p.add_argument("--score_path", required=True)
    p.add_argument("--submission_zip")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=40)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ssl_path", default=None)
    args = p.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        p.error("Invalid batch size or workers")
    set_random_seed(1234)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    ids, _ = read_protocol(args.protocol_path, require_label=False)
    if not ids or len(set(ids)) != len(ids):
        p.error("Evaluation protocol must have unique IDs")
    for utt in ids:
        if not (Path(args.eval_data_path) / utt).is_file():
            raise FileNotFoundError(Path(args.eval_data_path) / utt)
    from model.model import Model
    model = Model(args, device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    dataset = SpoofAudioDataset(ids, args.eval_data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, pin_memory=device.type == "cuda")
    output = Path(args.score_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    written = 0
    with temporary.open("w", encoding="utf-8") as stream, torch.no_grad():
        for waveform, utt_ids in tqdm(loader, desc="Evaluating RTC+Noisy", unit="batch"):
            scores = model(waveform.to(device)).float().softmax(1)[:, 0]
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Non-finite fake scores")
            for utt, score in zip(utt_ids, scores.cpu().tolist()):
                if utt != ids[written]:
                    raise ValueError("Output order differs from official protocol")
                stream.write(f"{utt} {score:.10f}\n")
                written += 1
    if written != len(ids):
        raise ValueError("Missing evaluation scores")
    temporary.replace(output)
    print(f"Scores saved: {output}; utterances={written}")
    if args.submission_zip:
        archive = Path(args.submission_zip)
        archive.parent.mkdir(parents=True, exist_ok=True)
        tmp_zip = Path(str(archive) + ".tmp")
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as package:
            package.write(output, arcname="scores.txt")
        tmp_zip.replace(archive)
        print(f"Submission ZIP saved: {archive}; root member: scores.txt")


if __name__ == "__main__":
    main()
