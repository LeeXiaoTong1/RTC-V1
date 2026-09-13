"""Extend the current V1+RTC-pair training with noise-before-RTC pairs."""
import json
import math
import os
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from main_train import build_arg_parser, build_loader
from utils.data_utils import build_dataset_from_protocol, set_random_seed
from utils.rtc_data import BalancedPairBatchSampler, RTCPairDataset
from utils.rtc_pairs import load_pairs
from rtc_noisy.common import assert_noise_disjoint, atomic_json, sha256
from rtc_noisy.data import NoisyDevDataset, NoisyPairDataset, load_cache
from rtc_noisy.training import selection_key, train_epoch, validate


def build_parser():
    p = build_arg_parser()
    p.description = __doc__
    p.set_defaults(track="xlsr_aasist_RTC_noisy_pair", batch_size=24, num_epochs=30, earlystop_epoch=10)
    p.add_argument("--rtc_pairs", required=True)
    p.add_argument("--train_noisy_cache", required=True)
    p.add_argument("--dev_noisy_cache", required=True)
    p.add_argument("--train_noise_manifest", required=True)
    p.add_argument("--rtc_pairs_per_batch", type=int, default=4)
    p.add_argument("--noisy_pairs_per_batch", type=int, default=4)
    p.add_argument("--rtc_weight", type=float, default=.1)
    p.add_argument("--noisy_weight", type=float, default=.1)
    p.add_argument("--noisy_warmup_epochs", type=float, default=2.)
    p.add_argument("--rtc_temperature", type=float, default=.1)
    p.add_argument("--grad_clip", type=float, default=0., help="0 preserves the previous optimizer behavior")
    p.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    p.add_argument("--ssl_path", default=None)
    p.add_argument("--check_data", action="store_true")
    return p


def save_weights(path, model):
    temporary = Path(str(path) + ".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def main():
    p = build_parser()
    args = p.parse_args()
    for name in ("rtc_weight", "noisy_weight", "noisy_warmup_epochs", "grad_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"{name} must be finite and nonnegative")
    if not math.isfinite(args.rtc_temperature) or args.rtc_temperature <= 0:
        p.error("rtc_temperature must be finite and positive")
    if args.batch_size < 2 or args.num_workers < 0 or args.num_epochs < 1 or args.earlystop_epoch < 1:
        p.error("Invalid batch size, workers, epochs or patience")
    set_random_seed(args.seed, args)
    real_pairs = load_pairs(args.rtc_pairs, args.train_protocol, args.train_data_path)
    train_rows, train_cache = load_cache(args.train_noisy_cache, "train", args.train_protocol, args.train_data_path)
    dev_rows, dev_cache = load_cache(args.dev_noisy_cache, "dev", args.dev_protocol, args.dev_data_path)
    assert_noise_disjoint(train_cache, dev_cache)
    if train_cache["noise"]["manifest_sha256"] != sha256(args.train_noise_manifest):
        p.error("Ordinary MUSAN and noisy-RTC cache must use the same training noise manifest")
    # Set only this process's manifest; retain the user's ordinary B probability
    # and SNR settings, which the supplied shell script fixes explicitly.
    os.environ["RTC_B_NOISE_MANIFEST"] = str(Path(args.train_noise_manifest).resolve())
    ordinary, train_ids, _ = build_dataset_from_protocol(
        args.train_protocol, args.train_data_path, mode="train", args=args, algo=args.algo)
    clean_dev, dev_ids, _ = build_dataset_from_protocol(args.dev_protocol, args.dev_data_path, mode="dev", args=args)
    noisy_train = NoisyPairDataset(train_rows, args.train_data_path)
    steps = math.ceil(len(ordinary) / args.batch_size)
    real_sampler = BalancedPairBatchSampler(real_pairs, args.rtc_pairs_per_batch, steps, args.seed + 1)
    noisy_sampler = BalancedPairBatchSampler(noisy_train.sources, args.noisy_pairs_per_batch, steps, args.seed + 2)
    effective = args.batch_size + 2 * args.rtc_pairs_per_batch + 2 * args.noisy_pairs_per_batch
    known_offline = {row["offline"] for row in real_pairs}
    unpaired_offline = sum(row["offline"] not in known_offline for row in noisy_train.sources)
    print(f"Train trials={len(train_ids)}; Dev trials={len(dev_ids)}; official RTC pairs={len(real_pairs)}")
    print(f"Noisy-RTC Offline sources={len(noisy_train)}, including {unpaired_offline} without Online pairs")
    print(f"Batch: {args.batch_size} ordinary + {args.rtc_pairs_per_batch} RTC pairs x2 "
          f"+ {args.noisy_pairs_per_batch} noisy pairs x2 = {effective} waveforms; {steps} steps/epoch")
    print(f"RawBoost algo={args.algo}; ordinary MUSAN={'enabled' if ordinary.env_noise is not None else 'DISABLED'}")
    if args.check_data:
        print("Data checks passed; no model loaded and no training started.")
        return
    if not args.model_path or not Path(args.model_path).is_file():
        p.error("model_path must be a V1/RTC-pair state_dict checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; fix the environment or explicitly select CPU")
    if device.type == "cuda" and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        p.error("BF16 unsupported; use --amp none")
    from model.model import Model
    model = Model(args, device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True), strict=True)
    print(f"Model loaded: {args.model_path}\nDevice: {device}")
    ordinary_loader = build_loader(ordinary, args.batch_size, args.num_workers, shuffle=True)
    pair_workers = min(2, args.num_workers)
    pair_kwargs = {"num_workers": pair_workers, "pin_memory": device.type == "cuda"}
    if pair_workers:
        pair_kwargs["persistent_workers"] = True
    real_loader = DataLoader(RTCPairDataset(real_pairs, args.train_data_path), batch_sampler=real_sampler, **pair_kwargs)
    noisy_loader = DataLoader(noisy_train, batch_sampler=noisy_sampler, **pair_kwargs)
    dev_kwargs = {"batch_size": effective, "shuffle": False, "num_workers": min(4, args.num_workers),
                  "pin_memory": device.type == "cuda"}
    clean_dev_loader = DataLoader(clean_dev, **dev_kwargs)
    noisy_dev_loader = DataLoader(NoisyDevDataset(dev_rows), **dev_kwargs)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([.1, .9], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    output = Path(args.out_path) / f"{args.track}_epoch{args.num_epochs}_bs{effective}_{stamp}"
    ckpt = output / "ckpt"
    ckpt.mkdir(parents=True)
    config = vars(args).copy()
    config.update(effective_audio_batch=effective, steps_per_epoch=steps,
                  initial_checkpoint_sha256=sha256(args.model_path), torch_version=str(torch.__version__),
                  train_cache_config=train_cache, dev_cache_config=dev_cache,
                  train_cache_manifest_sha256=sha256(Path(args.train_noisy_cache) / "manifest.jsonl"),
                  dev_cache_manifest_sha256=sha256(Path(args.dev_noisy_cache) / "manifest.jsonl"),
                  rtc_pairs_sha256=sha256(args.rtc_pairs), unpaired_offline_sources=unpaired_offline,
                  noise_environment={k: v for k, v in os.environ.items() if k.startswith("RTC_B_")},
                  selection="0.3 clean Online macro-F1 + 0.7 mean(four noisy simulated macro-F1s)",
                  inference="one checkpoint; baseline first-64600 crop/repeat; fake probability column 0")
    atomic_json(output / "config.json", config)
    initial = validate(clean_dev_loader, noisy_dev_loader, model, device, criterion)
    best, no_improve = selection_key(initial), 0
    best_path = ckpt / "best_model.pth"
    save_weights(best_path, model)
    atomic_json(ckpt / "best_model.meta.json", {"epoch": 0, "dev": initial})
    print("Initial dev (LOCAL proxy): " + json.dumps(initial), flush=True)
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        log.write(json.dumps({"epoch": 0, "dev": initial, "best": True}) + "\n")
        log.flush()
        for epoch in range(1, args.num_epochs + 1):
            real_sampler.set_epoch(epoch)
            noisy_sampler.set_epoch(epoch)
            train = train_epoch((ordinary_loader, real_loader, noisy_loader), model, optimizer, criterion, device, args, epoch)
            dev = validate(clean_dev_loader, noisy_dev_loader, model, device, criterion)
            key = selection_key(dev)
            improved = key > best
            log.write(json.dumps({"epoch": epoch, "train": train, "dev": dev, "best": improved}) + "\n")
            log.flush()
            save_weights(ckpt / "last_model.pth", model)
            atomic_json(ckpt / "last_model.meta.json", {"epoch": epoch, "dev": dev})
            print(f"Epoch {epoch}/{args.num_epochs} CE={train['ce']:.6f} "
                  f"RTCReal={train['rtc_real']:.6f} RTCNoisy={train['rtc_noisy']:.6f} "
                  f"NoisyWeight={train['noise_weight']:.4f} TrainAcc={train['acc']:.2f}% "
                  f"CleanOnlineF1={100*dev['clean']['online']['macro_f1']:.4f}% "
                  f"NoisySimF1={100*dev['noisy_simulated']['macro_f1']:.4f}% "
                  f"DevProxy={100*dev['proxy']:.4f}% TrainMinutes={train['seconds']/60:.2f}", flush=True)
            print("Noisy Dev bands: " + json.dumps({name: 100 * row["macro_f1"]
                  for name, row in dev["noisy_simulated"]["bands"].items()}), flush=True)
            if improved:
                best, no_improve = key, 0
                save_weights(best_path, model)
                save_weights(ckpt / f"epoch_{epoch}_proxy_{dev['proxy']:.6f}.pth", model)
                atomic_json(ckpt / "best_model.meta.json", {"epoch": epoch, "dev": dev})
                print(f"Saved best model: {best_path}", flush=True)
            else:
                no_improve += 1
            if no_improve >= args.earlystop_epoch:
                print(f"Early stopping at epoch {epoch}")
                break
    print(f"Best checkpoint: {best_path}\nCompatible with the existing run_eval_rtc.sh.")


if __name__ == "__main__":
    main()
