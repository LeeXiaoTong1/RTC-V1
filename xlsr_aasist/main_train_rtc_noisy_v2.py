"""V2: separate class/condition weights, rotate one view, validate held-out RTC."""
import json
import math
import os
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from main_train import build_loader
from main_train_rtc_noisy import build_parser as v1_parser, save_weights
from rtc_noisy.common import atomic_json, sha256
from rtc_noisy.data import NoisyDevDataset
from rtc_noisy_v2.cache import (RotatingNoisyDataset, check_suite, inspect_initial_history, load_v2_cache)
from rtc_noisy_v2.plan import PLAN_ID, group_coefficients
from rtc_noisy_v2.sampling import RotatingViewBatchSampler
from rtc_noisy_v2.training import selection_key, train_epoch, validate
from utils.data_utils import build_dataset_from_protocol, class_weights_from_labels, set_random_seed
from utils.rtc_data import BalancedPairBatchSampler, RTCPairDataset
from utils.rtc_pairs import load_pairs
from utils.w2vbert_tuning import configure_trainable_top_layers, split_trainable_params


def build_parser():
    p = v1_parser()
    p.description = __doc__
    p.set_defaults(track="xlsr_aasist_RTC_noisy_v2")
    p.add_argument("--noisy_ce_weight", type=float, default=.3)
    p.add_argument("--dev_heldout_cache", required=True)
    p.add_argument("--extra_train_noisy_cache", action="append", default=[],
                   help="Optional fresh bank with another generation; one view per source remains")
    p.add_argument("--encoder_lr", type=float, default=None)
    p.add_argument("--backend_lr", type=float, default=None)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--lr_patience", type=int, default=2,
                   help="Consecutive bad DevRobustProxy epochs before reducing LR")
    p.add_argument("--min_encoder_lr", type=float, default=1e-8)
    p.add_argument("--min_backend_lr", type=float, default=2e-7)
    p.add_argument("--encoder_trainable_layers", type=int, default=4,
                   help="Fine-tune only the final N of 24 w2v-BERT layers")
    return p


def build_grouped_optimizer(model, encoder_lr, backend_lr, weight_decay):
    encoder_params, backend_params = split_trainable_params(model)
    if not encoder_params:
        raise RuntimeError("Stage 3 expects at least one trainable w2v-BERT layer")
    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": encoder_lr, "name": "ssl"},
        {"params": backend_params, "lr": backend_lr, "name": "backend"},
    ], weight_decay=weight_decay)
    return optimizer, encoder_params, backend_params


def summarize_dev(metrics):
    return {"CleanOnlineF1": 100*metrics["clean"]["online"]["macro_f1"],
            "NoisySeenF1": 100*metrics["noisy_seen"]["macro_f1"],
            "NoisyHeldoutF1": 100*metrics["noisy_heldout"]["macro_f1"],
            "DevProxyV1": 100*metrics["proxy_v1"], "DevRobustProxy": 100*metrics["robust_proxy"]}


def main():
    p = build_parser()
    args = p.parse_args()
    args.encoder_lr = args.lr if args.encoder_lr is None else args.encoder_lr
    args.backend_lr = args.lr if args.backend_lr is None else args.backend_lr
    for name in ("rtc_weight", "noisy_weight", "noisy_warmup_epochs", "grad_clip", "weight_decay"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"{name} must be finite and nonnegative")
    if not 0 <= args.noisy_ce_weight <= 1:
        p.error("noisy_ce_weight must lie in [0,1]")
    for name in ("encoder_lr", "backend_lr", "rtc_temperature"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"{name} must be finite and positive")
    if not 0 < args.lr_factor < 1 or args.lr_patience < 0:
        p.error("Require 0 < lr_factor < 1 and lr_patience >= 0")
    if args.batch_size < 2 or args.num_workers < 0 or args.num_epochs < 1 or args.earlystop_epoch < 1:
        p.error("Invalid batch size, workers, epochs or patience")
    for name in ("rtc_pairs_per_batch", "noisy_pairs_per_batch"):
        if getattr(args, name) < 2 or getattr(args, name) % 2:
            p.error(f"{name} must be even and >=2")

    set_random_seed(args.seed, args)
    real_pairs = load_pairs(args.rtc_pairs, args.train_protocol, args.train_data_path)
    folders = [args.train_noisy_cache, *args.extra_train_noisy_cache]
    banks = [load_v2_cache(path, "train", args.train_protocol, args.train_data_path) for path in folders]
    seen = load_v2_cache(args.dev_noisy_cache, "dev_seen", args.dev_protocol, args.dev_data_path)
    heldout = load_v2_cache(args.dev_heldout_cache, "dev_heldout", args.dev_protocol, args.dev_data_path)
    check_suite(banks, seen, heldout)
    manifest_hash = sha256(args.train_noise_manifest)
    if any(config["noise"]["manifest_sha256"] != manifest_hash for _, config in banks):
        p.error("All Train banks and ordinary MUSAN must use the same noise manifest")
    os.environ["RTC_B_NOISE_MANIFEST"] = str(Path(args.train_noise_manifest).resolve())

    ordinary, train_ids, train_labels = build_dataset_from_protocol(
        args.train_protocol, args.train_data_path, mode="train", args=args, algo=args.algo)
    clean_dev, dev_ids, _ = build_dataset_from_protocol(
        args.dev_protocol, args.dev_data_path, mode="dev", args=args)
    class_weights, class_counts = class_weights_from_labels(train_labels)
    args.class_weights = class_weights.tolist()

    noisy_train = RotatingNoisyDataset([rows for rows, _ in banks], args.train_data_path)
    args.cache_banks = len(banks)
    steps = math.ceil(len(ordinary)/args.batch_size)
    effective = args.batch_size + 2*args.rtc_pairs_per_batch + 2*args.noisy_pairs_per_batch
    real_sampler = BalancedPairBatchSampler(real_pairs, args.rtc_pairs_per_batch, steps, args.seed+1)
    source_sampler = BalancedPairBatchSampler(noisy_train.sources, args.noisy_pairs_per_batch, steps, args.seed+2)
    rotation = RotatingViewBatchSampler(source_sampler, noisy_train.sources, args.seed+3, len(banks))

    print(f"Batch: {args.batch_size} ordinary + {args.rtc_pairs_per_batch} real pairs x2 + "
          f"{args.noisy_pairs_per_batch} noisy pairs x2 = {effective} waveforms; steps={steps}")
    print(f"Train trials={len(train_ids)}; Dev trials={len(dev_ids)}; noisy sources={len(noisy_train)}; banks={len(banks)}")
    print(f"Class counts [fake, real]: {class_counts.tolist()}")
    print(f"Auto ordinary CE weights [fake, real]: {[round(x, 6) for x in class_weights.tolist()]}; paired CE=equal")
    print("CE coefficients [ordinary, real pair, noisy reference, noisy processed]: " +
          str(group_coefficients(args.batch_size, 2*args.rtc_pairs_per_batch, args.noisy_pairs_per_batch, args.noisy_ce_weight)))
    print("One noisy view/source; 21 training combinations / 6 held-out combinations")
    print(f"Ordinary MUSAN={'enabled' if ordinary.env_noise is not None else 'DISABLED'}; RawBoost={args.algo}")

    if args.check_data:
        rotation.set_epoch(1)
        print("First rotation tickets (source, bank, band):", next(iter(rotation)))
        rotation.discard_epoch()
        print("V2 data checks passed; no model loaded")
        return
    if not args.model_path or not Path(args.model_path).is_file():
        p.error("Provide an existing state_dict checkpoint; recommended: the pre-FFmpeg real RTC pair model")

    history = inspect_initial_history(args.model_path)
    print("Initialization history:", json.dumps(history, ensure_ascii=False))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable; no silent CPU fallback")
    if device.type == "cuda" and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        p.error("BF16 is unsupported; use --amp none")

    from model.model import Model
    model = Model(args, device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True), strict=True)
    tuning = configure_trainable_top_layers(model, args.encoder_trainable_layers)
    print(f"Model loaded: {args.model_path}\nDevice: {device}")
    print(f"w2v-BERT trainable layers: {tuning['trainable_layers']}/{tuning['total_layers']} "
          f"({tuning['trainable_params']}/{tuning['total_params']} params)")

    ordinary_loader = build_loader(ordinary, args.batch_size, args.num_workers, shuffle=True)
    worker_count = min(2, args.num_workers)
    kwargs = {"num_workers": worker_count, "pin_memory": device.type == "cuda"}
    if worker_count:
        kwargs["persistent_workers"] = True
    real_loader = DataLoader(RTCPairDataset(real_pairs, args.train_data_path), batch_sampler=real_sampler, **kwargs)
    noisy_loader = DataLoader(noisy_train, batch_sampler=rotation, **kwargs)
    dev_kwargs = {"batch_size": effective, "shuffle": False, "num_workers": min(4, args.num_workers),
                  "pin_memory": device.type == "cuda"}
    clean_loader = DataLoader(clean_dev, **dev_kwargs)
    seen_loader = DataLoader(NoisyDevDataset(seen[0]), **dev_kwargs)
    heldout_loader = DataLoader(NoisyDevDataset(heldout[0]), **dev_kwargs)

    dev_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer, encoder_params, backend_params = build_grouped_optimizer(
        model, args.encoder_lr, args.backend_lr, args.weight_decay)
    print(f"SSL encoder parameters: {sum(p.numel() for p in encoder_params)} | LR={args.encoder_lr:.2e}")
    print(f"AASIST/backend parameters: {sum(p.numel() for p in backend_params)} | LR={args.backend_lr:.2e}")
    print(f"Adaptive LR: monitor=DevRobustProxy; bad_epochs={args.lr_patience}; factor={args.lr_factor}; "
          f"min=[{args.min_encoder_lr:.2e}, {args.min_backend_lr:.2e}]; restore_best=True")

    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    output = Path(args.out_path)/f"{args.track}_epoch{args.num_epochs}_bs{effective}_{stamp}"
    ckpt = output/"ckpt"
    ckpt.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update(plan_id=PLAN_ID, initial_checkpoint_sha256=sha256(args.model_path),
                  initialization_history=history, effective_audio_batch=effective,
                  class_counts=class_counts.tolist(), class_weights=class_weights.tolist(),
                  w2vbert_tuning=tuning,
                  train_cache_configs=[cfg for _, cfg in banks], dev_seen_config=seen[1],
                  dev_heldout_config=heldout[1],
                  train_manifest_hashes=[sha256(Path(path)/"manifest.jsonl") for path in folders],
                  dev_manifest_hashes=[sha256(Path(path)/"manifest.jsonl")
                                       for path in (args.dev_noisy_cache, args.dev_heldout_cache)],
                  pair_manifest_sha256=sha256(args.rtc_pairs),
                  noise_environment={k:v for k,v in os.environ.items() if k.startswith("RTC_B_")},
                  torch_version=str(torch.__version__),
                  selection="0.3 clean Online + 0.35 noisy seen + 0.35 noisy heldout",
                  inference="unchanged single-checkpoint first-64600 crop/repeat, spoof column 0")
    atomic_json(output/"config.json", config)

    dev = validate(clean_loader, seen_loader, heldout_loader, model, device, dev_criterion)
    print("Initial dev V2:", json.dumps(summarize_dev(dev)), flush=True)
    best, no_improve = selection_key(dev), 0
    best_monitor = dev["robust_proxy"]
    lr_bad_epochs = 0

    def save_named(name, epoch, metrics):
        save_weights(ckpt/f"{name}.pth", model)
        atomic_json(ckpt/f"{name}.meta.json", {"epoch":epoch, "dev":metrics,
                    "sampling_state":rotation.state_dict(), "initialization_history":history,
                    "note":"weights and committed view history; optimizer/RNG are not saved for exact resume"})

    save_named("best_model", 0, dev)
    with (output/"metrics.jsonl").open("w", encoding="utf-8") as log:
        log.write(json.dumps({"epoch":0,"dev":dev,"best":True})+"\n")
        log.flush()
        for epoch in range(1, args.num_epochs+1):
            used_encoder_lr, used_backend_lr = optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]
            real_sampler.set_epoch(epoch)
            rotation.set_epoch(epoch)
            train = train_epoch((ordinary_loader, real_loader, noisy_loader), model, optimizer, device, args, epoch)
            rotation.commit_epoch(train["steps"])
            dev = validate(clean_loader, seen_loader, heldout_loader, model, device, dev_criterion)
            key = selection_key(dev)
            improved = key > best

            save_named("last_model", epoch, dev)
            if improved:
                best, no_improve = key, 0
                save_named("best_model", epoch, dev)
                print(f"Saved best model: {ckpt/'best_model.pth'}", flush=True)
            else:
                no_improve += 1

            monitor = dev["robust_proxy"]
            if monitor > best_monitor:
                best_monitor = monitor
                lr_bad_epochs = 0
            else:
                lr_bad_epochs += 1

            next_encoder_lr, next_backend_lr = used_encoder_lr, used_backend_lr
            reduced = False
            if lr_bad_epochs >= args.lr_patience:
                next_encoder_lr = max(used_encoder_lr * args.lr_factor, args.min_encoder_lr)
                next_backend_lr = max(used_backend_lr * args.lr_factor, args.min_backend_lr)
                reduced = next_encoder_lr < used_encoder_lr or next_backend_lr < used_backend_lr
                if reduced:
                    optimizer.param_groups[0]["lr"] = next_encoder_lr
                    optimizer.param_groups[1]["lr"] = next_backend_lr
                    model.load_state_dict(
                        torch.load(ckpt/"best_model.pth", map_location=device, weights_only=True),
                        strict=True,
                    )
                    optimizer.state.clear()
                    lr_bad_epochs = 0
                    no_improve = 0

            print(f"Epoch {epoch}/{args.num_epochs} CE={train['ce']:.6f} "
                  f"CEOrd={train['ce_ordinary']:.6f} CEReal={train['ce_real_pair']:.6f} "
                  f"CERef={train['ce_noisy_reference']:.6f} CENoisy={train['ce_noisy_processed']:.6f} "
                  f"RTCReal={train['rtc_real']:.6f} RTCNoisy={train['rtc_noisy']:.6f} "
                  f"NoisyCEWeight={args.noisy_ce_weight:.3f} NoisyWeight={train['noise_weight']:.4f} "
                  f"TrainAcc={train['acc']:.2f}% "
                  f"EncoderLR={used_encoder_lr:.2e}->{next_encoder_lr:.2e} "
                  f"BackendLR={used_backend_lr:.2e}->{next_backend_lr:.2e} "
                  f"TrainMinutes={train['seconds']/60:.2f}", flush=True)
            print("Dev V2:", json.dumps(summarize_dev(dev)), flush=True)
            if reduced:
                print("LR reduced after consecutive bad DevRobustProxy epochs; restored best_model.pth and reset Adam state.", flush=True)
            elif lr_bad_epochs >= args.lr_patience:
                print("DevRobustProxy plateaued, but both learning rates are already at their configured minimum.", flush=True)
            for role in ("noisy_seen", "noisy_heldout"):
                print(role + " bands:", json.dumps({name:{"F1":100*v["macro_f1"],
                      "FakeAsReal":100*v["fake_as_real_rate"], "RealAsFake":100*v["real_as_fake_rate"]}
                      for name,v in dev[role]["bands"].items()}), flush=True)

            log.write(json.dumps({"epoch":epoch,"train":train,"dev":dev,"best":improved,
                                  "lr":{"encoder":next_encoder_lr,"backend":next_backend_lr}})+"\n")
            log.flush()
            if no_improve >= args.earlystop_epoch:
                print(f"Early stopping at epoch {epoch}")
                break
    print(f"Best checkpoint: {ckpt/'best_model.pth'}; use the w2v-BERT evaluation script")


if __name__ == "__main__":
    main()
