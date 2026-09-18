"""Fine-tune V1 with clean, officially matched Offline/Online utterances.

One forward: [ordinary augmented examples, clean Offline pairs, clean Online pairs].
Ordinary CE uses protocol-derived class weights; balanced RTC-pair CE is unweighted.
A warm-started RTC contrastive loss aligns verified Offline/Online counterparts.
"""
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from main_train import build_arg_parser, build_loader
from utils.data_utils import build_dataset_from_protocol, class_weights_from_labels, set_random_seed
from utils.rtc_data import BalancedPairBatchSampler, RTCPairDataset
from utils.rtc_loss import rtc_pair_contrastive_loss
from utils.rtc_pairs import load_pairs
from utils.w2vbert_tuning import configure_trainable_top_layers, split_trainable_params


def build_parser():
    parser = build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(track="xlsr_aasist_V1_RTC_pair", batch_size=24,
                        num_epochs=10, earlystop_epoch=5)
    parser.add_argument("--rtc_pairs", required=True, help="Validated train pair JSONL")
    parser.add_argument("--rtc_pairs_per_batch", type=int, default=4)
    parser.add_argument("--rtc_weight", type=float, default=0.05)
    parser.add_argument("--rtc_warmup_epochs", type=float, default=3.0,
                        help="Linearly ramp RTC contrastive weight from 0 to rtc_weight")
    parser.add_argument("--rtc_temperature", type=float, default=0.1)
    parser.add_argument("--ssl_path", default=None, help="SSL pretrained model path")
    parser.add_argument("--encoder_lr", type=float, default=None)
    parser.add_argument("--backend_lr", type=float, default=None)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2,
                        help="Consecutive bad validation epochs before reducing LR")
    parser.add_argument("--min_encoder_lr", type=float, default=1e-8)
    parser.add_argument("--min_backend_lr", type=float, default=1e-7)
    parser.add_argument("--encoder_trainable_layers", type=int, default=4,
                        help="Fine-tune only the final N of 24 w2v-BERT layers")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--class_weight_power", type=float, default=0.5)
    parser.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    parser.add_argument("--selection_metric", choices=["online_f1", "dev_loss"], default="online_f1")
    parser.add_argument("--check_data", action="store_true", help="Check data without loading the model")
    return parser


def build_grouped_optimizer(model, encoder_lr, backend_lr, weight_decay):
    encoder_params, backend_params = split_trainable_params(model)
    if not encoder_params:
        raise RuntimeError("Stage 2 expects at least one trainable w2v-BERT layer")
    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": encoder_lr, "name": "ssl"},
        {"params": backend_params, "lr": backend_lr, "name": "backend"},
    ], weight_decay=weight_decay)
    return optimizer, encoder_params, backend_params


def train_epoch(train_loader, pair_loader, model, optimizer, device, criterion,
                rtc_weight=0.1, rtc_warmup_epochs=2.0, temperature=0.1,
                amp=True, grad_clip=1.0, epoch=1):
    """Stage-2 objective with condition-aware classification.

    Ordinary samples follow protocol-derived class weights because that stream is
    imbalanced. RTC pair batches are deliberately 1:1 real/fake, so their CE is
    unweighted. This avoids multiplying real RTC gradients by the global class
    imbalance ratio.
    """
    if len(train_loader) != len(pair_loader):
        raise ValueError("Pair and main loaders must have the same number of steps")
    model.train()
    totals = dict(loss=0., ce=0., ce_ordinary=0., ce_pair=0., rtc=0.,
                  rtc_weight=0., correct=0, examples=0, pairs=0,
                  valid_anchors=0, steps=0)
    start_time = time.perf_counter()
    steps = len(train_loader)

    for step, (batch, pair_batch) in enumerate(
        tqdm(zip(train_loader, pair_loader), total=steps, desc="Training RTC", unit="batch")
    ):
        audio, labels, _ = batch
        offline, online, pair_labels = pair_batch
        ordinary_count, pair_count = audio.shape[0], offline.shape[0]
        audio = torch.cat([audio, offline, online], dim=0).to(device, non_blocking=True)
        labels = torch.cat([labels, pair_labels, pair_labels]).long().to(device, non_blocking=True)
        pair_labels = pair_labels.long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if amp and device.type == "cuda" else nullcontext()
        with context:
            logits, features = model(audio, return_features=True)

        per_sample_ce = F.cross_entropy(logits.float(), labels, reduction="none")
        ordinary_labels = labels[:ordinary_count]
        ordinary_weights = criterion.weight[ordinary_labels]
        ce_ordinary = (per_sample_ce[:ordinary_count] * ordinary_weights).sum() / ordinary_weights.sum()
        ce_pair = per_sample_ce[ordinary_count:].mean()
        ce = (
            ordinary_count * ce_ordinary + (2 * pair_count) * ce_pair
        ) / (ordinary_count + 2 * pair_count)

        off_features = features[ordinary_count:ordinary_count + pair_count]
        on_features = features[ordinary_count + pair_count:]
        rtc, stats = rtc_pair_contrastive_loss(
            off_features, on_features, pair_labels, temperature
        )
        if rtc_warmup_epochs <= 0:
            current_rtc_weight = rtc_weight
        else:
            progress = ((epoch - 1) * steps + step + 1) / (rtc_warmup_epochs * steps)
            current_rtc_weight = rtc_weight * min(1.0, progress)

        loss = ce + current_rtc_weight * rtc
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; check configuration and audio")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()

        count = labels.numel()
        totals["loss"] += loss.detach().item() * count
        totals["ce"] += ce.detach().item()
        totals["ce_ordinary"] += ce_ordinary.detach().item()
        totals["ce_pair"] += ce_pair.detach().item()
        totals["rtc"] += rtc.detach().item()
        totals["rtc_weight"] += current_rtc_weight
        totals["correct"] += (logits.argmax(dim=1) == labels).sum().item()
        totals["examples"] += count
        totals["pairs"] += pair_count
        totals["valid_anchors"] += stats["valid_anchors"]
        totals["steps"] += 1

    if not totals["steps"]:
        raise ValueError("Empty training loader")
    return {
        "loss": totals["loss"] / totals["examples"],
        "ce": totals["ce"] / totals["steps"],
        "ce_ordinary": totals["ce_ordinary"] / totals["steps"],
        "ce_pair": totals["ce_pair"] / totals["steps"],
        "rtc": totals["rtc"] / totals["steps"],
        "rtc_weight": totals["rtc_weight"] / totals["steps"],
        "acc": 100 * totals["correct"] / totals["examples"],
        "examples": totals["examples"], "pairs": totals["pairs"],
        "valid_anchors": totals["valid_anchors"], "steps": totals["steps"],
        "seconds": time.perf_counter() - start_time,
    }


def confusion_metrics(cm):
    """Rows=true, columns=predicted; 0=spoof, 1=real. Macro-F1 in [0,1]."""
    count = int(cm.sum())
    if not count:
        return None
    denominator = cm.sum(0) + cm.sum(1)
    f1 = torch.where(denominator > 0, 2 * cm.diag() / denominator, 0.)
    return {"macro_f1": f1.mean().item(), "acc": 100 * cm.diag().sum().item() / count,
            "count": count, "confusion": cm.tolist()}


@torch.no_grad()
def evaluate_dev(loader, model, device, criterion):
    model.eval()
    matrices = {name: torch.zeros(2, 2, dtype=torch.float64) for name in ("all", "online", "offline")}
    loss_numerator = 0.
    loss_denominator = 0.
    for audio, labels, ids in tqdm(loader, desc="Validating", unit="batch"):
        labels = labels.long().to(device)
        logits = model(audio.to(device, non_blocking=True)).float()
        predicted = (logits.softmax(dim=1)[:, 0] < 0.5).long()
        mass = criterion.weight[labels].sum().item() if criterion.weight is not None else labels.numel()
        loss_numerator += criterion(logits, labels).item() * mass
        loss_denominator += mass
        codes = (2 * labels + predicted).cpu()
        matrices["all"] += torch.bincount(codes, minlength=4).reshape(2, 2)
        for domain in ("online", "offline"):
            mask = torch.tensor([domain in Path(utt).parts for utt in ids], dtype=torch.bool)
            matrices[domain] += torch.bincount(codes[mask], minlength=4).reshape(2, 2)
    if loss_denominator == 0:
        raise ValueError("Empty dev loader")
    result = {name: confusion_metrics(cm) for name, cm in matrices.items()}
    result["loss"] = loss_numerator / loss_denominator
    return result


def selection_key(dev, metric):
    if metric == "dev_loss":
        return (-dev["loss"],)
    if dev["online"] is None:
        raise ValueError("No online paths in dev protocol; online_f1 selection needs the Online subset")
    return (dev["online"]["macro_f1"], -dev["loss"])


def scheduler_value(dev, metric):
    return dev["loss"] if metric == "dev_loss" else dev["online"]["macro_f1"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.encoder_lr = args.lr if args.encoder_lr is None else args.encoder_lr
    args.backend_lr = args.lr if args.backend_lr is None else args.backend_lr
    if not math.isfinite(args.rtc_weight) or args.rtc_weight < 0:
        parser.error("--rtc_weight must be finite and non-negative")
    if not math.isfinite(args.rtc_temperature) or args.rtc_temperature <= 0:
        parser.error("--rtc_temperature must be finite and positive")
    if not math.isfinite(args.rtc_warmup_epochs) or args.rtc_warmup_epochs < 0:
        parser.error("--rtc_warmup_epochs must be finite and non-negative")
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        parser.error("--grad_clip must be finite and non-negative")
    if not 0 <= args.class_weight_power <= 1:
        parser.error("--class_weight_power must lie in [0,1]")
    if not math.isfinite(args.encoder_lr) or args.encoder_lr <= 0:
        parser.error("--encoder_lr must be finite and positive")
    if not math.isfinite(args.backend_lr) or args.backend_lr <= 0:
        parser.error("--backend_lr must be finite and positive")
    if not 0 < args.lr_factor < 1 or args.lr_patience < 0:
        parser.error("Require 0 < lr_factor < 1 and lr_patience >= 0")
    if args.batch_size < 2 or args.num_epochs < 1 or args.num_workers < 0 or args.earlystop_epoch < 1:
        parser.error("Require batch_size >= 2, num_epochs/earlystop_epoch >= 1, num_workers >= 0")

    set_random_seed(args.seed, args)
    pairs = load_pairs(args.rtc_pairs, args.train_protocol, args.train_data_path)
    train_set, train_ids, train_labels = build_dataset_from_protocol(
        args.train_protocol, args.train_data_path, mode="train", args=args, algo=args.algo)
    dev_set, dev_ids, _ = build_dataset_from_protocol(
        args.dev_protocol, args.dev_data_path, mode="dev", args=args)
    class_weights, class_counts = class_weights_from_labels(
        train_labels, power=args.class_weight_power
    )

    for root, ids in ((args.train_data_path, train_ids), (args.dev_data_path, dev_ids)):
        for utt in ids:
            path = Path(root) / utt
            if not path.is_file():
                raise FileNotFoundError(path)
    if args.selection_metric == "online_f1" and not any("online" in Path(utt).parts for utt in dev_ids):
        parser.error("Dev protocol has no Online subset for --selection_metric online_f1")

    train_loader = build_loader(train_set, args.batch_size, args.num_workers, shuffle=True)
    effective_batch = args.batch_size + 2 * args.rtc_pairs_per_batch
    dev_loader = build_loader(dev_set, effective_batch, args.num_workers, shuffle=False)
    pair_sampler = BalancedPairBatchSampler(pairs, pairs_per_batch=args.rtc_pairs_per_batch,
                                           steps_per_epoch=len(train_loader), seed=args.seed + 1)
    pair_workers = min(2, args.num_workers)
    pair_loader = DataLoader(RTCPairDataset(pairs, args.train_data_path), batch_sampler=pair_sampler,
                            num_workers=pair_workers, pin_memory=True,
                            **({"persistent_workers": True} if pair_workers else {}))

    print(f"Train trials: {len(train_ids)}; Dev trials: {len(dev_ids)}; official RTC pairs: {len(pairs)}")
    print(f"Batch: {args.batch_size} V1 samples + {args.rtc_pairs_per_batch} clean pairs x 2 = {effective_batch} audios")
    print(f"Class counts [fake, real]: {class_counts.tolist()}")
    print(f"CE weights power={args.class_weight_power} [fake, real]: "
          f"{[round(x, 6) for x in class_weights.tolist()]}")
    print(f"RTC weight={args.rtc_weight}; temperature={args.rtc_temperature}; selection={args.selection_metric}")
    if train_set.env_noise is None:
        print("MUSAN augmentation is disabled: set RTC_B_NOISE_MANIFEST to retain V1 noise augmentation.")
    if args.check_data:
        print("Data checks passed. No model loaded and no training started.")
        return
    if not args.model_path or not Path(args.model_path).is_file():
        parser.error("Provide an existing V1 checkpoint with --model_path")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; fix the PyTorch/driver environment or explicitly use --device cpu")
    if device.type == "cuda" and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("This GPU does not support BF16; use --amp none")

    from model.model import Model
    model = Model(args, device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True), strict=True)
    model.ssl_model.deterministic_finetune = True
    tuning = configure_trainable_top_layers(model, args.encoder_trainable_layers)
    print(f"Model loaded: {args.model_path}\nDevice: {device}")
    print(f"w2v-BERT trainable layers: {tuning['trainable_layers']}/{tuning['total_layers']} "
          f"({tuning['trainable_params']}/{tuning['total_params']} params)")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer, encoder_params, backend_params = build_grouped_optimizer(
        model, args.encoder_lr, args.backend_lr, args.weight_decay)
    print(f"SSL encoder parameters: {sum(p.numel() for p in encoder_params)} | LR={args.encoder_lr:.2e}")
    print(f"AASIST/backend parameters: {sum(p.numel() for p in backend_params)} | LR={args.backend_lr:.2e}")
    print(f"Adaptive LR: monitor={args.selection_metric}; bad_epochs={args.lr_patience}; "
          f"factor={args.lr_factor}; min=[{args.min_encoder_lr:.2e}, {args.min_backend_lr:.2e}]; "
          "restore_best=True")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    exp_root = Path(args.out_path) / f"{args.track}_epoch{args.num_epochs}_bs{effective_batch}_{timestamp}"
    ckpt_dir = exp_root / "ckpt"
    ckpt_dir.mkdir(parents=True)
    config = vars(args).copy()
    config.update(effective_audio_batch=effective_batch, pair_count=len(pairs),
                  class_counts=class_counts.tolist(), class_weights=class_weights.tolist(),
                  class_weight_power=args.class_weight_power,
                  w2vbert_tuning=tuning,
                  train_protocol_sha256=sha256_file(args.train_protocol),
                  dev_protocol_sha256=sha256_file(args.dev_protocol),
                  pair_manifest_sha256=sha256_file(args.rtc_pairs),
                  initial_checkpoint_sha256=sha256_file(args.model_path),
                  torch_version=torch.__version__,
                  noise_environment={k: v for k, v in os.environ.items() if k.startswith("RTC_B_")},
                  noise_manifest_sha256=getattr(train_set.env_noise, "manifest_sha256", None))
    (exp_root / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    initial_dev = evaluate_dev(dev_loader, model, device, criterion)
    best = selection_key(initial_dev, args.selection_metric)
    best_monitor = scheduler_value(initial_dev, args.selection_metric)
    best_path = ckpt_dir / "best_model.pth"
    torch.save(model.state_dict(), best_path)
    no_improve = 0
    lr_bad_epochs = 0

    with (exp_root / "metrics.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"epoch": 0, "dev": initial_dev, "best": True}) + "\n")
        log.flush()
        print("Initial dev: " + json.dumps(initial_dev))
        for epoch in range(1, args.num_epochs + 1):
            used_encoder_lr, used_backend_lr = optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]
            pair_sampler.set_epoch(epoch)
            train = train_epoch(
                train_loader, pair_loader, model, optimizer, device, criterion,
                rtc_weight=args.rtc_weight,
                rtc_warmup_epochs=args.rtc_warmup_epochs,
                temperature=args.rtc_temperature,
                amp=args.amp == "bf16",
                grad_clip=args.grad_clip,
                epoch=epoch,
            )
            dev = evaluate_dev(dev_loader, model, device, criterion)
            key = selection_key(dev, args.selection_metric)
            improved = key > best

            torch.save(model.state_dict(), ckpt_dir / "last_model.pth")
            if improved:
                best, no_improve = key, 0
                torch.save(model.state_dict(), best_path)
                torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}_dev_loss_{dev['loss']:.6f}.pth")
                print(f"Saved best model: {best_path}")
            else:
                no_improve += 1

            monitor = scheduler_value(dev, args.selection_metric)
            monitor_improved = (monitor < best_monitor) if args.selection_metric == "dev_loss" else (monitor > best_monitor)
            if monitor_improved:
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
                    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True), strict=True)
                    optimizer.state.clear()
                    lr_bad_epochs = 0
                    no_improve = 0
            online_f1 = "N/A" if dev["online"] is None else f"{100 * dev['online']['macro_f1']:.4f}%"
            print(f"Epoch {epoch}/{args.num_epochs} CE={train['ce']:.6f} "
                  f"CEOrd={train['ce_ordinary']:.6f} CEPair={train['ce_pair']:.6f} "
                  f"RTC={train['rtc']:.6f} RTCWeight={train['rtc_weight']:.4f} "
                  f"TrainAcc={train['acc']:.2f}% DevLoss={dev['loss']:.6f} DevOnlineF1={online_f1} "
                  f"EncoderLR={used_encoder_lr:.2e}->{next_encoder_lr:.2e} "
                  f"BackendLR={used_backend_lr:.2e}->{next_backend_lr:.2e}")
            if reduced:
                print("LR reduced after consecutive bad validation epochs; restored best_model.pth and reset Adam state.")
            elif lr_bad_epochs >= args.lr_patience:
                print("Validation plateaued, but both learning rates are already at their configured minimum.")

            row = {"epoch": epoch, "train": train, "dev": dev, "best": improved,
                   "lr": {"encoder": next_encoder_lr, "backend": next_backend_lr}}
            log.write(json.dumps(row) + "\n")
            log.flush()

            if no_improve >= args.earlystop_epoch:
                print(f"Early stopping at epoch {epoch}")
                break
    print(f"Experiment: {exp_root}\nBest checkpoint (includes initial V1 as a candidate): {best_path}")


if __name__ == "__main__":
    main()
