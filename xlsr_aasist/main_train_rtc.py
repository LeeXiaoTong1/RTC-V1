"""Fine-tune V1 with clean, officially matched Offline/Online utterances.

One forward: [V1 augmented examples, clean Offline pairs, clean Online pairs].
One CE over all examples, plus an optional RTC loss on the clean pairs only.
--rtc_weight 0 is the matched-sampling CE control, NOT the original V1 run.
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from main_train import build_arg_parser, build_loader
from utils.data_utils import build_dataset_from_protocol, set_random_seed
from utils.rtc_data import BalancedPairBatchSampler, RTCPairDataset
from utils.rtc_loss import rtc_pair_contrastive_loss
from utils.rtc_pairs import load_pairs


def build_parser():
    parser = build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(track="xlsr_aasist_V1_RTC_pair", batch_size=32,
                        num_epochs=5, earlystop_epoch=3)
    parser.add_argument("--rtc_pairs", required=True, help="Validated train pair JSONL")
    parser.add_argument("--rtc_pairs_per_batch", type=int, default=4)
    parser.add_argument("--rtc_weight", type=float, default=0.1)
    parser.add_argument("--rtc_temperature", type=float, default=0.1)
    parser.add_argument("--ssl_path", default=None, help="SSL pretrained model path")
    parser.add_argument("--encoder_lr", type=float, default=None,
                        help="SSL encoder LR; defaults to --lr when omitted")
    parser.add_argument("--backend_lr", type=float, default=None,
                        help="AASIST/backend LR; defaults to --lr when omitted")
    parser.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    parser.add_argument("--selection_metric", choices=["online_f1", "dev_loss"], default="online_f1")
    parser.add_argument("--check_data", action="store_true", help="Check data without loading the model")
    return parser


def build_grouped_optimizer(model, encoder_lr, backend_lr, weight_decay):
    encoder_params = [p for p in model.ssl_model.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder_params}
    backend_params = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    if not encoder_params or not backend_params:
        raise RuntimeError("Could not split SSL encoder and AASIST/backend parameters")
    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": encoder_lr},
        {"params": backend_params, "lr": backend_lr},
    ], weight_decay=weight_decay)
    return optimizer, encoder_params, backend_params


def train_epoch(train_loader, pair_loader, model, optimizer, device, criterion,
                rtc_weight=0.1, temperature=0.1, amp=True):
    if len(train_loader) != len(pair_loader):
        raise ValueError("Pair and main loaders must have the same number of steps")
    model.train()
    totals = dict(loss=0., ce=0., rtc=0., correct=0, examples=0, pairs=0,
                  valid_anchors=0, steps=0)
    start = time.perf_counter()
    for batch, pair_batch in tqdm(zip(train_loader, pair_loader), total=len(train_loader),
                                 desc="Training RTC", unit="batch"):
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
        ce = criterion(logits.float(), labels)
        off_features = features[ordinary_count:ordinary_count + pair_count]
        on_features = features[ordinary_count + pair_count:]
        rtc, stats = rtc_pair_contrastive_loss(off_features, on_features, pair_labels, temperature)
        loss = ce + rtc_weight * rtc
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; check the logged configuration and audio")
        loss.backward()
        optimizer.step()
        count = labels.numel()
        totals["loss"] += loss.detach().item() * count
        totals["ce"] += ce.detach().item() * count
        totals["rtc"] += rtc.detach().item()
        totals["correct"] += (logits.argmax(dim=1) == labels).sum().item()
        totals["examples"] += count
        totals["pairs"] += pair_count
        totals["valid_anchors"] += stats["valid_anchors"]
        totals["steps"] += 1
    if not totals["steps"]:
        raise ValueError("Empty training loader")
    return {"loss": totals["loss"] / totals["examples"],
            "ce": totals["ce"] / totals["examples"],
            "rtc": totals["rtc"] / totals["steps"],
            "acc": 100 * totals["correct"] / totals["examples"],
            "examples": totals["examples"], "pairs": totals["pairs"],
            "valid_anchors": totals["valid_anchors"], "steps": totals["steps"],
            "seconds": time.perf_counter() - start}


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
    if not math.isfinite(args.encoder_lr) or args.encoder_lr <= 0:
        parser.error("--encoder_lr must be finite and positive")
    if not math.isfinite(args.backend_lr) or args.backend_lr <= 0:
        parser.error("--backend_lr must be finite and positive")
    if args.batch_size < 2 or args.num_epochs < 1 or args.num_workers < 0 or args.earlystop_epoch < 1:
        parser.error("Require batch_size >= 2, num_epochs/earlystop_epoch >= 1, num_workers >= 0")
    set_random_seed(args.seed, args)
    pairs = load_pairs(args.rtc_pairs, args.train_protocol, args.train_data_path)
    train_set, train_ids, _ = build_dataset_from_protocol(
        args.train_protocol, args.train_data_path, mode="train", args=args, algo=args.algo)
    dev_set, dev_ids, _ = build_dataset_from_protocol(
        args.dev_protocol, args.dev_data_path, mode="dev", args=args)
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
    print(f"Model loaded: {args.model_path}\nDevice: {device}")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], device=device))
    optimizer, encoder_params, backend_params = build_grouped_optimizer(
        model, args.encoder_lr, args.backend_lr, args.weight_decay)
    print(f"SSL encoder parameters: {sum(p.numel() for p in encoder_params)} | LR={args.encoder_lr:.2e}")
    print(f"AASIST/backend parameters: {sum(p.numel() for p in backend_params)} | LR={args.backend_lr:.2e}")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    exp_root = Path(args.out_path) / f"{args.track}_epoch{args.num_epochs}_bs{effective_batch}_{timestamp}"
    ckpt_dir = exp_root / "ckpt"
    ckpt_dir.mkdir(parents=True)
    config = vars(args).copy()
    config.update(effective_audio_batch=effective_batch, pair_count=len(pairs),
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
    best_path = ckpt_dir / "best_model.pth"
    torch.save(model.state_dict(), best_path)
    no_improve = 0
    with (exp_root / "metrics.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"epoch": 0, "dev": initial_dev, "best": True}) + "\n")
        log.flush()
        print("Initial dev: " + json.dumps(initial_dev))
        for epoch in range(1, args.num_epochs + 1):
            pair_sampler.set_epoch(epoch)
            train = train_epoch(train_loader, pair_loader, model, optimizer, device, criterion,
                                args.rtc_weight, args.rtc_temperature, args.amp == "bf16")
            dev = evaluate_dev(dev_loader, model, device, criterion)
            key = selection_key(dev, args.selection_metric)
            improved = key > best
            row = {"epoch": epoch, "train": train, "dev": dev, "best": improved}
            log.write(json.dumps(row) + "\n")
            log.flush()
            online_f1 = "N/A" if dev["online"] is None else f"{100 * dev['online']['macro_f1']:.4f}%"
            print(f"Epoch {epoch}/{args.num_epochs} CE={train['ce']:.6f} RTC={train['rtc']:.6f} "
                  f"TrainAcc={train['acc']:.2f}% DevLoss={dev['loss']:.6f} DevOnlineF1={online_f1} "
                  f"EncoderLR={args.encoder_lr:.2e} BackendLR={args.backend_lr:.2e}")
            torch.save(model.state_dict(), ckpt_dir / "last_model.pth")
            if improved:
                best, no_improve = key, 0
                torch.save(model.state_dict(), best_path)
                torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}_dev_loss_{dev['loss']:.6f}.pth")
                print(f"Saved best model: {best_path}")
            else:
                no_improve += 1
            if no_improve >= args.earlystop_epoch:
                print(f"Early stopping at epoch {epoch}")
                break
    print(f"Experiment: {exp_root}\nBest checkpoint (includes initial V1 as a candidate): {best_path}")


if __name__ == "__main__":
    main()
