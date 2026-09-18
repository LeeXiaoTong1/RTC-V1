"""Stage 1: stable base training for w2v-BERT 2.0 + AASIST.

Design:
- AASIST first learns on frozen pretrained features.
- Only the final w2v-BERT layers are then fine-tuned.
- Class imbalance uses softened (sqrt inverse-frequency) CE weights.
- Model selection targets official Dev Online Macro-F1.
- LR decay always rolls back to the best checkpoint before continuing.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from main_train import build_arg_parser, build_loader, train_epoch
from main_train_rtc import evaluate_dev
from model.model_w2vbert import Model
from utils.data_utils import (
    build_dataset_from_protocol,
    class_weights_from_labels,
    set_random_seed,
)
from utils.w2vbert_tuning import configure_trainable_top_layers, split_trainable_params

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def build_parser():
    parser = build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(
        track="w2vbert_aasist_base",
        batch_size=40,
        num_epochs=30,
        earlystop_epoch=6,
    )
    parser.add_argument("--ssl_path", default=None)
    parser.add_argument("--encoder_trainable_layers", type=int, default=8)
    parser.add_argument("--encoder_warmup_epochs", type=int, default=2)
    parser.add_argument("--encoder_lr", type=float, default=5e-7)
    parser.add_argument("--backend_lr", type=float, default=5e-5)
    parser.add_argument("--class_weight_power", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--min_encoder_lr", type=float, default=5e-8)
    parser.add_argument("--min_backend_lr", type=float, default=5e-6)
    parser.add_argument(
        "--selection_metric",
        choices=["online_f1", "dev_loss"],
        default="online_f1",
    )
    return parser


def selection_key(dev, metric):
    if metric == "dev_loss":
        return (-dev["loss"],)
    return (dev["online"]["macro_f1"], -dev["loss"])


def monitor_value(dev, metric):
    return dev["loss"] if metric == "dev_loss" else dev["online"]["macro_f1"]


def monitor_better(value, best_value, metric):
    return value < best_value if metric == "dev_loss" else value > best_value


def main():
    args = build_parser().parse_args()
    if not 0 <= args.class_weight_power <= 1:
        raise ValueError("class_weight_power must lie in [0,1]")
    if args.encoder_warmup_epochs < 0:
        raise ValueError("encoder_warmup_epochs must be >= 0")
    if args.encoder_lr <= 0 or args.backend_lr <= 0:
        raise ValueError("encoder_lr and backend_lr must be positive")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be nonnegative")
    if not 0 < args.lr_factor < 1 or args.lr_patience < 1:
        raise ValueError("Require 0 < lr_factor < 1 and lr_patience >= 1")

    set_random_seed(args.seed, args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for p in (args.train_data_path, args.dev_data_path, args.train_protocol, args.dev_protocol):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    train_set, train_files, train_labels = build_dataset_from_protocol(
        args.train_protocol, args.train_data_path, mode="train", args=args, algo=args.algo
    )
    dev_set, dev_files, _ = build_dataset_from_protocol(
        args.dev_protocol, args.dev_data_path, mode="dev", args=args
    )
    train_loader = build_loader(train_set, args.batch_size, args.num_workers, shuffle=True)
    dev_loader = build_loader(dev_set, args.batch_size, args.num_workers, shuffle=False)

    class_weights, class_counts = class_weights_from_labels(
        train_labels, power=args.class_weight_power
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model = Model(args, device).to(device)
    tuning = configure_trainable_top_layers(model, args.encoder_trainable_layers)
    encoder_params, backend_params = split_trainable_params(model)
    if not encoder_params:
        raise RuntimeError("Stage 1 requires trainable final w2v-BERT layers")

    # During the first epochs the encoder parameters exist in the optimizer but
    # LR=0. AASIST learns how to read pretrained features before the SSL
    # representation is allowed to move.
    initial_encoder_lr = 0.0 if args.encoder_warmup_epochs > 0 else args.encoder_lr
    optimizer = torch.optim.Adam(
        [
            {"params": encoder_params, "lr": initial_encoder_lr, "name": "w2vbert"},
            {"params": backend_params, "lr": args.backend_lr, "name": "aasist"},
        ],
        weight_decay=args.weight_decay,
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    exp_root = Path(args.out_path) / (
        f"{args.track}_epoch{args.num_epochs}_bs{args.batch_size}_{timestamp}"
    )
    ckpt_dir = exp_root / "ckpt"
    log_dir = exp_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter else None

    print(f"Device: {device}")
    print(f"Train trials: {len(train_files)}; Dev trials: {len(dev_files)}")
    print(f"Class counts [fake, real]: {class_counts.tolist()}")
    print(
        f"CE weights power={args.class_weight_power}: "
        f"{[round(x, 6) for x in class_weights.tolist()]}"
    )
    print(
        f"w2v-BERT trainable layers: {tuning['trainable_layers']}/{tuning['total_layers']} "
        f"({tuning['trainable_params']}/{tuning['total_params']} params)"
    )
    print(
        f"Stage1 LR: encoder=0 for first {args.encoder_warmup_epochs} epochs, "
        f"then {args.encoder_lr:.2e}; AASIST={args.backend_lr:.2e}"
    )
    print(
        f"Adaptive LR: monitor={args.selection_metric}, bad_epochs={args.lr_patience}, "
        f"factor={args.lr_factor}, restore_best=True"
    )

    config = vars(args).copy()
    config.update(
        class_counts=class_counts.tolist(),
        class_weights=class_weights.tolist(),
        w2vbert_tuning=tuning,
        stage="stage1_base",
    )
    (exp_root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_path = ckpt_dir / "best_model.pth"
    best_key = None
    best_monitor = None
    bad_epochs = 0
    no_improve = 0

    for epoch in range(1, args.num_epochs + 1):
        if epoch == args.encoder_warmup_epochs + 1 and optimizer.param_groups[0]["lr"] == 0:
            optimizer.param_groups[0]["lr"] = args.encoder_lr
            optimizer.state.clear()
            print(
                f"Encoder warmup finished: enabled w2v-BERT LR={args.encoder_lr:.2e}; "
                "reset Adam state."
            )

        used_encoder_lr = optimizer.param_groups[0]["lr"]
        used_backend_lr = optimizer.param_groups[1]["lr"]

        train_loss, train_acc = train_epoch(
            train_loader,
            model,
            optimizer,
            device,
            criterion,
            grad_clip=args.grad_clip,
        )
        dev = evaluate_dev(dev_loader, model, device, criterion)
        key = selection_key(dev, args.selection_metric)
        monitor = monitor_value(dev, args.selection_metric)

        improved = best_key is None or key > best_key
        if improved:
            best_key = key
            best_monitor = monitor
            bad_epochs = 0
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            torch.save(
                model.state_dict(),
                ckpt_dir / f"epoch_{epoch}_online_f1_{dev['online']['macro_f1']:.6f}.pth",
            )
            print(f"Saved best model: {best_path}")
        else:
            no_improve += 1
            if monitor_better(monitor, best_monitor, args.selection_metric):
                best_monitor = monitor
                bad_epochs = 0
            else:
                bad_epochs += 1

        next_encoder_lr = optimizer.param_groups[0]["lr"]
        next_backend_lr = optimizer.param_groups[1]["lr"]
        reduced = False
        # Do not trigger LR rollback during the frozen-encoder warmup.
        if epoch > args.encoder_warmup_epochs and bad_epochs >= args.lr_patience:
            next_encoder_lr = max(
                optimizer.param_groups[0]["lr"] * args.lr_factor,
                args.min_encoder_lr,
            )
            next_backend_lr = max(
                optimizer.param_groups[1]["lr"] * args.lr_factor,
                args.min_backend_lr,
            )
            reduced = (
                next_encoder_lr < optimizer.param_groups[0]["lr"]
                or next_backend_lr < optimizer.param_groups[1]["lr"]
            )
            if reduced:
                optimizer.param_groups[0]["lr"] = next_encoder_lr
                optimizer.param_groups[1]["lr"] = next_backend_lr
                model.load_state_dict(
                    torch.load(best_path, map_location=device, weights_only=True),
                    strict=True,
                )
                optimizer.state.clear()
                bad_epochs = 0
                no_improve = 0

        online_f1 = 100 * dev["online"]["macro_f1"]
        all_f1 = 100 * dev["all"]["macro_f1"]
        message = (
            f"Epoch {epoch}/{args.num_epochs} "
            f"TrainLoss={train_loss:.6f} TrainAcc={train_acc:.2f}% "
            f"DevLoss={dev['loss']:.6f} DevAllF1={all_f1:.2f}% "
            f"DevOnlineF1={online_f1:.2f}% "
            f"EncoderLR={used_encoder_lr:.2e}->{optimizer.param_groups[0]['lr']:.2e} "
            f"BackendLR={used_backend_lr:.2e}->{optimizer.param_groups[1]['lr']:.2e}"
        )
        print(message)
        if reduced:
            print(
                "LR reduced after validation plateau; restored best_model.pth "
                "and reset Adam state."
            )
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().replace(microsecond=0)}] {message}\n")

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("F1/dev_online", dev["online"]["macro_f1"], epoch)
            writer.add_scalar("F1/dev_all", dev["all"]["macro_f1"], epoch)
            writer.add_scalar("LR/encoder", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("LR/backend", optimizer.param_groups[1]["lr"], epoch)

        if no_improve >= args.earlystop_epoch:
            print(f"Early stopping at epoch {epoch}")
            break

    if writer:
        writer.close()
    print(f"Experiment saved to: {exp_root}")
    print(f"Best model: {best_path}")


if __name__ == "__main__":
    main()
