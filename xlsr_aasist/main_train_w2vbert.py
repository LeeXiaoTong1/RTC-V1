"""Stage-1 trainer for w2v-BERT 2.0 + AASIST.

- Separate encoder/backend learning rates.
- Class weights are derived from the actual Train protocol.
- Validation-driven LR reduction restores the best checkpoint before continuing.
"""
import os
from datetime import datetime

import torch
from torch import nn

from main_train import build_arg_parser, build_loader, train_epoch, evaluate_dev
from model.model_w2vbert import Model
from utils.data_utils import build_dataset_from_protocol, class_weights_from_labels, set_random_seed
from utils.w2vbert_tuning import configure_trainable_top_layers, split_trainable_params

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def build_parser():
    parser = build_arg_parser()
    parser.description = "Train w2v-BERT 2.0 + AASIST with adaptive LR and protocol-derived class weights"
    parser.set_defaults(track="w2vbert_aasist_base")
    parser.add_argument("--ssl_path", default=None)
    parser.add_argument("--encoder_lr", type=float, default=1e-6)
    parser.add_argument("--backend_lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2,
                        help="Consecutive bad validation epochs before reducing LR")
    parser.add_argument("--min_encoder_lr", type=float, default=1e-7)
    parser.add_argument("--min_backend_lr", type=float, default=1e-5)
    parser.add_argument("--encoder_trainable_layers", type=int, default=24,
                        help="How many final w2v-BERT Conformer layers are trainable")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser


def main():
    args = build_parser().parse_args()
    if args.encoder_lr <= 0 or args.backend_lr <= 0:
        raise ValueError("encoder_lr and backend_lr must be positive")
    if not 0 < args.lr_factor < 1 or args.lr_patience < 0:
        raise ValueError("Require 0 < lr_factor < 1 and lr_patience >= 0")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be nonnegative")

    set_random_seed(args.seed, args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for path in [args.train_data_path, args.dev_data_path, args.train_protocol, args.dev_protocol]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    exp_name = f"{args.track}_epoch{args.num_epochs}_bs{args.batch_size}_{timestamp}"
    exp_root = os.path.join(args.out_path, exp_name)
    ckpt_dir = os.path.join(exp_root, "ckpt")
    log_dir = os.path.join(exp_root, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    train_set, train_files, train_labels = build_dataset_from_protocol(
        args.train_protocol, args.train_data_path, mode="train", args=args, algo=args.algo
    )
    dev_set, dev_files, _ = build_dataset_from_protocol(
        args.dev_protocol, args.dev_data_path, mode="dev", args=args
    )
    train_loader = build_loader(train_set, args.batch_size, args.num_workers, shuffle=True)
    dev_loader = build_loader(dev_set, args.batch_size, args.num_workers, shuffle=False)

    class_weights, class_counts = class_weights_from_labels(train_labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model = Model(args, device).to(device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print(f"Model loaded: {args.model_path}")

    tuning = configure_trainable_top_layers(model, args.encoder_trainable_layers)
    encoder_params, backend_params = split_trainable_params(model)

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": args.encoder_lr, "name": "w2vbert"},
        {"params": backend_params, "lr": args.backend_lr, "name": "aasist"},
    ], weight_decay=args.weight_decay)
    writer = SummaryWriter(log_dir=log_dir) if SummaryWriter else None

    print(f"Device: {device}")
    print(f"Train trials: {len(train_files)}")
    print(f"Dev trials: {len(dev_files)}")
    print(f"Class counts [fake, real]: {class_counts.tolist()}")
    print(f"Auto CE weights [fake, real]: {[round(x, 6) for x in class_weights.tolist()]}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"w2v-BERT trainable layers: {tuning['trainable_layers']}/{tuning['total_layers']} | "
          f"trainable params={tuning['trainable_params']} | LR={args.encoder_lr:.2e}")
    print(f"AASIST parameters: {sum(p.numel() for p in backend_params)} | LR={args.backend_lr:.2e}")
    print(f"Adaptive LR: monitor=DevLoss; bad_epochs={args.lr_patience}; factor={args.lr_factor}; "
          f"min=[{args.min_encoder_lr:.2e}, {args.min_backend_lr:.2e}]; restore_best=True")

    best_dev_loss = float("inf")
    best_model_path = None
    no_improve_count = 0
    lr_bad_epochs = 0

    for epoch in range(1, args.num_epochs + 1):
        used_encoder_lr, used_backend_lr = optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]
        train_loss, train_acc = train_epoch(
            train_loader, model, optimizer, device, criterion, grad_clip=args.grad_clip
        )
        dev_loss, dev_acc = evaluate_dev(dev_loader, model, device, criterion)

        improved = dev_loss < best_dev_loss
        if improved:
            best_dev_loss = dev_loss
            no_improve_count = 0
            lr_bad_epochs = 0
            best_model_path = os.path.join(ckpt_dir, f"epoch_{epoch}_dev_loss_{dev_loss:.6f}.pth")
            torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
            print(f"Saved best model: {best_model_path}")
        else:
            no_improve_count += 1
            lr_bad_epochs += 1

        next_encoder_lr, next_backend_lr = used_encoder_lr, used_backend_lr
        reduced = False
        restored = False
        if lr_bad_epochs >= args.lr_patience:
            next_encoder_lr = max(used_encoder_lr * args.lr_factor, args.min_encoder_lr)
            next_backend_lr = max(used_backend_lr * args.lr_factor, args.min_backend_lr)
            reduced = next_encoder_lr < used_encoder_lr or next_backend_lr < used_backend_lr
            if reduced:
                optimizer.param_groups[0]["lr"] = next_encoder_lr
                optimizer.param_groups[1]["lr"] = next_backend_lr
                best_path = os.path.join(ckpt_dir, "best_model.pth")
                model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True), strict=True)
                optimizer.state.clear()
                lr_bad_epochs = 0
                no_improve_count = 0
                restored = True

        message = (
            f"Epoch {epoch}/{args.num_epochs} "
            f"TrainLoss={train_loss:.6f} TrainAcc={train_acc:.2f}% "
            f"DevLoss={dev_loss:.6f} DevAcc={dev_acc:.2f}% "
            f"EncoderLR={used_encoder_lr:.2e}->{next_encoder_lr:.2e} "
            f"BackendLR={used_backend_lr:.2e}->{next_backend_lr:.2e}"
        )
        print(message)
        if reduced:
            print("LR reduced after consecutive bad DevLoss epochs; restored best_model.pth and reset Adam state.")
        elif lr_bad_epochs >= args.lr_patience:
            print("Validation plateaued, but both learning rates are already at their configured minimum.")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().replace(microsecond=0)}] {message}\n")

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Loss/dev", dev_loss, epoch)
            writer.add_scalar("Acc/dev", dev_acc, epoch)
            writer.add_scalar("LR/encoder", next_encoder_lr, epoch)
            writer.add_scalar("LR/backend", next_backend_lr, epoch)

        if no_improve_count >= args.earlystop_epoch:
            print(f"Early stopping at epoch {epoch}. Best dev_loss={best_dev_loss:.6f}")
            break

    if writer:
        writer.close()
    print(f"Experiment saved to: {exp_root}")
    print(f"Best model: {best_model_path}")


if __name__ == "__main__":
    main()
