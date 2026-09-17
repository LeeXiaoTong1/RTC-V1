"""Stage-1 trainer for w2v-BERT 2.0 + AASIST with differential learning rates."""
import os
from datetime import datetime

import torch
from torch import nn

from main_train import build_arg_parser, build_loader, train_epoch, evaluate_dev
from model.model_w2vbert import Model
from utils.data_utils import build_dataset_from_protocol, set_random_seed

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def build_parser():
    parser = build_arg_parser()
    parser.description = "Train w2v-BERT 2.0 + AASIST with separate encoder/backend learning rates"
    parser.set_defaults(track="w2vbert_aasist_base")
    parser.add_argument("--ssl_path", default=None)
    parser.add_argument("--encoder_lr", type=float, default=1e-6,
                        help="Learning rate for pretrained w2v-BERT 2.0")
    parser.add_argument("--backend_lr", type=float, default=1e-4,
                        help="Learning rate for AASIST and classifier")
    return parser


def main():
    args = build_parser().parse_args()
    if args.encoder_lr <= 0 or args.backend_lr <= 0:
        raise ValueError("encoder_lr and backend_lr must be positive")

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

    train_set, train_files, _ = build_dataset_from_protocol(
        args.train_protocol, args.train_data_path, mode="train", args=args, algo=args.algo
    )
    dev_set, dev_files, _ = build_dataset_from_protocol(
        args.dev_protocol, args.dev_data_path, mode="dev", args=args
    )
    train_loader = build_loader(train_set, args.batch_size, args.num_workers, shuffle=True)
    dev_loader = build_loader(dev_set, args.batch_size, args.num_workers, shuffle=False)

    model = Model(args, device).to(device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print(f"Model loaded: {args.model_path}")

    encoder_params = list(model.ssl_model.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    backend_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": args.encoder_lr, "name": "w2vbert"},
        {"params": backend_params, "lr": args.backend_lr, "name": "aasist"},
    ], weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))
    writer = SummaryWriter(log_dir=log_dir) if SummaryWriter else None

    print(f"Device: {device}")
    print(f"Train trials: {len(train_files)}")
    print(f"Dev trials: {len(dev_files)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"w2v-BERT parameters: {sum(p.numel() for p in encoder_params)} | LR={args.encoder_lr:.2e}")
    print(f"AASIST parameters: {sum(p.numel() for p in backend_params)} | LR={args.backend_lr:.2e}")

    best_dev_loss = float("inf")
    best_model_path = None
    no_improve_count = 0

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_epoch(train_loader, model, optimizer, device, criterion)
        dev_loss, dev_acc = evaluate_dev(dev_loader, model, device, criterion)

        message = (
            f"Epoch {epoch}/{args.num_epochs} "
            f"TrainLoss={train_loss:.6f} TrainAcc={train_acc:.2f}% "
            f"DevLoss={dev_loss:.6f} DevAcc={dev_acc:.2f}% "
            f"EncoderLR={args.encoder_lr:.2e} BackendLR={args.backend_lr:.2e}"
        )
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().replace(microsecond=0)}] {message}\n")

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Loss/dev", dev_loss, epoch)
            writer.add_scalar("Acc/dev", dev_acc, epoch)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            no_improve_count = 0
            best_model_path = os.path.join(ckpt_dir, f"epoch_{epoch}_dev_loss_{dev_loss:.6f}.pth")
            torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
            print(f"Saved best model: {best_model_path}")
        else:
            no_improve_count += 1

        if no_improve_count >= args.earlystop_epoch:
            print(f"Early stopping at epoch {epoch}. Best dev_loss={best_dev_loss:.6f}")
            break

    if writer:
        writer.close()
    print(f"Experiment saved to: {exp_root}")
    print(f"Best model: {best_model_path}")


if __name__ == "__main__":
    main()
