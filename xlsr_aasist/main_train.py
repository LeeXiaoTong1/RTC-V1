import argparse
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import Model
from utils.data_utils import build_dataset_from_protocol, set_random_seed

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def build_loader(dataset, batch_size, num_workers, shuffle):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    return DataLoader(dataset, **kwargs)


def train_epoch(data_loader, model, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    num_total = 0

    for batch_x, batch_y, _ in tqdm(data_loader, desc="Training", unit="batch"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).long().to(device)
        batch_size = batch_x.size(0)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
        
        batch_loss.backward()
        optimizer.step()
        
        running_loss += batch_loss.item() * batch_size
        correct += (torch.argmax(batch_out, dim=1) == batch_y).sum().item()
        num_total += batch_size

    return running_loss / num_total, 100.0 * correct / num_total


def evaluate_dev(data_loader, model, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    num_total = 0

    with torch.no_grad():
        for batch_x, batch_y, _ in tqdm(data_loader, desc="Validating", unit="batch"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).long().to(device)
            batch_size = batch_x.size(0)

            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)

            running_loss += batch_loss.item() * batch_size
            correct += (torch.argmax(batch_out, dim=1) == batch_y).sum().item()
            num_total += batch_size

    return running_loss / num_total, 100.0 * correct / num_total



def main():
    parser = argparse.ArgumentParser(description="Train XLSR-AASIST spoof detector")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--dev_data_path", type=str, required=True)
    parser.add_argument("--train_protocol", type=str, required=True)
    parser.add_argument("--dev_protocol", type=str, required=True)
    parser.add_argument("--track", type=str, default="xlsr_aasist")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--out_path", type=str, default="./exp")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--earlystop_epoch", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cudnn-deterministic-toggle", action="store_false", default=True)
    parser.add_argument("--cudnn-benchmark-toggle", action="store_true", default=False)

    parser.add_argument("--algo", type=int, default=5)
    parser.add_argument("--nBands", type=int, default=5)
    parser.add_argument("--minF", type=int, default=20)
    parser.add_argument("--maxF", type=int, default=8000)
    parser.add_argument("--minBW", type=int, default=100)
    parser.add_argument("--maxBW", type=int, default=1000)
    parser.add_argument("--minCoeff", type=int, default=10)
    parser.add_argument("--maxCoeff", type=int, default=100)
    parser.add_argument("--minG", type=int, default=0)
    parser.add_argument("--maxG", type=int, default=0)
    parser.add_argument("--minBiasLinNonLin", type=int, default=5)
    parser.add_argument("--maxBiasLinNonLin", type=int, default=20)
    parser.add_argument("--N_f", type=int, default=5)
    parser.add_argument("--P", type=int, default=10)
    parser.add_argument("--g_sd", type=int, default=2)
    parser.add_argument("--SNRmin", type=int, default=10)
    parser.add_argument("--SNRmax", type=int, default=40)
    args = parser.parse_args()

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
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded: {args.model_path}")

    print(f"Device: {device}")
    print(f"Train trials: {len(train_files)}")
    print(f"Dev trials: {len(dev_files)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))
    writer = SummaryWriter(log_dir=log_dir) if SummaryWriter else None

    best_dev_loss = float("inf")
    best_model_path = None
    no_improve_count = 0

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_epoch(train_loader, model, optimizer, device, criterion)
        dev_loss, dev_acc = evaluate_dev(dev_loader, model, device, criterion)

        message = (
            f"Epoch {epoch}/{args.num_epochs} "
            f"TrainLoss={train_loss:.6f} TrainAcc={train_acc:.2f}% "
            f"DevLoss={dev_loss:.6f} DevAcc={dev_acc:.2f}%"
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
