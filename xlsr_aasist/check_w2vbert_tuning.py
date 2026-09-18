"""Verify that partial w2v-BERT fine-tuning really propagates gradients."""
import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from model.model_w2vbert import Model
from utils.w2vbert_tuning import configure_trainable_top_layers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ssl_path", default="/home/ubuntu/LXT/RTC/pretrained/w2v-bert-2.0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top_layers", type=int, default=8)
    p.add_argument("--batch", type=int, default=2)
    args = p.parse_args()

    device = torch.device(args.device)
    model = Model(SimpleNamespace(ssl_path=args.ssl_path), device).to(device)
    model.ssl_model.deterministic_finetune = True
    info = configure_trainable_top_layers(model, args.top_layers)
    model.train()

    wave = torch.randn(args.batch, 64600, device=device)
    labels = torch.arange(args.batch, device=device) % 2
    logits = model(wave)
    loss = F.cross_entropy(logits.float(), labels)
    loss.backward()

    layers = model.ssl_model.model.encoder.layers
    trainable_indices = []
    grad_indices = []
    for i, layer in enumerate(layers):
        if any(p.requires_grad for p in layer.parameters()):
            trainable_indices.append(i)
        if any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in layer.parameters() if p.requires_grad):
            grad_indices.append(i)

    expected = list(range(len(layers) - args.top_layers, len(layers)))
    print("tuning:", info)
    print("expected trainable layers:", expected)
    print("actual trainable layers:", trainable_indices)
    print("layers with nonzero gradients:", grad_indices)
    print("logits:", tuple(logits.shape), "loss:", float(loss.detach()))

    if trainable_indices != expected:
        raise RuntimeError("Trainable layer selection is incorrect")
    if grad_indices != expected:
        raise RuntimeError(
            "Gradient flow is broken: not every selected w2v-BERT layer received gradients"
        )
    print("PASS: partial w2v-BERT gradient flow is valid")


if __name__ == "__main__":
    main()
