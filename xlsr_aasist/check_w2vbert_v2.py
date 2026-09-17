"""Smoke-test the V2 w2v-BERT 2.0 + AASIST model before long training."""
import argparse
from types import SimpleNamespace

import torch

from model.model_w2vbert import Model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ssl_path", default="/home/ubuntu/LXT/RTC/pretrained/w2v-bert-2.0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--length", type=int, default=64600)
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model_args = SimpleNamespace(ssl_path=args.ssl_path)
    model = Model(model_args, device).to(device)
    model.eval()

    # Match the project's normal waveform tensor shape. The feature extractor
    # itself runs on CPU; the 160-D acoustic features are then transferred to GPU.
    wave = torch.randn(args.batch, args.length, dtype=torch.float32, device=device)
    with torch.inference_mode():
        ssl = model.ssl_model.extract_feat(wave)
        logits, readout = model(wave, return_features=True)

    print("w2v-BERT hidden:", tuple(ssl.shape))
    print("AASIST readout:", tuple(readout.shape))
    print("logits:", tuple(logits.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if ssl.ndim != 3 or ssl.shape[0] != args.batch or ssl.shape[-1] != 1024:
        raise RuntimeError("Unexpected w2v-BERT hidden shape")
    if logits.shape != (args.batch, 2):
        raise RuntimeError("Unexpected classifier output shape")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite logits")
    print("PASS")


if __name__ == "__main__":
    main()
