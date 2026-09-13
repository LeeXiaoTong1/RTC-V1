"""Check data pipeline dependencies and optional real detector gradients."""
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from rtc_noisy.simulator import LocalRTC, RTCSettings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rtc", action="store_true")
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ssl_path", default=None)
    args = p.parse_args()
    if args.rtc:
        rtc = LocalRTC()
        rng = np.random.RandomState(8)
        x = (rng.randn(70321) * .05).astype(np.float32)
        for nr, gain, bitrate in zip((6, 12, 18), (2, 4, 8), (16000, 24000, 32000)):
            cfg = RTCSettings(nr, gain, bitrate)
            a, b = rtc(x, cfg), rtc(x, cfg)
            if len(a) != len(x) or not np.array_equal(a, b):
                raise RuntimeError("RTC output length/state reproducibility check failed")
        print("Local RTC passed: deterministic output, preserved length; " + rtc.version)
    if args.checkpoint:
        from model.model import Model
        from utils.rtc_loss import rtc_pair_contrastive_loss
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        model = Model(SimpleNamespace(ssl_path=args.ssl_path), device).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        torch.manual_seed(1234)
        waveforms = torch.randn(4, 64600, device=device) * .05
        logits, features = model(waveforms, return_features=True)
        labels = torch.tensor([0, 1], device=device)
        loss, stats = rtc_pair_contrastive_loss(features[:2], features[2:], labels)
        grads = torch.autograd.grad(loss, tuple(p for p in model.parameters() if p.requires_grad), allow_unused=True)
        total = sum(g.abs().sum().item() for g in grads if g is not None)
        if logits.shape != (4, 2) or features.shape != (4, 160) or not np.isfinite(total) or total <= 0:
            raise RuntimeError("Unexpected detector output or missing contrastive gradients")
        print(f"Real detector passed: logits={tuple(logits.shape)}, features={tuple(features.shape)}, "
              f"valid anchors={stats['valid_anchors']}, gradient L1={total:.6g}; no parameter update")


if __name__ == "__main__":
    main()
