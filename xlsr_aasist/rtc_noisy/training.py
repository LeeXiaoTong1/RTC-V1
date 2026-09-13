"""Reuse the existing label-conditioned contrastive loss for BOTH pair types."""
import math
import time
from contextlib import nullcontext

import torch
from tqdm import tqdm

from main_train_rtc import confusion_metrics, evaluate_dev
from utils.rtc_loss import rtc_pair_contrastive_loss
from rtc_noisy.common import SNR_BANDS


def linear_weight(target, epoch, step, steps, warmup_epochs):
    if warmup_epochs <= 0:
        return target
    fraction = ((epoch - 1) + (step + 1) / steps) / warmup_epochs
    return target * min(1., fraction)


def paired_objective(model, batch, real_pair_batch, noisy_pair_batch, criterion,
                     device, rtc_weight, noisy_weight, temperature, amp):
    ordinary, labels, _ = batch
    real_off, real_on, real_labels = real_pair_batch
    noise_off, noise_on, noise_labels, bands = noisy_pair_batch
    ordinary_count, real_count, noise_count = len(ordinary), len(real_off), len(noise_off)
    waveforms = torch.cat([ordinary, real_off, real_on, noise_off, noise_on]).to(device, non_blocking=True)
    labels = torch.cat([labels, real_labels, real_labels, noise_labels, noise_labels]).long().to(device)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if amp and device.type == "cuda" else nullcontext()
    with context:
        logits, features = model(waveforms, return_features=True)
    logits = logits.float()
    if features.ndim != 2 or len(features) != len(waveforms):
        raise ValueError("Model must return one pre-classifier feature per input")
    start = ordinary_count
    real_loss, real_stats = rtc_pair_contrastive_loss(
        features[start:start + real_count], features[start + real_count:start + 2 * real_count],
        real_labels.to(device), temperature)
    start += 2 * real_count
    noisy_loss, noisy_stats = rtc_pair_contrastive_loss(
        features[start:start + noise_count], features[start + noise_count:], noise_labels.to(device), temperature)
    if real_stats["valid_anchors"] != 2 * real_count or noisy_stats["valid_anchors"] != 2 * noise_count:
        raise ValueError("Both pair batches must contain genuine and spoof negatives")
    ce = criterion(logits, labels)
    loss = ce + rtc_weight * real_loss + noisy_weight * noisy_loss
    stats = {"ce": float(ce.detach()), "rtc_real": float(real_loss.detach()),
             "rtc_noisy": float(noisy_loss.detach()), "loss": float(loss.detach()),
             "correct": int((logits.argmax(1) == labels).sum()), "examples": len(labels),
             "ordinary_examples": ordinary_count, "real_pairs": real_count, "noisy_pairs": noise_count,
             "valid_real_anchors": real_stats["valid_anchors"],
             "valid_noisy_anchors": noisy_stats["valid_anchors"],
             "noise_weight": noisy_weight, "snr_counts": torch.bincount(bands, minlength=4).tolist()}
    return loss, stats


def train_epoch(loaders, model, optimizer, criterion, device, args, epoch):
    steps = len(loaders[0])
    if not steps or any(len(loader) != steps for loader in loaders):
        raise ValueError("All three loaders must have the same nonzero number of steps")
    model.train()
    totals = {k: 0. for k in ("ce", "rtc_real", "rtc_noisy", "loss", "noise_weight")}
    counts = {k: 0 for k in ("correct", "examples", "ordinary_examples", "real_pairs", "noisy_pairs",
                             "valid_real_anchors", "valid_noisy_anchors")}
    bands = [0] * 4
    start = time.perf_counter()
    for step, (batch, real_pairs, noisy_pairs) in enumerate(tqdm(zip(*loaders), total=steps, desc="Training RTC+Noisy", unit="batch")):
        weight = linear_weight(args.noisy_weight, epoch, step, steps, args.noisy_warmup_epochs)
        optimizer.zero_grad(set_to_none=True)
        loss, stats = paired_objective(model, batch, real_pairs, noisy_pairs, criterion, device,
                                       args.rtc_weight, weight, args.rtc_temperature, args.amp == "bf16")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        for key in totals:
            totals[key] += stats[key]
        for key in counts:
            counts[key] += stats[key]
        bands = [x + y for x, y in zip(bands, stats["snr_counts"])]
    # These are optimizer-step averages, matching the objective being optimized.
    return {**{key: value / steps for key, value in totals.items()}, **counts,
            "acc": 100 * counts["correct"] / counts["examples"], "snr_counts": bands,
            "steps": steps, "seconds": time.perf_counter() - start}


@torch.no_grad()
def evaluate_noisy(loader, model, device, criterion):
    model.eval()
    cms = torch.zeros(4, 2, 2, dtype=torch.float64)
    numerators, denominators = [0.] * 4, [0.] * 4
    for waveform, labels, bands in tqdm(loader, desc="Validating simulated noisy RTC", unit="batch"):
        labels = labels.long().to(device)
        logits = model(waveform.to(device, non_blocking=True)).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite validation predictions")
        predicted = (logits.softmax(1)[:, 0] < .5).long()
        for band in range(4):
            mask = (bands == band).to(device)
            if not mask.any():
                continue
            y, z = labels[mask], logits[mask]
            mass = criterion.weight[y].sum().item()
            numerators[band] += criterion(z, y).item() * mass
            denominators[band] += mass
            codes = (2 * y + predicted[mask]).cpu()
            cms[band] += torch.bincount(codes, minlength=4).reshape(2, 2)
    result = {}
    for band, (low, high) in enumerate(SNR_BANDS):
        if not denominators[band] or (cms[band].sum(1) == 0).any():
            raise ValueError("Every noisy Dev band must contain both classes")
        result[f"{int(low)}_{int(high)}dB"] = {**confusion_metrics(cms[band]),
                                              "loss": numerators[band] / denominators[band]}
    return {"bands": result, "macro_f1": sum(row["macro_f1"] for row in result.values()) / 4,
            "pooled": confusion_metrics(cms.sum(0))}


def validate(clean_loader, noisy_loader, model, device, criterion):
    clean = evaluate_dev(clean_loader, model, device, criterion)
    if clean["online"] is None:
        raise ValueError("No real Online examples in Dev")
    noisy = evaluate_noisy(noisy_loader, model, device, criterion)
    proxy = .3 * clean["online"]["macro_f1"] + .7 * noisy["macro_f1"]
    if not math.isfinite(proxy):
        raise FloatingPointError("Invalid validation metric")
    return {"clean": clean, "noisy_simulated": noisy, "proxy": proxy}


def selection_key(metrics):
    # A local proxy, NEVER the official Progress/Eval weighted score.
    return (metrics["proxy"], metrics["clean"]["online"]["macro_f1"], -metrics["clean"]["loss"])
