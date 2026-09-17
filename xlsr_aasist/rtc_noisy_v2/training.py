import time
from contextlib import nullcontext

import torch
from tqdm import tqdm

from main_train_rtc import evaluate_dev
from rtc_noisy.training import evaluate_noisy, linear_weight
from utils.rtc_loss import rtc_pair_contrastive_loss
from .losses import grouped_classification


def error_rates(confusion):
    """Rows=true, columns=prediction; fake=0, real=1. Rates use class totals."""
    fake_total, real_total = sum(confusion[0]), sum(confusion[1])
    return {"fake_as_real_rate": confusion[0][1] / fake_total if fake_total else None,
            "real_as_fake_rate": confusion[1][0] / real_total if real_total else None}


def paired_objective(model, batch, real_batch, noisy_batch, device, args, noisy_weight):
    ordinary, ordinary_labels, _ = batch
    real_off, real_on, real_labels = real_batch
    noise_off, noise_on, noise_labels, bands, banks = noisy_batch
    n, r, s = len(ordinary), len(real_off), len(noise_off)
    waveforms = torch.cat([ordinary, real_off, real_on, noise_off, noise_on]).to(device, non_blocking=True)
    labels = torch.cat([ordinary_labels, real_labels, real_labels, noise_labels, noise_labels]).long().to(device)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp == "bf16" and device.type == "cuda" else nullcontext()
    with ctx:
        logits, features = model(waveforms, return_features=True)
    if features.ndim != 2 or len(features) != len(waveforms):
        raise ValueError("Expected one pre-classifier representation per waveform")
    ce, parts, coefficients = grouped_classification(
        logits, labels, n, r, s, args.noisy_ce_weight, args.class_weights)
    real_loss, real_stats = rtc_pair_contrastive_loss(features[n:n+r], features[n+r:n+2*r],
                                                    real_labels.to(device), args.rtc_temperature)
    a = n + 2*r
    noisy_loss, noisy_stats = rtc_pair_contrastive_loss(features[a:a+s], features[a+s:],
                                                       noise_labels.to(device), args.rtc_temperature)
    if real_stats["valid_anchors"] != 2*r or noisy_stats["valid_anchors"] != 2*s:
        raise ValueError("Both pair batches must contain both authenticity classes")
    loss = ce + args.rtc_weight * real_loss + noisy_weight * noisy_loss
    predicted = logits.float().argmax(1)
    codes = (2 * labels[-s:] + predicted[-s:]).detach().cpu()
    stats = {"loss": float(loss.detach()), "ce": float(ce.detach()),
             **{f"ce_{k}": float(v.detach()) for k, v in parts.items()},
             "rtc_real": float(real_loss.detach()), "rtc_noisy": float(noisy_loss.detach()),
             "noise_weight": noisy_weight, "noisy_ce_weight": args.noisy_ce_weight,
             "examples": len(labels), "correct": int((predicted == labels).sum()),
             "ordinary_examples": n, "real_pairs": r, "noisy_pairs": s,
             "snr_counts": torch.bincount(bands, minlength=4).tolist(),
             "bank_counts": torch.bincount(banks, minlength=args.cache_banks).tolist(),
             "noisy_confusion": torch.bincount(codes, minlength=4).reshape(2, 2).tolist(),
             "ce_coefficients": coefficients}
    return loss, stats


def train_epoch(loaders, model, optimizer, device, args, epoch):
    steps = len(loaders[0])
    if not steps or any(len(x) != steps for x in loaders):
        raise ValueError("All training streams must have equal nonzero steps")
    model.train()
    names = ("loss", "ce", "ce_ordinary", "ce_real_pair", "ce_noisy_reference", "ce_noisy_processed",
             "rtc_real", "rtc_noisy", "noise_weight", "noisy_ce_weight")
    totals = dict.fromkeys(names, 0.)
    counts = dict.fromkeys(("examples", "correct", "ordinary_examples", "real_pairs", "noisy_pairs"), 0)
    snr, banks = [0]*4, [0]*args.cache_banks
    cm = [[0, 0], [0, 0]]
    start = time.perf_counter()
    done = 0
    for step, batches in enumerate(tqdm(zip(*loaders), total=steps, desc="Training RTC+Noisy V2", unit="batch")):
        weight = linear_weight(args.noisy_weight, epoch, step, steps, args.noisy_warmup_epochs)
        optimizer.zero_grad(set_to_none=True)
        loss, stats = paired_objective(model, *batches, device, args, weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite total loss")
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        for k in totals:
            totals[k] += stats[k]
        for k in counts:
            counts[k] += stats[k]
        snr = [a+b for a, b in zip(snr, stats["snr_counts"])]
        banks = [a+b for a, b in zip(banks, stats["bank_counts"])]
        cm = [[cm[i][j] + stats["noisy_confusion"][i][j] for j in range(2)] for i in range(2)]
        done += 1
    if done != steps:
        raise RuntimeError("A training stream ended early; do not commit sampler history")
    return {**{k: v/steps for k, v in totals.items()}, **counts,
            "acc": 100*counts["correct"]/counts["examples"],
            "snr_counts": snr, "bank_counts": banks,
            "noisy_confusion": cm, **error_rates(cm),
            "steps": done, "seconds": time.perf_counter()-start}


def combine_validation(clean, seen, heldout):
    clean_f1 = clean["online"]["macro_f1"]
    legacy = .3*clean_f1 + .7*seen["macro_f1"]
    robust = .3*clean_f1 + .35*seen["macro_f1"] + .35*heldout["macro_f1"]
    return {"clean": clean, "noisy_seen": seen, "noisy_heldout": heldout,
            "proxy_v1": legacy, "robust_proxy": robust}


def validate(clean_loader, seen_loader, heldout_loader, model, device, criterion):
    clean = evaluate_dev(clean_loader, model, device, criterion)
    if clean["online"] is None:
        raise ValueError("Official Dev must contain Online audio")
    for name in ("all", "online", "offline"):
        if clean[name] is not None:
            clean[name].update(error_rates(clean[name]["confusion"]))
    seen = evaluate_noisy(seen_loader, model, device, criterion)
    heldout = evaluate_noisy(heldout_loader, model, device, criterion)
    for noisy in (seen, heldout):
        for value in [*noisy["bands"].values(), noisy["pooled"]]:
            value.update(error_rates(value["confusion"]))
    return combine_validation(clean, seen, heldout)


def selection_key(metrics):
    return (metrics["robust_proxy"], metrics["clean"]["online"]["macro_f1"], -metrics["clean"]["loss"])
