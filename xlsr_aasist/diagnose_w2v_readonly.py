"""Read-only diagnosis of an existing w2v_rebuild Stage checkpoint.

Run from xlsr_aasist: python diagnose_w2v_readonly.py --stage-dir "$RUN_DIR/stage1"
No optimizer, backward, model update, full validation, or checkpoint rewrite.
Uses a small balanced subset of official TRAIN/DEV, never Progress/Eval.
"""
import argparse
from collections import OrderedDict
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def summarize_scores(logits, labels):
    z, y = logits.detach().float().cpu(), labels.detach().long().cpu()
    p = z.softmax(1)[:, 0]
    pred = (p < 0.5).long()
    cm = torch.bincount(2*y+pred, minlength=4).reshape(2, 2)
    by_class = [p[y == c] for c in (0, 1)]
    return {"confusion": cm.tolist(), "mean_pfake_by_truth": [float(v.mean()) for v in by_class],
            "std_pfake_by_truth": [float(v.std(unbiased=False)) for v in by_class],
            "pfake_min_max": [float(p.min()), float(p.max())],
            "logit_margin_std": float((z[:, 0]-z[:, 1]).std(unbiased=False)),
            "cross_entropy": float(F.cross_entropy(z, y)),
            "pfake": p.tolist()}


def representation_summary(rows, labels):
    # Rows are equal-shape flattened activations (not manually normalized features).
    x = torch.cat(rows).float()
    y = labels.cpu()
    rms = x.square().mean().sqrt()
    centered = (x-x.mean(0, keepdim=True)).square().mean().sqrt()
    mean_gap = (x[y == 0].mean(0)-x[y == 1].mean(0)).square().mean().sqrt()
    within = sum((x[y == c]-x[y == c].mean(0, keepdim=True)).square().mean()
                 for c in (0, 1))/2
    return {"rms": float(rms), "between_sample_rms": float(centered),
            "relative_sample_spread": float(centered/(rms+1e-12)),
            "class_mean_gap_rms": float(mean_gap),
            "mean_gap_over_within_rms": float(mean_gap/(within.sqrt()+1e-12))}


def tensor_rows(output):
    if torch.is_tensor(output):
        return output.detach().float().cpu().flatten(1)
    if isinstance(output, (list, tuple)):
        return torch.cat([tensor_rows(x) for x in output if torch.is_tensor(x)], dim=1)
    raise TypeError("Unsupported hook output")


def checkpoint_deltas(backbone, model_dir):
    """Compare one projection matrix per encoder layer with the OFFICIAL local weights."""
    from safetensors import safe_open
    root = Path(model_dir)
    index_file = root/"model.safetensors.index.json"
    index = json.loads(index_file.read_text())["weight_map"] if index_file.exists() else None
    single = root/"model.safetensors"
    result = {}
    names = ["feature_projection.projection.weight"] + [
        f"encoder.layers.{i}.self_attn.linear_q.weight" for i in range(len(backbone.encoder.layers))]
    params = dict(backbone.named_parameters())
    for name in names:
        candidates = [name, "wav2vec2_bert."+name]
        if index is not None:
            stored = next((n for n in candidates if n in index), None)
            path = root/index[stored] if stored else None
        else:
            path, stored = single, None
        if path is None or not path.exists() or name not in params:
            result[name] = {"unavailable": "Original tensor/file missing; not evidence of a loading failure"}
            continue
        with safe_open(str(path), framework="pt", device="cpu") as f:
            if stored is None:
                stored = next((n for n in candidates if n in f.keys()), None)
            if stored is None:
                result[name] = {"unavailable": "Tensor key absent in original checkpoint"}
                continue
            original = f.get_tensor(stored).float()
        current = params[name].detach().float().cpu()
        if original.shape != current.shape:
            result[name] = {"shape_mismatch": [list(original.shape), list(current.shape)]}
            continue
        d = current-original
        result[name] = {"relative_l2_from_pretrained": float(d.norm()/(original.norm()+1e-12)),
                        "max_abs_change": float(d.abs().max()),
                        "changed_fraction": float((d != 0).float().mean())}
    return result


def load_log_summary(stage_dir):
    output = []
    path = stage_dir/"metrics.jsonl"
    if not path.exists():
        return output
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A currently running process may not have flushed its last line.
        item = {"epoch": row.get("epoch"), "train_loss": row.get("mean_train_loss")}
        for key, m in (("train", row.get("train", {})),
                       ("dev_online", row.get("dev", {}).get("online", {}))):
            item[key] = {k: m.get(k) for k in ("macro_f1", "balanced_ce", "confusion",
                           "mean_fake_score_by_class", "fake_prediction_fraction")}
        item["last_batch_grad_norm"] = row.get("last_batch", {}).get("grad_norm")
        output.append(item)
    return output


def balanced_ids(ids, labels, count, seed, online_only=False):
    rng = random.Random(seed)
    if online_only:
        ids = [x for x in ids if "online" in Path(x).parts]
    pools = [[x for x in ids if labels[x] == c] for c in (0, 1)]
    n = min(count, len(pools[0]), len(pools[1]))
    if n < 2:
        raise ValueError("Not enough examples from both classes")
    a, b = (rng.sample(pool, n) for pool in pools)
    return [x for pair in zip(a, b) for x in pair]


def collect_features(cfg, source, per_class):
    from utils.data_utils import read_protocol, SpoofAudioDataset
    from w2v_rebuild.data import FeatureCollator
    training = source.startswith("train")
    ids, labels = read_protocol(cfg["train_protocol" if training else "dev_protocol"], require_label=True)
    selected = balanced_ids(ids, labels, per_class, 8147, online_only=not training)
    augment = source == "train_aug"
    data = SpoofAudioDataset(selected, cfg["train_data_path" if training else "dev_data_path"],
                            labels=labels, args=SimpleNamespace(**cfg),
                            algo=cfg.get("algo", 5), use_rawboost=augment)
    rows = [data[i] for i in range(len(data))]
    waves = torch.stack([r[0] for r in rows])
    b = FeatureCollator(cfg["ssl_path"], "ordinary")(rows)
    # Verify official batch/single feature extraction does not mix samples.
    one = FeatureCollator(cfg["ssl_path"], "ordinary")([rows[0]])
    gap = float((b["features"][:1]-one["features"]).abs().max())
    return b, {"ids": selected, "feature_shape": list(b["features"].shape),
               "waveform_rms_by_sample": waves.square().mean(1).sqrt().tolist(),
               "feature_batch_single_max_abs": gap}


@torch.no_grad()
def inspect_batch(model, batch, microbatch, device):
    features, mask, labels = batch["features"], batch["mask"], batch["labels"]
    modules = OrderedDict((name, model.head.get_submodule(name)) for name in
                          ["projection", "first_norm", "conv", "last_norm", "gat_s", "gat_t",
                           "pool_s", "pool_t", "h11", "h12", "h21", "h22"])
    slots = OrderedDict((name, []) for name in ["input_features", "encoder_output", *modules, "readout"])
    handles = []
    for name, mod in modules.items():
        def hook(_, inputs, output, name=name):
            slots[name].append(tensor_rows(output))
        handles.append(mod.register_forward_hook(hook))
    z_fp, h_fp = [], []
    try:
        for i in range(0, len(features), microbatch):
            f, m = features[i:i+microbatch].to(device), mask[i:i+microbatch].to(device)
            h = model.backbone(input_features=f, attention_mask=m, return_dict=True).last_hidden_state
            z, r = model.head(h)
            slots["input_features"].append(tensor_rows(f))
            slots["encoder_output"].append(tensor_rows(h))
            slots["readout"].append(tensor_rows(r))
            z_fp.append(z.cpu()); h_fp.append(h.float().cpu())
    finally:
        for handle in handles:
            handle.remove()
    z_fp = torch.cat(z_fp)
    profiles = OrderedDict((k, representation_summary(v, labels)) for k, v in slots.items())
    del slots
    outputs = {"FP32_encoder_FP32_head": summarize_scores(z_fp, labels)}
    # Same weights, samples and eval mode. Isolate head arithmetic from encoder arithmetic.
    both, mixed, head_only = [], [], []
    for i, h_saved in zip(range(0, len(features), microbatch), h_fp):
        f, m = features[i:i+microbatch].to(device), mask[i:i+microbatch].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            hb = model.backbone(input_features=f, attention_mask=m, return_dict=True).last_hidden_state
            zb, _ = model.head(hb)
            zhead, _ = model.head(h_saved.to(device))
        zm, _ = model.head(hb.float())
        both.append(zb.float().cpu()); mixed.append(zm.float().cpu()); head_only.append(zhead.float().cpu())
    modes = {"BF16_encoder_BF16_head": torch.cat(both),
             "BF16_encoder_FP32_head": torch.cat(mixed),
             "FP32_encoder_BF16_head": torch.cat(head_only)}
    p0 = z_fp.softmax(1)[:,0]
    for name, z in modes.items():
        outputs[name] = summarize_scores(z, labels)
        outputs[name]["max_pfake_change_vs_FP32"] = float((z.softmax(1)[:,0]-p0).abs().max())
        outputs[name]["decision_changes_vs_FP32"] = int(((z.softmax(1)[:,0] >= .5) != (p0 >= .5)).sum())
    classifier = model.head.classifier
    d = (classifier.weight[0]-classifier.weight[1]).detach().float()
    return {"scores": outputs, "activation_profiles": profiles,
            "classifier_margin_weight_norm": float(d.norm()),
            "classifier_margin_bias": float((classifier.bias[0]-classifier.bias[1]).detach())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-dir", required=True)
    p.add_argument("--checkpoint", help="Defaults to last.pt; falls back to best_model.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--per-class", type=int, default=8)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--logs-only", action="store_true")
    p.add_argument("--out", help="New report JSON only; checkpoint files are never written")
    args = p.parse_args()
    stage = Path(args.stage_dir).resolve()
    report = {"stage_dir": str(stage), "logs": load_log_summary(stage)}
    print("\n=== Existing full-epoch Train/Dev results ===", flush=True)
    for row in report["logs"]:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    target = Path(args.out).resolve() if args.out else stage/"diagnosis_readonly.json"
    if target.suffix.lower() != ".json" or target.name in ("config.json", "completed.json", "preflight.json"):
        raise ValueError("Output must be a NEW diagnosis JSON, not a training file")
    if not args.logs_only:
        if args.per_class < 2 or args.microbatch < 1:
            raise ValueError("per-class >= 2 and microbatch >= 1 required")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        random.seed(8147); np.random.seed(8147); torch.manual_seed(8147)
        torch.set_num_threads(2)
        from w2v_rebuild.core import load_checkpoint
        from w2v_rebuild.model import Detector
        path = Path(args.checkpoint).resolve() if args.checkpoint else stage/"last.pt"
        if not path.exists() and not args.checkpoint:
            path = stage/"best_model.pt"
        print("\nCheckpoint:", path, flush=True)
        ckpt = load_checkpoint(path)
        cfg = ckpt["config"]
        report.update(checkpoint=str(path), checkpoint_epoch=ckpt.get("epoch"),
                      checkpoint_stage=ckpt.get("stage"), recorded_source_hashes=ckpt.get("source_hashes"))
        state, config = ckpt["model"], ckpt["model_config"]
        del ckpt  # Release the saved optimizer before constructing the 580M encoder.
        gc.collect()
        model = Detector.load(cfg["ssl_path"], config, checkpointing=False)
        model.load_state_dict(state, strict=True)
        del state; gc.collect()
        report["pretrained_weight_comparison"] = checkpoint_deltas(model.backbone, cfg["ssl_path"])
        model.to(device).eval()
        for q in model.parameters(): q.requires_grad_(False)
        os.environ["RTC_B_NOISE_MANIFEST"] = cfg["train_noise_manifest"]
        os.environ.setdefault("RTC_B_NOISE_PROB", "0.5")
        os.environ.setdefault("RTC_B_SNR_MIN", "10")
        os.environ.setdefault("RTC_B_SNR_MAX", "30")
        report["diagnostic_augmentation_env"] = {k:v for k,v in os.environ.items() if k.startswith("RTC_B_")}
        report["note"] = ("Balanced small Train/Dev subset; diagnostic, not a benchmark. "
                          "Same parameters and eval mode in every arithmetic comparison. "
                          "No optimizer/backward/checkpoint writes. Original training env overrides may differ.")
        for source in ("train_raw", "train_aug", "dev_online"):
            print("\n===", source, "===", flush=True)
            batch, meta = collect_features(cfg, source, args.per_class)
            result = inspect_batch(model, batch, args.microbatch, device)
            result["input"] = meta
            report[source] = result
            for name, scores in result["scores"].items():
                print(name, "CM=", scores["confusion"], "meanP(fake)|truth=", scores["mean_pfake_by_truth"],
                      "range=", scores["pfake_min_max"], flush=True)
            print("Sample spread by layer (not classification accuracy):", flush=True)
            for name, v in result["activation_profiles"].items():
                print(f"  {name:20s} relative_spread={v['relative_sample_spread']:.6g} class_gap/within={v['mean_gap_over_within_rms']:.5g}", flush=True)
            # Persist results after each section in case the next one errors.
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nReport:", target, flush=True)


if __name__ == "__main__":
    main()
