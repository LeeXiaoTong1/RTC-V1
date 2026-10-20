"""Frozen screening recipe followed by fixed-budget Train+Dev fitting."""
import argparse
import ast
import copy
import math
from pathlib import Path
from .common import ROOT, RECIPE, digest, read_json, schema_for
from w2v_v3189.config import end_learning_rates
from w2v_v318.config import runtime_versions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", help="Stopped V3.18.9 run; its verified best is loaded")
    p.add_argument("--resume", help="Existing V5 run; uses its recorded settings")
    p.add_argument("--phase", choices=("all", "screen", "final"), default="all",
                   help="final requires --resume of a completed screen")
    p.add_argument("--epochs", type=int, default=1, help="Epochs PER stage; default 1 screen + 1 final")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--eval-workers", type=int, default=None)
    p.add_argument("--prefetch-factor", type=int, default=1)
    p.add_argument("--microbatch", type=int, default=12)
    p.add_argument("--frame-budget", type=int, default=9000)
    p.add_argument("--eval-batch", type=int, default=12)
    p.add_argument("--window-batch", type=int, default=16)
    p.add_argument("--context-seconds", type=float, default=2.)
    p.add_argument("--screen-sources-per-group", type=int, default=64)
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=50001)
    p.add_argument("--submit-best", action="store_true", help="Export the final locked single model")
    p.add_argument("--upload-submission", action="store_true")
    p.add_argument("--upload-temp", action="store_true", help="Upload report only")
    p.add_argument("--submission-protocol")
    p.add_argument("--submission-audio-root")
    p.add_argument("--download-dir", default="/home/ubuntu/LXT/temp")
    return p


def validate(args, argv=None):
    if args.resume and args.parent_run:
        raise ValueError("Use --resume OR --parent-run")
    if args.phase == "final" and not args.resume:
        raise ValueError("Final fitting requires a completed V5 screening run")
    if (args.epochs < 1 or args.microbatch < 12 or
        min(args.frame_budget, args.eval_batch, args.window_batch,
            args.prefetch_factor, args.checkpoint_interval, args.screen_sources_per_group) < 1):
        raise ValueError("Positive budgets required; microbatch >=12 for inherited PCL")
    if any(x is not None and x < 0 for x in (args.workers, args.eval_workers)):
        raise ValueError("Worker counts must be nonnegative")
    if not math.isfinite(args.context_seconds) or not 0 <= args.context_seconds <= 5:
        raise ValueError("RTC context must be between 0 and 5 seconds")
    if not 0 <= args.seed < 2**32:
        raise ValueError("Seed must be uint32")
    if args.upload_submission and not args.submit_best:
        raise ValueError("--upload-submission requires --submit-best")
    if args.resume and argv is not None:
        allowed = {"--resume", "--phase", "--submit-best", "--upload-submission",
                   "--upload-temp", "--submission-protocol", "--submission-audio-root",
                   "--download-dir"}
        overrides = {a.split("=")[0] for a in argv if a.startswith("--")} - allowed
        if overrides:
            raise ValueError("Resume uses saved settings; remove " + ", ".join(sorted(overrides)))


def code_inventory(root=ROOT):
    root = Path(root)
    pending = list((root / "w2v_v5").glob("*.py"))
    seen = {}
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file() or path.name.startswith("test_"):
            continue
        seen[path] = digest(path)
        package = list(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".") for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package)-node.level+1] if node.level else []
                target = prefix + (node.module.split(".") if node.module else [])
                names = [target] + [target + a.name.split(".") for a in node.names if a.name != "*"]
            for parts in names:
                for depth in range(1, len(parts)+1):
                    item = root.joinpath(*parts[:depth])
                    pending.extend((item.with_suffix(".py"), item / "__init__.py"))
    return {str(p): h for p, h in seen.items()}


def training_configuration(old, origin, args):
    cfg = copy.deepcopy(old)
    for key in ("saved_manifests", "warm_start", "source_registry",
                "augmentation_budget_audit", "window_resource_audit"):
        cfg.pop(key, None)
    cfg.update(version="5", recipe=RECIPE, v5_stage="screen", epochs=args.epochs,
        warm_epochs=0, lora_layers=16, retained_lora_layers=old["lora_layers"],
        checkpoint_layers=16, checkpointing=True, experiment="E0", meta_beta=1.,
        meta_ramp_updates=1, inner_candidates=[], calibration_tasks=0,
        device=args.device, microbatch=args.microbatch, frame_budget=args.frame_budget,
        eval_batch=args.eval_batch, window_batch=args.window_batch,
        lr_warmup_updates=1, pcl_ramp_updates=1, lr_floor=.5,
        source_epoch_offset=int(old.get("source_epoch_offset", 0)) + origin["epoch"],
        parent_init=origin, data_files=origin["manifest_files"],
        initialization=dict(mode="v3189_best", run=origin["run"], selected=origin["tag"],
                            old_optimizer_loaded=False, all_parent_adapters_retained=True),
        classification_ce_weights=dict(online=.25, clean_rtc=.25, noisy=.5),
        paired_classification=True, robust_arm="baseline", robust_hard_fraction=0.,
        _candidate_diagnostics=False, sampling_seed=args.seed,
        v5_context_seconds=args.context_seconds,
        v5_screen_sources_per_group=args.screen_sources_per_group,
        v5_pcl_scope="parent_train", checkpoint_interval=args.checkpoint_interval,
        v5_tail_snapshots=4, v5_tail_start=.75,
        selection="0.3 Clean Macro-F1 + 0.7 pooled Noisy Macro-F1; fixed threshold 0.5",
        runtime_versions=runtime_versions())
    from .processing import policy
    cfg["v5_input"] = policy()
    cfg.update(end_learning_rates(old, origin["epoch"], origin["steps"]))
    # This worker estimator includes one classified Clean-RTC and one optional
    # PCL reference per Noisy source, without four unused receiver candidates.
    from w2v_v3189.workers import settings
    cfg.update(settings(args, old))
    schema_for(cfg)
    return cfg


def verify_inputs(cfg, *, training=True):
    from w2v_v3185.config import verify_unique
    from w2v_v3184.assets import configured_spec
    schema_for(cfg)
    configured_spec(cfg)
    verify_unique(cfg["code_fingerprints"],
                  {cfg["omni_checkpoint"]: cfg["omni_sha256"]},
                  cfg["data_files"] if training else {},
                  cfg["augmentation_files"] if training else {},
                  {cfg["ffmpeg"]: cfg["ffmpeg_sha256"]} if training else {},
                  cfg["scene_noise_files"] if training else {},
                  cfg.get("phoneme_files", {}) if training else {},
                  cfg.get("identity_input_files", {}) if training else {})
    if cfg["runtime_versions"] != runtime_versions():
        raise ValueError("Pinned runtime changed")
    if training and cfg.get("consistency") == "dual":
        from w2v_v3189.phonemes import bind_manifest
        parent_rows = read_json(Path(cfg["parent_init"]["run"]) / "train_rows.json")
        if bind_manifest(cfg["phoneme_manifest"], parent_rows) != cfg["phoneme_provenance"]:
            raise ValueError("Inherited parent Train phoneme assets changed")


if __name__ == "__main__":
    import sys
    args = parser().parse_args()
    validate(args, sys.argv[1:])
