"""V5.2 recipe plus input-adaptive fusion; initialize from a completed final.pt."""
import argparse
import ast
import copy
import math
from pathlib import Path
from .common import ROOT, RECIPE, digest, read_json, schema_for
from .parent import FINAL_KIND, DEFAULT_MARKER, continuation_rates, is_final_parent
from w2v_v318.config import runtime_versions
from w2v_v52.ce import CE_WEIGHTS
from w2v_v52.contrastive import configuration as contrastive_configuration


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", help="Completed V5.2 (or explicit V5/V5.1) run: verified final.pt. "
                   "Default: exp/" + DEFAULT_MARKER + "; explicit V3.18.9 uses best for controls")
    p.add_argument("--resume", help="Existing V5.3 run; uses its recorded settings")
    p.add_argument("--phase", choices=("all", "screen", "final"), default="all",
                   help="V5 parent: all/final continue directly; legacy parent: screen then final")
    p.add_argument("--epochs", type=int, default=1,
                   help="V5 parent: total continuation epochs (default 1); legacy: epochs per stage")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--eval-workers", type=int, default=None)
    p.add_argument("--prefetch-factor", type=int, default=1)
    p.add_argument("--microbatch", type=int, default=12)
    p.add_argument("--frame-budget", type=int, default=9000)
    p.add_argument("--eval-batch", type=int, default=12)
    p.add_argument("--window-batch", type=int, default=16)
    p.add_argument("--context-seconds", type=float, default=None,
                   help="Inherit V5 parent receiver context; legacy default 2 seconds")
    p.add_argument("--screen-sources-per-group", type=int, default=64)
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--contrastive-weight", type=float, default=None)
    p.add_argument("--contrastive-temperature", type=float, default=None)
    p.add_argument("--contrastive-ramp-updates", type=int, default=None)
    p.add_argument("--contrastive-bank-per-group", type=int, default=None)
    p.add_argument("--contrastive-sources-per-group", type=int, default=None)
    p.add_argument("--fusion-mode", choices=("dynamic", "static"), default="dynamic",
                   help="Static is the matched V5.2 control; same data/loss/budget")
    p.add_argument("--fusion-width", type=int, default=128)
    p.add_argument("--seed", type=int, default=50001)
    p.add_argument("--submit-best", action="store_true", help="Export the final locked single model")
    p.add_argument("--upload-submission", action="store_true")
    p.add_argument("--upload-temp", action="store_true", help="Upload report only")
    p.add_argument("--submission-protocol")
    p.add_argument("--submission-audio-root")
    p.add_argument("--download-dir", default="/home/ubuntu/LXT/temp")
    return p


def validate(args, argv=None):
    contrastive_configuration({k: v for k,v in vars(args).items() if v is not None})
    if args.fusion_width < 1:
        raise ValueError("Positive conditioner width required")
    if args.resume and args.parent_run:
        raise ValueError("Use --resume OR --parent-run")
    if (args.epochs < 1 or args.microbatch < 12 or
        min(args.frame_budget, args.eval_batch, args.window_batch,
            args.prefetch_factor, args.checkpoint_interval, args.screen_sources_per_group) < 1):
        raise ValueError("Positive budgets required; microbatch >=12 for inherited PCL")
    if any(x is not None and x < 0 for x in (args.workers, args.eval_workers)):
        raise ValueError("Worker counts must be nonnegative")
    if args.context_seconds is not None and (
            not math.isfinite(args.context_seconds) or not 0 <= args.context_seconds <= 5):
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
    pending = list((root / "w2v_v53").glob("*.py"))
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
    final_parent = origin.get("kind") == FINAL_KIND
    if final_parent and args.phase == "screen":
        raise ValueError("V5 final already trained on Dev; screening on trained Dev is forbidden")
    if not final_parent and args.phase == "final":
        raise ValueError("Legacy parent needs screening first; resume its completed V5.3 screen")
    cfg = copy.deepcopy(old)
    for key in ("saved_manifests", "warm_start", "source_registry",
                "augmentation_budget_audit", "window_resource_audit"):
        cfg.pop(key, None)
    cfg.pop("v5_recipe_lock_sha256", None)
    cfg.update(version="5.3", recipe=RECIPE, v5_stage="final" if final_parent else "screen", epochs=args.epochs,
        warm_epochs=0, lora_layers=16, retained_lora_layers=old.get("retained_lora_layers", old["lora_layers"]),
        checkpoint_layers=16, checkpointing=True, experiment="E0", meta_beta=1.,
        meta_ramp_updates=1, inner_candidates=[], calibration_tasks=0,
        device=args.device, microbatch=args.microbatch, frame_budget=args.frame_budget,
        eval_batch=args.eval_batch, window_batch=args.window_batch,
        lr_warmup_updates=1, pcl_ramp_updates=1, lr_floor=.5,
        source_epoch_offset=int(old.get("source_epoch_offset", 0)) + origin["epoch"],
        parent_init=origin, data_files=origin["manifest_files"],
        initialization=dict(mode=FINAL_KIND if final_parent else "v3189_best", run=origin["run"], selected=origin["tag"],
                            old_optimizer_loaded=False, all_parent_adapters_retained=True),
        classification_ce_weights=dict(CE_WEIGHTS),
        paired_classification=True, robust_arm="baseline", robust_hard_fraction=0.,
        _candidate_diagnostics=False, sampling_seed=args.seed,
        v5_context_seconds=(args.context_seconds if args.context_seconds is not None
                            else old.get("v5_context_seconds", 2.)),
        v5_screen_sources_per_group=args.screen_sources_per_group,
        v5_pcl_scope="parent_train", checkpoint_interval=args.checkpoint_interval,
        v5_tail_snapshots=4, v5_tail_start=.75,
        selection="0.3 Clean Macro-F1 + 0.7 pooled Noisy Macro-F1; fixed threshold 0.5",
        runtime_versions=runtime_versions())
    if old.get('version') == '5.2':
        from w2v_v52.common import schema_for as parent_schema
        parent_schema(old)
        # The V5.2 coefficient was ALREADY reduced; inheritance is not a second reduction.
        for key in ('v52_parent_pcl_weights', 'v52_noise_pcl_scale',
                    'pcl_noise_weight', 'pcl_platform_weight'):
            cfg[key] = copy.deepcopy(old[key])
    else:
        cfg['v52_parent_pcl_weights'] = dict(noise=float(old.get('pcl_noise_weight', .15)),
                                            platform=float(old.get('pcl_platform_weight', .045)))
        cfg['v52_noise_pcl_scale'] = .2
        cfg['pcl_noise_weight'] = cfg['v52_parent_pcl_weights']['noise'] * .2
        cfg['pcl_platform_weight'] = cfg['v52_parent_pcl_weights']['platform']
    for key, default in dict(contrastive_weight=.01, contrastive_temperature=.2,
                contrastive_ramp_updates=100, contrastive_bank_per_group=32,
                contrastive_sources_per_group=3).items():
        value = getattr(args, key)
        cfg[key] = value if value is not None else old.get(key, default)
    cfg.update(adaptive_fusion_mode=args.fusion_mode, adaptive_fusion_width=args.fusion_width,
               adaptive_fusion_seed=(args.seed + 3000) % 2**32,
               adaptive_fusion_condition='frozen_frontend_masked_mean_std_v1')
    cfg['contrastive_policy'] = 'A_only_same_source_online_noisy_opposite_class_same_language_v1'
    from w2v_v5.processing import policy
    cfg["v5_input"] = copy.deepcopy(old["v5_input"]) if final_parent else policy()
    if final_parent:
        cfg["source_registry"] = copy.deepcopy(old["source_registry"])
        cfg["v5_pcl_scope"] = old["v5_pcl_scope"]
        cfg["v5_tail_snapshots"] = old["v5_tail_snapshots"]
        cfg["v5_tail_start"] = old["v5_tail_start"]
        cfg["selection"] = "None: parent trained on all Train+Dev; inherit locked input/weight rule"
    cfg.update(continuation_rates(old, origin))
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
                  cfg.get("identity_input_files", {}) if training else {},
                  {cfg["parent_init"]["checkpoint"]: cfg["parent_init"]["checkpoint_sha256"]}
                  if training and is_final_parent(cfg) else {})
    if cfg["runtime_versions"] != runtime_versions():
        raise ValueError("Pinned runtime changed")
    if training and cfg.get("consistency") == "dual":
        if is_final_parent(cfg):
            from w2v_v5.manifests import bind_parent_phonemes
            bind_parent_phonemes(cfg, read_json(cfg["parent_init"]["training_rows"]))
            return
        from w2v_v3189.phonemes import bind_manifest
        parent_rows = read_json(Path(cfg["parent_init"]["run"]) / "train_rows.json")
        if bind_manifest(cfg["phoneme_manifest"], parent_rows) != cfg["phoneme_provenance"]:
            raise ValueError("Inherited parent Train phoneme assets changed")


def compact_configuration(cfg):
    """Keep interpretable settings in reports even when asset hashes exceed 2 MiB."""
    keys = ('version', 'recipe', 'v5_stage', 'epochs', 'sampling_seed', 'source_epoch_offset',
            'adaptive_fusion_mode', 'adaptive_fusion_width', 'adaptive_fusion_seed',
            'adaptive_fusion_condition', 'retained_lora_layers', 'lora_layers',
            'classification_ce_weights', 'pcl_noise_weight', 'pcl_platform_weight',
            'v52_parent_pcl_weights', 'v52_noise_pcl_scale', 'consistency',
            'contrastive_weight', 'contrastive_temperature', 'contrastive_ramp_updates',
            'contrastive_bank_per_group', 'contrastive_sources_per_group', 'contrastive_policy',
            'v5_input', 'v5_context_seconds', 'v5_pcl_scope', 'selection',
            'microbatch', 'frame_budget', 'stream_sources', 'source_draws_per_stream',
            'fusion_lr', 'layer_mix_lr', 'lora_lr', 'head_lr', 'lr_floor',
            'v5_tail_snapshots', 'v5_tail_start')
    parent = cfg.get('parent_init', {})
    return dict(settings={k: cfg[k] for k in keys if k in cfg},
        parent={k: parent[k] for k in ('version', 'kind', 'run', 'checkpoint', 'checkpoint_sha256', 'tag') if k in parent},
        note='Only layer weighting changes; report telemetry is not a generalization score')


if __name__ == "__main__":
    import sys
    args = parser().parse_args()
    validate(args, sys.argv[1:])
