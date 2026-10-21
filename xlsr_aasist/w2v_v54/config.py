"""Paired temporal feature repair, continuing a completed V5.3 final checkpoint."""
import argparse
import ast
import copy
import math
from pathlib import Path
from .common import ROOT, RECIPE, digest, read_json, schema_for
from .parent import FINAL_KIND, DEFAULT_PARENT, continuation_rates, is_final_parent
from w2v_v318.config import runtime_versions
from w2v_v3191.ce import CE_WEIGHTS


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", help="Completed V5.3 run with verified final.pt; default exp/" + DEFAULT_PARENT)
    p.add_argument("--resume", help="Existing V5.4 run; uses its recorded settings")
    p.add_argument("--phase", choices=("all", "final"), default="all",
                   help="Continue final Train+Dev; no screening")
    p.add_argument("--epochs", type=int, default=1,
                   help="Total continuation epochs, including repair warmup (default 1)")
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
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--fusion-mode", choices=("dynamic", "static"), default="dynamic",
                   help="Must match parent fusion mode; use --repair-arm for controls")
    p.add_argument("--fusion-width", type=int, default=128)
    p.add_argument('--repair-arm', choices=('baseline', 'ce_only', 'paired'), default='paired')
    p.add_argument("--seed", type=int, default=50001)
    p.add_argument("--submit-best", action="store_true", help="Export the final locked single model")
    p.add_argument("--upload-submission", action="store_true")
    p.add_argument("--upload-temp", action="store_true", help="Upload report only")
    p.add_argument("--submission-protocol")
    p.add_argument("--submission-audio-root")
    p.add_argument("--download-dir", default="/home/ubuntu/LXT/temp")
    return p


def validate(args, argv=None):
    if args.fusion_width < 1:
        raise ValueError("Positive conditioner width required")
    if args.resume and args.parent_run:
        raise ValueError("Use --resume OR --parent-run")
    if (args.epochs < 1 or args.microbatch < 12 or
        min(args.frame_budget, args.eval_batch, args.window_batch,
            args.prefetch_factor, args.checkpoint_interval) < 1):
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
    pending = list((root / "w2v_v54").glob("*.py"))
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
    if old.get('version') != '5.3':
        raise ValueError('V5.4 initializes only from a completed V5.3 final')
    final_parent = origin.get("kind") == FINAL_KIND
    if not final_parent or old.get('classification_ce_weights') != CE_WEIGHTS:
        raise ValueError('Expected completed V5.3 final with Online25/CleanRTC25/Noisy50')
    if args.context_seconds is not None and args.context_seconds != old['v5_context_seconds']:
        raise ValueError('V5.4 retains the parent receiver context; do not change the simulator recipe')
    cfg = copy.deepcopy(old)
    for key in ("saved_manifests", "warm_start", "source_registry",
                "augmentation_budget_audit", "window_resource_audit"):
        cfg.pop(key, None)
    cfg.pop("v5_recipe_lock_sha256", None)
    cfg.update(version="5.4", recipe=RECIPE, v5_stage="final", epochs=args.epochs,
        warm_epochs=0, lora_layers=16, retained_lora_layers=old.get("retained_lora_layers", old["lora_layers"]),
        checkpoint_layers=16, checkpointing=True, experiment="E0", meta_beta=1.,
        meta_ramp_updates=1, inner_candidates=[], calibration_tasks=0,
        device=args.device, microbatch=args.microbatch, frame_budget=args.frame_budget,
        eval_batch=args.eval_batch, window_batch=args.window_batch,
        lr_warmup_updates=1, pcl_ramp_updates=1, lr_floor=.5,
        source_epoch_offset=int(old.get("source_epoch_offset", 0)) + origin["epoch"],
        parent_init=origin, data_files=origin["manifest_files"],
        initialization=dict(mode=FINAL_KIND, run=origin["run"], selected=origin["tag"],
                            old_optimizer_loaded=False, all_parent_adapters_retained=True),
        classification_ce_weights=dict(CE_WEIGHTS),
        paired_classification=True, robust_arm="baseline", robust_hard_fraction=0.,
        _candidate_diagnostics=False, sampling_seed=args.seed,
        v5_context_seconds=(args.context_seconds if args.context_seconds is not None
                            else old.get("v5_context_seconds", 2.)),
        v5_pcl_scope="parent_train", checkpoint_interval=args.checkpoint_interval,
        v5_tail_snapshots=4, v5_tail_start=.75,
        selection="None: completed parent already trained on Train+Dev",
        runtime_versions=runtime_versions())
    for key in list(cfg):
        if key.startswith(('v52_', 'contrastive_')):
            cfg.pop(key)
    cfg['training_recipe_base'] = '5.1'
    cfg['contrastive_weight'] = 0.
    cfg['v54_parent_pcl_weights'] = dict(noise=float(old.get('pcl_noise_weight', .15)),
                                        platform=float(old.get('pcl_platform_weight', .045)))
    cfg['pcl_noise_weight'] = cfg['v54_parent_pcl_weights']['noise']
    cfg['pcl_platform_weight'] = cfg['v54_parent_pcl_weights']['platform']
    if args.fusion_mode != old['adaptive_fusion_mode'] or args.fusion_width != old['adaptive_fusion_width']:
        raise ValueError('Retain the parent V5.3 fusion geometry/mode')
    cfg.update(adaptive_fusion_mode=old['adaptive_fusion_mode'], adaptive_fusion_width=old['adaptive_fusion_width'],
               adaptive_fusion_seed=old['adaptive_fusion_seed'],
               adaptive_fusion_condition='frozen_frontend_masked_mean_std_v1')
    cfg.update(repair_arm=args.repair_arm, repair_seed=(args.seed+4000) % 2**32,
        repair_feature_weight=.05, repair_kd_weight=.02, repair_temperature=3.,
        repair_teacher_confidence=.9, repair_teacher_advantage=.02,
        repair_min_frames=4, repair_lr=1e-4, repair_gate_lr=5e-5,
        repair_scale_sources_per_group=128, repair_scale_recipes=2,
        repair_probe_sources_per_group=16, repair_min_group_coverage=.05,
        repair_audit_revision=2,
        repair_warmup_max=200, repair_warmup_fraction=.1,
        repair_feature_ramp=50, repair_kd_ramp=200,
        pcl_variance_reference_weight=max(cfg['v54_parent_pcl_weights'].values()),
        lora_dropout=0.)
    if args.repair_arm != 'baseline':
        cfg['pcl_noise_weight'] = 0.
    cfg['repair_objective'] = 'A-only global lambda; detached raw clean frame target; deterministic gated output teacher'
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
            'v54_parent_pcl_weights', 'training_recipe_base', 'consistency', 'contrastive_weight',
            'v5_input', 'v5_context_seconds', 'v5_pcl_scope', 'selection',
            'microbatch', 'frame_budget', 'stream_sources', 'source_draws_per_stream',
            'fusion_lr', 'layer_mix_lr', 'lora_lr', 'head_lr', 'lr_floor',
            'v5_tail_snapshots', 'v5_tail_start')
    keys += ('repair_arm', 'repair_seed', 'repair_feature_weight', 'repair_kd_weight',
             'repair_temperature', 'repair_teacher_confidence', 'repair_teacher_advantage',
             'repair_lr', 'repair_gate_lr', 'repair_warmup_max', 'repair_warmup_fraction',
             'repair_feature_ramp', 'repair_kd_ramp', 'repair_scale_sources_per_group',
             'repair_scale_recipes', 'pcl_variance_reference_weight', 'repair_objective', 'lora_dropout')
    keys += ('repair_probe_sources_per_group','repair_min_group_coverage','repair_audit_revision')
    parent = cfg.get('parent_init', {})
    return dict(settings={k: cfg[k] for k in keys if k in cfg},
        parent={k: parent[k] for k in ('version', 'kind', 'run', 'checkpoint', 'checkpoint_sha256', 'tag') if k in parent},
        note='V5.3 continuation: paired feature repair replaces simulated PCL; report telemetry is not a generalization score')


if __name__ == "__main__":
    import sys
    args = parser().parse_args()
    validate(args, sys.argv[1:])
