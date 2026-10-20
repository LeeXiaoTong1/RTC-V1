"""V3.19.1: matched paired-classification controls for processing robustness."""
import ast
import copy
import math
from pathlib import Path
import torch
from w2v_v3189.config import parser as parent_parser, validate_locked_cli, end_learning_rates
from w2v_v318.config import runtime_versions
from w2v_v3184.assets import configured_spec
from w2v_v3185.config import verify_unique
from .common import ROOT, RECIPE, schema_for, digest, read_json


def parser():
    p = parent_parser()
    p.description = __doc__
    for action in p._actions:
        if action.dest == "parent_run":
            action.help = "Stopped V3.18.9 run; defaults to exp/w2v_v3189_20261016_183410_a437"
        elif action.dest == "resume":
            action.help = "Resume this V3.19.1 arm with saved optimizer and RNG"
    p.set_defaults(epochs=2)
    p.add_argument("--lora-layers", type=int, choices=(16,), default=16,
                   help="Train final 16 layers; retain all learned prefix adapters frozen")
    p.add_argument("--experiment", choices=("E0",), default="E0",
                   help="Ordinary paired classification; V3.19 MetaDG is a separate experiment")
    p.add_argument("--arm", dest="robust_arm", choices=("baseline", "expanded", "hard"), default="hard")
    p.add_argument("--hard-fraction", dest="robust_hard_fraction", type=float, default=.5)
    p.add_argument("--seed", type=int, default=None, help="Repeat seed; never changes frozen evaluation processing")
    p.add_argument("--rnnoise-model", help="Optional local copy of the pinned public RNNoise weight")
    p.add_argument("--asset-cache", default=str(ROOT/"assets"/"v3191"),
                   help="The pinned 0.3 MB denoiser is downloaded once and verified")
    p.add_argument("--meta-beta", type=float, choices=(1.,), default=1.,
                   help="Fixed equal A/B CE contribution; no meta update")
    p.add_argument("--lr-floor", type=float, default=.5,
                   help="Decay from the selected parent-best endpoint to this fraction")
    return p


def validate(args):
    from w2v_v3189.config import validate as parent_validate
    parent_validate(args)
    if args.meta_beta != 1.:
        raise ValueError("A/B classification must have equal weight")
    if not math.isfinite(args.lr_floor) or not 0 < args.lr_floor <= 1:
        raise ValueError("Require 0 < lr_floor <= 1")
    if not math.isfinite(args.robust_hard_fraction) or not 0 <= args.robust_hard_fraction <= 1:
        raise ValueError("Require 0 <= hard-fraction <= 1")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")


def code_inventory(root=ROOT):
    root = Path(root)
    pending = list((root / "w2v_v3191").glob("*.py"))
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
                    candidate = root.joinpath(*parts[:depth])
                    pending.extend((candidate.with_suffix(".py"), candidate / "__init__.py"))
    return {str(p): value for p, value in seen.items()}


def training_configuration(old, origin, args):
    validate(args)
    cfg = copy.deepcopy(old)
    for key in ("saved_manifests", "warm_start", "source_registry", "augmentation_budget_audit", "window_resource_audit"):
        cfg.pop(key, None)
    cfg.update(version="3.19.1", recipe=RECIPE, epochs=args.epochs, warm_epochs=0,
               lora_layers=args.lora_layers, retained_lora_layers=old["lora_layers"],
               checkpoint_layers=args.lora_layers, checkpointing=True,
               experiment=args.experiment, meta_beta=args.meta_beta,
               meta_ramp_updates=1, inner_candidates=[], calibration_tasks=0,
               device=args.device, backend_precision=args.backend_precision,
               microbatch=args.microbatch, frame_budget=args.frame_budget,
               eval_batch=args.eval_batch, window_batch=args.window_batch,
               lr_warmup_updates=1, pcl_ramp_updates=1, lr_floor=args.lr_floor,
               source_epoch_offset=old.get("source_epoch_offset", 0) + origin["epoch"],
               parent_init=origin,
               data_source=dict(run=origin["run"], files=origin["manifest_files"], checkpoint_loaded=True),
               data_files=origin["manifest_files"],
               initialization=dict(mode="v3189_best", run=origin["run"], selected=origin["tag"],
                                   old_detector_loaded=True, checkpoint_loaded=True, old_optimizer_loaded=False,
                                   transfer="All parent LoRA/fusion/AASIST/classifier/PCL scale; freeze adapter prefix"),
               classification_objective="paired_Clean_RTC_Noisy_CE_with_bounded_balanced_condition_selection",
               robust_arm=args.robust_arm, robust_candidate_count=4,
               robust_hard_fraction=args.robust_hard_fraction, robust_protocol_seed=3191001,
               robust_train_mechanisms=["spectral_color", "packet_repeat", "rnnoise"],
               robust_eval_mechanisms=["spectral_wiener", "color_packet"],
               robust_clean_guard=.002, robust_recall_guard=.005,
               robust_min_gain=.001, robust_bootstrap_draws=2000,
               checkpoint_interval=200,
               sampling_seed=old["seed"] if args.seed is None else args.seed,
               selection="Held-mechanism Select macro F1 with parent Clean and two-class recall guards; Audit only after selection",
               runtime_versions=runtime_versions())
    # Keep cfg.seed fixed: historical dev_partition(), phoneme assets and saved
    # waveform identities depend on it. Repeats use a separate sampling/RNG seed.
    cfg["paired_classification"] = True
    cfg["asset_cache"] = str(Path(args.asset_cache).expanduser().resolve())
    cfg["rnnoise_model"] = args.rnnoise_model
    if args.rtc_activity != old.get("rtc_activity_policy", "mixed"):
        raise ValueError("This continuation preserves the parent's RTC activity recipe")
    if args.backend_precision != old.get("backend_precision", "fp32"):
        raise ValueError("Keep parent backend precision for an unchanged starting detector")
    cfg.update(end_learning_rates(old, origin["epoch"], origin["steps"]))
    cfg["learning_rate_origin"] = dict(epoch=origin["epoch"], tag=origin["tag"],
        start={k: cfg[k] for k in ("lora_lr", "head_lr", "fusion_lr", "layer_mix_lr")},
        policy="Fresh Adam moments; selected-best endpoint rates; no head/LoRA warmup; short cosine decay")
    from .workers import settings
    cfg.update(settings(args, old))
    schema_for(cfg)
    return cfg


def configuration(args):
    from .parent import metadata
    old, origin, train, dev = metadata(args.parent_run)
    if not args.device.startswith("cuda") or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Production training requires the BF16 CUDA server")
    cfg = training_configuration(old, origin, args)
    from .assets import prepare
    cfg.update(prepare(cfg["asset_cache"], cfg["rnnoise_model"]))
    from .augment import runtime_preflight
    cfg["robust_runtime"] = runtime_preflight(cfg)
    # Use the corrected shared PCM graph, but do not adopt V3.18.10's changed
    # validation panels or seed. The parent Dev waveforms/partition remain fixed.
    from w2v_v31810.validation import _AudioIdentity, _speech_partition
    from w2v_v31810.identity import registry_config, verify_registry
    from w2v_aasist.data import read_protocol
    identity = _AudioIdentity()
    print("[Startup] verifying Train/Dev PCM identities and official pairing", flush=True)
    official = read_protocol(cfg["official_dev_protocol"], cfg["official_dev_root"])
    identity.file(cfg["official_dev_protocol"])
    _, audit, train, registry = _speech_partition(cfg, train, dev, official, identity)
    cfg["source_registry"] = registry_config(registry)
    cfg["source_registry_audit"] = audit
    cfg["identity_input_files"] = identity.input_files
    verify_registry(train, registry, cfg, {"source_registry_hash": registry["registry_hash"]})
    from .resources import audit_training
    cfg["window_resource_audit"] = audit_training(train, dev, cfg)
    from .sampling import Plan
    plan = Plan(train, cfg["stream_sources"], cfg["sampling_seed"], cfg["group_quotas"],
                cfg["view_factors"], cfg["refresh_base_variants"], cfg=cfg)
    cfg["augmentation_budget_audit"] = plan.audit_budget(cfg["epochs"])
    cfg["code_fingerprints"] = code_inventory()
    verify_inputs(cfg)
    return cfg, train, dev, registry


def verify_inputs(cfg, *, training=True):
    schema_for(cfg)
    configured_spec(cfg)
    verify_unique(cfg["code_fingerprints"], {cfg["omni_checkpoint"]: cfg["omni_sha256"]},
                  cfg["data_files"] if training else {},
                  cfg["augmentation_files"] if training else {},
                  {cfg["ffmpeg"]: cfg["ffmpeg_sha256"]} if training else {},
                  cfg["scene_noise_files"] if training else {},
                  cfg.get("phoneme_files", {}) if training else {},
                  cfg.get("identity_input_files", {}) if training else {})
    if cfg["runtime_versions"] != runtime_versions():
        raise ValueError("Pinned runtime changed")
    if training and cfg["consistency"] == "dual":
        from .phonemes import bind_manifest
        rows = read_json(Path(cfg["parent_init"]["run"]) / "train_rows.json")
        if bind_manifest(cfg["phoneme_manifest"], rows) != cfg["phoneme_provenance"]:
            raise ValueError("Inherited phoneme inventory changed")
    if training:
        verify_unique(cfg.get("robust_asset_files", {}))
        from .augment import runtime_preflight
        if runtime_preflight(cfg) != cfg["robust_runtime"]:
            raise ValueError("Condition processor runtime changed after preparation")


if __name__ == "__main__":
    import sys
    args = parser().parse_args()
    validate(args)
    validate_locked_cli(args, sys.argv[1:])
