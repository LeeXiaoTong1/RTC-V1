"""V3.18.9 best -> one additional epoch, last-16 LoRA, first-order MetaDG."""
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
from .calibration import CANDIDATES


def parser():
    p = parent_parser()
    p.description = __doc__
    for action in p._actions:
        if action.dest == "parent_run":
            action.help = "Stopped V3.18.9 run; defaults to exp/w2v_v3189_20261016_183410_a437"
        elif action.dest == "resume":
            action.help = "Resume this V3.19 MetaDG run, with saved optimizer/alpha/RNG"
    p.set_defaults(epochs=1)
    p.add_argument("--lora-layers", type=int, choices=(16,), default=16,
                   help="Train final 16 layers; retain all learned prefix adapters frozen")
    p.add_argument("--experiment", choices=("E0", "E2"), default="E2",
                   help="E2: source-isolated MetaDG; E0: same tasks/losses with zero inner step")
    p.add_argument("--meta-beta", type=float, default=.5)
    p.add_argument("--meta-ramp-updates", type=int, default=100)
    p.add_argument("--lr-floor", type=float, default=.5,
                   help="Decay from the selected parent-best endpoint to this fraction")
    return p


def validate(args):
    from w2v_v3189.config import validate as parent_validate
    parent_validate(args)
    if not math.isfinite(args.meta_beta) or not 0 < args.meta_beta <= 1 or args.meta_ramp_updates < 1:
        raise ValueError("Require 0 < meta_beta <= 1 and positive meta ramp")
    if not math.isfinite(args.lr_floor) or not 0 < args.lr_floor <= 1:
        raise ValueError("Require 0 < lr_floor <= 1")


def code_inventory(root=ROOT):
    root = Path(root)
    pending = list((root / "w2v_v319_meta").glob("*.py"))
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
    cfg.update(version="3.19", recipe=RECIPE, epochs=args.epochs, warm_epochs=0,
               lora_layers=args.lora_layers, retained_lora_layers=old["lora_layers"],
               checkpoint_layers=args.lora_layers, checkpointing=True,
               experiment=args.experiment, meta_beta=args.meta_beta,
               meta_ramp_updates=args.meta_ramp_updates, inner_candidates=list(CANDIDATES),
               inner_grad_norm=old.get("max_grad_norm", 1.), calibration_tasks=12,
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
               classification_objective="source_isolated_first_order_metadg_with_parent_PCL_on_A",
               selection="Inherited raw Dev Select weighted F1; same cached audio; parent retained unless improved",
               runtime_versions=runtime_versions())
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
    plan = Plan(train, cfg["stream_sources"], cfg["seed"], cfg["group_quotas"],
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


if __name__ == "__main__":
    import sys
    args = parser().parse_args()
    validate(args)
    validate_locked_cli(args, sys.argv[1:])
