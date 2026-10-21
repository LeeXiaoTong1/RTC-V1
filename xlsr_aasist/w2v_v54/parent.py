"""Initialize from the verified V5.3 final; never silently choose a different run."""
import copy
from pathlib import Path

from .common import ROOT, read_json, digest, fingerprint, verify_files
from w2v_v5.processing import validate_policy
from w2v_v31810.identity import verify_registry

DEFAULT_PARENT = "w2v_v53_20261021_200839_f923"
FINAL_KIND = "v5_submitted_final"


def is_final_parent(cfg):
    return cfg.get("parent_init", {}).get("kind") == FINAL_KIND


def parent_delivery(run):
    version = read_json(Path(run) / 'config.json').get('version')
    if version == '5.3':
        from w2v_v53 import delivery
    else:
        raise ValueError('V5.4 requires a completed V5.3 run with verified final.pt')
    return delivery


def locked_weights(recipe):
    """Only the parent's already selected rule is eligible; no new Dev search."""
    weights = recipe.get("weights")
    allowed = (
        dict(name="anchor_tail_50", source="tail", parent_weight=.5),
        dict(name="tail_mean", source="tail", parent_weight=0.),
        dict(name="last", source="last", parent_weight=0.),
    )
    if weights not in allowed:
        raise ValueError("Parent has no recognized locked single-model weight rule")
    return copy.deepcopy(weights)


def _final_metadata(run):
    cfg, meta = parent_delivery(run).final_identity(run)
    recipe = read_json(run / "recipe_lock.json")
    locked_weights(recipe)
    if validate_policy(cfg["v5_input"]) != recipe.get("input"):
        raise ValueError("Parent final input policy differs from its locked recipe")
    if cfg.get("v5_manifest_phase") != "final":
        raise ValueError("Parent must record its full Train+Dev final training scope")
    stage = run / "final"
    if fingerprint(read_json(stage / "config.json")) != fingerprint(cfg):
        raise ValueError("Parent final stage/config identity mismatch")
    saved = cfg["saved_manifests"]
    if not {"train_rows.json", "source_registry.json"} <= set(saved):
        raise ValueError("Parent final Train+Dev manifests are missing")
    if any(Path(name).name != name for name in saved):
        raise ValueError("Parent manifest names must stay within its final stage")
    verify_files({str(stage / name): checksum for name, checksum in saved.items()})
    rows = read_json(stage / "train_rows.json")
    registry = read_json(stage / "source_registry.json")
    verify_registry(rows, registry, cfg, {"source_registry_hash": registry["registry_hash"]})
    if registry["dev"] or not rows or any(
            row.get("v5_origin_split") not in ("train", "dev") or
            (row.get("v5_origin_split") == "dev" and row.get("v5_pcl_eligible"))
            for row in rows):
        raise ValueError("Parent final data lineage/phoneme scope is invalid")
    from w2v_v31810.sampling import quota_schedule
    steps = len(quota_schedule(tuple(cfg["source_draws_per_stream"]), cfg["stream_sources"]))
    files = dict(cfg.get("data_files", {}))
    names = ("final_config.json", "completed.json", "recipe_lock.json", "final/config.json")
    files.update({str(run / name): digest(run / name) for name in names})
    files.update({str(stage / name): checksum for name, checksum in saved.items()})
    origin = dict(kind=FINAL_KIND, version=cfg['version'], run=str(run), checkpoint=str(run / "final.pt"),
        checkpoint_sha256=meta["checkpoint_sha256"], config_sha256=meta["final_config_sha256"],
        recipe_sha256=meta["recipe_sha256"], tag=meta["selected"], epoch=cfg["epochs"],
        steps=steps, manifest_files=files, training_rows=str(stage / "train_rows.json"),
        source_registry=str(stage / "source_registry.json"), locked_recipe=recipe)
    return cfg, origin, rows, []


def metadata(value=None):
    if value:
        run = Path(value).expanduser().resolve()
    else:
        run = (ROOT / 'exp' / DEFAULT_PARENT).resolve()
        if not run.is_dir():
            raise FileNotFoundError('V5.3 parent not found: ' + str(run) +
                                    '; pass --parent-run /path/to/completed_V5.3_run')
    if run.is_file() or run.name == "final":
        raise ValueError("Pass the V5 run directory containing final.pt, not final/last.pt")
    cfg = read_json(run / "config.json")
    if cfg.get("version") != "5.3":
        raise ValueError('Use a completed V5.3 final; older/paused checkpoints cannot initialize V5.4')
    # Artifact upload failure does not undo completed model fitting.
    return _final_metadata(run)


def load(run, expected=None):
    run = Path(run).expanduser().resolve()
    cfg, origin, _, _ = _final_metadata(run)
    if expected is not None and origin != expected:
        raise ValueError("Submitted parent changed after V5.4 preparation")
    selected_cfg, model, meta = parent_delivery(run).selected(run)
    if (fingerprint(selected_cfg) != fingerprint(cfg) or
            meta["checkpoint_sha256"] != origin["checkpoint_sha256"] or
            digest(run / "final.pt") != origin["checkpoint_sha256"]):
        raise ValueError("Submitted parent changed while loading")
    # Adapt the historical initializer interface without loading its last/screen
    # checkpoint, optimizer, or pre-final ancestor weights.
    return cfg, {"best_model": model}, origin, origin["checkpoint_sha256"]


def continuation_rates(old, origin):
    """Use the final continuation schedule, not the V3.18.9 warmup scheduler."""
    if origin.get("kind") != FINAL_KIND:
        from w2v_v3189.config import end_learning_rates
        return end_learning_rates(old, origin["epoch"], origin["steps"])
    from types import SimpleNamespace
    from w2v_v3191.schedule import schedule
    keys = ("lora_early_lr", "lora_middle_lr", "lora_lr", "head_lr", "fusion_lr",
            "layer_mix_lr", "classifier_lr", "evidence_lr", "pool_lr")
    groups = [dict(name=key, initial_lr=old[key], lr=old[key]) for key in keys]
    schedule(SimpleNamespace(param_groups=groups), old, origin["epoch"] - 1,
             origin["steps"] - 1, origin["steps"])
    return {group["name"]: float(group["lr"]) for group in groups}
