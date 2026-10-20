"""Checkpoint identity distinct from the historical w2v_v319 package."""
from w2v_v3189.common import *
from w2v_v3189.common import schema_for as _parent_schema, RECIPE as _parent_recipe

RECIPE = "v3191_v3189_best_paired_condition_risk_v1"


def schema_for(cfg):
    if cfg.get("version") != "3.19.1" or cfg.get("recipe") != RECIPE:
        raise ValueError("Expected V3.19.1 paired-condition experiment; historical w2v_v319 is different")
    _parent_schema(dict(cfg, version="3.18.9", recipe=_parent_recipe))
    depth = cfg["encoder_layers"]
    retained = cfg["retained_lora_layers"]
    active = cfg["lora_layers"]
    if not 1 <= active <= retained <= depth:
        raise ValueError("Trainable LoRA suffix must be contained in the retained parent adapters")
    if not 0 <= cfg["checkpoint_layers"] <= active or cfg["warm_epochs"] != 0:
        raise ValueError("Continuation has no head warmup and checkpoints only trainable suffix layers")
    if cfg["experiment"] != "E0":
        raise ValueError("V3.19.1 uses ordinary committed updates; MetaDG remains a separate V3.19 control")
    if cfg.get("robust_arm") not in ("baseline", "expanded", "hard"):
        raise ValueError("Choose baseline, expanded, or hard condition training")
    if cfg.get("robust_candidate_count") != 4:
        raise ValueError("The registered experiment scores exactly four candidate views")
    if not 0 <= cfg.get("robust_hard_fraction", -1) <= 1:
        raise ValueError("Invalid bounded hard-source fraction")
    return "rtc_v3191_paired_condition_risk_v1"
