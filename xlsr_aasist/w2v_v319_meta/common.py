"""Checkpoint identity distinct from the historical w2v_v319 package."""
from w2v_v3189.common import *
from w2v_v3189.common import schema_for as _parent_schema, RECIPE as _parent_recipe

RECIPE = "v319_v3189_best_suffix16_first_order_metadg_v1"


def schema_for(cfg):
    if cfg.get("version") != "3.19" or cfg.get("recipe") != RECIPE:
        raise ValueError("Expected V3.19 parent-best MetaDG; historical w2v_v319 is a different experiment")
    _parent_schema(dict(cfg, version="3.18.9", recipe=_parent_recipe))
    depth = cfg["encoder_layers"]
    retained = cfg["retained_lora_layers"]
    active = cfg["lora_layers"]
    if not 1 <= active <= retained <= depth:
        raise ValueError("Trainable LoRA suffix must be contained in the retained parent adapters")
    if not 0 <= cfg["checkpoint_layers"] <= active or cfg["warm_epochs"] != 0:
        raise ValueError("Continuation has no head warmup and checkpoints only trainable suffix layers")
    if cfg["experiment"] not in ("E0", "E2"):
        raise ValueError("Use E2 MetaDG or matched E0 control")
    return "rtc_v319_parent_best_suffix_metadg_v1"
