"""Version-isolated artifacts; no changes to any historical run."""
from w2v_v3189.common import *

RECIPE = "v5_competition_parent_screen_then_train_dev_v1"


def schema_for(cfg):
    if cfg.get("version") != "5" or cfg.get("recipe") != RECIPE:
        raise ValueError("Expected a V5 run; older runs are initialization only")
    if cfg.get("v5_stage") not in ("screen", "final"):
        raise ValueError("Unknown V5 data stage")
    if cfg.get("lora_layers") != 16 or cfg["retained_lora_layers"] < 16:
        raise ValueError("Retain parent adapters and train the final 16 layers")
    if cfg.get("experiment") != "E0" or cfg.get("robust_arm") != "baseline":
        raise ValueError("V5 uses paired CE, without meta adaptation or candidate mining")
    return "rtc_v5_omni_competition_v1"
