"""Version-isolated artifacts; no changes to any historical run."""
from w2v_v3189.common import *

RECIPE = "v52_online60_noisy20_source_contrast_v1"


def schema_for(cfg):
    from .ce import classification_weights
    from .contrastive import configuration
    if cfg.get("version") != "5.2" or cfg.get("recipe") != RECIPE:
        raise ValueError("Expected a V5.2 run; older runs are initialization only")
    if cfg.get("v5_stage") not in ("screen", "final"):
        raise ValueError("Unknown V5.2 data stage")
    if cfg.get("lora_layers") != 16 or cfg["retained_lora_layers"] < 16:
        raise ValueError("Retain parent adapters and train the final 16 layers")
    if cfg.get("experiment") != "E0" or cfg.get("robust_arm") != "baseline":
        raise ValueError("V5.2 uses paired CE, without meta adaptation or candidate mining")
    classification_weights(cfg)
    configuration(cfg)
    if cfg.get('v52_noise_pcl_scale') != .2:
        raise ValueError('V5.2 scales the parent noise PCL coefficient by 0.2')
    original = cfg['v52_parent_pcl_weights']
    import math
    if any(not math.isfinite(value) or value < 0 for value in original.values()):
        raise ValueError('Parent PCL coefficients must be finite and nonnegative')
    if (cfg.get('pcl_noise_weight') != original['noise'] * .2
            or cfg.get('pcl_platform_weight') != original['platform']):
        raise ValueError('PCL coefficients disagree with the recorded parent weights')
    return "rtc_v52_omni_online_contrast_v1"
