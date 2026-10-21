"""Version-isolated artifacts; no changes to any historical run."""
from w2v_v3189.common import *

RECIPE = "v53_v51_input_adaptive_layer_fusion_v2"


def schema_for(cfg):
    from w2v_v3191.ce import classification_weights
    if cfg.get("version") != "5.3" or cfg.get("recipe") != RECIPE:
        raise ValueError("Expected V5.3 on the V5.1 recipe; old V5.2-based V5.3 runs cannot resume here")
    if cfg.get("v5_stage") not in ("screen", "final"):
        raise ValueError("Unknown V5.3 data stage")
    if (cfg.get("parent_init", {}).get("kind") == "v5_submitted_final"
            and cfg.get("v5_stage") != "final"):
        raise ValueError("Submitted V5 parent already trained on Dev; no new screening stage")
    if cfg.get("lora_layers") != 16 or cfg["retained_lora_layers"] < 16:
        raise ValueError("Retain parent adapters and train the final 16 layers")
    if cfg.get("experiment") != "E0" or cfg.get("robust_arm") != "baseline":
        raise ValueError("V5.3 uses paired CE, without meta adaptation or candidate mining")
    classification_weights(cfg)
    if cfg.get('training_recipe_base') != '5.1' or cfg.get('contrastive_weight') != 0.:
        raise ValueError('V5.3 retains V5.1 objectives without V5.2 contrastive learning')
    if any(key.startswith('v52_') for key in cfg):
        raise ValueError('V5.2 PCL scaling cannot enter the V5.1 recipe')
    original = cfg['v53_parent_pcl_weights']
    import math
    if any(not math.isfinite(value) or value < 0 for value in original.values()):
        raise ValueError('Parent PCL coefficients must be finite and nonnegative')
    if (cfg.get('pcl_noise_weight') != original['noise']
            or cfg.get('pcl_platform_weight') != original['platform']):
        raise ValueError('PCL coefficients disagree with the recorded parent weights')
    if cfg.get('adaptive_fusion_mode') not in ('dynamic', 'static'):
        raise ValueError('Unknown fusion mode')
    if type(cfg.get('adaptive_fusion_width')) is not int or cfg['adaptive_fusion_width'] < 1:
        raise ValueError('Positive conditioner width required')
    if type(cfg.get('adaptive_fusion_seed')) is not int or not 0 <= cfg['adaptive_fusion_seed'] < 2**32:
        raise ValueError('Conditioner seed must be uint32')
    if cfg.get('adaptive_fusion_condition') != 'frozen_frontend_masked_mean_std_v1':
        raise ValueError('Unknown acoustic condition; checkpoint geometry cannot change')
    return "rtc_v53_v51_omni_adaptive_layer_fusion_v2"
