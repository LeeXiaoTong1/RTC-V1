"""Version-isolated artifacts; no changes to any historical run."""
from w2v_v3189.common import *

RECIPE = "v54_paired_feature_repair_v1"


def schema_for(cfg):
    from w2v_v3191.ce import classification_weights
    if cfg.get("version") != "5.4" or cfg.get("recipe") != RECIPE:
        raise ValueError("Expected V5.4 paired-feature-repair configuration")
    if cfg.get("v5_stage") != "final":
        raise ValueError("Unknown V5.4 data stage")
    if (cfg.get("parent_init", {}).get("kind") == "v5_submitted_final"
            and cfg.get("v5_stage") != "final"):
        raise ValueError("Submitted V5 parent already trained on Dev; no new screening stage")
    if cfg.get("lora_layers") != 16 or cfg["retained_lora_layers"] < 16:
        raise ValueError("Retain parent adapters and train the final 16 layers")
    if cfg.get("experiment") != "E0" or cfg.get("robust_arm") != "baseline":
        raise ValueError("V5.4 uses paired CE, without meta adaptation or candidate mining")
    classification_weights(cfg)
    if cfg.get('training_recipe_base') != '5.1' or cfg.get('contrastive_weight') != 0.:
        raise ValueError('V5.4 retains V5.1 objectives without V5.2 contrastive learning')
    if any(key.startswith('v52_') for key in cfg):
        raise ValueError('V5.2 PCL scaling cannot enter the V5.1 recipe')
    original = cfg['v54_parent_pcl_weights']
    import math
    if any(not math.isfinite(value) or value < 0 for value in original.values()):
        raise ValueError('Parent PCL coefficients must be finite and nonnegative')
    if (cfg.get('pcl_noise_weight') != (original['noise'] if cfg.get('repair_arm') == 'baseline' else 0.)
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
    if cfg.get('repair_arm') not in ('baseline', 'ce_only', 'paired'):
        raise ValueError('Unknown repair experiment')
    if cfg.get('lora_dropout') != 0.:
        raise ValueError('Paired features require deterministic encoder adapters throughout training')
    if cfg.get('pcl_variance_reference_weight') != max(original.values()):
        raise ValueError('Preserve the original platform variance coefficient')
    for key in ('repair_feature_weight', 'repair_kd_weight', 'repair_lr', 'repair_gate_lr'):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError('Invalid repair coefficient: ' + key)
    if cfg.get('v5_stage') != 'final' or cfg.get('parent_init', {}).get('version') != '5.3':
        raise ValueError('V5.4 continues a completed V5.3 final, with no Dev selection')
    return "rtc_v54_paired_feature_repair_v1"


def _saved(name):
    return name.startswith(('head.', 'fusion.', 'pcl_scale.', 'repair.')) or name.endswith(('.lora_a', '.lora_b'))


def partial_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if _saved(k)}


def apply_partial(model, values):
    expected = {k: v for k, v in model.state_dict().items() if _saved(k)}
    if set(expected) != set(values):
        raise ValueError('V5.4 requires all detector, fusion, repair and scale tensors')
    for name, value in values.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid V5.4 tensor: ' + name)
    with torch.no_grad():
        for name, value in values.items():
            expected[name].copy_(value)
    if hasattr(model, 'pcl_scale'):
        model.pcl_scale.sync_state()
