"""V5.5 is an independent continuation of the submitted V5.1 final."""
from w2v_v3189.common import *  # noqa: F401,F403

RECIPE = 'v55_v51_hard_positive_guidance_triplet_v1'


def schema_for(cfg):
    if cfg.get('version') != '5.5' or cfg.get('recipe') != RECIPE:
        raise ValueError('Expected a V5.5 run')
    if cfg.get('training_recipe_base') != '5.1' or cfg.get('v5_stage') != 'final':
        raise ValueError('V5.5 continues V5.1 final, without a new Dev screen')
    if cfg.get('parent_init', {}).get('kind') != 'v5_submitted_final':
        raise ValueError('Only the verified V5.1 submitted final is a V5.5 parent')
    if cfg.get('classification_ce_weights') != dict(online=.25, clean_rtc=.25, noisy=.5):
        raise ValueError('Preserve V5.1 Online25/CleanRTC25/Noisy50 CE')
    if cfg.get('lora_layers') != 16 or cfg.get('retained_lora_layers', 0) < 16:
        raise ValueError('Preserve parent adapters; update the final 16 layers')
    if cfg.get('experiment') != 'E0' or cfg.get('robust_arm') != 'baseline':
        raise ValueError('V5.5 preserves ordinary V5.1 paired training')
    if cfg.get('temporal_stats', True) is not True:
        raise ValueError('V5.5 requires the V5.1 384-dimensional temporal/statistical readout')
    if any(k.startswith(('v52_', 'v53_', 'v54_', 'adaptive_fusion_', 'repair_', 'contrastive_')) for k in cfg):
        raise ValueError('V5.2/V5.3/V5.4 settings cannot enter V5.5')
    return 'rtc_v55_v51_hard_positive_v1'
