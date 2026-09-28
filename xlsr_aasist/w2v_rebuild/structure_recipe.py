"""Shared, versioned recipe binding for the diagnostic and optional training."""
from pathlib import Path


def existing_diverse_cache(config):
    banks = config.get('extra_train_noisy_cache') or []
    if len(banks) > 1:
        raise ValueError('Structure screening expects one recorded diverse Train bank')
    return Path(banks[0] if banks else Path(config['dev_noisy_cache']).parent/'train_g1').expanduser().resolve()


def structure_config(config, cache=None):
    from start_w2v_coverage import coverage_config
    result = coverage_config(config, cache or existing_diverse_cache(config))
    result.update(local_structure_weight=.02, local_structure_warmup_steps=100,
                  noisy_selection=True)
    return result


def structure_recipe_contract(values):
    """Bind effective parser defaults too; a changed recipe needs a new screen.

    Source config and file contents are independently hashed by the audit. This
    records behavior, including augmentation and inference settings, without
    binding the new output directory or initialization bookkeeping.
    """
    from .train import parser
    excluded = {'help', 'out', 'init', 'resume', 'finetune_from', 'check_data',
                'preflight', 'profile_steps'}
    settings = {action.dest: values.get(action.dest, action.default)
                for action in parser()._actions if action.dest not in excluded}
    settings['noise_environment'] = dict(values.get('noise_environment', {}))
    return {'format': 'w2v_local_structure_recipe_v1', 'settings': settings}
