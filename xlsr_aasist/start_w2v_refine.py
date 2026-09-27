"""One conservative Stage3 refinement from the original best; no cache generation/deletion.

Read paths from a previous improved run's config. Preview by default; --run starts
one new experiment. The rejected run supplies paths only, never model weights.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess

from recover_w2v_storage import recovery_lock, training_command
from w2v_rebuild.core import atomic_json, sha256

ROOT = Path(__file__).resolve().parent


def refine_config(config):
    if config.get('stage') != 3 or not config.get('adaptation'):
        raise ValueError('Expected the previous Stage3 adaptation config')
    baseline = config.get('baseline_path') or config.get('finetune_from')
    if not baseline:
        raise ValueError('Previous run does not identify its original baseline')
    result = dict(config)
    result.update(ordinary_sampling='legacy', noisy_bank_policy='cycle',
                  extra_train_noisy_cache=[], consistency_weight=0.,
                  adaptation_control=False, noisy_extra_fraction=0., noisy_mix_warmup_epochs=1.,
                  trainable_encoder_layers=4, real_ce_weight=1.25, guard_baseline=True,
                  encoder_lr=1e-7, head_lr=2e-6, warmup_epochs=.25,
                  epochs=3, patience=1, earlystop=2,
                  init=None, resume=None, finetune_from=baseline, baseline_path=baseline,
                  preflight=False, check_data=False, no_feature_cache=not bool(config.get('feature_cache')))
    # Ordinary augmentation and the original noisy CE/contrast weights stay unchanged.
    # Dev caches stay exactly the same, so scores remain comparable with the failed run.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', required=True, help='Previous improved/recovered run directory (contains stage3/config.json)')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source = Path(args.from_run).expanduser().resolve()
    config = json.loads((source/'stage3'/'config.json').read_text(encoding='utf-8'))
    values = refine_config(config)
    baseline = Path(values['baseline_path']).expanduser().resolve()
    if baseline.name != 'best_model.pt' or baseline.parent.name != 'stage3' or not baseline.is_file():
        raise ValueError('Original stage3/best_model.pt is missing')
    if baseline.parent.parent == source:
        raise ValueError('Use the original baseline, not the rejected adaptation run')
    # Prevent a quiet switch of the original checkpoint after the previous run.
    print(f'Checking original baseline: {baseline}', flush=True)
    digest = sha256(baseline)
    expected = config.get('init_sha256')
    if not expected or digest != expected:
        raise ValueError('Original baseline SHA256 differs from the previous run (or is missing)')
    run = ROOT/'exp'/('w2v_refine_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
    for path in (source, baseline.parent):
        if run == path or path in run.parents:
            raise ValueError('Refinement needs a separate new output directory')
    command = training_command(values, run/'stage3', baseline, False,
                               keep_feature_cache=bool(values.get('feature_cache')))
    print(f'ORIGINAL_BEST={baseline}\nBASELINE_SHA256={digest}\nRUN={run}', flush=True)
    print('Recipe: legacy ordinary + inverse-frequency CE; real cost=1.25; original noisy train bank only.', flush=True)
    print('Last 4 encoder layers + head; LR=1e-7/2e-6; consistency OFF.', flush=True)
    print('Fixed Dev/cache reuse; max 3 epochs, stop after 2 without an eligible best.', flush=True)
    print('All source checkpoints and audio caches are preserved.', flush=True)
    if not args.run:
        print('Preview only. Add --run to start this single refinement.')
        return
    (ROOT/'exp').mkdir(exist_ok=True)
    with recovery_lock():
        run.mkdir(exist_ok=False)
        atomic_json({'source_run': str(source), 'baseline': str(baseline), 'baseline_sha256': digest,
                     'command': command, 'recipe': values}, run/'refinement_plan.json')
        env = os.environ.copy()
        env.update({key: str(value) for key, value in config.get('noise_environment', {}).items()})
        env.setdefault('OMP_NUM_THREADS', '1')
        env['TOKENIZERS_PARALLELISM'] = 'false'
        # One process: DataBundle is checked once, and the first actual train batch
        # audits gradients/updates. No duplicate preflight model/data initialization.
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        report = json.loads((run/'stage3'/'completed.json').read_text(encoding='utf-8'))
        print('REFINEMENT_RESULT='+report['status'], flush=True)
        print('BEST_MODEL='+report['best_model'], flush=True)
        print('ORIGINAL_BEST_PRESERVED='+str(sha256(baseline) == digest), flush=True)


if __name__ == '__main__':
    main()
