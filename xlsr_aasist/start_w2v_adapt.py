"""One new adaptation from the original best, reusing existing diverse audio.

The previous run supplies paths only. Neither its last.pt nor any original
checkpoint/cache is modified. Preview by default; --run launches once.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess

from recover_w2v_storage import recovery_lock, training_command
from start_w2v_refine import refine_config
from w2v_rebuild.core import atomic_json, sha256

ROOT = Path(__file__).resolve().parent


def adapt_config(config, diverse_cache):
    result = refine_config(config)
    result.update(extra_train_noisy_cache=[str(diverse_cache)], noisy_bank_policy='cycle',
                  noisy_extra_fraction=.2, noisy_mix_warmup_epochs=1., adaptation_control=True,
                  guard_baseline=True, epochs=3, earlystop=2, patience=1)
    return result


def check_diverse_cache(path):
    path = Path(path).expanduser().resolve()
    for name in ('config.json', 'manifest.jsonl'):
        if not (path/name).is_file():
            raise FileNotFoundError(f'Existing diverse train cache missing: {path/name}; use --diverse-cache')
    config = json.loads((path/'config.json').read_text(encoding='utf-8'))
    if (config.get('role') != 'train' or config.get('processing', {}).get('profile') != 'diverse'
            or type(config.get('generation')) is not int or config['generation'] < 1):
        raise ValueError('Expected an existing diverse TRAIN generation; never train on Dev caches')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', required=True, help='Previous refine/recovered experiment directory')
    parser.add_argument('--diverse-cache', help='Existing train_g1; default is the sibling of fixed Dev seen cache')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source = Path(args.from_run).expanduser().resolve()
    config = json.loads((source/'stage3'/'config.json').read_text(encoding='utf-8'))
    cache = check_diverse_cache(args.diverse_cache or Path(config['dev_noisy_cache']).parent/'train_g1')
    values = adapt_config(config, cache)
    baseline = Path(values['baseline_path']).expanduser().resolve()
    if baseline.name != 'best_model.pt' or baseline.parent.name != 'stage3' or not baseline.is_file():
        raise ValueError('Original stage3/best_model.pt is missing')
    if baseline.parent.parent == source:
        raise ValueError('The initialization must be the original best, not the rejected run')
    print(f'Checking original baseline: {baseline}', flush=True)
    digest = sha256(baseline)
    if not config.get('init_sha256') or digest != config['init_sha256']:
        raise ValueError('Original best SHA256 differs from the previous run')
    if cache == Path(config['train_noisy_cache']).resolve():
        raise ValueError('The diverse bank must differ from the original train bank')
    run = ROOT/'exp'/('w2v_adapt_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
    for path in (source, baseline.parent, cache):
        if run == path or path in run.parents:
            raise ValueError('Use a new output directory outside source checkpoints and caches')
    command = training_command(values, run/'stage3', baseline, False,
                               keep_feature_cache=bool(values.get('feature_cache')))
    print(f'ORIGINAL_BEST={baseline}\nBASELINE_SHA256={digest}\nRUN={run}', flush=True)
    print(f'DIVERSE_TRAIN_CACHE={cache}', flush=True)
    print('Last 4 encoder layers + head; LR=1e-7/2e-6; real cost=1.25; consistency OFF.', flush=True)
    print('Diverse noisy pairs ramp 0 -> 20% during epoch 1, then stay at 20%; total noisy CE stays 30%.', flush=True)
    print('Original best retained; candidate_best.pt preserves the best score even if target guards reject it.', flush=True)
    print('Max 3 epochs; progress-based early stopping permits training after an LR reduction within this budget.', flush=True)
    if not args.run:
        print('Preview only. Add --run to start. Existing audio/features are reused; no cache generation.')
        return
    (ROOT/'exp').mkdir(exist_ok=True)
    with recovery_lock():
        run.mkdir(exist_ok=False)
        atomic_json({'source_run': str(source), 'baseline': str(baseline), 'baseline_sha256': digest,
                     'command': command, 'recipe': values}, run/'adaptation_plan.json')
        env = os.environ.copy()
        env.update({key: str(value) for key, value in config.get('noise_environment', {}).items()})
        env.setdefault('OMP_NUM_THREADS', '1')
        env['TOKENIZERS_PARALLELISM'] = 'false'
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        report = json.loads((run/'stage3'/'completed.json').read_text(encoding='utf-8'))
        print('ADAPTATION_RESULT='+report['status'], flush=True)
        print('BEST_MODEL='+report['best_model'], flush=True)
        print('SCORE_CANDIDATE='+str(report.get('candidate_model')), flush=True)
        print('ORIGINAL_BEST_PRESERVED='+str(sha256(baseline) == digest), flush=True)


if __name__ == '__main__':
    main()
