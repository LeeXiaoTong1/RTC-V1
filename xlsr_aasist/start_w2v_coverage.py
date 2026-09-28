"""One coverage-repair epoch from the preserved original Stage3 best.

Read-only preview by default. Reuses complete existing audio caches and the
same fixed Dev. --upload-temp exports diagnostics only, never audio or weights.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess

from recover_w2v_storage import recovery_lock, training_command
from start_w2v_adapt import check_diverse_cache
from start_w2v_en import (english_config, verified_baseline, reference_metrics,
                          guard_output, run_training, finish_report)
from w2v_rebuild.core import atomic_json, sha256

ROOT = Path(__file__).resolve().parent


def coverage_config(config, diverse_cache):
    result = english_config(config, diverse_cache)
    result.update(coverage_training=True, coverage_prefix_probability=.5)
    return result


def coverage_command(values, out, baseline):
    command = training_command(values, out, baseline, False,
                               keep_feature_cache=bool(values.get('feature_cache')))
    for flag in ('--coverage_training', '--coverage_prefix_probability', '--language_weighting'):
        if flag not in command:
            raise RuntimeError('Incomplete coverage update: training engine lacks '+flag)
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', required=True, help='Prior adaptation/English run supplying paths, not weights')
    parser.add_argument('--diverse-cache', help='Existing diverse Train bank; defaults to recorded extra cache')
    parser.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    parser.add_argument('--upload-temp', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    source = Path(args.from_run).expanduser().resolve()
    config = json.loads((source/'stage3'/'config.json').read_text(encoding='utf-8'))
    banks = config.get('extra_train_noisy_cache') or []
    if len(banks) > 1 and not args.diverse_cache:
        raise ValueError('Specify --diverse-cache when the previous run has multiple extra banks')
    cache = check_diverse_cache(args.diverse_cache or (banks[0] if banks else
                               Path(config['dev_noisy_cache']).parent/'train_g1'))
    if cache == Path(config['train_noisy_cache']).expanduser().resolve():
        raise ValueError('Diverse Train bank must differ from the original bank')
    values = coverage_config(config, cache)
    baseline, digest = verified_baseline(config, source)
    reference = reference_metrics(source, config)
    run = ROOT/'exp'/('w2v_coverage_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
    protected = [source, baseline.parent, cache, config['train_noisy_cache'],
                 config['dev_noisy_cache'], config['dev_heldout_cache']]
    if config.get('feature_cache'):
        protected.append(config['feature_cache'])
    guard_output(run, protected)
    command = coverage_command(values, run/'stage3', baseline)
    print(f'ORIGINAL_BEST={baseline}\nBASELINE_SHA256={digest}\nRUN={run}', flush=True)
    print('One epoch; last 4 encoder layers + head; LR=1e-7/2e-6; real cost=1.25.', flush=True)
    print('Ordinary: full traversal; 50% prefix / 50% energy-screened random proposal; one 64600-sample view.', flush=True)
    print('Noisy: shared bank/family/SNR per balanced batch; en targets fake=40%, real=35%; CE language multiplier=1.', flush=True)
    print('Reuse all caches. Diverse mixture still ramps 0 -> 20%; noisy CE remains 30%. No backend change.', flush=True)
    print('Coverage counts and fixed Dev group scores will be saved locally and included in the diagnostic ZIP.', flush=True)
    if not args.run:
        print('Preview only. Add --run to train. No checkpoint deletion, cache generation or model load.', flush=True)
        return 0
    (ROOT/'exp').mkdir(exist_ok=True)
    with recovery_lock():
        run.mkdir(exist_ok=False)
        atomic_json({'source_run': str(source), 'baseline': str(baseline), 'baseline_sha256': digest,
                     'command': command, 'recipe': values, 'reference': reference}, run/'coverage_plan.json')
        env = os.environ.copy()
        env.update({key: str(value) for key, value in config.get('noise_environment', {}).items()})
        env.setdefault('OMP_NUM_THREADS', '1')
        env['TOKENIZERS_PARALLELISM'] = 'false'
        error, returncode = None, 1
        try:
            returncode = run_training(command, env, run/'coverage_execution.log')
            if returncode:
                error = f'Training exited with status {returncode}'
            elif not (run/'stage3'/'completed.json').is_file():
                error, returncode = 'Training returned without completed.json', 1
        except (OSError, subprocess.SubprocessError) as exc:
            error = f'{type(exc).__name__}: {exc}'
        try:
            preserved = sha256(baseline) == digest
        except OSError:
            preserved = False
        if not preserved:
            error, returncode = 'Original best changed; do not use these results before inspection', 1
        status = {'status': 'complete' if not error else 'failed', 'training_complete': not error,
                  'returncode': returncode, 'error': error, 'original_best_preserved': preserved}
        try:
            finish_report(run, reference, args.download_dir, status, args.upload_temp)
        except Exception as exc:
            print(f'REPORT_EXPORT_FAILED={type(exc).__name__}: {exc}', flush=True)
        print('ORIGINAL_BEST_PRESERVED='+str(preserved), flush=True)
        if error:
            print('COVERAGE_TRAINING_RESULT=failed\n'+error, flush=True)
            return returncode or 1
        completed = json.loads((run/'stage3'/'completed.json').read_text(encoding='utf-8'))
        print('COVERAGE_TRAINING_RESULT='+completed['status'], flush=True)
        print('BEST_MODEL='+completed['best_model'], flush=True)
        print('SCORE_CANDIDATE='+str(completed.get('candidate_model')), flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
