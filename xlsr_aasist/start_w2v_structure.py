"""Screen local structure first; train only from an explicitly reviewed audit.

Default mode runs read-only inference on a small source-grouped Train/Dev
subset. It does not start a fine-tune, generate audio, or modify checkpoints.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

from recover_w2v_storage import recovery_lock, training_command
from start_w2v_adapt import check_diverse_cache
from start_w2v_en import (verified_baseline, reference_metrics, guard_output,
                          run_training, finish_report)
from w2v_rebuild.core import atomic_json, sha256
from w2v_rebuild.structure_recipe import (existing_diverse_cache, structure_config,
                                         structure_recipe_contract)

ROOT = Path(__file__).resolve().parent


def structure_command(values, out, baseline):
    command = training_command(values, out, baseline, False,
                               keep_feature_cache=bool(values.get('feature_cache')))
    for flag in ('--coverage_training', '--language_weighting',
                 '--local_structure_weight', '--noisy_selection'):
        if flag not in command:
            raise RuntimeError('Incomplete local-structure update: training engine lacks '+flag)
    return command


def audit_command(args, source, out):
    command = [sys.executable, '-u', str(ROOT/'audit_w2v_structure.py'),
               '--from-run', str(source), '--out', str(out),
               '--download-dir', args.download_dir, '--train-per-group', str(args.train_per_group),
               '--dev-per-group', str(args.dev_per_group), '--seed', str(args.audit_seed),
               '--device', args.device, '--microbatch', str(args.audit_microbatch)]
    if args.upload_temp:
        command.append('--upload-temp')
    return command


def reviewed_manifest(directory, baseline, digest, source, values):
    from audit_w2v_structure import validate_reviewed_audit
    directory = Path(directory).expanduser().resolve(strict=True)
    manifest = validate_reviewed_audit(directory, digest)
    if manifest.get('format') != 'w2v_structure_audit_v1':
        raise ValueError('Unknown local-structure diagnostic format')
    if Path(manifest.get('baseline_path', '')).resolve() != baseline.resolve():
        raise ValueError('Reviewed diagnostic uses a different original best path')
    source_config = (source/'stage3'/'config.json').resolve()
    if manifest.get('input_sha256', {}).get(str(source_config)) != sha256(source_config):
        raise ValueError('Reviewed diagnostic source config differs or was not fingerprinted')
    expected = structure_recipe_contract(values)
    if manifest.get('training_recipe_contract') != expected:
        raise ValueError('Training recipe differs from the reviewed diagnostic; screen this recipe again')
    return directory, manifest


def _environment(config):
    env = os.environ.copy()
    env.update({key: str(value) for key, value in config.get('noise_environment', {}).items()})
    env.setdefault('OMP_NUM_THREADS', '1')
    env['TOKENIZERS_PARALLELISM'] = 'false'
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', required=True, help='Previous run supplying paths, never candidate weights')
    parser.add_argument('--mode', choices=['audit', 'train'], default='audit')
    parser.add_argument('--reviewed-audit', help='Completed ready_for_review diagnostic directory; train mode only')
    parser.add_argument('--review-note', default='', help='Optional interpretation retained in the training plan')
    parser.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    parser.add_argument('--upload-temp', action='store_true', help='Upload only the diagnostic report ZIP to temp.sh')
    parser.add_argument('--preview', action='store_true', help='Print plan without GPU work or creating a run directory')
    parser.add_argument('--train-per-group', type=int, default=64)
    parser.add_argument('--dev-per-group', type=int, default=32)
    parser.add_argument('--audit-seed', type=int, default=1729)
    parser.add_argument('--device', default='cuda:0', help='Read-only audit inference device')
    parser.add_argument('--audit-microbatch', type=int, default=4)
    args = parser.parse_args(argv)
    if args.mode == 'train' and not args.reviewed_audit:
        parser.error('train mode requires --reviewed-audit from a completed, reviewed diagnostic')
    if args.mode == 'audit' and (args.reviewed_audit or args.review_note):
        parser.error('--reviewed-audit/--review-note apply only to explicit train mode')
    if min(args.train_per_group, args.dev_per_group, args.audit_microbatch) < 1:
        parser.error('Audit sample counts and microbatch must be positive')
    source = Path(args.from_run).expanduser().resolve(strict=True)
    config = json.loads((source/'stage3'/'config.json').read_text(encoding='utf-8'))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2)
    run = ROOT/'exp'/('w2v_structure_'+args.mode+'_'+stamp)
    if args.mode == 'audit':
        command = audit_command(args, source, run)
        print(f'MODE=read_only_audit\nREPORT_DIR={run}', flush=True)
        print('Small Train/Dev source-grouped screen only. No training starts automatically.', flush=True)
        if args.preview:
            print('Preview only. No model load, checkpoint/cache modification or output creation.', flush=True)
            print('COMMAND='+json.dumps(command), flush=True)
            return 0
        (ROOT/'exp').mkdir(exist_ok=True)
        with recovery_lock():
            return subprocess.run(command, cwd=ROOT, env=_environment(config), check=False).returncode

    cache = check_diverse_cache(existing_diverse_cache(config))
    if cache == Path(config['train_noisy_cache']).expanduser().resolve():
        raise ValueError('Diverse Train bank must differ from the original bank')
    values = structure_config(config, cache)
    baseline, digest = verified_baseline(config, source)
    audit_dir, manifest = reviewed_manifest(args.reviewed_audit, baseline, digest, source, values)
    reference = reference_metrics(source, config)
    protected = [source, audit_dir, baseline.parent, cache, config['train_noisy_cache'],
                 config['dev_noisy_cache'], config['dev_heldout_cache']]
    if config.get('feature_cache'):
        protected.append(config['feature_cache'])
    guard_output(run, protected)
    command = structure_command(values, run/'stage3', baseline)
    print(f'MODE=reviewed_training\nORIGINAL_BEST={baseline}\nBASELINE_SHA256={digest}\nRUN={run}', flush=True)
    print('One epoch; last 4 encoder layers + head; LR=1e-7/2e-6; local structure weight=0.02 (provisional).', flush=True)
    print('Coverage recipe and CE budgets retained; local structure replaces only noisy global InfoNCE.', flush=True)
    print('Promotion: noisy and weighted Dev F1 improve; Online F1 does not decrease; recall slices are diagnostics.', flush=True)
    if args.preview:
        print('Preview only. Verified the reviewed diagnostic; no training/output creation.', flush=True)
        return 0
    (ROOT/'exp').mkdir(exist_ok=True)
    with recovery_lock():
        run.mkdir(exist_ok=False)
        audit_summary = json.loads((audit_dir/'summary.json').read_text(encoding='utf-8'))
        atomic_json({'source_run': str(source), 'baseline': str(baseline), 'baseline_sha256': digest,
                     'command': command, 'recipe': values, 'reference': reference,
                     'reviewed_audit': str(audit_dir), 'audit_manifest': manifest,
                     'audit_summary': audit_summary, 'review_note': args.review_note,
                     'review_action': 'explicit --mode train --reviewed-audit'}, run/'structure_plan.json')
        error, returncode = None, 1
        try:
            returncode = run_training(command, _environment(config), run/'structure_execution.log')
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
            error, returncode = 'Original best changed; inspect these results before use', 1
        status = {'status': 'complete' if not error else 'failed', 'training_complete': not error,
                  'returncode': returncode, 'error': error, 'original_best_preserved': preserved}
        try:
            finish_report(run, reference, args.download_dir, status, args.upload_temp)
        except Exception as exc:
            print(f'REPORT_EXPORT_FAILED={type(exc).__name__}: {exc}', flush=True)
        print('ORIGINAL_BEST_PRESERVED='+str(preserved), flush=True)
        if error:
            print('STRUCTURE_TRAINING_RESULT=failed\n'+error, flush=True)
            return returncode or 1
        completed = json.loads((run/'stage3'/'completed.json').read_text(encoding='utf-8'))
        print('STRUCTURE_TRAINING_RESULT='+completed['status'], flush=True)
        print('BEST_MODEL='+completed['best_model'], flush=True)
        print('SCORE_CANDIDATE='+str(completed.get('candidate_model')), flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
