"""One English-weighted Stage3 epoch from the preserved original best.

Preview by default. Existing adaptation paths and augmentation are reused; only
the class-conditional language CE budget changes. Reports contain no weights or
audio. --upload-temp explicitly uploads that diagnostic ZIP, not the experiment.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess

from recover_w2v_storage import recovery_lock, training_command
from start_w2v_adapt import adapt_config, check_diverse_cache
from w2v_rebuild.core import atomic_json, sha256

ROOT = Path(__file__).resolve().parent


def english_config(config, diverse_cache):
    result = adapt_config(config, diverse_cache)
    result.update(epochs=1, language_weighting=True, en_real_budget=.35, en_fake_budget=.40)
    return result


def verified_baseline(config, source):
    baseline = Path(config.get('baseline_path') or config.get('finetune_from') or '').expanduser().resolve()
    if baseline.name != 'best_model.pt' or baseline.parent.name != 'stage3' or not baseline.is_file():
        raise ValueError('Original stage3/best_model.pt is missing')
    if baseline.parent.parent == Path(source).resolve():
        raise ValueError('The initialization must be the original best, not the reference run')
    print(f'Checking original baseline SHA256: {baseline}', flush=True)
    digest = sha256(baseline)
    if not config.get('init_sha256') or digest != config['init_sha256']:
        raise ValueError('Original best SHA256 differs from the previous run (or is missing)')
    return baseline, digest


def reference_metrics(reference_run, config):
    """Select the recorded candidate epoch; never substitute the last epoch."""
    source = Path(reference_run).expanduser().resolve()
    result = {'source_run': str(source), 'available': False}
    try:
        stage = source/'stage3'
        reference_config = json.loads((stage/'config.json').read_text(encoding='utf-8'))
        if not config.get('init_sha256') or reference_config.get('init_sha256') != config['init_sha256']:
            raise ValueError('Reference uses a different or unspecified original baseline')
        for key in ('dev_data_path', 'dev_protocol', 'dev_noisy_cache', 'dev_heldout_cache'):
            if not config.get(key) or reference_config.get(key) != config[key]:
                raise ValueError(f'Reference fixed Dev setting differs: {key}')
        for label, recorded in (('Source', config), ('Reference', reference_config)):
            if not isinstance(recorded.get('data_fingerprints'), dict) or not recorded['data_fingerprints']:
                raise ValueError(f'{label} data_fingerprints missing; reference content cannot be verified')
        if reference_config['data_fingerprints'] != config['data_fingerprints']:
            raise ValueError('Reference recorded data_fingerprints differ from the source run')
        for key in ('eval_microbatch', 'eval_batch'):
            if type(reference_config.get(key)) is not int or reference_config[key] < 1:
                raise ValueError(f'Reference inference setting missing or invalid: {key}')
            if reference_config[key] != config.get(key):
                raise ValueError(f'Reference inference setting differs from source: {key}')
        completed = json.loads((stage/'completed.json').read_text(encoding='utf-8'))
        epoch = completed.get('candidate_epoch')
        if type(epoch) is not int or epoch < 1:
            raise ValueError('Reference has no recorded candidate epoch')
        if not (stage/'candidate_best.pt').is_file():
            raise ValueError('Reference candidate_best.pt is missing')
        rows = [json.loads(line) for line in (stage/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        matches = [row for row in rows if row.get('epoch') == epoch]
        if len(matches) != 1 or not matches[0].get('candidate_saved'):
            raise ValueError('Recorded candidate epoch does not identify one saved metrics row')
        metrics = matches[0]
        if not isinstance(metrics.get('dev'), dict) or 'robust_f1' not in metrics['dev']:
            raise ValueError('Reference candidate has no complete Dev metrics')
        result.update(available=True, epoch=epoch, metrics=metrics,
                      candidate_model=str(stage/'candidate_best.pt'),
                      data_fingerprints=reference_config['data_fingerprints'],
                      eval_microbatch=reference_config['eval_microbatch'],
                      eval_batch=reference_config['eval_batch'],
                      dev_inputs={key: reference_config.get(key) for key in
                                  ('dev_protocol', 'dev_noisy_cache', 'dev_heldout_cache', 'ssl_path', 'dev_data_path')},
                      origin='completed.json candidate_epoch + metrics.jsonl; no new inference')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result['reason'] = str(exc)
    return result


def guard_output(run, protected):
    run = Path(run).resolve()
    for value in protected:
        path = Path(value).expanduser().resolve()
        if run == path or path in run.parents:
            raise ValueError(f'Use a new output directory outside source checkpoints and caches: {path}')


def english_command(values, out, baseline):
    command = training_command(values, out, baseline, False,
                               keep_feature_cache=bool(values.get('feature_cache')))
    # An incomplete pull must fail visibly rather than silently run without weights.
    for flag in ('--language_weighting', '--en_real_budget', '--en_fake_budget'):
        if flag not in command:
            raise RuntimeError('Training engine lacks '+flag+'; pull the complete English-weighting update')
    return command


def run_training(command, env, log_path):
    """Preserve child output in the report while keeping nohup progress visible."""
    with Path(log_path).open('x', encoding='utf-8') as output:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                              errors='replace', bufsize=1) as process:
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end='', flush=True)
            return process.wait()


def finish_report(run, reference, download_dir, status, upload_temp=False):
    """Training failure remains a failure even when partial diagnostics export."""
    stage = Path(run)/'stage3'
    stage.mkdir(exist_ok=True)
    atomic_json(status, stage/'launcher_status.json')
    from w2v_rebuild.group_metrics import export_language_report
    report = export_language_report(stage, reference_metrics=reference, download_dir=download_dir)
    archive = report.get('download_archive') or report['archive']
    print('REPORT='+str(report['report'])+'\nDOWNLOAD_ZIP='+str(archive), flush=True)
    if upload_temp:
        from audit_w2v_train import upload_report
        try:
            print('Uploading diagnostic ZIP to temp.sh (no audio or checkpoints).', flush=True)
            link = upload_report(archive)
            (Path(run)/'temp_download_url.txt').write_text(link+'\n', encoding='utf-8')
            print('TEMP_DOWNLOAD_URL='+link+'\nUPLOAD_COMPLETE=True', flush=True)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            # A network failure must not turn a successful GPU run into a failure.
            print('UPLOAD_COMPLETE=False\nLocal diagnostic ZIP is preserved.\n'+str(exc), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', required=True, help='Previous adaptation run containing stage3/config.json')
    parser.add_argument('--reference-run', help='Recorded candidate comparison; defaults to --from-run')
    parser.add_argument('--diverse-cache', help='Existing diverse Train cache; defaults to recorded extra bank')
    parser.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    parser.add_argument('--upload-temp', action='store_true', help='Upload only the diagnostic ZIP to temp.sh')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    source = Path(args.from_run).expanduser().resolve()
    config = json.loads((source/'stage3'/'config.json').read_text(encoding='utf-8'))
    banks = config.get('extra_train_noisy_cache') or []
    if len(banks) > 1 and not args.diverse_cache:
        raise ValueError('Expected one recorded diverse bank; specify --diverse-cache explicitly')
    cache = check_diverse_cache(args.diverse_cache or (banks[0] if banks else Path(config['dev_noisy_cache']).parent/'train_g1'))
    values = english_config(config, cache)
    baseline, digest = verified_baseline(config, source)
    if cache == Path(config['train_noisy_cache']).expanduser().resolve():
        raise ValueError('The diverse bank must differ from the original Train bank')
    reference = reference_metrics(args.reference_run or source, config)
    run = ROOT/'exp'/('w2v_en_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
    protected = [source, baseline.parent, cache, config['train_noisy_cache'],
                 config['dev_noisy_cache'], config['dev_heldout_cache']]
    if args.reference_run:
        protected.append(args.reference_run)
    if config.get('feature_cache'):
        protected.append(config['feature_cache'])
    guard_output(run, protected)
    command = english_command(values, run/'stage3', baseline)
    print(f'ORIGINAL_BEST={baseline}\nBASELINE_SHA256={digest}\nRUN={run}', flush=True)
    print('English CE budgets within true classes: real=35%, fake=40%; class budgets retained.', flush=True)
    print('One epoch; last 4 encoder layers + head; LR=1e-7/2e-6; real cost=1.25.', flush=True)
    print('Existing augmentation/caches/Dev/thresholds unchanged; diverse mixture ramps 0 -> 20%.', flush=True)
    print('REFERENCE='+str(reference.get('epoch')) if reference['available'] else 'REFERENCE_UNAVAILABLE='+reference['reason'], flush=True)
    if not args.run:
        print('Preview only. Add --run to train once. No cache generation or checkpoint deletion.', flush=True)
        return 0
    (ROOT/'exp').mkdir(exist_ok=True)
    with recovery_lock():
        run.mkdir(exist_ok=False)
        atomic_json({'source_run': str(source), 'baseline': str(baseline), 'baseline_sha256': digest,
                     'command': command, 'recipe': values, 'reference': reference}, run/'en_plan.json')
        env = os.environ.copy()
        env.update({key: str(value) for key, value in config.get('noise_environment', {}).items()})
        env.setdefault('OMP_NUM_THREADS', '1')
        env['TOKENIZERS_PARALLELISM'] = 'false'
        error, returncode = None, 1
        try:
            # Data validation and baseline evaluation happen once inside training.
            returncode = run_training(command, env, run/'en_execution.log')
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
            error, returncode = 'Original checkpoint changed during training; inspect before using results', 1
        status = {'status': 'complete' if not error else 'failed', 'training_complete': not error,
                  'returncode': returncode, 'error': error, 'original_best_preserved': preserved}
        try:
            finish_report(run, reference, args.download_dir, status, args.upload_temp)
        except Exception as exc:
            # Retain the true GPU result; all raw files still exist in this run.
            print(f'REPORT_EXPORT_FAILED={type(exc).__name__}: {exc}', flush=True)
        print('ORIGINAL_BEST_PRESERVED='+str(preserved), flush=True)
        if error:
            print('EN_TRAINING_RESULT=failed\n'+error, flush=True)
            return returncode or 1
        completed = json.loads((run/'stage3'/'completed.json').read_text(encoding='utf-8'))
        print('EN_TRAINING_RESULT='+completed['status'], flush=True)
        print('BEST_MODEL='+completed['best_model'], flush=True)
        print('SCORE_CANDIDATE='+str(completed.get('candidate_model')), flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
