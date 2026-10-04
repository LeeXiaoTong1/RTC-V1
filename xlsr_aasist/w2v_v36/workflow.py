"""Extract once, tune on Train sources, compare on fixed Dev, save a tiny patch."""
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import secrets
import shutil
import traceback
import zipfile

import torch
from live_progress import phase, publish
from w2v_aasist.launch import ROOT, check_environment, run_lock, upload_archive
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.runtime import atomic_json, sha256
from .config import parser, configuration, verify_files
from .patch import load_base, final_layer, save_patch, load_selected


def announce(message):
    print('[Phase] '+message, flush=True)
    phase(message)


def export_report(run, destination, upload=False):
    run, destination = Path(run), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/(run.name+'_report.zip')
    temporary = archive.with_suffix('.zip.tmp')
    # Explicit top-level diagnostics only: no audio, feature matrices or base weights.
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run.iterdir()):
            if path.is_file() and not path.is_symlink() and (
                    path.suffix in ('.json', '.jsonl', '.md', '.log')
                    or path.name.startswith('dev_scores_') and path.suffix == '.npz'):
                z.write(path, path.name)
    os.replace(temporary, archive)
    print('REPORT_ZIP='+str(archive), flush=True)
    if upload:
        url = upload_archive(archive)
        (run/'temp_download_url.txt').write_text(url+'\n', encoding='utf-8')
    return archive


def report(run, result):
    entries = [('baseline', result['baseline'])]
    entries += [(c['tag'], c['metrics']) for c in result['candidates'] if c.get('metrics')]
    lines = ['# V3.6 frozen best + final classifier correction', '',
        'Fixed full Dev proxy, not an official Progress/Eval score.',
        'Clean is official Online only; Noisy is mean Seen/Heldout; Weighted = 0.3 Clean + 0.7 Noisy.',
        'FP32, full recordings, threshold 0.5. Same features for baseline and both candidates.',
        'Regularization is selected on a source-group Train holdout, not Dev.', '',
        '| Model | Clean | Noisy | Weighted | EN online real | EN seen real | EN heldout real |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for tag, metrics in entries:
        groups = metrics.get('groups', {})
        values = [100*metrics[k] for k in ('clean_f1', 'noisy_f1', 'weighted_f1')]
        values += [100*groups.get(g, {}).get('recall', [0.,0.])[1]
                   for g in ('online/en', 'seen/en', 'heldout/en')]
        lines.append('| '+tag+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
        print('[Dev] '+tag+' '+' '.join(f'{k}={100*metrics[k]:.3f}'
              for k in ('clean_f1', 'noisy_f1', 'weighted_f1')), flush=True)
        for group in ('online/en', 'seen/en', 'heldout/en'):
            if group in groups:
                rec = groups[group]['recall']
                print(f'  {group} fake={100*rec[0]:.3f}% real={100*rec[1]:.3f}%', flush=True)
    for candidate in result['candidates']:
        summary = (candidate['tag']+': status='+str(candidate.get('status','unknown'))+
            ', lambda='+str(candidate.get('selected_regularization','unavailable'))+
            ', eligible='+str(candidate.get('eligible',False))+
            ', reasons='+str(candidate.get('guardrails') or candidate.get('reason') or 'none'))
        lines += ['', summary]
        print('[Selection] '+summary, flush=True)
    lines += ['', 'Selected: '+result['selected'],
        'Baseline fallback: '+str(result['selected']=='baseline'),
        'Deployment requires both best_patch.pt AND the SHA-bound original base checkpoint.',
        'The tuning holdout was seen by the earlier encoder training; it is not an unseen-domain test.',
        'Only an official submission can establish progress toward Weighted >97.']
    target = Path(run)/'report.md'
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    os.replace(temporary, target)


def run_experiment(cfg, run):
    from .data import build_records
    from .features import extract_cache
    from .fit import fit_candidates
    announce('V3.6 checking existing full Train/Dev; no audio generation')
    records = build_records(cfg['source_config'], cfg['dev_config'])
    atomic_json(run/'coverage.json', records['coverage'])
    fingerprint_path = run/'data_fingerprints.json'
    if fingerprint_path.is_file():
        if json.loads(fingerprint_path.read_text(encoding='utf-8')) != records['fingerprints']:
            raise ValueError('Data metadata changed since extraction started')
    else:
        atomic_json(fingerprint_path, records['fingerprints'])
    identity = dict(base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
                    data_fingerprints=records['fingerprints'], code_fingerprints=cfg['code_fingerprints'])
    announce('V3.6 loading verified best; all upstream weights frozen')
    model = load_base(cfg, cfg['device'])
    layer = final_layer(model)
    w, b = layer.weight.detach().cpu().numpy().copy(), layer.bias.detach().cpu().numpy().copy()
    # Reserve both splits before spending GPU time, including duplicated JSON
    # during atomic metadata writes. Existing committed files are already paid for.
    estimate = sum(len(records[s])*(w.shape[1]+2)*4 +
                   2*len(json.dumps(records[s], ensure_ascii=False).encode('utf-8'))
                   for s in ('train','dev'))
    present = sum(p.stat().st_size for p in (run/'features').rglob('*') if p.is_file())
    required = max(0, estimate-present)+256*1024**2
    available = shutil.disk_usage(run).free
    print(f'V36_TOTAL_FEATURE_STORAGE estimated_GiB={estimate/1024**3:.3f} '
          f'additional_required_GiB={required/1024**3:.3f} free_GiB={available/1024**3:.3f}',flush=True)
    if available < required:
        raise OSError('Insufficient space for both feature splits; no audio or checkpoint was deleted')
    atomic_json(run/'original_classifier.json', dict(weight=w.tolist(), bias=b.tolist()))
    print(f'V36_CLASSIFIER_PARAMETERS={w.size+b.size}; backbone gradients=False', flush=True)
    bundles = {}
    for split in ('train', 'dev'):
        announce('V3.6 extracting '+split+' full-recording features once')
        bundles[split] = extract_cache(model, records[split], cfg, run/'features'/split,
                                      dict(identity, split=split))
    del model, layer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    announce('V3.6 fitting two anchored final classifiers on cached Train features')
    result = fit_candidates(bundles['train'], bundles['dev'], w, b, cfg, run)
    verify_files(records['fingerprints'])
    if sha256(cfg['base_checkpoint']) != cfg['base_checkpoint_sha256']:
        raise RuntimeError('Protected base changed; no patch promoted')
    verify_files(cfg['code_fingerprints'])
    report(run, result)
    patch = save_patch(run/'best_patch.pt', cfg, result)
    done = dict(version='3.6', status='complete', selected=result['selected'],
        baseline_fallback=patch['baseline_fallback'], base_checkpoint=cfg['base_checkpoint'],
        base_checkpoint_sha256=cfg['base_checkpoint_sha256'], patch_sha256=sha256(run/'best_patch.pt'))
    atomic_json(run/'completed.json', done)
    print('V36_SELECTED='+result['selected'], flush=True)
    print('V36_BASELINE_FALLBACK='+str(done['baseline_fallback']), flush=True)
    return done


def main():
    args = parser().parse_args()
    check_environment(gpu=args.device.startswith('cuda'))
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'), run_lock(ROOT/'exp'/'.v36-workflow.lock'):
        ensure_idle()
        if args.resume:
            run = Path(args.resume).expanduser().resolve()
            cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
            if cfg.get('version') != '3.6':
                raise ValueError('Resume requires a V3.6 run')
            verify_files(cfg['code_fingerprints'])
        else:
            cfg = configuration(args)
            run = ROOT/'exp'/('w2v_v36_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True, exist_ok=False)
            atomic_json(run/'config.json', cfg)
        (ROOT/'exp'/'.latest_v36_run').write_text(str(run)+'\n', encoding='utf-8')
        print('V36_RUN='+str(run), flush=True)
        try:
            if (run/'completed.json').is_file():
                load_selected(run)
                print('V36_ALREADY_COMPLETE=True', flush=True)
            else:
                verify_files({cfg['dev_config_path']: cfg['dev_config_sha256']})
                run_experiment(cfg, run)
            publish('V3.6 complete', status='complete', force=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
            publish('V3.6 failed; committed feature rows can be resumed', status='failed', force=True)
            raise
        finally:
            try:
                export_report(run, args.download_dir, args.upload_temp)
            except Exception as exc:
                print('REPORT_EXPORT_FAILED='+str(exc)+'; diagnostics retained in '+str(run), flush=True)


if __name__ == '__main__':
    main()
