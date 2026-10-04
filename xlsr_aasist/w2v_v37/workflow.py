"""Full official Train language correction with reusable vectors and no new WAVs."""
from datetime import datetime
import gc
import json
from pathlib import Path
import secrets
import shutil
import traceback

import torch
from threadpoolctl import threadpool_limits

from live_progress import publish
from w2v_aasist.launch import ROOT,check_environment,run_lock
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.runtime import atomic_json,sha256
from w2v_v36.workflow import announce,export_report
from w2v_v36.data import build_records
from w2v_v36.features import extract_cache
from .config import parser,configuration,verify_files
from .features import reusable_cache,record_reuse
from .patch import load_base,final_layer,save_patch,load_selected


def report(run,result):
    entries=[('baseline',result['baseline'])]
    entries += [(c['tag'],c['metrics']) for c in result['candidates'] if c.get('metrics')]
    lines=['# V3.7 internal language correction','',
        'Full fixed Dev proxy, not an official Progress/Eval score.',
        'Clean: Online only. Noisy: mean Seen/Heldout. Weighted: 0.3 Clean + 0.7 Noisy.',
        'One frozen detector. Train-only language teacher; no external teacher at inference.',
        'Hyperparameters: grouped Train holdout, followed by refit on all official Train.',
        'The earlier encoder saw Train; this holdout is not an independent domain test.','',
        '| Model | Clean | Noisy | Weighted | EN online real | EN seen real | EN heldout real |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for tag,metrics in entries:
        groups=metrics.get('groups',{})
        values=[100*metrics[k] for k in ('clean_f1','noisy_f1','weighted_f1')]
        values += [100*groups.get(g,{}).get('recall',[0.,0.])[1] for g in ('online/en','seen/en','heldout/en')]
        lines.append('| '+tag+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
        print('[Dev] V3.7 '+tag+' '+' '.join(f'{k}={100*metrics[k]:.3f}' for k in ('clean_f1','noisy_f1','weighted_f1')),flush=True)
        for group in ('online/en','seen/en','heldout/en','online/zh','seen/zh','heldout/zh'):
            if group in groups:
                recall=groups[group]['recall']
                print(f'  {group} fake={100*recall[0]:.3f}% real={100*recall[1]:.3f}%',flush=True)
    for candidate in result['candidates']:
        text=(candidate['tag']+': status='+candidate.get('status','unknown')+
            ', eligible='+str(candidate.get('eligible',False))+
            ', reasons='+str(candidate.get('guardrails') or candidate.get('reason') or 'none'))
        lines += ['',text]
        print('[Selection] '+text,flush=True)
    selected=result['selected']
    applied=result['selected_patch'].get('language_state') is not None
    lines += ['', 'Selected: '+selected,'Baseline fallback: '+str(selected=='baseline'),
        'Language correction applied: '+str(applied),
        'A head-only control win is not evidence that language correction helped.',
        'Deployment requires best_patch.pt AND the SHA-bound original best. Teacher weights are not needed.',
        'This teacher-distilled gated correction is an adaptation, not a reproduction of the language orthogonalization paper.',
        'The fixed Dev is used for model selection. Official Weighted >97 remains unverified.']
    temporary=Path(run)/'report.md.tmp'
    temporary.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    temporary.replace(Path(run)/'report.md')


def _storage(run,records,bundles):
    # Count vectors, row metadata, language assets and atomic-write reserves;
    # already borrowed hidden vectors require no second disk allocation.
    feature_bytes=0
    for split in ('train','dev'):
        if split not in bundles:
            feature_bytes += len(records[split])*514*4 + 2*len(json.dumps(records[split]).encode())
    language_bytes=len(records['train'])*256*4+2*len(json.dumps(records['train']).encode())
    existing=sum(p.stat().st_size for folder in ('features','language')
                 for p in (run/folder).rglob('*') if p.is_file())
    required=max(0,feature_bytes+language_bytes-existing)+384*1024**2
    free=shutil.disk_usage(run).free
    print(f'V37_STORAGE vector_and_metadata_GiB={(feature_bytes+language_bytes)/1024**3:.3f} '
          f'additional_required_GiB={required/1024**3:.3f} free_GiB={free/1024**3:.3f}; no new WAV caches',flush=True)
    if free < required:
        raise OSError('Insufficient space for V3.7 compact caches and reserve; no data was deleted')


def run_experiment(cfg,run):
    bundles={}
    try:
        return _run_experiment(cfg,run,bundles)
    finally:
        closed=set()
        for bundle in bundles.values():
            for name in ('x','logits','lid'):
                array=bundle.get(name)
                mapped=getattr(array,'_mmap',None)
                if mapped is not None and id(mapped) not in closed:
                    mapped.close();closed.add(id(mapped))


def _run_experiment(cfg,run,bundles):
    from .language import ensure_language_assets
    from .language_cache import extract_language_cache
    from .fit import fit_candidates
    from .deployment import validate_deployment
    run=Path(run)
    announce('V3.7 checking complete official Train and fixed full Dev; reuse existing audio')
    records=build_records(cfg['source_config'],cfg['dev_config'])
    atomic_json(run/'coverage.json',records['coverage'])
    saved=run/'data_fingerprints.json'
    if saved.is_file() and json.loads(saved.read_text(encoding='utf-8'))!=records['fingerprints']:
        raise ValueError('Data identity changed after V3.7 started')
    atomic_json(saved,records['fingerprints'])
    identity=dict(base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
                  data_fingerprints=records['fingerprints'],code_fingerprints=cfg['code_fingerprints'])
    for split in ('train','dev'):
        reused=reusable_cache(cfg.get('feature_run'),split,records[split],cfg,records['fingerprints'])
        if reused is not None:
            bundles[split]=reused
    _storage(run,records,bundles)
    if sha256(cfg['base_checkpoint'])!=cfg['base_checkpoint_sha256']:
        raise ValueError('Protected original best changed before extraction')
    announce('V3.7 preparing frozen language teacher and official Train vectors')
    assets=ensure_language_assets(cfg)
    atomic_json(run/'language_teacher.json',assets)
    language=extract_language_cache(records['train'],dict(cfg,language_assets=assets),run/'language'/'train',
        dict(identity,split='train',teacher=assets))
    bundles['train_language']=language
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    announce('V3.7 loading protected submitted best; encoder and MultiConv remain frozen')
    model=load_base(cfg,cfg['device'])
    layer=final_layer(model)
    w=layer.weight.detach().cpu().numpy().copy()
    b=layer.bias.detach().cpu().numpy().copy()
    if w.shape != (2,512) or b.shape != (2,):
        raise ValueError('V3.7 requires the submitted MultiConv final Linear(512,2); no features extracted')
    atomic_json(run/'classifier_info.json',dict(weight_shape=list(w.shape),bias_shape=list(b.shape),
                parameters=int(w.size+b.size),base_checkpoint_sha256=cfg['base_checkpoint_sha256']))
    for split in ('train','dev'):
        if split not in bundles:
            announce('V3.7 extracting '+split+' full-wave detector vectors once')
            bundles[split]=extract_cache(model,records[split],cfg,run/'features'/split,dict(identity,split=split))
        record_reuse(run,split,bundles[split])
    del model,layer
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    if [(r['source_id'],r['condition']) for r in language['rows']] != [
            (r['source_id'],r['condition']) for r in bundles['train']['rows']]:
        raise ValueError('Language and detector row identities/order differ')
    bundles['train']['lid']=language['lid']
    announce('V3.7 fitting internal language branch and guarded classifier on all cached Train')
    previous_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(int(cfg.get('fit_threads',4)))
        with threadpool_limits(limits=int(cfg.get('fit_threads',4))):
            result=fit_candidates(bundles['train'],bundles['dev'],w,b,cfg,run)
            announce('V3.7 replaying candidates through actual deployment module before selection')
            result=validate_deployment(result,bundles['dev'],w,b,cfg,run)
    finally:
        torch.set_num_threads(previous_threads)
    verify_files(records['fingerprints'])
    verify_files(cfg['code_fingerprints'])
    if sha256(cfg['base_checkpoint'])!=cfg['base_checkpoint_sha256']:
        raise ValueError('Protected original best changed; nothing promoted')
    report(run,result)
    state=save_patch(run/'best_patch.pt',cfg,result)
    done=dict(version='3.7',status='complete',selected=result['selected'],
        baseline_fallback=state['baseline_fallback'],language_debias_applied=state['language_debias_applied'],
        base_checkpoint=cfg['base_checkpoint'],base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
        patch_sha256=sha256(run/'best_patch.pt'),external_teacher_at_inference=False)
    atomic_json(run/'completed.json',done)
    print('V37_SELECTED='+result['selected'],flush=True)
    print('V37_BASELINE_FALLBACK='+str(done['baseline_fallback']),flush=True)
    print('V37_LANGUAGE_DEBIAS_APPLIED='+str(done['language_debias_applied']),flush=True)
    return done


def main():
    args=parser().parse_args()
    check_environment(gpu=args.device.startswith('cuda'))
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'),run_lock(ROOT/'exp'/'.v37-workflow.lock'):
        ensure_idle()
        if args.resume:
            run=Path(args.resume).expanduser().resolve()
            cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
            if cfg.get('version')!='3.7':raise ValueError('Resume requires a V3.7 run')
            verify_files(cfg['code_fingerprints'])
        else:
            cfg=configuration(args)
            run=ROOT/'exp'/('w2v_v37_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True,exist_ok=False)
            atomic_json(run/'config.json',cfg)
        (ROOT/'exp'/'.latest_v37_run').write_text(str(run)+'\n',encoding='utf-8')
        print('V37_RUN='+str(run),flush=True)
        try:
            if (run/'completed.json').is_file():
                load_selected(run)
                print('V37_ALREADY_COMPLETE=True',flush=True)
            else:
                verify_files({cfg['dev_config_path']:cfg['dev_config_sha256']})
                run_experiment(cfg,run)
            publish('V3.7 complete',status='complete',force=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            publish('V3.7 failed; committed vectors retained for resume',status='failed',force=True)
            raise
        finally:
            try:export_report(run,args.download_dir,args.upload_temp)
            except Exception as exc:print('REPORT_EXPORT_FAILED='+str(exc)+'; report retained in '+str(run),flush=True)


if __name__=='__main__':main()
