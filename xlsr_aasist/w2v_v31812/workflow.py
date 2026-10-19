"""Full-source diagnosis, frozen residual training, then a gated final audit."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import signal
import sys
import traceback
import zipfile
import numpy as np
import torch
from w2v_v31810.common import (ROOT, read_json, atomic_json, atomic_save, fingerprint,
                              digest, verify_files, apply_partial, seed_all)
from w2v_v31810.config import verify_inputs
from w2v_v31810.identity import verify_registry
from w2v_v31810.model import load_model
from w2v_v31810.validation import _panels
from . import VERSION, SCHEMA
from .cache import extract, concatenate, save_npz, read_complete
from .continuation import prepare_continuation, load_classifier, inherited_review
from .partition import split_sources, full_sweep
from .training import fit, predict
from .metrics import report, guard, paired_changes, bootstrap

# Verify large immutable inputs once per process. Raw audio reads retain the
# parent's per-file size/mtime checks, and every new resume verifies hashes again.
_VERIFIED_INPUTS = set()


def parent_run(value=None):
    if value:return Path(value).expanduser().resolve()
    marker=ROOT/'exp'/'.latest_v31810_run'
    if marker.is_file():return Path(marker.read_text().strip()).resolve()
    candidates=[]
    for path in (ROOT/'exp').glob('w2v_v31810_*'):
        if (path/'completed.json').exists() or (path/'paused.json').exists():candidates.append(path)
    if len(candidates)==1:return candidates[0].resolve()
    raise ValueError('Specify --parent-run with a completed or safely paused V3.18.10 run: '+str(candidates))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run');p.add_argument('--run');p.add_argument('--resume')
    p.add_argument('--continue-from',help='Continue the released v1 feature review in a new run, reusing all completed caches and probes')
    p.add_argument('--upload-report',action='store_true',help='Upload the completed lightweight report to temp.sh')
    p.add_argument('--device',default='cuda:0');p.add_argument('--train-device',default='cuda:0')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--microbatch',type=int,default=2)
    p.add_argument('--frame-budget',type=int,default=4000)
    p.add_argument('--cache-views',type=int,default=2,help='Full-source cached visits; same parent DSP, finite views, default 2')
    p.add_argument('--epochs',type=int,default=20,help='Maximum cheap cached-feature epochs, not 3B training')
    p.add_argument('--patience',type=int,default=5);p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--weight-decay',type=float,default=1e-4)
    return p


def code_identity():
    return {p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}


def prepare(args):
    if args.resume and args.continue_from:
        raise ValueError('Choose --resume for this revision or --continue-from for the released v1 review')
    if args.resume:
        run=Path(args.resume).resolve();cfg=read_json(run/'config.json')
        if cfg['schema']!=SCHEMA or cfg['code']!=code_identity():raise ValueError('Resume code/schema changed')
        verify_files({str(run/k):v for k,v in cfg['artifacts'].items()})
        return run,cfg
    if args.workers<0 or min(args.microbatch,args.frame_budget,args.cache_views,args.epochs,args.patience)<1:
        raise ValueError('Worker/count/budget arguments are invalid')
    if args.batch_size<16 or args.batch_size%8 or not 0<args.lr<=.1 or args.weight_decay<0:
        raise ValueError('Expected batch multiple of eight and positive small learning rate')
    if args.continue_from:
        return prepare_continuation(args,code_identity())
    from w2v_v31810.evaluate import selected
    source=parent_run(args.parent_run);original,weights,meta=selected(source,'best')
    if original.get('omni_arch')!='3b' or not original.get('temporal_stats',True):
        raise ValueError('Expected Omni3B and complete 384D temporal-statistics parent')
    train=read_json(source/'train_rows.json');protocol=read_json(source/'validation_protocol.json')
    registry=read_json(source/'source_registry.json')
    verify_registry(train,registry,original,protocol)
    if fingerprint(protocol)!=original['validation_protocol_hash']:raise ValueError('Parent validation protocol changed')
    run=Path(args.run).resolve() if args.run else ROOT/'exp'/('w2v_v31812_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if run.exists() and any(run.iterdir()):raise ValueError('Run directory exists; use --resume to continue it')
    run.mkdir(parents=True,exist_ok=True)
    split=split_sources(train,original)
    artifacts={}
    for name,value in (('parent_config.json',original),('train_rows.json',train),('validation_protocol.json',protocol),
                       ('source_registry.json',registry),('residual_split.json',split)):
        atomic_json(run/name,value);artifacts[name]=digest(run/name)
    # A self-contained frozen partial parent protects against later parent cleanup
    # or resumed training; the multi-GB official base stays a verified shared asset.
    atomic_save(run/'parent.pt',weights);artifacts['parent.pt']=digest(run/'parent.pt')
    cfg=dict(schema=SCHEMA,version=VERSION,parent=str(source),parent_checkpoint=meta['checkpoint_sha256'],
        parent_tag=meta['selected'],device=args.device,train_device=args.train_device,workers=args.workers,
        microbatch=args.microbatch,frame_budget=args.frame_budget,cache_views=args.cache_views,
        epochs=args.epochs,patience=args.patience,batch_size=args.batch_size,lr=args.lr,weight_decay=args.weight_decay,
        upload_report=args.upload_report,
        label_smoothing=original.get('label_smoothing',.02),backend_precision=original.get('backend_precision','fp32'),
        source_draws_per_stream=original['source_draws_per_stream'],seeds=[3181201,3181202,3181203],
        artifacts=artifacts,code=code_identity(),split_hash=split['hash'],
        train_objective='Eight equally weighted cell CE means; parent/classifier frozen; no meta/PCL/XEUS',
        finite_cache_warning='Cached visits rotate; no new waveform augmentation during residual epochs. These are not new sources.',
        stage3='Not launched automatically. Requires cross-seed external benefit and paired-source Audit evidence.')
    atomic_json(run/'config.json',cfg)
    return run,cfg


def runtime_config(run,cfg):
    original=read_json(run/'parent_config.json')
    return dict(original,device=cfg['device'],workers=cfg['workers'],eval_workers=cfg['workers'],
                microbatch=cfg['microbatch'],frame_budget=cfg['frame_budget'],prefetch_factor=1)


def load_parent(run,cfg):
    original=read_json(run/'parent_config.json')
    verify_files({str(run/'parent.pt'):cfg['artifacts']['parent.pt']})
    # Include pinned code, official checkpoint and runtime identity on every resume.
    verification_key=fingerprint(original)
    if verification_key not in _VERIFIED_INPUTS:
        verify_inputs(original)
        _VERIFIED_INPUTS.add(verification_key)
    live=runtime_config(run,cfg)
    parent=load_model(live)
    apply_partial(parent,torch.load(run/'parent.pt',map_location='cpu',weights_only=True))
    parent.set_phase(False);parent.requires_grad_(False).eval()
    return parent,live


def fixed_cache(parent,part,base_cfg,run,cfg,name,identity,stop):
    local=dict(base_cfg,scene_noise_records={'dev':part['scene_noise_records']})
    if cfg.get('continued_from') and name!='dev_audit':
        origin=cfg['continued_from']
        return read_complete(part['rows'],part['tickets'],local,Path(origin['run'])/'cache'/name,
                             origin['identity'],fixed=True,stop=stop)
    return extract(parent,part['rows'],part['tickets'],local,run/'cache'/name,identity,fixed=True,stop=stop)


def save_scores(path,data,logits):
    rows=data['rows']
    save_npz(path,baseline=data['logits'],logits=logits,
        ids=np.asarray([r['id'] for r in rows],dtype=np.str_),
        canonical_source=np.asarray([r['canonical_source'] for r in rows],dtype=np.str_),
        language=np.asarray([r['language'] for r in rows],dtype=np.str_),
        label=np.asarray([r['label'] for r in rows],dtype=np.int64),
        panel=np.asarray([r['validation_panel'] for r in rows],dtype=np.str_))


def review_global(baseline,probe,*,scope='external'):
    a=baseline['groups']['official_online/en'];b=probe['groups']['official_online/en']
    # A recall gain alone could simply shift decisions toward Real. Require the
    # full ROC ordering to remain comparable; this is a predeclared screening rule.
    if scope not in ('inner','external'):raise ValueError('Unknown feature-review scope')
    recall_ok=(b['real_false_positive']<=a['real_false_positive']+1e-12 if scope=='inner'
               else b['real_false_positive']<a['real_false_positive']-1e-12)
    return recall_ok and b['auc']>=a['auc']-.01


def print_metrics(label,metrics):
    print(f'{label} | SelectS={100*metrics["score"]:.2f} | fixed threshold=0.5',flush=True)
    print('Group                 FakeR   RealR      F1     AUC',flush=True)
    for key,m in metrics['groups'].items():
        print(f'{key:22s} {100*m["fake_recall"]:6.2f} {100*m["real_recall"]:7.2f} '
              f'{100*m["macro_f1"]:7.2f} {100*m["auc"]:7.2f}',flush=True)


def execute(run,cfg,stop=lambda:False):
    identity=fingerprint(cfg)
    if (run/'completed.json').exists():
        completed=read_json(run/'completed.json')
        if completed['identity']!=identity:raise ValueError('Completed experiment identity changed')
        print('Already completed: '+str(run/'summary.json'));return
    seed_all(3181200)
    train_rows=read_json(run/'train_rows.json');protocol=read_json(run/'validation_protocol.json')
    split=read_json(run/'residual_split.json');held=set(split['validation']);fit_sources=set(split['fit'])
    if held&fit_sources:raise ValueError('Residual source split overlaps')
    external=set(protocol['select']['speech_sources'])|set(protocol['audit']['speech_sources'])
    if external&(held|fit_sources):raise ValueError('Parent Train/Dev source overlap')
    if set(protocol['select']['speech_sources'])&set(protocol['audit']['speech_sources']):
        raise ValueError('Parent Select/Audit source overlap')
    if cfg.get('continued_from'):
        parent=None;live=runtime_config(run,cfg);classifier=load_classifier(run,cfg)
        print('Residual continuation: no Omni/AASIST loading for completed Train/Select evidence',flush=True)
    else:parent,live=load_parent(run,cfg)
    from w2v_v31810.sampling import Plan
    plan=Plan(train_rows,cfg=live,seed=live['seed'])
    # Parent sampler supplies exact existing variant definitions and RTC controls.
    train_parts=[]
    for view in range(cfg['cache_views']):
        tickets=[t for t in full_sweep(train_rows,plan,view) if t['canonical_source'] in fit_sources]
        if cfg.get('continued_from'):
            origin=cfg['continued_from']
            part=read_complete(train_rows,tickets,live,Path(origin['run'])/'cache'/f'train_{view}',origin['identity'],stop=stop)
        else:part=extract(parent,train_rows,tickets,live,run/'cache'/f'train_{view}',identity,stop=stop)
        train_parts.append(part)
    train=concatenate(train_parts);del train_parts
    inner_rows=[dict(r,domain=r['condition']) for r in train_rows if r['canonical_source'] in held]
    inner=_panels(inner_rows,'select');inner['scene_noise_records']=live['scene_noise_records']['train']
    validation=fixed_cache(parent,inner,live,run,cfg,'train_inner',identity,stop)
    selection=fixed_cache(parent,protocol['select'],live,run,cfg,'dev_select',identity,stop)
    if parent is not None:classifier=copy.deepcopy(parent.head.aasist.out_layer).cpu()
    # Extraction ended. Release the 3B parent during all cheap feature training.
    del parent
    if torch.cuda.is_available():torch.cuda.empty_cache()
    base_inner=report(validation['rows'],validation['logits']);base_select=report(selection['rows'],selection['logits'])
    stage1=dict(baseline=dict(inner=base_inner,select=base_select),probes={},
        coverage=dict(train_sources=len(fit_sources),inner_sources=len(held),train_cached_views=len(train['rows']),
                      select_sources=len(protocol['select']['speech_sources']),all_fit_sources=set(r['canonical_source'] for r in train['rows'])==fit_sources),
        split_interpretation=split['interpretation'])
    if not stage1['coverage']['all_fit_sources']:raise ValueError('Full-source review omitted a Train source')
    if cfg.get('continued_from'):
        previous=inherited_review(run,cfg,validation,selection,report)
        stage1['probes']=previous['probes'];stage1['inherited_from']=cfg['continued_from']
        print('Both completed probes reused; starting no additional probe epochs',flush=True)
    else:
        for kind in ('global','local'):
            model,best=fit(kind,train,validation,classifier,cfg,run/('probe_'+kind),3181201,identity,stop)
            zi=predict(model,validation,cfg['train_device']);zs=predict(model,selection,cfg['train_device'])
            stage1['probes'][kind]=dict(inner=report(validation['rows'],zi),select=report(selection['rows'],zs),epoch=best['epoch'])
            save_scores(run/f'probe_{kind}_select.npz',selection,zs)
            del model
    stage1['checks']={scope:bool(review_global(base,stage1['probes']['global'][scope],scope='inner' if scope=='inner' else 'external'))
                      for scope,base in (('inner',base_inner),('select',base_select))}
    stage1['supported']=all(stage1['checks'].values())
    stage1['screening_rule']='Train-inner EN-real FPR must not increase (zero errors may remain zero); Select EN-real FPR must decrease; EN AUC at most 0.01 lower on each split. Diagnostic screen only.'
    for label,m in [('Original',base_select)]+[(kind,stage1['probes'][kind]['select']) for kind in ('global','local')]:
        print_metrics('Dev Select '+label,m)
    print('Feature-review checks: '+str(stage1['checks']),flush=True)
    atomic_json(run/'feature_review.json',stage1)
    if not stage1['supported']:
        finish(run,identity,dict(stage='feature_review',decision='stop_feature_review_screen',
            joint_eligible=False,feature_review=stage1));return
    results=[];chosen=None
    for seed in cfg['seeds']:
        model,best=fit('residual',train,validation,classifier,cfg,run/f'residual_{seed}',seed,identity,stop)
        zs=predict(model,selection,cfg['train_device']);metrics=report(selection['rows'],zs)
        decision=guard(base_select,metrics)
        print_metrics(f'Dev Select residual seed={seed}',metrics)
        print('External checks: '+str(decision['checks']),flush=True)
        item=dict(seed=seed,epoch=best['epoch'],inner=best['metrics'],inner_guard=best['guard'],select=metrics,
                  select_guard=decision,paired_cohorts=paired_changes(selection['rows'],selection['logits'],zs))
        results.append(item);save_scores(run/f'residual_{seed}_select.npz',selection,zs)
        # Seed choice is also Train-inner only; no selecting the best Dev seed.
        rank=tuple(best['rank'])
        if chosen is None or rank>chosen['rank']:
            chosen=dict(seed=seed,rank=rank,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        result=item,select_logits=zs)
        del model
    atomic_json(run/'residual_results.json',results)
    stable=sum(r['inner_guard']['passed'] and r['select_guard']['passed'] for r in results)>=2
    selected_ok=chosen['result']['inner_guard']['passed'] and chosen['result']['select_guard']['passed']
    payload=dict(schema=SCHEMA,identity=identity,seed=chosen['seed'],epoch=chosen['result']['epoch'],
        residual={k.removeprefix('residual.'):v for k,v in chosen['model'].items() if k.startswith('residual.')},
        parent_sha256=cfg['artifacts']['parent.pt'])
    atomic_save(run/'residual.pt',payload)
    summary=dict(stage='residual',results=results,chosen_seed=chosen['seed'],selection_rule='Train-inner only',
        stable_across_seeds=stable,joint_eligible=False,decision='stop_external_tradeoff_or_instability',
        residual_sha256=digest(run/'residual.pt'))
    if stable and selected_ok:
        # Only this already selected checkpoint may open the unchanged Audit.
        parent,live=load_parent(run,cfg)
        audit=fixed_cache(parent,protocol['audit'],live,run,cfg,'dev_audit',identity,stop)
        del parent
        if torch.cuda.is_available():torch.cuda.empty_cache()
        from .model import CachedDetector, Residual
        candidate=CachedDetector(classifier,Residual(),cfg['backend_precision']).to(cfg['train_device'])
        candidate.load_state_dict(chosen['model']);za=predict(candidate,audit,cfg['train_device'])
        bm=report(audit['rows'],audit['logits']);cm=report(audit['rows'],za)
        checked=guard(bm,cm);uncertainty=bootstrap(audit['rows'],audit['logits'],za)
        summary['audit']=dict(baseline=bm,candidate=cm,guard=checked,uncertainty=uncertainty,
            paired_cohorts=paired_changes(audit['rows'],audit['logits'],za),
            historical_warning=protocol.get('historical_dev_warning'))
        save_scores(run/'audit_predictions.npz',audit,za)
        summary['joint_eligible']=bool(checked['passed'] and uncertainty['supported'])
        summary['decision']='eligible_for_limited_joint_review' if summary['joint_eligible'] else 'stop_audit_insufficient_or_tradeoff'
    finish(run,identity,summary)


def finish(run,identity,summary):
    summary.update(version=VERSION,identity=identity,fixed_threshold=.5,
        joint_training_launched=False,ensemble=False,meta=False,pcl=False,xeus=False)
    atomic_json(run/'summary.json',summary)
    atomic_json(run/'completed.json',dict(identity=identity,decision=summary['decision'],summary_sha256=digest(run/'summary.json')))
    with zipfile.ZipFile(run/'report.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.log') and 'cache' not in path.relative_to(run).parts:
                archive.write(path,str(path.relative_to(run)))
        for path in sorted(run.glob('*.npz')):archive.write(path,path.name)
    print('V3.18.12 decision: '+summary['decision'],flush=True)
    print('REPORT='+str(run/'report.zip'),flush=True)
    if (run/'config.json').is_file() and read_json(run/'config.json').get('upload_report'):
        try:
            from w2v_aasist.launch import upload_archive
            url=upload_archive(run/'report.zip')
            (run/'report_url.txt').write_text(url+'\n',encoding='ascii')
            print('REPORT_URL='+url,flush=True)
        except Exception as exc:
            print('Report is saved; upload failed: '+str(exc),flush=True)


class Tee:
    def __init__(self,stream,file):self.stream,self.file=stream,file
    def write(self,value):self.stream.write(value);self.file.write(value);self.file.flush()
    def flush(self):self.stream.flush();self.file.flush()


def main():
    args=parser().parse_args()
    from w2v_v31810.processes import ensure_idle
    from w2v_aasist.launch import run_lock
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle();run,cfg=prepare(args)
        (ROOT/'exp').mkdir(exist_ok=True)
        (ROOT/'exp'/'.latest_v31812_run').write_text(str(run)+'\n')
        requested=[False]
        def stop_signal(signum,frame):requested[0]=True
        signal.signal(signal.SIGTERM,stop_signal)
        def stop():return requested[0] or (run/'stop_requested').exists()
        if (run/'stop_requested').exists():
            raise ValueError('Remove this run\'s stop_requested file before resuming')
        old=sys.stdout
        with (run/'training.log').open('a',encoding='utf-8',buffering=1) as log:
            sys.stdout=Tee(old,log)
            try:
                print('RUN='+str(run),flush=True);execute(run,cfg,stop)
            except (KeyboardInterrupt,InterruptedError):
                atomic_json(run/'paused.json',dict(identity=fingerprint(cfg),resume=str(run)))
                print('Paused. Completed cache shards and residual epochs retained.',flush=True)
            except Exception:
                (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
                raise
            finally:sys.stdout=old


if __name__=='__main__':main()
