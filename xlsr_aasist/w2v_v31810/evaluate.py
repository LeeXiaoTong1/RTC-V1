"""Export the unadapted V3.18.10 model with fixed, unfitted probabilities."""
import argparse
from datetime import datetime
import os
import shutil
from pathlib import Path
import torch
from w2v_aasist.data import read_protocol
from w2v_aasist.evaluate import package_scores
from w2v_aasist.launch import run_lock
from .processes import ensure_idle
from .common import ROOT,RECIPE,schema_for,read_json,atomic_json,digest,verify_files,fingerprint,apply_partial,seed_all
from .config import verify_inputs
from .model import load_model
from .inference import infer


def selected(run,kind):
    if kind not in ('best','last'):raise ValueError('Choose best or last')
    run=Path(run);cfg=read_json(run/'config.json')
    status_file=run/'completed.json' if (run/'completed.json').is_file() else run/'paused.json'
    done=read_json(status_file)
    if done.get('version')!='3.18.10' or done.get('recipe')!=RECIPE or done.get('status') not in ('complete','paused'):
        raise ValueError('A completed or safely paused V3.18.10 run is required')
    if done['status']=='paused' and kind!='best':raise ValueError('Paused runs export completed best only')
    if digest(run/'last.pt')!=done['checkpoint_sha256']:raise ValueError('Checkpoint hash changed')
    state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    if state.get('schema')!=schema_for(cfg) or state['identity']!=fingerprint(cfg):raise ValueError('V3.18.10 checkpoint/config mismatch')
    tag=state['best_tag' if kind=='best' else 'last_tag']
    if not tag or tag!=done['best_tag' if kind=='best' else 'last_tag'] or not any(e['tag']==tag for e in state['history']):
        raise ValueError('No completed, recorded V3.18.10 epoch available for this selection')
    return cfg,state['best_model' if kind=='best' else 'model'],dict(done,selected=tag,kind=kind,
        baseline_fallback=False,best_origin='fresh_official_omni')


def export(run,out,kind='best',device='cuda:0',workers=4,protocol=None,audio_root=None,
           dev=False,raw=False,model_factory=load_model,verify=True,announce=True,audit=False):
    if workers<0:raise ValueError('workers must be nonnegative')
    if audit and (dev or kind!='best'):raise ValueError('Locked audit is available only for the final selected best')
    run,out=Path(run).resolve(),Path(out).resolve()
    if out==run or run in out.parents:raise ValueError('Export outside the training run')
    if out.exists() and any(out.iterdir()):raise ValueError('Export directory is not empty')
    original,weights,meta=selected(run,kind)
    cfg=dict(original,device=device,workers=workers,eval_workers=workers)
    if audit:
        result=read_json(run/'locked_audit.json')
        expected=dict(checkpoint_sha256=meta['checkpoint_sha256'],selected=meta['selected'],protocol_sha256=original['validation_protocol_hash'])
        if meta['status']!='complete' or result['identity']!=expected:raise ValueError('No verified locked audit for the completed best')
        out.mkdir(parents=True,exist_ok=True)
        artifact=result.get('metrics',{}).get('predictions')
        if artifact:
            if artifact['file']!='locked_audit_scores.npz':raise ValueError('Unexpected locked audit prediction artifact')
            saved=run/artifact['file']
            if saved.is_symlink() or not saved.is_file() or digest(saved)!=artifact['sha256']:
                raise ValueError('Locked audit prediction artifact changed or missing')
            shutil.copyfile(saved,out/artifact['file'])
        atomic_json(out/'metrics.json',result)
        print('LOCKED_AUDIT='+str(out/'metrics.json'),flush=True)
        return out/'metrics.json'
    seed_all(cfg['seed'])
    if dev:
        path=run/'validation_protocol.json'
        verify_files({str(path):original['saved_manifests']['validation_protocol.json']})
        fixed=read_json(path)
        if fingerprint(fixed)!=original['validation_protocol_hash']:raise ValueError('Locked protocol changed')
        if verify:verify_files(fixed['input_files'])
        protocol_hash=original['validation_protocol_hash'];rows=None
    else:
        protocol_hash=digest(protocol);rows=read_protocol(protocol,audio_root,labeled=False)
    if verify:verify_inputs(original,training=False)
    model=model_factory(cfg);apply_partial(model,weights);del weights
    model.set_phase(True);model.requires_grad_(False).eval()
    out.mkdir(parents=True,exist_ok=True)
    print(f'SUBMISSION_SELECTED=V3.18.10 {meta["selected"]}; kind={kind}; fixed_identity; no target adaptation',flush=True)
    if dev:
        from .validation import evaluate_protocol
        result=evaluate_protocol(model,fixed,cfg,split='select',
            predictions_path=out/f'selection_scores_{meta["selected"]}.npz',checkpoint_tag=meta['selected'])
        atomic_json(out/'metrics.json',dict(selected=meta['selected'],checkpoint_sha256=meta['checkpoint_sha256'],**result))
        print('SELECTION_SCORE='+format(result['score']*100,'.4f'),flush=True)
        path=out/'metrics.json'
    else:
        from .resources import audit_inference
        resource_audit=audit_inference(rows,cfg)
        def progress(completed,total):
            atomic_json(run/'progress.json',dict(status='exporting',phase='Submission inference',
                epoch=meta['completed_epochs'],epochs=original['epochs'],step=completed,steps=total))
        progress(0,len(rows));logits,_=infer(model,rows,cfg,progress=progress)
        probabilities=torch.from_numpy(logits).softmax(-1)[:,0].tolist()
        path=package_scores([r['id'] for r in rows],probabilities,out)
        calibration=dict(selected_candidate='fixed_identity',fit=False,scale=1.,bias=0.,threshold=.5,
            checkpoint_sha256=meta['checkpoint_sha256'],selected=meta['selected'])
        metadata=dict(meta,version='3.18.10',recipe=cfg['recipe'],experiment=cfg['experiment'],run=str(run),
            initialization=cfg['initialization'],frontend=cfg['omni_provenance'],
            calibration=calibration,score='P(fake=0)',threshold=.5,count=len(rows),protocol_sha256=protocol_hash,
            id_order_sha256=fingerprint([r['id'] for r in rows]),zip_sha256=digest(path),
            classification_objective=cfg['classification_objective'],noisy_views_per_source=1,
            pooling=cfg['pooling'],classification_readout='320D window ASP plus 64D within-window temporal variation',
            classifier_input_dim=int(model.head.aasist.out_layer.in_features),
            normalization=getattr(model.head.aasist,'normalization',None),fusion=model.fusion.describe(),
            lora_layers=cfg['lora_layers'],consistency='none',inference_requires_phonemes=False,
            inference_requires_language=False,target_adaptation=False,full_waveform_input=True,
            eval_amp='SSL BF16 / AASIST '+cfg.get('backend_precision','fp32').upper()+' mixed / pooling FP32',
            resource_audit=resource_audit,validation_protocol_sha256=original['validation_protocol_hash'])
        atomic_json(out/'submission_meta.json',metadata)
        if announce:print('SUBMISSION_ZIP='+str(path),flush=True)
    verify_files({str(run/'last.pt'):meta['checkpoint_sha256']})
    if not dev:verify_files({str(protocol):protocol_hash})
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run');p.add_argument('--checkpoint',choices=('best','last'),default='best')
    choice=p.add_mutually_exclusive_group();choice.add_argument('--dev',action='store_true');choice.add_argument('--audit',action='store_true')
    p.add_argument('--raw',action='store_true',help='Compatibility alias: scores are always raw, unfitted softmax')
    p.add_argument('--out');p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--upload-temp',action='store_true');p.add_argument('--report-dir')
    root='/home/ubuntu/LXT/RTC/xlsr_aasist/dataset'
    p.add_argument('--protocol',default=os.environ.get('EVAL_PROTOCOL',root+'/progress.txt'))
    p.add_argument('--audio-root',default=os.environ.get('EVAL_AUDIO_ROOT',root+'/wav/progress'))
    args=p.parse_args();run=args.run or (ROOT/'exp'/'.latest_v31810_run').read_text().strip()
    suffix=('_audit_' if args.audit else '_dev_' if args.dev else '_submission_')+args.checkpoint+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    out=args.out or '/home/ubuntu/LXT/temp/'+Path(run).name+suffix
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle();path=export(run,out,args.checkpoint,args.device,args.workers,args.protocol,args.audio_root,args.dev,args.raw,audit=args.audit)
        if args.upload_temp and not (args.dev or args.audit):
            from .delivery import deliver_export
            deliver_export(run,path.parent,args.report_dir)


if __name__=='__main__':main()
