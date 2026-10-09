"""V3.18-only best/last export, explicit raw/calibrated provenance, no fallback."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import numpy as np
import torch
from w2v_aasist.data import read_protocol
from w2v_aasist.evaluate import package_scores
from w2v_aasist.launch import run_lock,upload_archive
from w2v_aasist.full_workflow import ensure_idle
from .common import ROOT,schema_for,read_json,atomic_json,digest,verify_files,fingerprint,apply_partial,seed_all
from .config import verify_inputs
from .model import load_model
from .inference import infer
from .calibration import apply as calibrate
from .monitor import summarize,measure,table_lines


def selected(run,kind):
    if kind not in ('best','last'):raise ValueError('Choose best or last')
    run=Path(run);cfg=read_json(run/'config.json');done=read_json(run/'completed.json')
    if done.get('version')!='3.18' or done.get('status')!='complete':raise ValueError('A completed V3.18 run is required')
    if digest(run/'last.pt')!=done['checkpoint_sha256']:raise ValueError('Checkpoint hash changed')
    state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    if state.get('schema')!=schema_for(cfg) or state['identity']!=fingerprint(cfg):raise ValueError('V3.18 checkpoint/config mismatch')
    tag=state['best_tag' if kind=='best' else 'last_tag']
    if tag!=done['best_tag' if kind=='best' else 'last_tag'] or not any(e['tag']==tag for e in state['history']):
        raise ValueError('Checkpoint is not a trained epoch of this V3.18 run')
    return cfg,state['best_model' if kind=='best' else 'model'],dict(done,selected=tag,kind=kind)


def export(run,out,kind='best',device='cuda:0',workers=4,protocol=None,audio_root=None,dev=False,raw=False,model_factory=load_model,verify=True):
    if workers<0:raise ValueError('workers must be nonnegative')
    run,out=Path(run).resolve(),Path(out).resolve()
    if run==out or run in out.parents:raise ValueError('Export outside the training run')
    if out.exists() and any(out.iterdir()):raise ValueError('Export directory is not empty')
    original,weights,meta=selected(run,kind)
    if verify:verify_inputs(original)
    cfg=dict(original,device=device,workers=workers);seed_all(cfg['seed'])
    model=model_factory(cfg);apply_partial(model,weights);del weights
    model.set_phase(True);model.requires_grad_(False).eval()
    if dev:
        verify_files({str(run/'dev_rows.json'):original['saved_manifests']['dev_rows.json']})
        rows=read_json(run/'dev_rows.json');protocol_hash=None
    else:protocol_hash=digest(protocol);rows=read_protocol(protocol,audio_root,labeled=False)
    out.mkdir(parents=True,exist_ok=True)
    print(f'SUBMISSION_SELECTED=V3.18 {meta["selected"]}; kind={kind}; calibrated={not raw}; baseline_fallback=False',flush=True)
    logits,_=infer(model,rows,cfg)
    calibration=None
    if not raw:
        calibration=read_json(run/f'calibration_{kind}.json')
        if calibration['checkpoint_sha256']!=meta['checkpoint_sha256'] or calibration['selected']!=meta['selected']:
            raise ValueError('Calibration belongs to another checkpoint')
        scores_logits=calibrate(logits,calibration)
    else:scores_logits=logits
    metadata=dict(meta,version='3.18',run=str(run),initialization=cfg['initialization'],frontend=cfg['omni_provenance'],
        calibration=calibration,score='P(fake=0)',threshold=.5,count=len(rows),protocol_sha256=protocol_hash,
        input=f'full 16kHz waveform; whole-wave layer_norm; final {cfg["encoder_dim"]}-dim SSL',
        local_evidence=cfg['local_evidence'],window=cfg['window'],hop=cfg['hop'],eval_amp='SSL BF16 / AASIST FP32')
    if dev:
        raw_metrics=measure(rows,logits);metrics=measure(rows,scores_logits)
        for line in table_lines('[Dev] V3.18 '+meta['selected']+(' raw' if raw else ' calibrated'),summarize(rows,scores_logits)):print(line,flush=True)
        print('Full Dev Weighted='+f'{100*metrics["weighted_f1"]:.3f}'+'; includes calibration fit subset; see separated metrics below',flush=True)
        partition=read_json(run/'dev_partition.json')
        separated={}
        for part in ('select','calibration'):
            ix=partition[part];part_rows=[rows[i] for i in ix]
            separated[part]=dict(raw=measure(part_rows,logits[ix]),calibrated=measure(part_rows,scores_logits[ix]))
        atomic_json(out/'metrics.json',dict(raw=raw_metrics,reported=metrics,partitions=separated,
            note='Full Dev includes calibration fitting rows. Selection rows excluded from fitting; already used for checkpoint selection.'))
        atomic_json(out/'validation_meta.json',metadata)
        np.savez_compressed(out/'scores.npz',raw_logits=logits,reported_logits=scores_logits)
        result=out/'metrics.json'
    else:
        probabilities=torch.from_numpy(scores_logits).softmax(-1)[:,0].tolist()
        result=package_scores([r['id'] for r in rows],probabilities,out)
        atomic_json(out/'submission_meta.json',dict(metadata,zip_sha256=digest(result)))
        print('SUBMISSION_ZIP='+str(result),flush=True)
    verify_files({str(run/'last.pt'):meta['checkpoint_sha256']})
    if protocol_hash:verify_files({str(protocol):protocol_hash})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run');p.add_argument('--checkpoint',choices=('best','last'),default='best')
    p.add_argument('--dev',action='store_true');p.add_argument('--raw',action='store_true')
    p.add_argument('--out');p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--upload-temp',action='store_true')
    root='/home/ubuntu/LXT/RTC/xlsr_aasist/dataset'
    p.add_argument('--protocol',default=os.environ.get('EVAL_PROTOCOL',root+'/progress.txt'))
    p.add_argument('--audio-root',default=os.environ.get('EVAL_AUDIO_ROOT',root+'/wav/progress'))
    args=p.parse_args()
    run=args.run or (ROOT/'exp'/'.latest_v318_run').read_text().strip()
    suffix=('_dev_' if args.dev else '_submission_')+args.checkpoint+('_raw_' if args.raw else '_calibrated_')+datetime.now().strftime('%Y%m%d_%H%M%S')
    out=args.out or '/home/ubuntu/LXT/temp/'+Path(run).name+suffix
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle();path=export(run,out,args.checkpoint,args.device,args.workers,args.protocol,args.audio_root,args.dev,args.raw)
        if args.upload_temp and not args.dev:
            try:print('TEMP_DOWNLOAD_URL='+upload_archive(path),flush=True)
            except Exception as exc:print('UPLOAD_FAILED='+str(exc)+'; local submission retained',flush=True)


if __name__=='__main__':main()
