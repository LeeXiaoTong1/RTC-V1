"""Export the trained V3.17 winner, with no historical-parent fallback."""
import argparse
import os
from pathlib import Path
import torch
from w2v_aasist.evaluate import package_scores
from w2v_v3.data import read_protocol
from w2v_v39.common import ROOT,atomic_json,digest,verify_files
from .config import verify_inputs
from .model import load_model
from .state import load_selected,apply_partial,KINDS
from .output import infer,print_metrics


def export(run,protocol,audio_root,out,device='cuda:0',workers=2,checkpoint_kind='best',dev=False):
    run,out=Path(run).expanduser().resolve(),Path(out).expanduser().resolve()
    if out==run or run in out.parents:raise ValueError('Output must be outside the training run')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):raise ValueError('Output exists; choose a new --out path')
    checkpoint,meta=load_selected(run,checkpoint_kind)
    original=checkpoint['config'];verify_inputs(original)
    cfg=dict(original,device=device,workers=workers,feature_workers=workers)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    model=load_model(cfg,training=False);apply_partial(model,checkpoint['candidate']['state'])
    model.requires_grad_(False).eval();del checkpoint
    if dev:
        from w2v_v316_tfcl.data import bundles
        with bundles(cfg) as (_,bundle):rows=bundle['rows']
        protocol_hash=None
    else:
        protocol_hash=digest(protocol);rows=read_protocol(protocol,audio_root,labeled=False)
    out.mkdir(parents=True,exist_ok=True)
    print('SELECTED='+meta['selected']+'; origin='+meta['origin']['version']+'; fallback=False',flush=True)
    logits=infer(model,rows,cfg,'V3.17 '+('Dev' if dev else 'submission'),out)
    verify_inputs(original);verify_files({meta['checkpoint_path']:meta['checkpoint_sha256']})
    record=dict(version='3.17',selected=meta['selected'],origin=meta['origin'],
        checkpoint_kind=checkpoint_kind,checkpoint_sha256=meta['checkpoint_sha256'],
        initialization=cfg['initialization_provenance'],
        baseline_fallback=False,score='P(fake)',threshold=.5,count=len(rows),input_policy='full utterance',
        eval_amp='none',training_only_tfcl_removed_at_inference=True,protocol_sha256=protocol_hash)
    if dev:
        import numpy as np
        from w2v_v316_tfcl.metrics import measure
        value=measure(rows,logits,target=cfg['matched_fake_recall'])
        print_metrics(meta['selected'],value)
        np.savez_compressed(out/'dev_scores.npz',logits=logits)
        atomic_json(out/'metrics.json',value);atomic_json(out/'validation_meta.json',record)
        return out/'metrics.json'
    verify_files({str(protocol):protocol_hash})
    scores=torch.from_numpy(logits).softmax(-1)[:,0].tolist()
    archive=package_scores([r['id'] for r in rows],scores,out)
    atomic_json(out/'submission_meta.json',dict(record,zip_sha256=digest(archive)))
    print('SUBMISSION_ZIP='+str(archive),flush=True)
    return archive


def main():
    from w2v_aasist.full_workflow import ensure_idle
    from w2v_aasist.launch import run_lock,upload_archive
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run');p.add_argument('--checkpoint',choices=KINDS,default='best')
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--out');p.add_argument('--dev',action='store_true');p.add_argument('--upload-temp',action='store_true')
    root='/home/ubuntu/LXT/RTC/xlsr_aasist/dataset'
    p.add_argument('--protocol',default=os.environ.get('EVAL_PROTOCOL',root+'/progress.txt'))
    p.add_argument('--audio-root',default=os.environ.get('EVAL_AUDIO_ROOT',root+'/wav/progress'))
    args=p.parse_args()
    if args.workers<0:p.error('workers must be nonnegative')
    run=args.run or os.environ.get('V317_RUN')
    if not run:run=(ROOT/'exp'/'.latest_v317_run').read_text(encoding='utf-8').strip()
    out=args.out or os.environ.get('SUBMISSION_DIR') or '/home/ubuntu/LXT/temp/'+Path(run).name+(
        '_dev_' if args.dev else '_submission_')+args.checkpoint
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        archive=export(run,args.protocol,args.audio_root,out,args.device,args.workers,args.checkpoint,args.dev)
        if args.upload_temp and not args.dev:
            try:
                url=upload_archive(archive)
                (Path(out)/'temp_download_url.txt').write_text(url+'\n',encoding='utf-8')
            except Exception as exc:print('UPLOAD_FAILED='+str(exc)+'; local ZIP retained',flush=True)


if __name__=='__main__':main()
