"""Export an explicit Omni selection with exactly the validated numerical path."""
import argparse
from pathlib import Path
import torch
from w2v_aasist.evaluate import package_scores
from w2v_aasist.data import read_protocol
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.launch import run_lock,upload_archive
from w2v_v39.common import ROOT,read_json,atomic_json,digest,verify_files
from .model import load_model
from .state import KINDS,load_selected,apply_partial
from .inference import infer


def export(args):
    run=Path(args.run).expanduser().resolve(); out=Path(args.out).expanduser().resolve()
    if out==run or run in out.parents: raise ValueError('Submission must be outside the run')
    cfg,weights,meta=load_selected(run,args.checkpoint)
    verify_files(cfg['code_fingerprints']); verify_files({cfg['omni_checkpoint']:cfg['omni_sha256']})
    cfg=dict(cfg,**read_json(run/'execution_plan.json')['selected'])
    cfg.update(device=args.device,workers=args.workers)
    torch.set_num_threads(1); torch.cuda.set_device(torch.device(args.device))
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('Recorded BF16 inference unavailable')
    model=load_model(cfg); apply_partial(model,weights); del weights
    model.requires_grad_(False).eval()
    protocol_hash=digest(args.protocol)
    rows=read_protocol(args.protocol,args.audio_root,labeled=False)
    logits=infer(model,rows,cfg,'V3.16 submission')
    scores=torch.from_numpy(logits).softmax(-1)[:,0].tolist()
    verify_files({args.protocol:protocol_hash,meta['checkpoint_path']:meta['checkpoint_sha256'],cfg['omni_checkpoint']:cfg['omni_sha256']})
    archive=package_scores([r['id'] for r in rows],scores,out)
    atomic_json(out/'submission_meta.json',dict(version='3.16',selected=meta['selected'],checkpoint_kind=args.checkpoint,
        checkpoint_sha256=meta['checkpoint_sha256'],omni_sha256=cfg['omni_sha256'],count=len(rows),
        protocol_sha256=protocol_hash,zip_sha256=digest(archive),score='P(fake)',threshold=.5,
        eval_amp='bf16',input_policy='full waveform; exact-length CNN frontend',automatic_baseline_fallback=False))
    print('SUBMISSION_SELECTED='+meta['selected']+'\nSUBMISSION_ZIP='+str(archive),flush=True)
    if args.upload_temp:
        try:
            url=upload_archive(archive); (out/'temp_download_url.txt').write_text(url+'\n',encoding='utf-8')
        except Exception as exc: print('UPLOAD_FAILED='+str(exc)+'; local ZIP retained',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','protocol','audio-root','out'): p.add_argument('--'+key,required=True)
    p.add_argument('--checkpoint',choices=KINDS,default='best_guarded')
    p.add_argument('--workers',type=int,default=2); p.add_argument('--device',default='cuda:0')
    p.add_argument('--upload-temp',action='store_true'); args=p.parse_args()
    if args.workers<0: p.error('workers must be nonnegative')
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle(); export(args)


if __name__=='__main__': main()
