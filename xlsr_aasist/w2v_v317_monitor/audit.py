"""Add full Train metrics for a completed V3.17 best or last checkpoint."""
import argparse
from pathlib import Path
import torch
from w2v_v39.common import ROOT,digest
from w2v_v317.config import verify_inputs
from w2v_v317.state import load_selected,apply_partial
from w2v_v317.model import load_model
from w2v_v317.data import bundles
from .panel import evaluate_panel
from .history import collect,console_summary
from .render import write_artifacts


def audit(run,kind='best',device='cuda:0',workers=2):
    run=Path(run).resolve();checkpoint,meta=load_selected(run,kind)
    original=checkpoint['config'];verify_inputs(original)
    cfg=dict(original,device=device,workers=workers,feature_workers=workers)
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    model=load_model(cfg,training=False);apply_partial(model,checkpoint['candidate']['state'])
    print('TRAIN_DIAGNOSTIC_SELECTED='+meta['selected']+'; V3.17; full Train, fixed noisy recipes',flush=True)
    with bundles(cfg) as (train,_):
        result=evaluate_panel(model,train['rows'],cfg,run,meta['selected'],meta['checkpoint_sha256'])
    verify_inputs(original)
    if digest(run/'last.pt')!=meta['checkpoint_sha256']:raise ValueError('Checkpoint changed during diagnostic inference')
    report=collect(run);path=write_artifacts(report,run/'diagnostics')
    print(console_summary(report,all_epochs=True),flush=True);print('CURVES='+str(path),flush=True)
    return result


def main():
    from w2v_aasist.full_workflow import ensure_idle
    from w2v_aasist.launch import run_lock
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run');parser.add_argument('--checkpoint',choices=('best','last'),default='best')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--upload-temp',action='store_true');parser.add_argument('--download-dir',default='/home/ubuntu/LXT/temp')
    args=parser.parse_args()
    if args.workers<0:parser.error('workers must be nonnegative')
    run=args.run or (ROOT/'exp'/'.latest_v317_run').read_text(encoding='utf-8').strip()
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle();audit(run,args.checkpoint,args.device,args.workers)
        from .archive import export_report
        export_report(run,args.download_dir,args.upload_temp)


if __name__=='__main__':main()
