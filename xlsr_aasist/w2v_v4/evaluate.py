"""Single detector, no XEUS/phoneme dependencies or fitted score calibration."""
import argparse
from pathlib import Path
import torch
from .common import ROOT,read_json,atomic_json,digest,apply_partial,fingerprint
from .model import load_model
from w2v_v31810.inference import infer
from w2v_aasist.data import read_protocol
from w2v_aasist.evaluate import package_scores
from w2v_aasist.launch import upload_archive,run_lock
from w2v_v31810.processes import ensure_idle
from .config import verify
from . import SCHEMA,VERSION


def selected(run):
    run=Path(run);cfg=read_json(run/'config.json')
    status=run/'completed.json' if (run/'completed.json').exists() else run/'paused.json'
    done=read_json(status)
    if done['version']!=VERSION or done['status'] not in ('complete','paused') or done['checkpoint_sha256']!=digest(run/'last.pt'):
        raise ValueError('Need a completed or safely paused V4 checkpoint')
    state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    if state['schema']!=SCHEMA or state['identity']!=fingerprint(cfg) or state['best_tag']!=done['best_tag']:
        raise ValueError('Checkpoint/config mismatch')
    return cfg,state['best_model'],done


def export(run,protocol,audio_root,out,device='cuda:0',workers=2,upload=False,model_factory=load_model):
    run=Path(run);out=Path(out);cfg,weights,done=selected(run)
    sha=digest(protocol)
    if (out/'submission_meta.json').exists():
        meta=read_json(out/'submission_meta.json');archive=out/'submission.zip'
        if (meta.get('checkpoint_sha256')!=done['checkpoint_sha256'] or meta.get('protocol_sha256')!=sha or
            meta.get('audio_root')!=str(Path(audio_root).resolve()) or meta.get('zip_sha256')!=digest(archive)):
            raise ValueError('Existing export differs from this checkpoint/protocol')
    else:
        verify(run,training=False)
        if out.exists() and any(out.iterdir()):raise ValueError('Export destination is not empty')
        rows=read_protocol(protocol,audio_root,labeled=False)
        cfg=dict(cfg,device=device,workers=workers,eval_workers=workers)
        model=model_factory(cfg);apply_partial(model,weights);model.requires_grad_(False).eval()
        logits,_=infer(model,rows,cfg);scores=torch.from_numpy(logits).softmax(-1)[:,0].tolist()
        out.mkdir(parents=True,exist_ok=True);archive=package_scores([r['id'] for r in rows],scores,out)
        atomic_json(out/'submission_meta.json',dict(version=VERSION,checkpoint_sha256=done['checkpoint_sha256'],selected=done['best_tag'],
            protocol_sha256=sha,audio_root=str(Path(audio_root).resolve()),zip_sha256=digest(archive),count=len(rows),
            score='P(fake=0)',threshold=.5,calibration_fit=False,independent_models=1,teacher_at_inference=False,pcl_at_inference=False))
    if upload:
        link=upload_archive(archive);(out/'temp_download_url.txt').write_text(link+'\n',encoding='utf-8')
        print('TEMP_DOWNLOAD_URL='+link,flush=True)
    print('SUBMISSION_ZIP='+str(archive),flush=True);return archive


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run');p.add_argument('--out')
    p.add_argument('--protocol',required=True);p.add_argument('--audio-root',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=2);p.add_argument('--upload-temp',action='store_true');a=p.parse_args()
    run=Path(a.run or (ROOT/'exp/.latest_v4_run').read_text().strip());out=a.out or str(run.parent/(run.name+'_submission'))
    ensure_idle()
    with run_lock(ROOT/'exp/.aasist-launch.lock'):
        ensure_idle();export(run,a.protocol,a.audio_root,out,a.device,a.workers,a.upload_temp)

if __name__=='__main__':main()
