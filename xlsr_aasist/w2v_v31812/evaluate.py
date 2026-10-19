"""Export one selected residual detector with the unchanged fixed .5 convention."""
import argparse
from pathlib import Path
import numpy as np
import torch
from w2v_aasist.data import read_protocol
from w2v_aasist.evaluate import package_scores
from w2v_v31810.common import ROOT, read_json, digest, fingerprint, atomic_json, verify_files
from w2v_v31810.data import Waves, loader, microbatches, close
from . import SCHEMA
from .workflow import load_parent, code_identity
from .model import Residual, evidence


def export(run,out,protocol,audio_root,device='cuda:0'):
    run,out=Path(run).resolve(),Path(out).resolve()
    cfg=read_json(run/'config.json');done=read_json(run/'completed.json');summary=read_json(run/'summary.json')
    if cfg['schema']!=SCHEMA or cfg['code']!=code_identity() or done['identity']!=fingerprint(cfg):
        raise ValueError('Experiment code/config changed')
    if digest(run/'summary.json')!=done['summary_sha256'] or not summary['joint_eligible']:
        raise ValueError('This experiment has not passed the predeclared external checks; do not export a failed residual')
    if digest(run/'residual.pt')!=summary['residual_sha256']:raise ValueError('Selected residual changed')
    if out.exists() and any(out.iterdir()):raise ValueError('Export into a new empty directory')
    if out==run or run in out.parents:raise ValueError('Keep submission outside the experiment')
    payload=torch.load(run/'residual.pt',map_location='cpu',weights_only=True)
    if payload['schema']!=SCHEMA or payload['identity']!=fingerprint(cfg) or payload['parent_sha256']!=cfg['artifacts']['parent.pt']:
        raise ValueError('Selected residual/parent identity mismatch')
    parent,live=load_parent(run,dict(cfg,device=device))
    residual=Residual().to(device);residual.load_state_dict(payload['residual']);residual.eval()
    rows=read_protocol(protocol,audio_root,labeled=False);protocol_hash=digest(protocol)
    dataset=Waves(rows,live,'train');batches=loader(dataset,live,batch_size=live['eval_batch'])
    scores=[]
    try:
        with torch.no_grad():
            for units in batches:
                ordered=[r for u in units for r in u];positions={id(r):i for i,r in enumerate(ordered)}
                result=np.empty((len(ordered),2),np.float32)
                for micro in microbatches(units,live['microbatch'],live['frame_budget']):
                    examples=[r for u in micro for r in u]
                    z=evidence(parent,[r['wave'] for r in examples],[r['original_samples'] for r in examples],residual)[2]
                    result[[positions[id(r)] for r in examples]]=z.cpu().numpy()
                scores.extend(torch.from_numpy(result).softmax(-1)[:,0].tolist())
                print(f'Submission {len(scores)}/{len(rows)}',flush=True)
    finally:close(batches)
    if len(scores)!=len(rows):raise ValueError('Submission row coverage changed')
    verify_files({str(protocol):protocol_hash})
    out.mkdir(parents=True,exist_ok=True)
    result=package_scores([r['id'] for r in rows],scores,out)
    atomic_json(out/'submission_meta.json',dict(version='3.18.12',identity=payload['identity'],
        parent_sha256=payload['parent_sha256'],residual_sha256=summary['residual_sha256'],
        score='P(fake=0)',threshold=.5,calibration=False,ensemble=False,encoder_passes=1,
        final_classifiers=1,protocol_sha256=protocol_hash,count=len(rows),zip_sha256=digest(result)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run');p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0')
    root='/home/ubuntu/LXT/RTC/xlsr_aasist/dataset'
    p.add_argument('--protocol',default=root+'/progress.txt');p.add_argument('--audio-root',default=root+'/wav/progress')
    args=p.parse_args();run=args.run or (ROOT/'exp'/'.latest_v31812_run').read_text().strip()
    from w2v_v31810.processes import ensure_idle
    from w2v_aasist.launch import run_lock
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle();print('SUBMISSION_ZIP='+str(export(run,args.out,args.protocol,args.audio_root,args.device)),flush=True)


if __name__=='__main__':main()
