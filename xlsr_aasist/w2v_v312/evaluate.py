"""One full-wave detector; no language label, adversary or teacher at inference."""
import argparse
from pathlib import Path

import torch

from w2v_aasist.evaluate import package_scores
from w2v_v3.data import read_protocol
from w2v_v39.common import ROOT, atomic_json, digest, verify_files
from .model import load_model
from .state import load_selected, apply_partial
from .train import infer


def export(run, protocol, audio_root, out, device='cuda:0', workers=2):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint, done = load_selected(run)
    cfg = dict(checkpoint['config'], device=device, workers=workers, feature_workers=workers)
    verify_files(cfg['code_fingerprints'])
    protocol_hash, weight_hash = digest(protocol), digest(Path(run)/'best.pt')
    model = load_model(cfg,device,training=False)
    if checkpoint['model'] is not None:
        apply_partial(model,checkpoint['model'])
    model.requires_grad_(False).eval()
    rows = read_protocol(protocol,audio_root,labeled=False)
    logits, _ = infer(model,rows,cfg,'V3.12 submission inference')
    ids = [row['id'] for row in rows]
    scores = torch.from_numpy(logits).softmax(-1)[:,0].tolist()
    verify_files({str(protocol):protocol_hash,str(Path(run)/'best.pt'):weight_hash,
                  cfg['base_checkpoint']:cfg['base_checkpoint_sha256']})
    verify_files(cfg['code_fingerprints'])
    archive = package_scores(ids,scores,out)
    atomic_json(Path(out)/'submission_meta.json',dict(version='3.12',selected=done['selected'],
        checkpoint_sha256=weight_hash,base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
        baseline_fallback=done['baseline_fallback'],language_probe_evidence=done['language_probe_evidence'],
        protocol_sha256=protocol_hash,zip_sha256=digest(archive),count=len(ids),score='P(fake)',threshold=.5,
        input_policy='full utterance',eval_amp='none',external_teacher_at_inference=False))
    print('SUBMISSION_ZIP='+str(archive),flush=True)
    print('SUBMISSION_SELECTED='+done['selected'],flush=True)
    return archive


def main():
    from w2v_aasist.full_workflow import ensure_idle
    from w2v_aasist.launch import run_lock, upload_archive
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run','protocol','audio-root','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--upload-temp',action='store_true')
    args = p.parse_args()
    if args.workers < 0:
        p.error('workers must be nonnegative')
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        archive = export(args.run,args.protocol,args.audio_root,args.out,args.device,args.workers)
        if args.upload_temp:
            try:
                url = upload_archive(archive)
                Path(args.out,'temp_download_url.txt').write_text(url+'\n',encoding='utf-8')
            except Exception as exc:
                print('UPLOAD_FAILED='+str(exc)+'; local ZIP retained: '+str(archive),flush=True)


if __name__ == '__main__':
    main()
