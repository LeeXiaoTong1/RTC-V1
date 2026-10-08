"""Export one explicitly selected detector; no ensemble or language metadata."""
import argparse
from pathlib import Path

import torch

from w2v_aasist.evaluate import package_scores
from w2v_v3.data import read_protocol
from w2v_v39.common import ROOT, atomic_json, digest, verify_files
from .model import load_model
from .config import verify_parent
from .state import KINDS, load_selected, apply_candidate
from .train import infer


def export(run, protocol, audio_root, out, device='cuda:0', workers=2, checkpoint_kind='best_guarded'):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run, out = Path(run).expanduser().resolve(), Path(out).expanduser().resolve()
    if out == run or run in out.parents:
        raise ValueError('Submission output must be outside the training run')
    checkpoint, done = load_selected(run, checkpoint_kind)
    cfg = dict(checkpoint['config'], device=device, workers=workers, feature_workers=workers)
    verify_files(cfg['code_fingerprints'])
    weight_path = Path(done['checkpoint_path'])
    protocol_hash, weight_hash = digest(protocol), digest(weight_path)
    print('SUBMISSION_SELECTED='+done['selected']+'; kind='+checkpoint_kind, flush=True)
    print('SUBMISSION_STARTING_CHECKPOINT='+cfg['starting_checkpoint'], flush=True)
    print('SUBMISSION_PARENT='+cfg['parent_run']+'; selected='+cfg['parent_selected_tag']+
          '; selector='+cfg['parent_selector'],flush=True)
    model = load_model(cfg, device, training=False)
    apply_candidate(model, checkpoint['candidate'])
    model.requires_grad_(False).eval()
    rows = read_protocol(protocol, audio_root, labeled=False)
    logits, _ = infer(model, rows, cfg, 'V3.15.1 submission inference')
    scores = torch.from_numpy(logits).softmax(-1)[:, 0].tolist()
    verify_files({str(protocol):protocol_hash, str(weight_path):weight_hash,
                  cfg['base_checkpoint']:cfg['base_checkpoint_sha256'],
                  cfg['starting_checkpoint']:cfg['starting_checkpoint_sha256'],
                  cfg['parent_checkpoint']:cfg['parent_checkpoint_sha256']})
    verify_parent(cfg)
    verify_files(cfg['code_fingerprints'])
    archive = package_scores([r['id'] for r in rows], scores, out)
    atomic_json(out/'submission_meta.json', dict(version='3.15.1', selected=done['selected'],
        checkpoint_kind=checkpoint_kind, checkpoint_path=str(weight_path), checkpoint_sha256=weight_hash,
        starting_checkpoint_sha256=cfg['starting_checkpoint_sha256'], starting_tag=cfg['starting_tag'],
        parent_run=cfg['parent_run'],parent_selector=cfg['parent_selector'],
        parent_selected_tag=cfg['parent_selected_tag'],parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],
        base_checkpoint_sha256=cfg['base_checkpoint_sha256'], baseline_fallback=done['baseline_fallback'],
        protocol_sha256=protocol_hash, zip_sha256=digest(archive), count=len(rows), score='P(fake)',
        threshold=.5, input_policy='full utterance', eval_amp='none', external_teacher_at_inference=False))
    print('SUBMISSION_ZIP='+str(archive), flush=True)
    return archive


def main():
    from w2v_aasist.full_workflow import ensure_idle
    from w2v_aasist.launch import run_lock, upload_archive
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'protocol', 'audio-root', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--checkpoint', choices=KINDS, default='best_guarded')
    p.add_argument('--upload-temp', action='store_true')
    args = p.parse_args()
    if args.workers < 0:
        p.error('workers must be nonnegative')
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        output = Path(args.out).with_name(Path(args.out).name+'_'+args.checkpoint)
        archive = export(args.run, args.protocol, args.audio_root, output, args.device, args.workers, args.checkpoint)
        if args.upload_temp:
            try:
                url = upload_archive(archive)
                (output/'temp_download_url.txt').write_text(url+'\n', encoding='utf-8')
            except Exception as exc:
                print('UPLOAD_FAILED='+str(exc)+'; local ZIP retained: '+str(archive), flush=True)


if __name__ == '__main__':
    main()
