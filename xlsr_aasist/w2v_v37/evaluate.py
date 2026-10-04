"""Export the selected V3.7 base+patch as a normal single-model submission."""
import argparse
from pathlib import Path
import torch

from w2v_aasist.evaluate import package_scores
from w2v_aasist.launch import ROOT, run_lock, upload_archive
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.runtime import atomic_json, sha256
from w2v_aasist.progress import progress
from w2v_v3.data import read_protocol
from w2v_v3.step import predict
from .patch import load_selected, load_base, apply_patch
from .config import verify_files


def export(run, protocol, audio_root, out, device='cuda:0', workers=2):
    from w2v_v36.features import inference_batches
    patch, done = load_selected(run)
    cfg = dict(patch['config'], device=device, workers=workers)
    verify_files(cfg['code_fingerprints'])
    protocol_digest = sha256(protocol)
    patch_digest = sha256(Path(run)/'best_patch.pt')
    model = apply_patch(load_base(cfg, device), patch)
    rows = read_protocol(protocol, audio_root, labeled=False)
    ids, scores = [], []
    with torch.inference_mode():
        for examples in progress(inference_batches(rows, cfg),
                total=(len(rows)+cfg['eval_batch']-1)//cfg['eval_batch'],
                label='V3.7 submission inference', every=200):
            logits = predict(model, examples, torch.device(device), 'none', cfg['microbatch'], cfg['frame_budget'])
            probabilities = logits.float().softmax(-1)[:, 0].cpu().tolist()
            ids.extend(row['id'] for row in examples)
            scores.extend(probabilities)
    if (ids != [r['id'] for r in rows] or sha256(protocol) != protocol_digest
            or sha256(cfg['base_checkpoint']) != cfg['base_checkpoint_sha256']
            or sha256(Path(run)/'best_patch.pt') != patch_digest):
        raise RuntimeError('Protocol order or immutable weights changed during inference')
    verify_files(cfg['code_fingerprints'])
    archive = package_scores(ids, scores, out)
    atomic_json(Path(out)/'submission_meta.json', dict(version='3.7',
        patch=str((Path(run)/'best_patch.pt').resolve()), patch_sha256=patch_digest,
        base_checkpoint=cfg['base_checkpoint'], base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
        checkpoint_tag=done['selected'], baseline_fallback=done['baseline_fallback'],
        language_debias_applied=patch['language_debias_applied'], external_teacher_at_inference=False,
        protocol_sha256=protocol_digest, zip_sha256=sha256(archive), count=len(ids),
        score='P(fake)', threshold=.5, input_policy='full utterance', eval_amp='none',
        metric_definition=cfg['metric_definition']))
    print('SUBMISSION_ZIP='+str(archive), flush=True)
    print('SUBMISSION_BASELINE_FALLBACK='+str(done['baseline_fallback']), flush=True)
    print('SUBMISSION_LANGUAGE_DEBIAS_APPLIED='+str(patch['language_debias_applied']),flush=True)
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'protocol', 'audio-root', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--upload-temp', action='store_true')
    args = p.parse_args()
    if args.workers < 0:
        p.error('workers must be nonnegative')
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        archive = export(args.run, args.protocol, args.audio_root, args.out, args.device, args.workers)
        if args.upload_temp:
            try:
                url = upload_archive(archive)
                (Path(args.out)/'temp_download_url.txt').write_text(url+'\n', encoding='utf-8')
            except Exception as exc:
                print('UPLOAD_FAILED='+str(exc)+'; local ZIP retained: '+str(archive), flush=True)


if __name__ == '__main__':
    main()
