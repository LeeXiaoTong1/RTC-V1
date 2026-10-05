"""Single-model full-wave export; no language ID or external teacher is needed."""
import argparse
from pathlib import Path

import torch

from .common import ROOT, atomic_json, digest, verify_files
from .patch import load_selected, apply_patch


def export(run, protocol, audio_root, out, device='cuda:0', workers=2):
    from w2v_aasist.evaluate import package_scores
    from w2v_aasist.progress import progress
    from w2v_v3.data import read_protocol
    from w2v_v3.step import predict
    from w2v_v36.features import inference_batches
    from w2v_v36.patch import load_base
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    patch, done = load_selected(run)
    cfg = dict(patch['config'], device=device, workers=workers)
    verify_files(cfg['code_fingerprints'])
    protocol_hash, patch_hash = digest(protocol), digest(Path(run) / 'best_patch.pt')
    model = apply_patch(load_base(cfg, device), patch)
    rows = read_protocol(protocol, audio_root, labeled=False)
    ids, scores = [], []
    with torch.inference_mode():
        for examples in progress(inference_batches(rows, cfg),
                total=(len(rows)+cfg['eval_batch']-1)//cfg['eval_batch'],
                label='V3.8 submission inference', every=200):
            logits = predict(model, examples, torch.device(device), 'none', cfg['microbatch'], cfg['frame_budget'])
            ids.extend(row['id'] for row in examples)
            scores.extend(logits.float().softmax(-1)[:, 0].cpu().tolist())
    if ids != [r['id'] for r in rows]:
        raise ValueError('Submission protocol order changed')
    verify_files({str(protocol): protocol_hash, str(Path(run) / 'best_patch.pt'): patch_hash,
                  cfg['base_checkpoint']: cfg['base_checkpoint_sha256']})
    verify_files(cfg['code_fingerprints'])
    archive = package_scores(ids, scores, out)
    atomic_json(Path(out) / 'submission_meta.json', dict(version='3.8',
        patch=str((Path(run) / 'best_patch.pt').resolve()), patch_sha256=patch_hash,
        base_checkpoint=cfg['base_checkpoint'], base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
        checkpoint_tag=done['selected'], baseline_fallback=done['baseline_fallback'],
        language_debias_applied=patch['language_debias_applied'], external_teacher_at_inference=False,
        protocol_sha256=protocol_hash, zip_sha256=digest(archive), count=len(ids),
        score='P(fake)', threshold=.5, input_policy='full utterance', eval_amp='none',
        metric_definition=cfg['metric_definition']))
    print('SUBMISSION_ZIP=' + str(archive), flush=True)
    print('SUBMISSION_SELECTED=' + done['selected'], flush=True)
    print('SUBMISSION_BASELINE_FALLBACK=' + str(done['baseline_fallback']), flush=True)
    return archive


def main():
    from w2v_aasist.full_workflow import ensure_idle
    from w2v_aasist.launch import run_lock, upload_archive
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'protocol', 'audio-root', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--upload-temp', action='store_true')
    args = p.parse_args()
    if args.workers < 0:
        p.error('workers must be nonnegative')
    ensure_idle()
    with run_lock(ROOT / 'exp' / '.aasist-launch.lock'):
        ensure_idle()
        archive = export(args.run, args.protocol, args.audio_root, args.out, args.device, args.workers)
        if args.upload_temp:
            try:
                url = upload_archive(archive)
                Path(args.out, 'temp_download_url.txt').write_text(url + '\n', encoding='utf-8')
            except Exception as exc:
                print('UPLOAD_FAILED=' + str(exc) + '; local ZIP retained: ' + str(archive), flush=True)


if __name__ == '__main__':
    main()
