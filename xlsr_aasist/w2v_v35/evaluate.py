"""Export a single full-utterance V3.5 model in official protocol order."""
import argparse
from pathlib import Path
import torch
from w2v_aasist.evaluate import package_scores
from w2v_aasist.launch import upload_archive
from w2v_aasist.runtime import atomic_json, sha256
from w2v_aasist.progress import progress
from w2v_v3.data import read_protocol, loader
from w2v_v3.model import Detector
from w2v_v3.step import predict
from . import SCHEMA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'protocol', 'audio-root', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--ssl-path')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--upload-temp', action='store_true')
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.workers < 0:
        raise ValueError('workers must be nonnegative')
    digest = sha256(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if state.get('schema') != SCHEMA or state.get('kind') != 'weights':
        raise ValueError('Submission requires a V3.5 weights checkpoint, not optimizer state')
    cfg = dict(state['config'], workers=args.workers, device=args.device)
    if args.ssl_path:
        cfg['ssl_path'] = str(Path(args.ssl_path).expanduser().resolve())
    hashes = {h for name,h in state['data_fingerprints'].items() if Path(name).name == 'preprocessor_config.json'}
    if len(hashes) != 1 or sha256(Path(cfg['ssl_path'])/'preprocessor_config.json') not in hashes:
        raise ValueError('Feature extractor differs from the saved model')
    rows = read_protocol(args.protocol, args.audio_root, labeled=False)
    model = Detector.from_checkpoint(state, checkpointing=False).to(device).eval()
    ids, scores = [], []
    with torch.inference_mode():
        batches = loader(rows, cfg)
        for examples in progress(batches, total=len(batches), label='V3.5 submission inference', every=200):
            logits = predict(model, examples, device, 'none', cfg['microbatch'], cfg['frame_budget'])
            for i,row in enumerate(examples):
                ids.append(row['id']); scores.append(float(logits[i].softmax(0)[0]))
    if ids != [r['id'] for r in rows] or sha256(args.checkpoint) != digest:
        raise RuntimeError('Protocol order or checkpoint changed during inference')
    archive = package_scores(ids, scores, args.out)
    fallback = state['tag'] == 'reference'
    atomic_json(Path(args.out)/'submission_meta.json', {
        'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': digest,
        'checkpoint_tag': state['tag'], 'reference_fallback': fallback,
        'reference_checkpoint_sha256': cfg['reference_checkpoint_sha256'],
        'pretrained_origin': cfg.get('pretrained_origin', {}),
        'protocol_sha256': sha256(args.protocol), 'zip_sha256': sha256(archive),
        'count': len(ids), 'score': 'P(fake)', 'threshold': .5,
        'input_policy': 'full utterance', 'eval_amp': 'none',
        'selection_metric': 'fixed full Dev Online Clean/Noisy weighted 0.3/0.7; not a platform score'})
    print('SUBMISSION_ZIP='+str(archive), flush=True)
    print('SUBMISSION_CHECKPOINT_TAG='+state['tag'], flush=True)
    print('SUBMISSION_REFERENCE_FALLBACK='+str(fallback), flush=True)
    if args.upload_temp:
        try:
            url = upload_archive(archive)
            (Path(args.out)/'temp_download_url.txt').write_text(url+'\n', encoding='utf-8')
        except Exception as exc:
            print('UPLOAD_FAILED='+str(exc)+'; local ZIP: '+str(archive), flush=True)


if __name__ == '__main__':
    main()
