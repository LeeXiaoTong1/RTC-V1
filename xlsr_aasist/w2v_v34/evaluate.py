"""Export one V3.4 checkpoint with auditable warm-start identity and official ID order."""
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
    checkpoint_hash = sha256(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if state.get('schema') != SCHEMA or state.get('kind') != 'weights':
        raise ValueError('V3.4 submission requires a V3.4 weights checkpoint, not last.pt')
    cfg = dict(state['config'], workers=args.workers, device=args.device)
    if args.ssl_path:
        cfg['ssl_path'] = str(Path(args.ssl_path).expanduser().resolve())
    hashes = [h for path,h in state['data_fingerprints'].items() if Path(path).name == 'preprocessor_config.json']
    if len(hashes) != 1 or sha256(Path(cfg['ssl_path'])/'preprocessor_config.json') != hashes[0]:
        raise ValueError('Feature extractor differs from training')
    rows = read_protocol(args.protocol, args.audio_root, labeled=False)
    model = Detector.from_checkpoint(state, checkpointing=False).to(device).eval()
    ids, scores = [], []
    with torch.inference_mode():
        batches = loader(rows, cfg)
        for examples in progress(batches, total=len(batches), label='V3.4 submission inference', every=200):
            z = predict(model, examples, device, 'none', cfg['microbatch'], cfg['frame_budget'])
            for i,row in enumerate(examples):
                ids.append(row['id']); scores.append(float(z[i].softmax(0)[0]))
    if ids != [r['id'] for r in rows]:
        raise RuntimeError('Submission order differs from official protocol')
    if sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during export')
    archive = package_scores(ids, scores, args.out)
    atomic_json(Path(args.out)/'submission_meta.json', {
        'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_tag': state['tag'], 'warm_checkpoint_sha256': cfg['warm_checkpoint_sha256'],
        'warm_checkpoint_tag': cfg['expected_warm_tag'], 'baseline_fallback': state['tag']=='baseline',
        'source_provenance': cfg.get('source_provenance', {}),
        'protocol_sha256': sha256(args.protocol), 'zip_sha256': sha256(archive),
        'count': len(ids), 'score': 'P(fake)', 'threshold': .5,
        'input_policy': 'full utterance', 'eval_amp': 'none'})
    print('SUBMISSION_ZIP='+str(archive), flush=True)
    print('SUBMISSION_CHECKPOINT_TAG='+state['tag'], flush=True)
    print('SUBMISSION_BASELINE_FALLBACK='+str(state['tag']=='baseline'), flush=True)
    if args.upload_temp:
        try:
            url = upload_archive(archive)
            (Path(args.out)/'temp_download_url.txt').write_text(url+'\n', encoding='utf-8')
        except Exception as exc:
            print('UPLOAD_FAILED='+str(exc)+'; local ZIP: '+str(archive), flush=True)


if __name__ == '__main__':
    main()
