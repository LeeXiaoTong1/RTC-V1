"""Inference-only submission export, using the trained input policy and fake probability."""
import argparse
import json
import math
import os
from pathlib import Path
import zipfile
import torch
from tqdm import tqdm
from .data import loader, read_protocol
from .model import Detector
from .runtime import amp_context, atomic_json, load_checkpoint, sha256


def package_scores(ids, scores, destination):
    if not ids or len(ids) != len(set(ids)) or len(ids) != len(scores):
        raise ValueError('Missing/duplicate/mismatched IDs')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
        raise ValueError('Scores must be finite P(fake) probabilities')
    if any(len(name.split()) != 1 for name in ids):
        raise ValueError('Whitespace in protocol IDs')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    score_file = destination / 'scores.txt'
    tmp = destination / 'scores.txt.tmp'
    with tmp.open('w', encoding='utf-8', newline='\n') as output:
        for name, score in zip(ids, scores):
            output.write(f'{name} {score:.10f}\n')
    os.replace(tmp, score_file)
    archive = destination / 'submission.zip'
    with zipfile.ZipFile(str(archive) + '.tmp', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(score_file, arcname='scores.txt')
    os.replace(str(archive) + '.tmp', archive)
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--protocol', required=True, help='Official unlabeled Eval ID list')
    p.add_argument('--audio-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ssl-path', help='Override location only; extractor hash must match training')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--upload-temp', action='store_true', help='Upload submission.zip only, never checkpoint/audio')
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    state = load_checkpoint(args.checkpoint)
    cfg = dict(state['config'])
    if device.type == 'cuda' and cfg['amp'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This checkpoint recipe uses BF16; select a BF16-capable GPU')
    cfg.update(workers=args.workers, device=args.device)
    if args.ssl_path:
        cfg['ssl_path'] = str(Path(args.ssl_path).resolve())
    expected = [v for k, v in state['data_fingerprints'].items() if Path(k).name == 'preprocessor_config.json']
    if len(expected) != 1 or sha256(Path(cfg['ssl_path']) / 'preprocessor_config.json') != expected[0]:
        raise ValueError('Feature extractor differs from training')
    rows = read_protocol(args.protocol, args.audio_root, labeled=False)
    model = Detector.from_config(state['model_config'], state['head_config'], checkpointing=False)
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    ids, scores = [], []
    with torch.inference_mode():
        for examples in tqdm(loader(rows, cfg), desc='Submission inference'):
            for ex in examples:
                with amp_context(device, cfg['amp']):
                    logits, _ = model(ex['features'].to(device), ex['mask'].to(device))
                ids.append(ex['id'])
                scores.append(float(logits.float().softmax(1)[0, 0]))
    if ids != [r['id'] for r in rows]:
        raise RuntimeError('Output IDs differ from official protocol order')
    archive = package_scores(ids, scores, args.out)
    atomic_json(Path(args.out) / 'submission_meta.json', {
        'checkpoint_sha256': sha256(args.checkpoint), 'protocol_sha256': sha256(args.protocol),
        'zip_sha256': sha256(archive), 'count': len(ids), 'score': 'P(fake)',
        'input_policy': cfg['input_policy'], 'max_seconds': cfg['max_seconds']})
    print('SUBMISSION_ZIP=' + str(archive), flush=True)
    if args.upload_temp:
        from .launch import upload_archive
        try:
            url = upload_archive(archive)
            (Path(args.out) / 'temp_download_url.txt').write_text(url + '\n', encoding='utf-8')
        except Exception as exc:
            print('UPLOAD_FAILED=' + str(exc) + '; submission remains at ' + str(archive), flush=True)


if __name__ == '__main__':
    main()
