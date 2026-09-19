"""Inference only. Preserve official IDs and write a ZIP containing only scores.txt."""
import argparse
from pathlib import Path
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .model import Detector, forward_chunks
from .data import FeatureCollator, worker_init
from .core import load_checkpoint, sha256, atomic_json


class EvaluationDataset(Dataset):
    def __init__(self, ids, directory):
        from utils.data_utils import SpoofAudioDataset
        self.inner = SpoofAudioDataset(ids, directory)  # first-64600 crop/repeat, no augmentation

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        wave, name = self.inner[i]
        return wave, 0, name  # dummy label is never used by inference


def package_scores(ids, scores, directory):
    if not ids or len(ids) != len(set(ids)) or len(ids) != len(scores):
        raise ValueError('Missing, duplicate, or mismatched evaluation IDs')
    if any(not 0 <= v <= 1 for v in scores):
        raise ValueError('Scores must be finite probabilities')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'scores.txt'
    tmp = path.with_suffix('.txt.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for name, value in zip(ids, scores):
            if len(name.split()) != 1:
                raise ValueError('Whitespace in utterance ID')
            f.write(f'{name} {value:.10f}\n')
    tmp.replace(path)
    archive = directory/'submission.zip'
    with zipfile.ZipFile(str(archive)+'.tmp', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname='scores.txt')
    Path(str(archive)+'.tmp').replace(archive)
    with zipfile.ZipFile(archive) as z:
        if z.namelist() != ['scores.txt'] or z.testzip() is not None:
            raise ValueError('Invalid submission ZIP')
    return path, archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ssl_path', required=True, help='Original directory for OFFICIAL feature extractor')
    p.add_argument('--eval_data_path', required=True)
    p.add_argument('--protocol_path', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    args = p.parse_args()
    from utils.data_utils import read_protocol
    ids, _ = read_protocol(args.protocol_path, require_label=False)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Expected unique unlabeled protocol IDs')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.microbatch < 1 or args.num_workers < 0:
        raise ValueError('Invalid batch/worker count')
    ckpt = load_checkpoint(args.checkpoint)
    preprocessor_key = str((Path(args.ssl_path)/'preprocessor_config.json').resolve())
    hashes = [v for k, v in ckpt['data_fingerprints'].items() if Path(k).name == 'preprocessor_config.json']
    if len(hashes) != 1 or sha256(preprocessor_key) != hashes[0]:
        raise ValueError('Use the SAME official feature extractor as training')
    model = Detector.load(args.ssl_path, ckpt['model_config'], checkpointing=False).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    kwargs = {'batch_size': 16, 'shuffle': False, 'num_workers': args.num_workers,
              'collate_fn': FeatureCollator(args.ssl_path)}
    if args.num_workers:
        kwargs.update(multiprocessing_context='spawn', worker_init_fn=worker_init)
    loader = DataLoader(EvaluationDataset(ids, args.eval_data_path), **kwargs)
    scores, written = [], []
    with torch.inference_mode():
        for b in tqdm(loader, desc='Inference only'):
            z, _ = forward_chunks(model, b['features'].to(device), b['mask'].to(device), args.microbatch)
            prob = z.float().softmax(1)[:, 0]
            if not torch.isfinite(prob).all():
                raise FloatingPointError('Non-finite scores')
            scores.extend(prob.cpu().tolist())
            written.extend(b['ids'])
    if written != ids:
        raise ValueError('Output ordering differs from protocol')
    score_file, archive = package_scores(ids, scores, args.out)
    atomic_json({'checkpoint_sha256': sha256(args.checkpoint), 'protocol_sha256': sha256(args.protocol_path),
                 'count': len(ids), 'zip_sha256': sha256(archive), 'fake_column': 0}, Path(args.out)/'submission_meta.json')
    print('Scores:', score_file, '\nUpload:', archive)


if __name__ == '__main__':
    main()
