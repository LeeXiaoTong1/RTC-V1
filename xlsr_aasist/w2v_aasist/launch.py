"""Discover the existing server recipe and launch one independent AASIST full audio run."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import importlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import zipfile
from .runtime import atomic_json, sha256

ROOT = Path(__file__).resolve().parent.parent
BASELINE_SHA256 = 'db3f8167742bf2fe41cfad028dec962d56f6c61295442870620421d7f3a9bbee'
BASELINE = '/home/ubuntu/LXT/RTC/xlsr_aasist/exp/w2v_rebuild_20260920_093548/stage3/best_model.pt'
RAW = dict(nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000, minCoeff=10,
           maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20,
           N_f=5, P=10, g_sd=2, SNRmin=10, SNRmax=40)


@contextmanager
def run_lock(path):
    """OS lock releases after a crash; never interpret a stale PID as ownership."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        if os.name == 'nt':
            import msvcrt
            f.seek(0)
            f.write(b'0')
            f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def check_environment(gpu=True):
    import torch
    import transformers
    for name in ('numpy', 'soundfile', 'scipy', 'tqdm', 'librosa', 'safetensors'):
        importlib.import_module(name)
    if transformers.__version__ != '4.38.2':
        raise RuntimeError('Run setup_w2v_aasist.sh inside sdd: transformers==4.38.2 is required')
    # Existing V2 cache readers are intentionally reused, not changed or regenerated.
    from rtc_noisy_v2.cache import check_metadata, check_suite
    from utils.data_utils import process_rawboost_feature
    if gpu and not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU visible in this Python environment')
    print('ENVIRONMENT_OK=True torch=' + str(torch.__version__) + ' transformers=' + transformers.__version__, flush=True)
    if torch.cuda.is_available():
        print('GPU=' + torch.cuda.get_device_name(0), flush=True)


def find_source(exp, explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_dir():
            path = path / 'stage3' / 'config.json'
        candidates = [path]
    else:
        candidates = sorted(Path(exp).glob('w2v_*/stage3/config.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding='utf-8'))
        required = ('train_protocol', 'dev_protocol', 'train_data_path', 'dev_data_path',
                    'ssl_path', 'train_noisy_cache', 'dev_noisy_cache', 'dev_heldout_cache')
        if all(value.get(k) for k in required):
            return path, value
    raise FileNotFoundError('No existing Stage3 config found. Pass --source-config /path/to/stage3/config.json')


def configuration(source, args):
    cfg = {k: str(Path(source[k]).expanduser().resolve()) for k in (
        'train_protocol', 'dev_protocol', 'train_data_path', 'dev_data_path', 'ssl_path',
        'dev_noisy_cache', 'dev_heldout_cache')}
    full_cache = getattr(args, 'full_noisy_cache', None)
    caches = [full_cache or source['train_noisy_cache']]
    if full_cache and args.include_extra_cache:
        raise ValueError('Full two-view mode replaces legacy Train caches; do not include extra cache')
    if args.include_extra_cache:
        extra = source.get('extra_train_noisy_cache') or []
        caches += [extra] if isinstance(extra, str) else extra
    cfg['train_caches'] = [str(Path(x).expanduser().resolve()) for x in dict.fromkeys(caches)]
    digest = sha256(args.baseline)
    if digest != BASELINE_SHA256:
        raise ValueError('Training must start from the original 91.68 AASIST best; MultiConv/other checkpoints are rejected')
    cfg.update(baseline=str(Path(args.baseline).expanduser().resolve()), baseline_sha256=digest,
               seed=args.seed, device=args.device, amp=args.amp, eval_amp='none', epochs=args.epochs,
               trainable_layers=args.trainable_layers, head_lr=args.head_lr, encoder_lr=args.encoder_lr,
               weight_decay=1e-4, noisy_weight=.3, grad_clip=1., ordinary_batch=args.ordinary_batch,
               noisy_batch=args.noisy_batch, eval_batch=16, workers=args.workers, max_seconds=0.,
               microbatch=args.microbatch, frame_budget=args.frame_budget, checkpointing=True,
               patience=2, lr_warmup_steps=100, rawboost=5,
               raw_config={key: source.get(key, value) for key, value in RAW.items()},
               score_column='P(fake)', decision_threshold=.5, input_policy='full utterance',
               composition_recipe='same-source: single 50%, temporal switch 25%, noisy prefix plus original tail 25%',
               algorithm='original w2v-BERT + AASIST weights; last-layer features; one-pass CE fine-tuning')
    if full_cache:
        cfg.update(full_noisy=True, composition_recipe='two full-duration processed views per Offline Train source; one view per noisy slot')
    return cfg


def upload_archive(archive):
    result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--max-time', '180',
                             '-F', 'file=@' + str(archive), 'https://temp.sh/upload'],
                            check=True, capture_output=True, text=True)
    import re
    urls = re.findall(r'https://temp\.sh/[^\s<>"\x27]+', result.stdout)
    if not urls:
        raise RuntimeError('Upload returned no temp.sh download link')
    print('TEMP_DOWNLOAD_URL=' + urls[0], flush=True)
    return urls[0]


def export_report(run, destination, upload=False):
    run, destination = Path(run), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / (run.name + '_report.zip')
    files = [p for p in run.iterdir() if p.is_file() and (p.suffix in ('.json', '.jsonl', '.md', '.log'))]
    with zipfile.ZipFile(str(archive) + '.tmp', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=p.name)
    os.replace(str(archive) + '.tmp', archive)
    print('REPORT_ZIP=' + str(archive), flush=True)
    if upload:
        url = upload_archive(archive)
        (run / 'temp_download_url.txt').write_text(url + '\n', encoding='utf-8')
    return archive


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-config')
    p.add_argument('--baseline', default=BASELINE)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', choices=['bf16', 'none'], default='bf16')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--trainable-layers', type=int, default=4)
    p.add_argument('--head-lr', type=float, default=2e-6)
    p.add_argument('--encoder-lr', type=float, default=1e-7)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--frame-budget', type=int, default=1600)
    p.add_argument('--ordinary-batch', type=int, default=24)
    p.add_argument('--noisy-batch', type=int, default=4)
    p.add_argument('--include-extra-cache', action='store_true', help='Opt in to diverse bank; default uses original Train bank')
    p.add_argument('--full-noisy-cache', help='Completed two-view full-duration Train cache; replaces old Train caches')
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true', help='Upload diagnostic ZIP only, no audio/model')
    p.add_argument('--run', action='store_true')
    p.add_argument('--check', action='store_true')
    p.add_argument('--cpu-check', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--resume', help='Existing AASIST full audio run directory, resuming last.pt')
    return p


def main():
    args = parser().parse_args()
    check_environment(gpu=not args.cpu_check)
    if args.check:
        return
    if args.cpu_check and args.device.startswith('cuda'):
        raise ValueError('--cpu-check does not silently change training device')
    if (args.workers < 0 or args.epochs < 1
            or not 0 <= args.trainable_layers <= 24 or args.ordinary_batch < 4
            or args.noisy_batch < 2 or args.noisy_batch % 2 or args.smoke_steps < 0):
        raise ValueError('Invalid training counts')
    if any(not math.isfinite(v) or v <= 0 for v in (args.head_lr, args.encoder_lr)):
        raise ValueError('Learning rates must be finite and positive')
    if args.microbatch < 1 or args.frame_budget < 12:
        raise ValueError('Invalid microbatch or frame budget')
    if args.resume:
        run = Path(args.resume).expanduser().resolve()
        cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        if not (run / 'last.pt').is_file():
            raise FileNotFoundError(run / 'last.pt')
        print('Resuming saved configuration; recipe overrides are not applied.', flush=True)
    else:
        source_path, source = find_source(ROOT / 'exp', args.source_config)
        print('SOURCE_CONFIG=' + str(source_path), flush=True)
        print('Hashing the protected original best.', flush=True)
        cfg = configuration(source, args)
        run = ROOT / 'exp' / ('w2v_aasist_full_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(2))
    print('RUN=' + str(run) + '\nORIGINAL_BEST=' + cfg['baseline'], flush=True)
    print(f'INPUT_POLICY={cfg["input_policy"]}; epochs={cfg["epochs"]}; {cfg["composition_recipe"]}', flush=True)
    if not args.run:
        print('Preview complete. Add --run to start.', flush=True)
        return
    protected = [Path(cfg['baseline']).parent, *map(Path, cfg['train_caches']),
                 Path(cfg['dev_noisy_cache']), Path(cfg['dev_heldout_cache'])]
    if any(run == p.resolve() or p.resolve() in run.parents for p in protected):
        raise ValueError('Output overlaps protected input data/checkpoints')
    with run_lock(ROOT / 'exp' / '.aasist-launch.lock'):
        if not args.resume:
            run.mkdir(parents=True, exist_ok=False)
            atomic_json(run / 'config.json', cfg)
        (ROOT / 'exp' / '.latest_aasist_run').write_text(str(run) + '\n', encoding='utf-8')
        command = [sys.executable, '-u', '-m', 'w2v_aasist.train', '--config', str(run/'config.json'), '--out', str(run)]
        if args.resume:
            command += ['--resume', str(run/'last.pt')]
        if args.smoke_steps:
            command += ['--smoke-steps', str(args.smoke_steps)]
        log = run / ('execution_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log')
        with log.open('x', encoding='utf-8') as output:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', bufsize=1)
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end='', flush=True)
            code = process.wait()
        try:
            export_report(run, args.download_dir, args.upload_temp)
        except Exception as exc:
            print('REPORT_EXPORT_FAILED=' + str(exc) + '; original report remains in ' + str(run), flush=True)
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
