"""Discover the existing server recipe and launch one independent MultiConv run."""
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
        raise RuntimeError('Run setup_w2v_multiconv.sh inside sdd: transformers==4.38.2 is required')
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
    caches = [source['train_noisy_cache']]
    if args.include_extra_cache:
        extra = source.get('extra_train_noisy_cache') or []
        caches += [extra] if isinstance(extra, str) else extra
    cfg['train_caches'] = [str(Path(x).expanduser().resolve()) for x in dict.fromkeys(caches)]
    cfg.update(baseline=str(Path(args.baseline).expanduser().resolve()),
               baseline_sha256=sha256(args.baseline), seed=args.seed, device=args.device, amp=args.amp,
               warmup_epochs=args.warmup_epochs, joint_epochs=args.joint_epochs,
               trainable_layers=args.trainable_layers, warmup_head_lr=args.warmup_head_lr,
               head_lr=args.head_lr, encoder_lr=args.encoder_lr, weight_decay=1e-4,
               cka_weight=args.cka_weight, noisy_weight=.3, grad_clip=1.,
               ordinary_batch=args.ordinary_batch, noisy_batch=args.noisy_batch,
               eval_batch=8, workers=args.workers, max_seconds=args.max_seconds,
               patience=3, lr_warmup_steps=200, rawboost=5,
               raw_config={key: source.get(key, value) for key, value in RAW.items()},
               score_column='P(fake)', decision_threshold=.5,
               input_policy='full utterance' if args.max_seconds == 0 else 'random Train crop / fixed Dev-Eval prefix',
               algorithm='w2v-BERT MultiConv: shared SwiGLU, four multi-kernel blocks, attentive mean/std, differentiable CKA')
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
    p.add_argument('--warmup-epochs', type=int, default=2)
    p.add_argument('--joint-epochs', type=int, default=4)
    p.add_argument('--trainable-layers', type=int, default=4)
    p.add_argument('--warmup-head-lr', type=float, default=1e-4)
    p.add_argument('--head-lr', type=float, default=1e-5)
    p.add_argument('--encoder-lr', type=float, default=2e-7)
    p.add_argument('--cka-weight', type=float, default=.05)
    p.add_argument('--ordinary-batch', type=int, default=16)
    p.add_argument('--noisy-batch', type=int, default=4)
    p.add_argument('--max-seconds', type=float, default=0., help='0=full utterance; positive explicitly caps Train/Dev/Eval length')
    p.add_argument('--include-extra-cache', action='store_true', help='Opt in to diverse bank; default uses original Train bank')
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true', help='Upload diagnostic ZIP only, no audio/model')
    p.add_argument('--run', action='store_true')
    p.add_argument('--check', action='store_true')
    p.add_argument('--cpu-check', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--resume', help='Existing MultiConv run directory, resuming last.pt')
    return p


def main():
    args = parser().parse_args()
    check_environment(gpu=not args.cpu_check)
    if args.check:
        return
    if args.cpu_check and args.device.startswith('cuda'):
        raise ValueError('--cpu-check does not silently change training device')
    if (args.workers < 0 or min(args.warmup_epochs, args.joint_epochs) < 1
            or not 1 <= args.trainable_layers <= 24 or args.ordinary_batch < 4
            or args.noisy_batch < 2 or args.noisy_batch % 2 or args.smoke_steps < 0):
        raise ValueError('Invalid training counts')
    if any(not math.isfinite(v) or v <= 0 for v in (args.warmup_head_lr, args.head_lr, args.encoder_lr)):
        raise ValueError('Learning rates must be finite and positive')
    if not math.isfinite(args.cka_weight) or args.cka_weight < 0 or not math.isfinite(args.max_seconds) or args.max_seconds < 0:
        raise ValueError('Invalid CKA weight / duration limit')
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
        run = ROOT / 'exp' / ('w2v_multiconv_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(2))
    print('RUN=' + str(run) + '\nORIGINAL_BEST=' + cfg['baseline'], flush=True)
    print(f'INPUT_POLICY={cfg["input_policy"]}; warmup={cfg["warmup_epochs"]}, joint={cfg["joint_epochs"]}', flush=True)
    if not args.run:
        print('Preview complete. Add --run to start.', flush=True)
        return
    protected = [Path(cfg['baseline']).parent, *map(Path, cfg['train_caches']),
                 Path(cfg['dev_noisy_cache']), Path(cfg['dev_heldout_cache'])]
    if any(run == p.resolve() or p.resolve() in run.parents for p in protected):
        raise ValueError('Output overlaps protected input data/checkpoints')
    with run_lock(ROOT / 'exp' / '.multiconv-launch.lock'):
        if not args.resume:
            run.mkdir(parents=True, exist_ok=False)
            atomic_json(run / 'config.json', cfg)
        (ROOT / 'exp' / '.latest_multiconv_run').write_text(str(run) + '\n', encoding='utf-8')
        command = [sys.executable, '-u', '-m', 'w2v_multiconv.train', '--run-dir', str(run)]
        if args.resume:
            command += ['--resume']
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
