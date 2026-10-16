"""CPU-only, read-only V3.18.9 audio comparison export while training continues."""
import argparse
from datetime import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
SCHEMA = 'rtc_v3189_audio_quartets_v1'
GROUPS = (('en', 1), ('en', 0), ('zh', 1), ('zh', 0))
STAGES = (
    ('01_offline.wav', 'Offline source'),
    ('02_official_online.wav', 'Official paired Online'),
    ('03_noisy_before_rtc.wav', 'Same Offline + noise, before RTC'),
    ('04_noisy_online.wav', 'That exact noisy input after simulated RTC'),
    ('05_rtc_activity_off.wav', 'Same noisy input and RTC settings, activity attenuation off'),
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_inputs(value=None):
    if value is None:
        marker = ROOT / 'exp/.latest_v3189_run'
        if not marker.is_file():
            raise FileNotFoundError('No V3.18.9 marker; pass --run /absolute/path/to/run')
        value = marker.read_text(encoding='utf-8').strip()
    if not value:
        raise ValueError('Empty run path')
    run = Path(value).expanduser()
    if not run.is_absolute():
        run = ROOT / run
    run = run.resolve()
    cfg = read_json(run / 'config.json')
    if cfg.get('version') != '3.18.9' or cfg.get('recipe') != 'v3189_activity_conditioned_rtc_v1':
        raise ValueError('This exporter requires the current V3.18.9 run')
    path = run / 'train_rows.json'
    checksum = digest(path)
    if checksum != cfg.get('saved_manifests', {}).get('train_rows.json'):
        raise ValueError('Train manifest does not match its saved fingerprint')
    rows = read_json(path)
    if any(row.get('split') != 'train' for row in rows):
        raise ValueError('Only Train recordings may be selected')
    if digest(path) != checksum:
        raise ValueError('Train manifest changed while reading')
    return run, cfg, rows, dict(config_sha256=digest(run / 'config.json'), train_rows_sha256=checksum)


def select_pairs(rows, per_group, seed, min_seconds, max_seconds):
    import soundfile as sf
    from w2v_v3186.sampling import canonical_pairs
    pairs = canonical_pairs(rows)
    selected = []
    for language, label in GROUPS:
        candidates = [i for i in pairs if (rows[i]['language'], rows[i]['label']) == (language, label)]
        candidates.sort(key=lambda i: (hashlib.sha256((str(seed) + ':' + rows[i]['source_id']).encode()).hexdigest(), rows[i]['id']))
        chosen = []
        for i in candidates:
            off, on = rows[i], rows[pairs[i]]
            sizes = [sf.info(row['audio']) for row in (off, on)]
            if all(min_seconds <= info.duration <= max_seconds for info in sizes):
                chosen.append((off, on))
                if len(chosen) == per_group:
                    break
        if len(chosen) < per_group:
            raise ValueError(f'{language}/{label}: only {len(chosen)} paired full recordings within duration limits; widen --min-seconds/--max-seconds')
        selected.extend(chosen)
    return selected


def generate_stages(source, recipe, bank, engine):
    import numpy as np
    from w2v_v3189.augment import generate
    captures = []
    received, metadata = generate(source, recipe, bank, engine, input_observer=lambda x: captures.append(x.copy()))
    if len(captures) != 1:
        raise RuntimeError('Expected one RTC call for one Noisy view')
    if not metadata.get('pair_eligible', False):
        raise RuntimeError('No valid speech/noise overlap; refusing an unaugmented example')
    for wave in (captures[0], received):
        if wave.shape != source.shape or not np.isfinite(wave).all():
            raise RuntimeError('Simulation changed the complete audio geometry or produced nonfinite samples')
    return captures[0], received, metadata


def wave_info(wave):
    import numpy as np
    return dict(samples=len(wave),seconds=len(wave)/16000.,sample_rate=16000,
        peak=float(np.abs(wave).max()),rms=float(np.sqrt(np.mean(wave.astype(np.float64)**2))))


def make_index(cases):
    text = ['<!doctype html><meta charset="utf-8"><title>V3.18.9 audio examples</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px}section{border-top:1px solid #ccc;padding:20px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}audio{width:100%}small{color:#555}code{word-break:break-all}</style>',
        '<h1>V3.18.9: Offline / Online / noisy input / RTC activity on and off</h1>',
        '<p>Extract the ZIP before opening this page. All recordings are complete 16 kHz mono float WAVs. No per-file loudness normalization is applied. Official Online is a real recording and can have a different duration or delay.</p>',
        '<p>These are new diagnostic draws from the training noise/RTC functions, using the original Offline branch without speed/EQ augmentation. They are not a capture of the live training batch.</p>']
    for case in cases:
        meta = case['augmentation']; rec = meta['recipe']
        snr = meta.get('measured_input_snr_db')
        snr_text = f'{snr:.2f} dB measured input SNR' if snr is not None else 'Echo uses delayed-speech gain, not additive SNR'
        text.append('<section><h2>'+html.escape(case['name'])+'</h2><p>'+html.escape(str(rec.get('category') or rec['noise_type']))+' | '+html.escape(rec['family'])+' | '+html.escape(snr_text)+'</p><div class="grid">')
        for filename, caption in STAGES:
            text.append('<div><p>'+html.escape(caption)+'</p><audio controls preload="none" src="'+html.escape(case['name']+'/'+filename,quote=True)+'"></audio></div>')
        text.append('</div></section>')
    return '\n'.join(text)


def export_examples(args, engine_factory=None):
    import numpy as np
    import soundfile as sf
    from w2v_v3184.data import read_wave
    from w2v_v3189.augment import NoiseBank, Engines, recipe
    run, cfg, rows, inputs = load_inputs(args.run)
    selected = select_pairs(rows, args.per_group, args.seed, args.min_seconds, args.max_seconds)
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root == run or run in output_root.parents:
        raise ValueError('Export outside the live training run')
    # Read only a progress snapshot; never open last.pt or touch the live RNG.
    try:
        epoch = int(read_json(run / 'progress.json').get('epoch', 1))
    except (OSError, ValueError, TypeError):
        epoch = 1
    recipe_epoch = int(cfg.get('source_epoch_offset', 0)) + max(0, epoch-1)
    if engine_factory is None and cfg.get('ffmpeg_sha256') and digest(cfg['ffmpeg']) != cfg['ffmpeg_sha256']:
        raise ValueError('Recorded FFmpeg binary changed')
    bank = NoiseBank(cfg['noise_records']['train'],16*1024**2,cfg.get('scene_noise_records',{}).get('train'))
    engine = (engine_factory or Engines)(cfg['ffmpeg'])
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / ('audio_examples_v3189_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(3))
    out.mkdir(exist_ok=False)
    cases = []; counts = {}
    for off, on in selected:
        language, label = off['language'], off['label']
        group = language.upper()+'_'+('REAL' if label == 1 else 'FAKE')
        counts[group] = counts.get(group, 0)+1
        name = group+f'_{counts[group]:02d}'
        folder = out / name; folder.mkdir()
        before = {row['condition']: digest(row['audio']) for row in (off, on)}
        source, online = read_wave(off), read_wave(on)
        if np.sqrt(np.mean(source.astype(np.float64)**2)) < 1e-7:
            raise ValueError('Selected Offline source is silent: '+off['id'])
        occurrence = f'audio-example:{args.seed}:{off["source_id"]}:{counts[group]}'
        lane = (counts[group]-1) % 2
        rec = recipe(cfg['seed'],occurrence,off['group_id'],lane,recipe_epoch,cfg['warm_epochs'],'train',
            stress_probability=cfg.get('rtc_stress_probability',.35),
            stress_ramp_epochs=cfg.get('rtc_stress_ramp_epochs',2.),
            scene_probability=cfg.get('scene_probability',1.),rtc_policy=cfg.get('rtc_policy','three-profile'),
            scene=None if args.scene == 'auto' else args.scene,
            activity_policy=cfg.get('rtc_activity_policy','mixed'),
            activity_profile=None if args.activity_profile=='auto' else args.activity_profile)
        noisy, received, metadata = generate_stages(source, rec, bank, engine)
        baseline = engine(noisy,rec)
        waves = (source,online,noisy,received,baseline)
        stats = {}
        for (filename,_), wave in zip(STAGES,waves):
            sf.write(folder/filename,wave,16000,format='WAV',subtype='FLOAT')
            decoded,sr = sf.read(folder/filename,dtype='float32')
            if sr != 16000 or not np.array_equal(decoded,wave):
                raise RuntimeError('WAV roundtrip changed the exported stage')
            stats[filename] = wave_info(wave)
        for row in (off,on):
            if digest(row['audio']) != before[row['condition']]:
                raise ValueError('A selected source changed during export')
        cases.append(dict(name=name,language=language,label=label,source_id=off['source_id'],
            offline_id=off['id'],online_id=on['id'],source_file_sha256=before,
            base_augmentation='original',occurrence=occurrence,noise_recipe_lane=lane,
            augmentation=metadata,audio=stats))
        print(f'EXPORTED {name} scene={rec.get("category") or rec["noise_type"]} rtc={rec["family"]}',flush=True)
    (out/'index.html').write_text(make_index(cases),encoding='utf-8')
    (out/'README.txt').write_text(
        'Extract this ZIP and open index.html, or listen to each WAV.\n'
        '01: original Offline decoded exactly as in training (16 kHz mono).\n'
        '02: official Online with the same canonical source_id and label.\n'
        '03: noisy Offline including the training common gain, BEFORE RTC.\n'
        '04: the exact waveform in 03 after the training RTC engine.\n'
        '05: the same input and RTC settings, but without V3.18.9 activity attenuation.\n'
        'Compare 04 and 05 at the same playback volume; no independent normalization.\n'
        'Full recordings; no cropping, repeated padding or per-file volume normalization.\n'
        'IEEE float32 WAV retains exact arrays; use a compatible player such as VLC.\n'
        'Original base branch only; dynamic speed/EQ/envelope variants are not included.\n'
        'Fresh diagnostic draws, not captured live training batches. No model scores are used to select samples.\n'
        'Noise category, SNR, gain and RTC settings are in manifest.json.\n',encoding='ascii')
    files = []
    for path in sorted(out.rglob('*')):
        if path.is_file():
            files.append(dict(path=path.relative_to(out).as_posix(),bytes=path.stat().st_size,sha256=digest(path)))
    manifest = dict(schema=SCHEMA,run_name=run.name,version=cfg['version'],inputs=inputs,
        groups=len(cases),seed=args.seed,recipe_epoch=recipe_epoch,scene_override=args.scene,
        activity_override=args.activity_profile,
        selection=dict(per_language_class=args.per_group,min_seconds=args.min_seconds,max_seconds=args.max_seconds,
            method='Canonical Train pairs; seeded order; full-length duration filter; no model scores'),
        live_training_modified=False,model_loaded=False,base_branch='original',cases=cases,files=files)
    write_json(out/'manifest.json',manifest)
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for item in files: z.write(out/item['path'],item['path'])
        z.write(out/'manifest.json','manifest.json')
    validate_archive(archive)
    print('AUDIO_DIR='+str(out),flush=True)
    print('AUDIO_ZIP='+str(archive),flush=True)
    return archive


def validate_archive(archive):
    with zipfile.ZipFile(archive) as z:
        if z.getinfo('manifest.json').file_size > 4*1024**2:
            raise ValueError('Oversized export manifest')
        manifest = json.loads(z.read('manifest.json'))
        if manifest.get('schema') != SCHEMA:
            raise ValueError('Upload requires this exporter\'s audio ZIP')
        files = manifest['files']; expected = [r['path'] for r in files]+['manifest.json']
        if len(set(expected)) != len(expected) or sorted(z.namelist()) != sorted(expected):
            raise ValueError('Unexpected files in the audio ZIP')
        for row in files:
            info = z.getinfo(row['path'])
            if info.file_size != row['bytes'] or info.file_size > 64*1024**2:
                raise ValueError('Archive member size changed')
            if hashlib.sha256(z.read(row['path'])).hexdigest() != row['sha256']:
                raise ValueError('Archive member hash changed: '+row['path'])
    return manifest


def upload(archive, runner=subprocess.run):
    archive = Path(archive).expanduser().resolve()
    validate_archive(archive)
    checksum = digest(archive)
    result = runner(['curl','--fail','--silent','--show-error','--max-time','180',
        '-F','file=@'+str(archive),'https://temp.sh/upload'],check=True,capture_output=True,text=True)
    urls = re.findall(r'https://temp\.sh/[A-Za-z0-9][^\s<>"\x27]*',result.stdout)
    urls = [url for url in urls if url.rstrip('/') != 'https://temp.sh/upload']
    if not urls: raise RuntimeError('Upload returned no temp.sh download URL')
    if digest(archive) != checksum: raise ValueError('Archive changed during upload')
    link = archive.with_suffix('.url.txt'); link.write_text(urls[0]+'\n',encoding='ascii')
    print('AUDIO_DOWNLOAD_URL='+urls[0],flush=True)
    print('LINK_FILE='+str(link),flush=True)
    return urls[0]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',help='Defaults to exp/.latest_v3189_run; may be actively training')
    p.add_argument('--output-root',default='/home/ubuntu/LXT/temp')
    p.add_argument('--per-group',type=int,default=1,help='Pairs per EN/ZH Real/Fake group; default 4 groups total')
    p.add_argument('--seed',type=int,default=318901)
    p.add_argument('--activity-profile',choices=('auto','off','mild','strong'),default='auto',
        help='auto follows training mixture; explicit profiles are diagnostic overrides')
    p.add_argument('--min-seconds',type=float,default=3.)
    p.add_argument('--max-seconds',type=float,default=20.)
    p.add_argument('--scene',choices=('auto','office','coffee','rain','footsteps','keyboard','echo'),default='auto')
    p.add_argument('--upload',action='store_true',help='Upload only this audio ZIP to temp.sh')
    p.add_argument('--retry-zip',help='Validate and upload an existing audio ZIP without regenerating it')
    return p


def main():
    args = parser().parse_args()
    if not 1 <= args.per_group <= 10 or not (math.isfinite(args.min_seconds) and math.isfinite(args.max_seconds)
            and 0 < args.min_seconds <= args.max_seconds <= 60):
        raise ValueError('Use 1..10 pairs/group and 0 < min-seconds <= max-seconds <= 60')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    if hasattr(os,'nice'): os.nice(10)
    if args.retry_zip:
        upload(args.retry_zip); return
    archive = export_examples(args)
    if args.upload:
        try: upload(archive)
        except Exception:
            print('UPLOAD_RETRY: bash export_audio_examples_v3189.sh --retry-zip '+json.dumps(str(archive)),file=sys.stderr,flush=True)
            raise


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print('AUDIO_EXPORT_FAILED: '+str(error),file=sys.stderr,flush=True)
        sys.exit(1)
