"""Download RNNoise office/coffee and ESC-50 noises; screen once, train from PCM."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import numpy as np
import soundfile as sf
from w2v_v3184.prepare_noise import (atomic_json, preparation_lock, download,
    convert_cached, summary)
from w2v_v3184.noise import SCHEMA, bind_manifest, validate_records
from .download import fetch, sha256
from .screen import (POLICY, SpeechScreen, infer_rate, decode_raw, safe_intervals,
                     assign_splits)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = HERE / 'sources.json'
EXCLUSIONS = HERE / 'exclusions.json'


def index_archive(archive, out, archive_hash, exclusions):
    """Extract only bounded regular office/coffee members. Never use extractall."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / 'index.json'
    identity = dict(archive_sha256=archive_hash, exclusions=sorted(exclusions), format=1)
    if receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved.get('identity') == identity:
            if all((out/r['file']).is_file() and sha256(out/r['file']) == r['source_sha256'] for r in saved['records']):
                return saved
    pattern = re.compile(r'(\d{13})-(office|coffee)\.(raw|txt)')
    rows, agents, counts, excluded = [], {}, Counter(), []
    names = set()
    with tarfile.open(archive, mode='r|gz', bufsize=1024*1024) as members:
        for member in members:
            name = PurePosixPath(member.name).name
            if name.endswith('.raw'):
                counts[name.rsplit('-',1)[-1][:-4]] += 1
            match = pattern.fullmatch(name)
            if not match:
                continue
            if member.name != 'rnnoise_contributions/'+name or not member.isfile():
                raise ValueError('Unexpected selected archive member: '+member.name)
            if name in names:
                raise ValueError('Duplicate archive member: '+name)
            names.add(name)
            if not 0 < member.size <= (8*1024**2 if name.endswith('.raw') else 4096):
                raise ValueError('Unexpected archive member size: '+name)
            if name in exclusions:
                excluded.append(name)
                continue
            source = members.extractfile(member)
            if source is None:
                raise ValueError('Unreadable archive member')
            data = source.read(member.size+1)
            if len(data) != member.size:
                raise ValueError('Truncated archive member: '+name)
            if name.endswith('.txt'):
                agents[name[:-4]+'.raw'] = hashlib.sha256(data.strip()).hexdigest()
                continue
            destination = out / name
            digest = hashlib.sha256(data).hexdigest()
            if not destination.is_file() or sha256(destination) != digest:
                tmp = destination.with_suffix('.tmp')
                tmp.write_bytes(data)
                tmp.replace(destination)
            rows.append(dict(recording=name, file=name, bytes=len(data),
                source_sha256=digest, timestamp_ms=int(match.group(1)), category=match.group(2)))
            if len(rows) % 100 == 0:
                print(f'[Archive] {len(rows)} office/coffee recordings indexed', flush=True)
    for row in rows:
        row['user_agent_sha256'] = agents.get(row['recording'], 'missing-user-agent')
    result = dict(identity=identity, records=sorted(rows,key=lambda r:r['recording']),
                  all_raw_categories=dict(counts), excluded_known_files=sorted(excluded))
    atomic_json(receipt, result)
    return result


def cached_screen(row, out, identity):
    key = 'rnnoise-'+Path(row['recording']).stem
    receipt = out / 'screening' / (key+'.json')
    bound = dict(identity, source_sha256=row['source_sha256'])
    if receipt.is_file():
        cached = json.loads(receipt.read_text())
        if cached.get('identity') == bound and all(
                (out/r['file']).is_file() and sha256(out/r['file']) == r['sha256'] for r in cached['records']):
            return cached
    return None


def screen_recording(row, raw_root, out, screen, identity, prepared=None):
    key = 'rnnoise-'+Path(row['recording']).stem
    receipt = out / 'screening' / (key+'.json')
    bound = dict(identity, source_sha256=row['source_sha256'])
    cached = cached_screen(row, out, identity)
    if cached is not None:
        return cached
    result = dict(identity=bound, recording=row['recording'], category=row['category'], records=[])
    try:
        if prepared is None:
            rate = infer_rate(row['bytes'])
            wave, clipping = decode_raw((raw_root/row['file']).read_bytes(), rate)
            probs = screen.probabilities(wave)
        else:
            rate, wave, clipping, probs = prepared
        intervals, speech_fraction = safe_intervals(probs, len(wave))
        result.update(inferred_sample_rate=rate, sample_rate_basis='near-minute length, not verified metadata',
                      source_clip_fraction=clipping, blocked_fraction=speech_fraction,
                      vad_max=float(np.max(probs)), input_seconds=len(wave)/16000)
        for number, (start, stop) in enumerate(intervals):
            piece = wave[start:stop]
            if float(np.sqrt(np.mean(piece.astype(np.float64)**2))) < POLICY['min_noise_rms']:
                continue
            pcm = np.rint(np.clip(piece,-1,1)*32767).astype('<i2')
            destination = out / 'audio' / f'{key}-{number:03d}.wav'
            tmp = destination.with_suffix('.tmp.wav')
            sf.write(tmp, pcm, 16000, subtype='PCM_16')
            tmp.replace(destination)
            result['records'].append(dict(row, key=f'{key}-{number:03d}',
                file=destination.relative_to(out).as_posix(), sha256=sha256(destination),
                pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), samples=len(pcm), seconds=len(pcm)/16000,
                inferred_sample_rate=rate, start_sample_16k=start, stop_sample_16k=stop,
                dataset='RNNoise contributions (filtered public 2025-01 archive)', license='CC0-1.0',
                page='https://media.xiph.org/rnnoise/', attribution='RNNoise noise contributors',
                category_basis='original filename label', speech_screen='Silero VAD, local policy v1'))
        result['status'] = 'accepted' if result['records'] else 'excluded_no_usable_non_speech_interval'
    except ValueError as exc:
        # Fail closed for a per-recording audio quality exclusion. Runtime/model
        # errors still propagate and prevent a misleading complete manifest.
        result['status'] = 'excluded_quality'
        result['reason'] = str(exc)
    atomic_json(receipt, result)
    return result


def screen_batch(rows, raw_root, out, screen, identity):
    """Reuse completed recordings and screen the remainder in one small batch."""
    results, pending = {}, []
    for row in rows:
        cached = cached_screen(row, out, identity)
        if cached is not None:
            results[row['recording']] = cached
            continue
        try:
            rate = infer_rate(row['bytes'])
            wave, clipping = decode_raw((raw_root/row['file']).read_bytes(), rate)
        except ValueError:
            # The regular path records the exclusion reason and its identity.
            results[row['recording']] = screen_recording(row,raw_root,out,screen,identity)
            continue
        pending.append((row,rate,wave,clipping))
    if pending:
        values = screen.probabilities_batch([item[2] for item in pending])
        for (row,rate,wave,clipping), probs in zip(pending,values):
            results[row['recording']] = screen_recording(row,raw_root,out,screen,identity,
                                                        (rate,wave,clipping,probs))
    return [results[row['recording']] for row in rows]


def write_outputs(out, identity, records, report, sources):
    split_records = {s:[r for r in records if r['split']==s] for s in ('train','dev')}
    validate_records(split_records)
    records.sort(key=lambda r:r['key'])
    result = dict(schema=SCHEMA, status='complete', variant='v3184-published-noise-sources-v1',
        preparation_identity=identity, records=records, summary=summary(records),
        provenance=sources['provenance'], screening_policy=POLICY,
        note='Contributor labels and automatic VAD are imperfect. Unknown organizer file/SNR/platform parameters are not reproduced.')
    atomic_json(out/'preparation_report.json', dict(report, manifest_summary=result['summary'], identity=identity))
    lines = ['RTCFake published-source noise bank for V3.18.4',
        'RNNoise contributions: CC0-1.0; https://media.xiph.org/rnnoise/README.txt',
        'Silero VAD: MIT; see SILERO_LICENSE.txt; preparation only.',
        'ESC-50: CC-BY-NC-3.0; underlying Freesound provenance follows.', '']
    for row in records:
        lines.extend([row['key']+' | '+row['license'],row['attribution'],row['page'],''])
    (out/'ATTRIBUTION.txt').write_text('\n'.join(lines),encoding='utf-8')
    shutil.copyfile(HERE/'SILERO_LICENSE.txt', out/'SILERO_LICENSE.txt')
    previews = ['<!doctype html><meta charset="utf-8"><title>Published-source noise bank</title>',
        '<h1>RNNoise + ESC-50</h1><p>Original labels; automatically screened. Counts of cropped segments are not counts of independent sources.</p>']
    for row in records:
        previews.append('<p>'+html.escape(row['category']+' / '+row['split']+' / '+row['key'])+
            '</p><audio controls preload="none" src="'+html.escape(row['file'])+'"></audio>')
    (out/'preview.html').write_text('\n'.join(previews),encoding='utf-8')
    atomic_json(out/'manifest.json', result)  # Final completion marker.
    bind_manifest(out/'manifest.json')
    print('NOISE_MANIFEST='+str(out/'manifest.json'),flush=True)
    print(json.dumps(result['summary']),flush=True)
    return result


def prepare(out, jobs=4, archive=None, reuse=None, ffmpeg=None):
    if not 1 <= jobs <= 8:
        raise ValueError('Use 1 to 8 download workers')
    out = Path(out).expanduser().resolve()
    sources = json.loads(SOURCES.read_text(encoding='utf-8'))
    exclusions = json.loads(EXCLUSIONS.read_text(encoding='utf-8'))
    blocked = set(exclusions['clear_speech']['files']+exclusions['reported_distortion']['files'])
    import torch
    import scipy
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_hash = sha256(ffmpeg)
    identity = dict(sources_sha256=sha256(SOURCES), exclusions_sha256=sha256(EXCLUSIONS),
        preparation_code={p.name:sha256(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')},
        torch_version=torch.__version__, scipy_version=scipy.__version__, numpy_version=np.__version__,
        soundfile_version=sf.__version__, ffmpeg_sha256=ffmpeg_hash)
    out.mkdir(parents=True, exist_ok=True)
    with preparation_lock(out/'.prepare.lock'):
        if (out/'manifest.json').exists():
            old = json.loads((out/'manifest.json').read_text())
            if old.get('preparation_identity') != identity:
                raise ValueError('Completed preparation identity changed; use a new output directory')
            bind_manifest(out/'manifest.json')
            print('NOISE_MANIFEST='+str(out/'manifest.json')+'; verified complete bank',flush=True)
            return old
        for name in ('raw','audio','screening'):
            (out/name).mkdir(exist_ok=True)
        archive_path = Path(archive).expanduser().resolve() if archive else out/'raw'/'rnnoise_contributions.tar.gz'
        if archive:
            print('[Verify] checking supplied 6.3 GiB archive SHA256',flush=True)
            if archive_path.stat().st_size != sources['archive']['bytes'] or sha256(archive_path) != sources['archive']['sha256']:
                raise ValueError('Supplied RNNoise archive does not match the pinned public release')
        else:
            if not archive_path.exists() and shutil.disk_usage(out).free < 2*sources['archive']['bytes']+3*1024**3:
                raise RuntimeError('Need about 16 GiB free for range assembly, raw selections and prepared audio')
            print('[Noise] RNNoise 6.3 GiB archive; original office/coffee labels; ESC-50 120 files; CPU screening only',flush=True)
            fetch(sources['archive'], archive_path, jobs=jobs)
        vad_path = out/'raw'/'silero_vad.jit'
        download(sources['vad'], vad_path)
        raw_root = out/'raw'/'rnnoise_selected'
        print('[Archive] reading selected office/coffee members (or verifying the saved index)',flush=True)
        indexed = index_archive(archive_path, raw_root, sources['archive']['sha256'], blocked)
        print('[Archive] selected recordings='+str(len(indexed['records']))+'; '+json.dumps(indexed['all_raw_categories']),flush=True)
        screen = SpeechScreen(vad_path)
        screen_identity = dict(screen_code=sha256(HERE/'screen.py'), prepare_code=sha256(HERE/'prepare.py'),
            vad_sha256=sources['vad']['sha256'], policy=POLICY,
            torch=torch.__version__, scipy=scipy.__version__, numpy=np.__version__)
        reports, records = [], []
        for start in range(0,len(indexed['records']),POLICY['vad_batch_size']):
            batch = indexed['records'][start:start+POLICY['vad_batch_size']]
            for result in screen_batch(batch,raw_root,out,screen,screen_identity):
                reports.append({k:v for k,v in result.items() if k not in ('records','identity')})
                records.extend(result['records'])
            number = start+len(batch)
            print(f'[Screen] {number}/{len(indexed["records"])} recordings; {len(records)} usable intervals',flush=True)
        records = assign_splits(records)
        reuse = Path(reuse).expanduser().resolve() if reuse else ROOT/'assets'/'noise_v3184'
        def esc_one(row):
            source = out/'raw'/(row['key']+'.wav')
            previous = reuse/'raw'/(row['key']+'.wav')
            if not source.exists() and previous.is_file() and sha256(previous)==row['sha256']:
                shutil.copyfile(previous, source)
            download(row, source)
            destination = out/'audio'/(row['key']+'.wav')
            info = convert_cached(source,destination,ffmpeg,row['sha256'],ffmpeg_hash)
            return dict(row,**info,file=destination.relative_to(out).as_posix(),source_sha256=row['sha256'])
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(esc_one,row) for row in sources['esc50']]
            try:
                for future in as_completed(futures):
                    records.append(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        report = dict(archive_counts=indexed['all_raw_categories'], known_exclusions_present=indexed['excluded_known_files'],
            recording_quality=dict(Counter(r['status'] for r in reports)), recording_reports=reports,
            limitations=['No donor/room identifiers; session grouping is heuristic.',
                'Raw sample rates are inferred for near-minute uploads; other uploads excluded.',
                'VAD is a screen, not a guarantee of absence of speech.',
                'Official noise file IDs and mixing/RTC parameters have not been published.'])
        return write_outputs(out, identity, records, report, sources)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default=str(ROOT/'assets'/'noise_v3184_rnnoise'))
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--archive',help='Optional already-downloaded exact public archive; never modified')
    p.add_argument('--reuse-esc',help='Optional previous small noise bank, to reuse pinned ESC downloads')
    p.add_argument('--ffmpeg')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--list-only',action='store_true')
    args = p.parse_args()
    if args.list_only:
        s = json.loads(SOURCES.read_text())
        print(json.dumps(dict(rnnoise_bytes=s['archive']['bytes'],esc_files=len(s['esc50']),
            vad_bytes=s['vad']['bytes'],source_categories=['office','coffee','rain','footsteps','keyboard'],
            echo='existing V3.18.4 delayed-speech generator',no_training_or_gpu=True),indent=2))
    elif args.verify_only:
        path = Path(args.out).expanduser().resolve()/'manifest.json'
        result = bind_manifest(path)
        print(json.dumps(result['scene_noise_summary'],indent=2))
    else:
        prepare(args.out,args.jobs,args.archive,args.reuse_esc,args.ffmpeg)


if __name__ == '__main__':
    main()
