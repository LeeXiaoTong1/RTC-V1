"""Pin useful weights, archive then remove only explicitly listed obsolete source files."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import os
import zipfile
from .checkpoints import retain
from .launch import BASELINE, ROOT
from .runtime import atomic_json


def clean_code(project, manifest, apply=False):
    project = Path(project).resolve()
    entries = json.loads(Path(manifest).read_text(encoding='utf-8'))['files']
    protected = {'.git', 'exp', 'checkpoints', 'dataset', 'data', 'pretrained', 'external_noise', 'code_archives'}
    planned = []
    # Validate ALL final absolute targets before any archive or removal.
    for entry in entries:
        relative = PurePosixPath(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or set(relative.parts) & protected:
            raise ValueError('Cleanup target is outside the source allowlist')
        path = project / str(relative)
        path.resolve().relative_to(project)
        if path.suffix.lower() not in {'.py','.sh','.ps1','.md','.txt'}:
            raise ValueError('Cleanup accepts source/document files only')
        if path.is_symlink():
            raise ValueError('Inspect symbolic-link source manually: ' + str(path))
        if path.is_file():
            if path.stat().st_size > 8 * 1024**2:
                raise ValueError('Unexpected large source file; nothing removed')
            data = path.read_bytes()
            planned.append((path, str(relative), data, hashlib.sha256(data).hexdigest()))
    result = {'apply': apply, 'files': [p[1] for p in planned],
              'weights_and_audio_removed': 0, 'archive': None}
    if apply and planned:
        folder = project / 'code_archives'
        folder.mkdir(exist_ok=True)
        archive = folder / ('obsolete_sources_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.zip')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for _, relative, data, _ in planned:
                z.writestr(relative, data)
        with zipfile.ZipFile(archive) as z:
            for path, relative, _, digest in planned:
                if hashlib.sha256(z.read(relative)).hexdigest() != digest or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise RuntimeError('Source/archive changed; nothing removed')
        # Explicit files only. Never recursively remove a directory that may contain user data.
        parents = set()
        for path, _, _, _ in planned:
            path.unlink()
            parents.update(p for p in path.parents if p != project and project in p.parents)
        for path in sorted(parents, key=lambda p: len(p.parts), reverse=True):
            path.resolve().relative_to(project)
            try:
                path.rmdir()  # Empty directories only; unrelated/untracked files remain.
            except OSError:
                pass
        result['archive'] = str(archive)
        atomic_json(folder / (archive.stem + '.json'), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default=BASELINE)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    print('Preserving original and best candidates BEFORE source cleanup.', flush=True)
    retain(args.baseline, [ROOT/'exp', Path(args.baseline).parents[2]],
           ROOT/'checkpoints'/'retained', apply=args.apply)
    result = clean_code(ROOT.parent, ROOT.parent/'cleanup_manifest.json', args.apply)
    print('SOURCE_CLEANUP=' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
