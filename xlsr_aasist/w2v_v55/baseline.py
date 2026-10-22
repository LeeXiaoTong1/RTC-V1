"""Pin the inherited V5.1 sources, allowing only platform line-ending changes."""
import hashlib
import json
from pathlib import Path


def source_digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def verify_baseline(root, expected=None):
    manifest = Path(__file__).with_name('v51_baseline.json')
    if expected is None:
        expected = json.loads(manifest.read_text(encoding='utf-8'))['files']
    root = Path(root).resolve()
    changed = []
    for name, checksum in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('V5.1 baseline manifest contains a path outside the checkout')
        if not path.is_file() or source_digest(path) != checksum:
            changed.append(name)
    if changed:
        raise ValueError('Inherited V5.1 code changed; restore the pinned baseline before V5.5: '
                         + ', '.join(changed[:12]))
    return manifest
