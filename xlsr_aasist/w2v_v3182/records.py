"""Read verified full-wave manifests, including an existing V3.18 run."""
from pathlib import Path
from w2v_v318.records import dev_partition, inherited, data_inputs as legacy_inputs
from .common import read_json, digest, verify_files


def data_inputs(source, _seen=None):
    source = Path(source).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if source in seen:
        raise ValueError('Cyclic data-run provenance')
    seen.add(source)
    cfg = read_json(source / 'config.json')
    if cfg.get('version') not in ('3.18', '3.18.2'):
        return legacy_inputs(source)
    if Path(cfg['data_run']).resolve() == source:
        raise ValueError('Data run cannot reference itself')
    # Recheck the original protocol and provenance, not merely the saved row count.
    original, train, dev, files = data_inputs(cfg['data_run'], seen)
    verify_files({str(source / n): h for n, h in cfg['saved_manifests'].items()})
    if train != read_json(source / 'train_rows.json') or dev != read_json(source / 'dev_rows.json'):
        raise ValueError('Saved V3.18 manifests differ from canonical official data')
    files[str(source / 'config.json')] = digest(source / 'config.json')
    return original, train, dev, files
