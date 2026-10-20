"""Reuse registered speech and augmentation assets, never an Omni checkpoint."""
from pathlib import Path
from .common import ROOT, read_json, verify_files
from w2v_v31810.identity import verify_registry


def resolve_parent(value=None):
    if value:
        run = Path(value).expanduser().resolve()
    else:
        marker = next((ROOT/'exp'/name for name in ('.latest_v31813_run', '.latest_v31810_run')
                       if (ROOT/'exp'/name).is_file()), None)
        if marker is None:
            raise FileNotFoundError('Specify --data-run with a registered V3.18.10/13 data run')
        run = Path(marker.read_text().strip()).resolve()
    verify_source(run)
    return run


def verify_source(run):
    run = Path(run)
    cfg = read_json(run/'config.json')
    if cfg.get('version') not in ('3.18.10', '3.18.13'):
        raise ValueError('Use a registered V3.18.10/13 run as DATA only; detector weights are incompatible')
    names = ('train_rows.json', 'validation_protocol.json', 'source_registry.json')
    recorded = cfg.get('saved_manifests', {})
    verify_files({str(run/name):recorded[name] for name in names if name in recorded})
    verify_registry(read_json(run/names[0]), read_json(run/names[2]), cfg, read_json(run/names[1]))
    return cfg


def resolve_data_run(value):
    run = Path(value).expanduser().resolve()
    cfg = read_json(run/'config.json')
    if cfg.get('data_recipe') != 'v4_source_transfer_v2':
        raise ValueError('Expected V4 prepared source registry, never the original unmodified Dev inventory')
    verify_files({str(run/name):sha for name,sha in cfg['prepared_files'].items()})
    verify_registry(read_json(run/'train_rows.json'),read_json(run/'source_registry.json'),cfg,read_json(run/'validation_protocol.json'))
    return run
