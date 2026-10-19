"""Resolve immutable V3.18.13 weights and V4's separately registered data."""
from pathlib import Path
from .common import ROOT, read_json, verify_files
from w2v_v31810.identity import verify_registry


def resolve_parent(value=None):
    run = Path(value).expanduser().resolve() if value else Path((ROOT/'exp/.latest_v31813_run').read_text().strip()).resolve()
    cfg = read_json(run/'config.json')
    if cfg.get('version') != '3.18.13' or cfg.get('lora_layers') != 16:
        raise ValueError('V4 requires a completed/safely paused V3.18.13 last-16-LoRA parent')
    return run


def resolve_data_run(value):
    run = Path(value).expanduser().resolve()
    cfg = read_json(run/'config.json')
    if cfg.get('data_recipe') != 'v4_source_transfer_v1':
        raise ValueError('Expected V4 prepared source registry, never the original unmodified Dev inventory')
    verify_files({str(run/name):sha for name,sha in cfg['prepared_files'].items()})
    verify_registry(read_json(run/'train_rows.json'),read_json(run/'source_registry.json'),cfg,read_json(run/'validation_protocol.json'))
    return run
