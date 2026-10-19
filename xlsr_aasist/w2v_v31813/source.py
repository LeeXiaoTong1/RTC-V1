"""Resolve the actual detector inspected by V3.18.11, never its diagnostics zip."""
from pathlib import Path
from w2v_v31810.common import ROOT, read_json

DIAGNOSTIC_PARENT='w2v_v31810_20261018_145711_recal_732feb'

def resolve_parent(value=None):
    run=Path(value).expanduser().resolve() if value else ROOT/'exp'/DIAGNOSTIC_PARENT
    cfg=read_json(run/'config.json')
    if cfg.get('version')!='3.18.10' or cfg.get('lora_layers')!=16:
        raise ValueError('Use the safely completed/paused V3.18.10 detector with last-16 LoRA; V3.18.11 has no separately trained best')
    return run

# Preparation reads the existing registered Train inventory only.
resolve_data_run=resolve_parent
