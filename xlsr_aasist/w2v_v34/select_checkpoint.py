"""Choose the completed V3.4 protected winner, including explicit baseline fallback."""
import argparse
import json
from pathlib import Path


def select(run):
    run = Path(run).expanduser().resolve()
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
    done = json.loads((run/'completed.json').read_text(encoding='utf-8'))
    if cfg.get('version') != '3.4' or not done.get('selection', {}).get('best_safe', {}).get('tag'):
        raise ValueError('Default export requires a completed V3.4 run and protected selection')
    checkpoint = run/'best_model.pt'
    if not checkpoint.is_file():
        raise ValueError('Selected V3.4 checkpoint is missing')
    from w2v_v3.train import read_state
    from . import SCHEMA
    state = read_state(checkpoint)
    if (state.get('schema') != SCHEMA or state.get('kind') != 'weights'
            or state.get('tag') != done['selection']['best_safe']['tag']
            or state.get('config') != cfg):
        raise ValueError('Checkpoint differs from the completed V3.4 selection')
    return checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    print(select(p.parse_args().run))


if __name__ == '__main__':
    main()
