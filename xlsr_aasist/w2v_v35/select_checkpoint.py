"""Select the completed single-model V3.5 winner, including an explicit reference fallback."""
import argparse
import json
from pathlib import Path


def select(run):
    run = Path(run).expanduser().resolve()
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
    done = json.loads((run/'completed.json').read_text(encoding='utf-8'))
    expected = done.get('selected_tag') or done.get('selection', {}).get('best_selected', {}).get('tag')
    if cfg.get('version') != '3.5' or not expected:
        raise ValueError('Default submission requires a completed V3.5 run with a selected model')
    from w2v_v3.train import read_state
    from . import SCHEMA
    path = run/'best_model.pt'
    state = read_state(path)
    if (state.get('schema') != SCHEMA or state.get('kind') != 'weights'
            or state.get('tag') != expected or state.get('config') != cfg):
        raise ValueError('Checkpoint differs from completed V3.5 selection')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    print(select(p.parse_args().run))


if __name__ == '__main__':
    main()
