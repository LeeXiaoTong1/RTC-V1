"""Select one completed experiment's protected winner, never combine predictions."""
import argparse
import json
from pathlib import Path


def select(run):
    run = Path(run).expanduser().resolve()
    report = json.loads((run/'comparison.json').read_text(encoding='utf-8'))
    if report.get('status') != 'complete':
        raise ValueError('Finish/resume both requested arms before default submission selection')
    arm = report.get('selected_arm')
    if arm not in ('control','candidate'):
        raise ValueError('No validated selected arm')
    checkpoint = run/arm/'best_model.pt'
    if (not report.get('checkpoint') or checkpoint.resolve() != Path(report['checkpoint']).resolve()
            or not checkpoint.is_file()):
        raise ValueError('Selected checkpoint path changed or missing')
    return checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    print(select(p.parse_args().run))


if __name__=='__main__': main()
