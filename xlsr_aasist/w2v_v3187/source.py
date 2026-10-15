"""Find reusable data manifests without loading a detector checkpoint."""
from pathlib import Path
from .common import ROOT, read_json, digest

REQUIRED = ('config.json', 'train_rows.json', 'dev_rows.json', 'dev_partition.json')


def inspect_data_run(path):
    run = Path(path).expanduser().resolve()
    for name in REQUIRED:
        if not (run / name).is_file():
            raise ValueError('Missing data artifact: ' + str(run / name))
    cfg = read_json(run / 'config.json')
    if cfg.get('version') not in ('3.18.4', '3.18.5', '3.18.6', '3.18.7'):
        raise ValueError('Use a V3.18.4 AUG5 / V3.18.5 / V3.18.6 data run')
    if cfg.get('pooling') != 'asp' or cfg.get('omni_arch') != '3b':
        raise ValueError('Data recipe must use Omni-3B and overlapping-window ASP')
    return run, cfg


def resolve_data_run(value=None, root=ROOT):
    if value:
        return inspect_data_run(value)[0]
    roots = [Path(root) / 'exp']
    base = Path(root).parents[1]
    roots += [base / name / 'xlsr_aasist' / 'exp' for name in ('RTC-w2v-improved', 'RTC')]
    # Prefer an explicit latest marker. A run can supply manifests even while
    # interrupted; no detector weights or optimizer state are inherited.
    for version in ('3186', '3185', '3184'):
        for directory in roots:
            pointer = directory / ('.latest_v' + version + '_run')
            if pointer.is_file():
                try:
                    return inspect_data_run(pointer.read_text(encoding='utf-8').strip())[0]
                except (ValueError, OSError):
                    pass
    found = {}
    for directory in roots:
        if directory.is_dir():
            for path in directory.glob('w2v_v318*'):
                try:
                    run, _ = inspect_data_run(path)
                    found[str(run)] = run
                except (ValueError, OSError):
                    pass
    if len(found) == 1:
        return next(iter(found.values()))
    listing = '\n'.join(sorted(found)) or '(none)'
    raise ValueError('Specify --data-run; compatible manifest runs:\n' + listing)


def metadata(path):
    run, cfg = inspect_data_run(path)
    files = {str(run / name): digest(run / name) for name in REQUIRED}
    return cfg, dict(run=str(run), files=files, checkpoint_loaded=False,
                    purpose='Data, assets and fixed Dev partition only')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-run')
    print(resolve_data_run(p.parse_args().data_run))
