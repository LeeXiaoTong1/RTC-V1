"""Fork an intact, pre-joint warmup checkpoint onto the smaller-alpha grid.

The failed run is read-only. This is a narrowly checked calibration repair,
not a general override of the checkpoint/code/config identity contract.
"""
import argparse
import copy
from datetime import datetime
from pathlib import Path
import re
import secrets
import shutil

import torch

from w2v_aasist.launch import run_lock
from .calibration import CANDIDATES, LEGACY_CANDIDATES, choose_alpha
from .common import ROOT, atomic_json, atomic_save, digest, fingerprint, read_json, schema_for, verify_files
from .config import code_inventory
from .identity import require_registry_config
from .processes import ensure_idle


# Exact published files from 91b3018. No model/optimizer/sampler change is
# permitted, and no already calibrated or jointly trained run can use this.
OLD_CODE = {
    'w2v_v31810/calibration.py': 'd76da87f058a98426449ad17d96acf91a834d4f57743d5f96513997a158552a3',
    'w2v_v31810/config.py': 'b33a20bb9ef1f7c4c1eb9f535f29d3af48c427b22422179c56eb9d68ca326e64',
    'w2v_v31810/train.py': '2dc100e89e2427bb9dc4f2e4650cd8e747a361ca0101e3270ff5f39095f10fd7',
}
ADDED_CODE = {'w2v_v31810/recover_calibration.py'}


def verify_code_upgrade(saved, current):
    def relative(inventory):
        result = {}
        for name, checksum in inventory.items():
            path = Path(name).resolve().relative_to(ROOT.resolve()).as_posix()
            if path in result:
                raise ValueError('Duplicate recorded code path: ' + path)
            result[path] = checksum
        return result
    old, new = relative(saved), relative(current)
    if set(old) - set(new) or set(new) - set(old) != ADDED_CODE:
        raise ValueError('Unexpected code inventory change; this repair only supports the published pre-joint run')
    for name, checksum in old.items():
        expected = OLD_CODE.get(name, new[name])
        if checksum != expected:
            raise ValueError('Unrelated or unknown code change blocks calibration recovery: ' + name)
    if not set(OLD_CODE) <= set(old):
        raise ValueError('Incomplete saved calibration code inventory')


def validate_boundary(run, cfg, state, failure):
    schema = schema_for(cfg)
    if state.get('schema') != schema or state.get('identity') != fingerprint(cfg):
        raise ValueError('Checkpoint/config identity mismatch')
    if (cfg.get('experiment') not in ('E1', 'E2') or cfg.get('warm_epochs') != 1 or
            cfg.get('calibration_tasks') != 12 or tuple(cfg.get('inner_candidates', ())) != LEGACY_CANDIDATES):
        raise ValueError('Recovery requires the original five-candidate, one-epoch warmup recipe')
    history = state.get('history', [])
    if (state.get('epoch') != 1 or state.get('next_step') != 0 or state.get('alpha') is not None or
            state.get('calibration') is not None or state.get('epoch_accumulator') is not None or
            len(history) != 1 or history[0].get('phase') != 'head_warmup'):
        raise ValueError('Only the completed warmup boundary before any joint update can be recovered')
    tag = state.get('last_tag', '')
    match = re.fullmatch(r'epoch_1_step_(\d+)', tag)
    if (not match or state.get('cursor') != int(match[1]) or history[0].get('tag') != tag or
            state.get('best_tag') != tag or state.get('best_model') is None):
        raise ValueError('Warmup checkpoint cursor/history mismatch')
    details = failure.get('details') or {}
    if (details.get('status') != 'failed' or details.get('task_count') != 12 or
            details.get('selected_alpha') is not None or
            details.get('candidates') != list(LEGACY_CANDIDATES) or
            len(details.get('records', [])) != 60 or
            choose_alpha(details['records'], LEGACY_CANDIDATES)[0] is not None):
        raise ValueError('Expected a completed five-candidate calibration failure report')
    if Path(failure.get('retained_checkpoint', '')).resolve() != run / 'last.pt':
        raise ValueError('Calibration failure belongs to another checkpoint')
    if (run / 'completed.json').exists():
        raise ValueError('Completed experiments cannot use calibration recovery')
    require_registry_config(cfg)
    required = {'train_rows.json', 'dev_rows.json', 'validation_protocol.json', 'source_registry.json'}
    if set(cfg.get('saved_manifests', {})) != required:
        raise ValueError('Missing or unexpected saved source manifests')
    verify_files({str(run / name): value for name, value in cfg['saved_manifests'].items()})


def recover(run):
    run = Path(run).expanduser().resolve()
    initial = {name: digest(run / name) for name in ('config.json', 'last.pt', 'inner_calibration_failure.json')}
    cfg = read_json(run / 'config.json')
    current_code = code_inventory()
    verify_code_upgrade(cfg.get('code_fingerprints', {}), current_code)
    state = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
    failure = read_json(run / 'inner_calibration_failure.json')
    validate_boundary(run, cfg, state, failure)
    names = set(cfg['saved_manifests']) | {
        'config.json', 'last.pt', 'inner_calibration_failure.json',
        'training_history.json', 'steps.jsonl', 'input_audit.json', 'parameter_inventory.json',
    }
    for pattern in ('coverage_epoch_1.json', 'performance_epoch_1_step_*.json',
                    'selection_epoch_1_step_*.json', 'selection_scores_epoch_1_step_*.npz',
                    'diagnostics/*.html', 'diagnostics/*.csv', 'submission_request.json', 'report_request.json'):
        names.update(path.relative_to(run).as_posix() for path in run.glob(pattern))
    for name in names:
        path = run / name
        if not path.is_file() or path.is_symlink() or path.resolve() != path.absolute():
            raise ValueError('Recovery input must be a regular in-run file: ' + name)
    checksums = {name: digest(run / name) for name in sorted(names)}
    if any(checksums[name] != value for name, value in initial.items()):
        raise ValueError('Recovery inputs changed while reading the saved boundary')
    required_bytes = sum((run / name).stat().st_size for name in names)
    reserve = cfg.get('free_reserve_bytes', 0) + cfg.get('disk_margin_bytes', 0)
    if shutil.disk_usage(run.parent).free < required_bytes * 2 + reserve:
        raise OSError('Need space for a recovery copy plus the configured free-space reserve')
    new_cfg = copy.deepcopy(cfg)
    new_cfg.update(inner_candidates=list(CANDIDATES), code_fingerprints=current_code)
    # Only identity metadata changes. Every model tensor, Adam moment, RNG,
    # GradScaler value, history entry and successful-update cursor is retained.
    state['identity'] = fingerprint(new_cfg)
    suffix = datetime.now().strftime('%Y%m%d_%H%M%S') + '_recal_' + secrets.token_hex(3)
    destination = run.parent / ('w2v_v31810_' + suffix)
    staging = run.parent / ('.w2v_v31810_' + suffix + '.partial')
    staging.mkdir(exist_ok=False)
    for name in sorted(names - {'config.json', 'last.pt', 'inner_calibration_failure.json', 'submission_request.json'}):
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / name, target)
    atomic_json(staging / 'config.json', new_cfg)
    atomic_save(staging / 'last.pt', state, reserve)
    atomic_json(staging / 'recovery' / 'original_calibration_failure.json', failure)
    if 'submission_request.json' in names:
        request = read_json(run / 'submission_request.json')
        request['out'] = str(Path(request['out']).parent / (destination.name + '_submission_best'))
        atomic_json(staging / 'submission_request.json', request)
    verify_files({str(run / name): checksum for name, checksum in checksums.items()})
    for name in names - {'config.json', 'last.pt', 'inner_calibration_failure.json', 'submission_request.json'}:
        if digest(staging / name) != checksums[name]:
            raise ValueError('Recovery copy changed: ' + name)
    atomic_json(staging / 'calibration_recovery.json', dict(
        status='ready', source_run=str(run), source_files=checksums,
        original_config_identity=fingerprint(cfg), recovered_config_identity=fingerprint(new_cfg),
        changed_config_fields=['inner_candidates', 'code_fingerprints'],
        original_candidates=list(LEGACY_CANDIDATES), candidates=list(CANDIDATES),
        completed_epochs=state['epoch'], cursor=state['cursor'], next_step=state['next_step'],
        joint_updates=0, policy='Retain warmup model/optimizer/RNG/scaler; recalibrate on Train A only'))
    staging.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    ensure_idle()
    with run_lock(ROOT / 'exp' / '.aasist-launch.lock'):
        ensure_idle()
        destination = recover(args.run)
    # stdout is a single path for RUN=$(python -m ...).
    print(destination)


if __name__ == '__main__':
    main()
