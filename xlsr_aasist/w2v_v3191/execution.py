"""Verified execution-only upgrade; never rewrite saved training configuration.

The compatibility manifest pins both the known predecessor and replacement
bytes. Unlisted edits still fail closed. Optimizer/RNG/cursor and the checkpoint
configuration identity continue to be loaded by the original resume checks.
"""
import hashlib
import json
from pathlib import Path

EXECUTION_POLICY = 'required_candidate_scores_v2'
DIAGNOSTIC_INTERVAL = 200
MANIFEST = 'w2v_v3191/execution_compat.json'


def diagnostic_step(step):
    """Zero-based epoch step; workers and trainer must use the same schedule."""
    if type(step) is not int or step < 0:
        raise ValueError('Candidate diagnostics require a nonnegative epoch step')
    return step % DIAGNOSTIC_INTERVAL == 0


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_execution_code(saved, root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    manifest = root / MANIFEST
    spec = json.loads(manifest.read_text(encoding='utf-8'))
    if spec.get('execution_policy') != EXECUTION_POLICY:
        raise ValueError('Unregistered V3.19.1 execution manifest')
    current = spec['current']
    # Also cover newly imported modules absent from the old saved inventory.
    for relative, expected in current.items():
        path = (root / relative).resolve()
        if (not path.is_relative_to(root / 'w2v_v3191')
                or _digest(path) != expected):
            raise ValueError('Execution upgrade code differs from registered bytes: ' + relative)
    upgraded = {}
    for absolute, expected in saved.items():
        path = Path(absolute).resolve()
        actual = _digest(path)
        if actual == expected:
            continue
        if not path.is_relative_to(root):
            raise ValueError('Changed code outside execution root: ' + str(path))
        relative = path.relative_to(root).as_posix()
        if (expected not in spec.get('previous', {}).get(relative, [])
                or actual != current.get(relative)):
            raise ValueError('Unregistered source change; refusing resume: ' + relative)
        upgraded[relative] = dict(previous=expected, current=actual)
    return dict(policy=EXECUTION_POLICY, diagnostic_interval=DIAGNOSTIC_INTERVAL,
                manifest_sha256=_digest(manifest), predecessor=spec['predecessor'],
                current_code=current, upgraded_files=upgraded)


def record_execution(run, cfg, state):
    """Attach execution provenance without editing config.json or Adam state."""
    from .common import atomic_json, fingerprint, read_json
    receipt = verify_execution_code(cfg.get('code_fingerprints', {}))
    path = Path(run) / 'execution_policy.json'
    value = read_json(path) if path.is_file() else dict(events=[])
    identity = fingerprint(cfg)
    events = value['events']
    if (not events or events[-1]['manifest_sha256'] != receipt['manifest_sha256']
            or events[-1]['configuration_identity'] != identity):
        events.append(dict(receipt, configuration_identity=identity,
                           from_update=state['cursor'], epoch=state['epoch'],
                           next_step=state['next_step'],
                           optimizer_rng_schedule_preserved=True))
        atomic_json(path, value)
    state['execution_policy'] = EXECUTION_POLICY
    print('[Execution] required candidate scores only; full diagnostics every '
          f'{DIAGNOSTIC_INTERVAL} epoch steps; resume keeps optimizer/RNG/LR/cursor',
          flush=True)
