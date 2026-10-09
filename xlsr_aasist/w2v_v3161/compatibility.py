"""Audited source-code migrations; never rewrite a completed run's manifest.

22e4252 -> 672953d moved the V3.16 parser to arguments.py and replaced only
the launch-budget check. All remaining function ASTs are identical. A completed
22e4252 run can therefore be read by the new launcher without changing weights,
data, model construction or training mathematics. Unknown edits remain errors.
"""
from pathlib import Path

from w2v_v39.common import ROOT, digest
from w2v_v316_tfcl.config import verify_inputs as strict_source_verification


# Both LF and CRLF byte encodings are recorded explicitly. This is not a
# general newline normalization or a blanket exception for config modules.
OLD_CONFIG_HASHES = frozenset((
    'bf73739a95b13bfa86451e832cab3e9a30d2b9d1cf865b0d6f62d52f10e0a811',
    'fe4f415d36243cf0f34b62c546c3362b9cc85b20639de6cd41b353bbc66730cc',
))
CURRENT_CONFIG_HASHES = frozenset((
    '63086572dd3eb6d81fc95e5a21e3b2ea95755bf96fd935ce17527b407dc02050',
    '4275c8c8e865243c0dd2501cca07fe872f5bc0e009aa028912b60110e25b5dda',
))
CONFIG_PATH = 'w2v_v316_tfcl/config.py'


def verify_source(cfg):
    """Return current code pins and the narrowly audited migration record.

    The old config dictionary (including its identity inside LAST) is retained
    verbatim. Only a temporary verification copy uses the approved code hash.
    All original checkpoint, data, pretrained and runtime checks still run.
    """
    current, migrations, failures = {}, [], []
    for name, expected in cfg['code_fingerprints'].items():
        path = Path(name)
        actual = digest(path) if path.is_file() else None
        current[name] = actual
        if actual == expected:
            continue
        try:
            relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            relative = None
        if (relative == CONFIG_PATH and expected in OLD_CONFIG_HASHES | CURRENT_CONFIG_HASHES
                and actual in CURRENT_CONFIG_HASHES):
            migrations.append(dict(path=name, recorded_sha256=expected, current_sha256=actual,
                reason='V3.16 launch-argument refactor only; remaining function ASTs unchanged',
                from_commit='22e42521becd463e08c4d94887622fec209f02d9',
                to_commit='672953da8960d42593d3b3d5109b4173483d66fd'))
        else:
            failures.append(f'{name}\n  recorded={expected}\n  current={actual or "MISSING"}')
    if failures:
        raise ValueError('Unreviewed source-code change; checkpoint was not modified:\n' + '\n'.join(failures))
    # This new import was absent from 22e4252's manifest. Pin it in the new run.
    arguments = ROOT/'w2v_v316_tfcl'/'arguments.py'
    current.setdefault(str(arguments.resolve()), digest(arguments))
    strict_source_verification(dict(cfg, code_fingerprints=current))
    return current, migrations
