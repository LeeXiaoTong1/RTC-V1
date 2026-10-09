"""PEP 440 release checks: official fairseq2 reports 0.6, equivalent to 0.6.0."""
from importlib.metadata import version
from packaging.version import InvalidVersion,Version


def require_version(name,wanted,actual=None):
    actual=version(name) if actual is None else actual
    try:
        # Accept release padding and local wheel tags, but not another release,
        # prerelease, development or postrelease. Native ABI is checked on import.
        expected=Version(wanted)
        installed=Version(actual)
        matches=(installed if expected.local else Version(installed.public))==expected
    except InvalidVersion:
        matches=False
    if not matches:
        raise RuntimeError(f'{name}: installed={actual}; required release={wanted}; use the sdd-v318 environment')
    return actual
