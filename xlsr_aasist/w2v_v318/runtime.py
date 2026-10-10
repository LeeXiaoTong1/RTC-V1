"""PEP 440 release checks: official fairseq2 reports 0.6, equivalent to 0.6.0."""
from importlib.metadata import version
from packaging.version import InvalidVersion,Version
from pathlib import Path


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


def bundled_ffmpeg():
    """Use the pinned wheel's binary, never a historical PATH/Conda fallback."""
    require_version('imageio-ffmpeg','0.6.0')
    import imageio_ffmpeg
    from imageio_ffmpeg._definitions import FNAME_PER_PLATFORM,get_platform
    name=FNAME_PER_PLATFORM.get(get_platform())
    if not name:raise RuntimeError('No bundled FFmpeg for this platform')
    path=Path(imageio_ffmpeg.__file__).resolve().parent/'binaries'/name
    if not path.is_file():raise FileNotFoundError('Bundled FFmpeg is missing; install the imageio-ffmpeg==0.6.0 binary wheel: '+str(path))
    return str(path)
