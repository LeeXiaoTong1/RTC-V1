"""PEP 440 release checks: official fairseq2 reports 0.6, equivalent to 0.6.0."""
from importlib.metadata import version
from packaging.version import InvalidVersion,Version
from pathlib import Path

PROFILES={
    'cu118':dict(torch='2.6.0+cu118',torchaudio='2.6.0+cu118',fairseq2n='0.6',cuda='11.8'),
    'cu126':dict(torch='2.8.0+cu126',torchaudio='2.8.0+cu126',fairseq2n='0.6+cu126',cuda='12.6'),
}


def execution_profile(torch_version=None):
    actual=version('torch') if torch_version is None else str(torch_version)
    for name,pins in PROFILES.items():
        if Version(actual)==Version(pins['torch']):return name
    raise RuntimeError('Unsupported V3.18 Torch build: '+actual+'; use setup_w2v_v318.sh --cuda 11.8 or --cuda 12.6')


def expected_requirements(profile=None):
    profile=profile or execution_profile()
    filename='requirements_w2v_v318_cu118.txt' if profile=='cu118' else 'requirements_w2v_v318.txt'
    if profile not in PROFILES:raise ValueError('Unknown execution profile: '+profile)
    result={}
    for line in (Path(__file__).resolve().parents[1]/filename).read_text(encoding='utf8').splitlines():
        line=line.split('#',1)[0].strip()
        if line:
            name,wanted=line.split('==',1);result[name]=wanted
    return result


def native_abi(profile=None):
    """Validate the extension's actual build, not only its package version."""
    profile=profile or execution_profile();pins=PROFILES[profile]
    import fairseq2n
    require_version('fairseq2n',pins['fairseq2n'])
    built=fairseq2n.torch_version();variant=fairseq2n.torch_variant()
    require_version('fairseq2n build Torch',pins['torch'],built)
    if variant!='CUDA '+pins['cuda']:raise RuntimeError('fairseq2n native CUDA variant mismatch: '+variant)
    return dict(profile=profile,torch=built,variant=variant,custom_cuda_kernels=fairseq2n.supports_cuda())


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
