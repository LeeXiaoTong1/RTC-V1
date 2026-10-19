"""Defer an unused text converter in the owned, pinned XEUS environment.

ESPnet's phoneme tokenizer imports g2p_en at module import time. That package
downloads NLTK resources even when SSLTask only encodes waveforms. Move this
one import to its existing lazy G2p_en call site, preserving the converter.
No encoder, model configuration, weights, or NLTK functions are changed.
"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

ESPNET_REVISION='7d395a8bb223f6bddd479d4d1dd6feddf49f82f4'
ORIGINAL_SHA256='ffb3b49c9789beaec0be3780b6b69aadc1c06f4c9fd9cc73e9dc2efe9ce7fc58'
PATCHED_SHA256='8f94d61289c700a661fdd64bedbcea084b8d542c26048d293f7ee255d475982e'
RELATIVE_PATH='espnet2/text/phoneme_tokenizer.py'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_atomic(path,data):
    temporary=path.with_name(path.name+'.v31813.partial')
    temporary.write_bytes(data);temporary.replace(path)


def patch_text_import(private,distribution):
    private=Path(private).resolve()
    # Never patch an inherited/global package or another task's environment.
    if private!=Path(sys.prefix).resolve() or private==Path(sys.base_prefix).resolve():
        raise RuntimeError('XEUS text-import compatibility requires its private environment')
    owner=json.loads((private/'owner.json').read_text(encoding='utf-8'))
    if owner.get('schema')!='v31813_xeus_env_v1' or owner.get('espnet')!=ESPNET_REVISION:
        raise RuntimeError('XEUS environment ownership/revision mismatch')
    target=Path(distribution.locate_file(RELATIVE_PATH)).resolve()
    if not target.is_relative_to(private) or distribution.version!='202506':
        raise RuntimeError('Refusing to patch ESPnet outside the owned pinned environment')
    source=target.read_bytes();before=sha256(source);changed=before==ORIGINAL_SHA256
    if before not in (ORIGINAL_SHA256,PATCHED_SHA256):
        raise RuntimeError('Unexpected ESPnet phoneme tokenizer; refusing an unverified text-import patch')
    backup=private/'v31813_phoneme_tokenizer.original.py'
    if backup.exists() and sha256(backup.read_bytes())!=ORIGINAL_SHA256:
        raise RuntimeError('Original ESPnet text-import backup changed')
    if changed:
        original='        if self.g2p is None:\n            self.g2p = g2p_en.G2p()'
        replacement='        if self.g2p is None:\n            import g2p_en\n\n            self.g2p = g2p_en.G2p()'
        updated=source.decode('utf-8').replace('import g2p_en\n','',1).replace(original,replacement,1).encode('utf-8')
        if sha256(updated)!=PATCHED_SHA256:raise RuntimeError('ESPnet text-import patch hash mismatch')
        if not backup.exists():write_atomic(backup,source)
        write_atomic(target,updated)
        print('XEUS: deferred unused English text converter; waveform inference needs no NLTK resource download.',flush=True)
    receipt=dict(schema='v31813_espnet_lazy_g2p_v1',espnet_revision=ESPNET_REVISION,
                 file=RELATIVE_PATH,original_sha256=ORIGINAL_SHA256,patched_sha256=PATCHED_SHA256)
    write_atomic(private/'v31813_text_import_patch.json',(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode())
    return changed


def load_ssl_task():
    patch_text_import(Path(sys.prefix),metadata.distribution('espnet'))
    from espnet2.tasks.ssl import SSLTask
    return SSLTask
