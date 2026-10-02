"""One audited I/O-only migration from the published V3.3 source version.

Config, data, dependencies and every other implementation remain exact-match.
The migration never rewrites a checkpoint or changes optimizer/RNG state.
"""
import hashlib
import json
from pathlib import Path

from w2v_aasist.runtime import atomic_json, sha256
from . import SCHEMA


SOURCE_RELEASE = '01c495c'
ERROR = ('Exact resume requires a V3 training checkpoint and identical config, '
         'metadata, code, versions; only the verified V3.3 transport repair is compatible')

# SHA256 of the exact UTF-8/LF blobs published at 01c495c. Including tests makes
# this an exact old release identity, rather than a filename-based exemption.
PUBLISHED_V33 = {
    'w2v_v33/__init__.py': 'd81dea0e833673d623194b53d9ae38002fdee482f7042a5d566c563a089822b8',
    'w2v_v33/cache.py': '29f3a488919cd2728dec7bcbb0d569946a9327e5af2569cb3bf6b0e0f7deca9a',
    'w2v_v33/config.py': '933e944623fd6cfa6b18614ff232a9d7f9440c294ec7a648e2c15c34c6920f64',
    'w2v_v33/control.py': 'c6ded6e4df3160801ff1fe556a353349f493ce6e79d58102906e007026e036a2',
    'w2v_v33/data.py': 'b5286e0a663b67dec3760b7ff4473cc02910b0581e03cac09be488e20f74631c',
    'w2v_v33/evaluate.py': 'b2d8fa043922fb6051d4c33117d56981a98f6522f905fef041e33b0c6f72eeb5',
    'w2v_v33/losses.py': '9774d5e6c927af3d020316397cc43d644965ad6f6ba57444c0ef237ef906608d',
    'w2v_v33/maintenance.py': '3de2c7ff0bab52025203d75ca311585c75eb50548f025f136c5cc95696ab4c81',
    'w2v_v33/model.py': '1c92064e972f3e480c4df14b8aa524f6d2dc39b8c09aafe8f9932aab46661ebf',
    'w2v_v33/select_checkpoint.py': '3291ac540dbd265a3bc6c07b5ac1891a6768d525a923bb121b9fd0031c6b1725',
    'w2v_v33/step.py': '4777d8ce3c98b9276b1b372abc3b3bb901f1c8244c3297a3bccbc560bbe5ed2c',
    'w2v_v33/stop.py': '5b8aceb4af7c1d9f8980f8ea361a84797dccb344e8e00ef83d17961bdb6f6786',
    'w2v_v33/test_cache.py': 'ae1139258c7043120b372a34ceee731c3dec32bd17b5566626cb3ae3698d1c84',
    'w2v_v33/test_control.py': '5a42125b6727122a93365c19f720d1573fba625e9d0f80d7f78fe3cbbc69a17c',
    'w2v_v33/test_data.py': '4a79e6fd5efa897aa4b7be0b60570afc5a175b1c6e1bab38ae29b1069a42fa81',
    'w2v_v33/test_model.py': 'da5e8a58e34db5a3872a380e2e5824a8e2a167d29c193e3c103d6a13ed73fc0c',
    'w2v_v33/test_step.py': 'f39db6d82bbac80cc4dd83ab0a48c5fb738b625598490045417d536e72299e58',
    'w2v_v33/test_train.py': 'd975c17abdfd51071c12d05e29762981dd17a3cb9c33f785ad05f31903d58b44',
    'w2v_v33/test_workflow.py': 'de8d2a797cd00d386254a191981d0811cf8f548cc69fe0271d2ecc2e6f8931de',
    'w2v_v33/train.py': 'b40ed71aa5d2db7abf5bb6ca59710cfdbc1872d438343219712e331a75702fc4',
    'w2v_v33/workflow.py': '8951eb221cedda53b8e7665b4b4a4ba601a09694a5c1341dc47afc7b62580bc9',
}
CHANGED_FILES = frozenset(('w2v_v33/data.py', 'w2v_v33/train.py'))
ADDED_FILES = frozenset(('w2v_v33/transport.py', 'w2v_v33/test_transport.py',
                         'w2v_v33/resume_compat.py', 'w2v_v33/test_resume_compat.py'))

# Final release hashes are pinned after the I/O patch is frozen. The executing
# validator checks its own file separately, avoiding an impossible self-hash.
TARGET_HASHES = {'w2v_v33/data.py': '119a36784be1f8a889cd775fa586de9d3caf4a9a66b11e5da2b9334ec6cac9af', 'w2v_v33/train.py': '875333582ef3627215352f6af958f80f8f1466f5d9be0decb52b8b810e42edd8', 'w2v_v33/transport.py': 'e7192d1283ddb093611bf708e47189025cd6abe99699cb5579fe79fdd8a2c181', 'w2v_v33/test_transport.py': 'f3709f918e9fadcd23963680f70dfabc17f6a803f7f2621bd11fa770feb0e683', 'w2v_v33/test_resume_compat.py': '2cd7808a0d814ea3944e59f8bf45ed4355401e0c481165f4768287e2d82bd869'}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def approved_code_transition(previous, current):
    """Accept exactly the known old release -> this reviewed transport patch."""
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return False
    before = {k: v for k, v in previous.items() if k.startswith('w2v_v33/')}
    after = {k: v for k, v in current.items() if k.startswith('w2v_v33/')}
    if before != PUBLISHED_V33:
        return False
    if {k: v for k, v in previous.items() if k not in before} != {
            k: v for k, v in current.items() if k not in after}:
        return False
    pinned = dict(TARGET_HASHES)
    pinned['w2v_v33/resume_compat.py'] = sha256(Path(__file__).resolve())
    if set(pinned) != CHANGED_FILES | ADDED_FILES:
        return False
    if any(len(value) != 64 for value in pinned.values()):
        return False
    expected = {**PUBLISHED_V33, **pinned}
    return after == expected


def validate_resume(state, cfg, fingerprints, codes, checkpoint, run):
    """Validate exact resume, or record the sole approved I/O migration."""
    if (state.get('schema') != SCHEMA or state.get('kind') != 'training' or
            state.get('config') != cfg or state.get('data_fingerprints') != fingerprints):
        raise ValueError(ERROR)
    old_codes = state.get('source_hashes')
    if old_codes == codes:
        return None
    if not approved_code_transition(old_codes, codes):
        raise ValueError(ERROR)
    checkpoint = Path(checkpoint).resolve()
    checkpoint_hash = sha256(checkpoint)
    changes = {name: {'before': old_codes.get(name), 'after': codes[name]}
               for name in sorted(CHANGED_FILES | ADDED_FILES)}
    record = dict(format='rtc_v33_transport_resume_compat_v1', source_release=SOURCE_RELEASE,
                  checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                  saved_tag=state.get('tag'), changed_sources=changes,
                  config_sha256=_digest(cfg), data_fingerprints_sha256=_digest(fingerprints),
                  previous_sources_sha256=_digest(old_codes), current_sources_sha256=_digest(codes),
                  checkpoint_rewritten=False, optimizer_or_training_state_modified=False,
                  note='Only CPU tensor transport is changed; all unsaved updates replay from the saved validation boundary.')
    destination = Path(run)/f'resume_transport_compat_{checkpoint_hash[:16]}.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != record:
            raise ValueError('Existing V3.3 transport resume audit differs; refusing to overwrite it')
    else:
        atomic_json(destination, record)
    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError('Resume checkpoint changed while recording transport compatibility')
    print(f'V33_TRANSPORT_RESUME_COMPAT=True source_release={SOURCE_RELEASE} '
          f'checkpoint_sha256={checkpoint_hash} audit={destination}', flush=True)
    return destination
