"""Reuse complete verified hidden vectors without copying or mutating V3.6."""
import json
from pathlib import Path

from w2v_aasist.runtime import sha256, atomic_json
from w2v_v36.features import load_cache, extract_cache, _rows, _digest


def reusable_cache(feature_run, split, records, cfg, fingerprints):
    if not feature_run:
        return None
    path = Path(feature_run).expanduser().resolve()/'features'/split
    if not (path/'complete.json').is_file():
        print(f'V37_FEATURE_REUSE {split}: no complete split; extract into this V3.7 run',flush=True)
        return None
    owner = json.loads((path/'owner.json').read_text(encoding='utf-8'))
    identity = owner['identity']
    if (identity.get('base_checkpoint_sha256') != cfg['base_checkpoint_sha256']
            or identity.get('split') != split or identity.get('data_fingerprints') != fingerprints):
        raise ValueError('Borrowed features belong to a different model/data split')
    rows = _rows(records)
    if _digest(rows) != owner['records_digest']:
        raise ValueError('Borrowed feature row identities/order differ')
    if owner['preprocessing_sha256'] != sha256(Path(cfg['ssl_path'])/'preprocessor_config.json'):
        raise ValueError('Borrowed feature preprocessing differs')
    recorded = identity.get('code_fingerprints',{})
    # Every module that defined V3.6's frozen forward must still be identical.
    # Classifier fitting/report code may evolve without changing these vectors.
    required = [('w2v_v36','features.py'),('w2v_v3','model.py'),('w2v_v3','data.py'),
                ('w2v_v3','step.py'),('w2v_aasist','model.py'),('w2v_aasist','data.py')]
    for folder,name in required:
        entries = [(p,d) for p,d in recorded.items() if Path(p).parent.name==folder and Path(p).name==name]
        if len(entries)!=1 or not Path(entries[0][0]).is_file() or sha256(entries[0][0])!=entries[0][1]:
            raise ValueError('Borrowed feature producer changed: '+folder+'/'+name)
    bundle = load_cache(path)
    print(f'V37_FEATURE_REUSE {split}: {len(rows)} verified rows from {path}',flush=True)
    bundle['reused_path'] = str(path)
    return bundle


def record_reuse(run, split, bundle):
    path = Path(run)/'feature_reuse.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    source = bundle.get('reused_path')
    data[split] = dict(path=source,borrowed_read_only=bool(source),
                      rows=len(bundle['rows']),records_digest=bundle['manifest']['records_digest'])
    atomic_json(path,data)
