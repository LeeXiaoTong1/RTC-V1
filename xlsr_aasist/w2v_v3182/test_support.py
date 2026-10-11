"""Synthetic fixtures; never a substitute for the real Omni/native acceptance check."""
import hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from w2v_v318.test_core import (tiny_cfg as old_cfg, make_fixture as old_fixture,
    TinyFrontend, TinyEncoder, TinyLayout, NativeBoundary)
from .common import digest
from .model import Detector


def tiny_cfg():
    cfg = old_cfg()
    cfg.update(version='3.18.2', variant='C1', source_sampling_power=.5, contrastive_weight=.05,
        contrastive_temperature=.1, contrastive_ramp_epochs=1., negative_mass=4.,
        view_ce_weights=[.1,.45,.45], group_loss_mass=.25, risk_coefficient=0.,
        microbatch=4, epochs=2, warm_epochs=1)
    return cfg


def tiny_model(cfg):
    return Detector(TinyFrontend(), TinyEncoder(), cfg, TinyLayout)


def metadata(counts=(7,3,11,5), missing=True):
    rows=[]
    for (lang,label), count in zip((('en',0),('en',1),('zh',0),('zh',1)),counts):
        for i in range(count):
            name=f'{lang}-{label}-{i}'
            sha=hashlib.sha256(name.encode()).hexdigest()
            for role in ('offline','online'):
                if role=='online' and missing and i==0: continue
                rows.append(dict(id=role+'/'+name,source_id='offline/'+name,group_id=sha,source_sha256=sha,
                    audio_sha256=sha if role=='offline' else hashlib.sha256(('online'+name).encode()).hexdigest(),
                    condition=role,language=lang,label=label,split='train',full_length=True))
    return rows


def units_from_plan(plan, epoch=0, samples=8500):
    units=[]
    for t in next(plan.batches(epoch)):
        base=plan.rows[t['index']]
        roles=('offline','online','noisy_a') if t['online_index'] is not None else ('offline','noisy_a','noisy_b')
        units.append([dict(base,role=role,condition=role,occurrence=t['occurrence'],source_mass=t['source_mass'],
            ce_weight=w*t['source_mass'],wave=np.random.default_rng(int(base['group_id'][:8],16)+v).normal(0,.05,samples).astype(np.float32))
            for v,(role,w) in enumerate(zip(roles,(.1,.45,.45)))])
    return units


def make_fixture(root):
    prior,train,dev=old_fixture(root)
    cfg=tiny_cfg()
    # Retain protocol/noise fixture paths, but bind training identities to actual
    # distinct waveform hashes and explicit Offline--Online relationships.
    for k in ('dev_pairs','official_dev_protocol','official_dev_root','noise_records'):
        cfg[k]=prior[k]
    sources={}
    for r in train:
        key=r['group_id'].removeprefix('online:')
        source_seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
        wave=np.random.default_rng(source_seed).normal(0,.04,8500).astype(np.float32)
        if r['condition']=='online': wave=np.convolve(wave,[.1,.8,.1],mode='same').astype(np.float32)
        sf.write(r['audio'],wave,16000,subtype='FLOAT');st=Path(r['audio']).stat()
        r.update(audio_sha256=digest(r['audio']),audio_size=st.st_size,audio_mtime_ns=st.st_mtime_ns)
        sources.setdefault(key,{})[r['condition']]=r
    for pair in sources.values():
        original=pair['offline']
        for row in pair.values():
            row.update(source_id=original['id'],source_sha256=original['audio_sha256'],group_id=original['audio_sha256'])
    return cfg,train,dev
