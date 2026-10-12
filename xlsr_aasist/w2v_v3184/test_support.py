"""Synthetic CPU fixtures; native Omni/CUDA acceptance is a separate requirement."""
import copy
import numpy as np
from w2v_v318.test_core import TinyFrontend,TinyEncoder,TinyLayout,NativeBoundary,tiny_cfg as legacy_cfg
from w2v_v3182.test_support import metadata
from .model import Detector


def tiny_cfg():
    cfg=legacy_cfg()
    cfg.update(version='3.18.4',warm_epochs=1,epochs=2,group_quotas=[1,1,1,1],fusion_layers=[1,2],
               fusion_dim=128,fusion_lr=.001,layer_mix_lr=.003,lr_warmup_updates=2,lr_floor=.1,
               warm_lr=.001,warm_fusion_lr=.001,warm_layer_mix_lr=.003,warm_evidence_lr=.001,
               bn_momentum=.01,init_mode='parent-transfer',
               rtc_stress_probability=.35,rtc_stress_ramp_epochs=2.)
    cfg.update(scene_probability=.8,rtc_policy='legacy',scene_panel_per_group=1)
    return cfg


def scene_fixture(root):
    """Distinct recording and PCM identities, all five categories in both splits."""
    import hashlib
    import soundfile as sf
    from .noise import RECORDED,sha256,SCHEMA,bind_manifest
    from .common import atomic_json
    root.mkdir(parents=True,exist_ok=True);rows=[]
    for split in ('train','dev'):
        for category in RECORDED:
            key=f'{split}-{category}';index=len(rows)
            pcm=np.round(np.random.default_rng(index+81).normal(0,500,16000)).astype('<i2')
            p=root/(key+'.wav');sf.write(p,pcm,16000,subtype='PCM_16')
            rows.append(dict(key=key,file=p.name,sha256=sha256(p),pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                category=category,split=split,original_sources=[key],source_group=key,license='synthetic fixture'))
    manifest=root/'manifest.json'
    atomic_json(manifest,dict(schema=SCHEMA,status='complete',records=rows,summary={},provenance='test fixture'))
    return bind_manifest(manifest)


def tiny_model(cfg):return Detector(TinyFrontend(),TinyEncoder(),cfg,TinyLayout)


def units(plan,epoch=0,samples=8500):
    values=[]
    for t in next(plan.batches(epoch)):
        r=plan.rows[t['index']];roles=('online',) if t['kind']=='online' else ('noisy_a','noisy_b')
        values.append([dict(r,role=role,condition=role,occurrence=t['occurrence'],
            wave=np.random.default_rng(t['index']*2+v).normal(0,.04,samples).astype(np.float32)) for v,role in enumerate(roles)])
    return values
