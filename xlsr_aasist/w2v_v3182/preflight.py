"""Native runtime checks plus an optional real-Omni three-view training update."""
import argparse
import hashlib
import sys
from unittest.mock import patch
import numpy as np
import torch
from .assets import DEFAULT_ARCH, MODELS, spec, default_assets, validate_assets
from .common import GROUPS, read_json, seed_all, partial_state, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--omni-size', choices=tuple(MODELS), default=DEFAULT_ARCH)
    p.add_argument('--assets')
    p.add_argument('--weights', action='store_true')
    args = p.parse_args()
    from w2v_v318.preflight import main as native_check
    with patch.object(sys, 'argv', ['native-check', '--omni-size', args.omni_size]):
        native_check()
    if not args.weights:
        return
    from w2v_v318.environment import check_cuda
    from .model import load_model, optimizer_for
    from .step import train_step
    check_cuda()
    item = spec(args.omni_size)
    assets = read_json(args.assets or default_assets(args.omni_size))
    validate_assets(assets,args.omni_size)
    if digest(assets['checkpoint']) != item['sha256']:
        raise ValueError('Public Omni checkpoint hash mismatch')
    seed_all(31801)
    cfg = dict(omni_checkpoint=assets['checkpoint'],omni_sha256=assets['sha256'],omni_provenance=assets,
        omni_arch=args.omni_size,encoder_dim=item['encoder_dim'],encoder_layers=item['encoder_layers'],
        device='cuda:0',lora_layers=16,lora_rank=16,lora_alpha=32.,lora_dropout=.05,
        local_evidence=True,window=128,hop=64,window_batch=32,checkpointing=True,
        lora_lr=1e-5,head_lr=3e-5,evidence_lr=1e-4,weight_decay=.01,stream_sources=4,
        microbatch=item['microbatch'],frame_budget=item['frame_budget'],label_smoothing=.02,
        contrastive_temperature=.1,negative_mass=4.,max_grad_norm=1.)
    units = []
    for i,(language,label) in enumerate(GROUPS):
        identity=hashlib.sha256(f'synthetic-{i}'.encode()).hexdigest()
        wave=(.1*np.sin(np.arange(16000)*(i+2)*.01)).astype(np.float32)
        units.append([dict(wave=wave+np.random.default_rng(10*i+j).normal(0,.001,len(wave)).astype(np.float32),
            source_id=identity,source_sha256=identity,group_id=identity,language=language,label=label,
            occurrence=identity,source_mass=.25,ce_weight=.25*w,split='train',role=role)
            for j,(role,w) in enumerate(zip(('offline','online','noisy_a'),(.1,.45,.45)))])
    model=load_model(cfg); optimizer=optimizer_for(model,cfg)
    model.set_phase(False); train_step(model,optimizer,units,cfg,0.)
    before=partial_state(model)
    model.set_phase(True); stats=train_step(model,optimizer,units,cfg,.05)
    after=partial_state(model)
    changed=[n for n in before if n.endswith('.lora_b') and not torch.equal(before[n],after[n])]
    if not changed:
        raise RuntimeError('No LoRA B update in the real three-view step')
    print('V3182_REAL_SMOKE_OK=True; synthetic waves only; no dataset performance claim',flush=True)
    print({k:stats[k] for k in ('loss','ce_objective','pair_loss','candidate_sources','replay_max_abs_error')},flush=True)
    print('PEAK_GPU_GIB='+str(torch.cuda.max_memory_allocated()/1024**3),flush=True)


if __name__=='__main__': main()
