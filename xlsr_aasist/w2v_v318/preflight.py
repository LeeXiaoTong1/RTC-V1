"""Validate native ABI and optionally run the selected real SSL/LoRA smoke test."""
import argparse
from importlib.metadata import version
import numpy as np
import torch
from .common import read_json,digest,seed_all
from .assets import DEFAULT_ARCH,MODELS,spec,default_assets,validate_assets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--omni-size',choices=tuple(MODELS),default=DEFAULT_ARCH)
    p.add_argument('--assets',help='Matching assets manifest; default follows --omni-size')
    p.add_argument('--weights',action='store_true',help='Load selected public SSL weights and run one forward/backward')
    args=p.parse_args()
    item=spec(args.omni_size)
    expected={'fairseq2':'0.6.0','omnilingual-asr':'0.2.0','webrtc-audio-processing':'0.1.3'}
    for name,want in expected.items():
        if version(name)!=want:raise RuntimeError(f'{name} must be {want}')
    if torch.__version__.split('+')[0]!='2.8.0':raise RuntimeError('torch 2.8.0 required by fairseq2n ABI')
    import omnilingual_asr
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    arch=get_wav2vec2_model_hub().get_arch_config(item['arch'])
    ec=arch.encoder_config
    if (ec.num_encoder_layers,ec.model_dim)!=(item['encoder_layers'],item['encoder_dim']):raise ValueError('Unexpected '+item['name']+' architecture')
    from .augment import Engines,recipe
    engine=Engines();seed_all(31801)
    wave=(.08*np.sin(np.arange(16000)*2*np.pi*240/16000)).astype(np.float32)
    for family in ('ffmpeg','webrtc','light','bypass','g711_mulaw','anlmdn'):
        r=recipe(1,'preflight','public-synthetic',0,0,family=family)
        output=engine(wave,r)
        if output.shape!=wave.shape or not np.isfinite(output).all():raise ValueError('Invalid '+family)
    print(f'V318_RUNTIME_OK=True; native processing engines passed; {item["name"]} architecture={ec.num_encoder_layers}x{ec.model_dim}',flush=True)
    if not args.weights:return
    assets=read_json(args.assets or default_assets(item['arch']))
    validate_assets(assets,item['arch'])
    if digest(assets['checkpoint'])!=item['sha256']:raise ValueError('SSL asset hash mismatch')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA/BF16 unavailable')
    from .model import load_model
    cfg=dict(omni_checkpoint=assets['checkpoint'],omni_sha256=assets['sha256'],omni_provenance=assets,
        omni_arch=item['arch'],encoder_dim=item['encoder_dim'],encoder_layers=item['encoder_layers'],device='cuda:0',lora_layers=16,lora_rank=16,
        lora_alpha=32.,lora_dropout=.05,local_evidence=True,window=128,hop=64,window_batch=32,checkpointing=True)
    model=load_model(cfg);model.set_phase(True);model.train()
    logits=model([np.tile(wave,2),np.tile(wave,3)])
    torch.nn.functional.cross_entropy(logits,torch.tensor([0,1],device='cuda')).backward()
    lora=[p for n,p in model.named_parameters() if n.endswith('.lora_b')]
    if not any(p.grad is not None and p.grad.abs().sum()>0 for p in lora):raise RuntimeError('No gradient reaches LoRA')
    if any(p.grad is not None for n,p in model.encoder.named_parameters() if '.base.' in n):raise RuntimeError('Frozen base received gradients')
    print(f'V318_REAL_{item["arch"].upper()}_SMOKE_OK=True; LoRA gradient reached; peak_allocated_GiB='+f'{torch.cuda.max_memory_allocated()/1024**3:.2f}')


if __name__=='__main__':main()
