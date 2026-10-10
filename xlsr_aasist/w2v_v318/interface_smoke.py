"""Actual installed fairseq2 factory/load/LoRA/optimizer check without public weights."""
from copy import deepcopy
from pathlib import Path
import tempfile
import numpy as np
import torch


def main(device='cpu'):
    import omnilingual_asr
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    from .model import Detector,optimizer_for
    from .common import partial_state,apply_partial,seed_all
    seed_all(31801)
    hub=get_wav2vec2_model_hub()
    for name,shape in (('1b',(48,1280)),('3b',(60,2048))):
        ec=hub.get_arch_config(name).encoder_config
        if (ec.num_encoder_layers,ec.model_dim)!=shape:raise ValueError('Unexpected '+name+' architecture')
    arch=deepcopy(hub.get_arch_config('3b'));ec=arch.encoder_config
    ec.model_dim=64;ec.num_encoder_layers=2;ec.ffn_inner_dim=128;ec.num_encoder_attn_heads=4
    ec.feature_dim=32;ec.feature_extractor_layer_descs=[(32,k,s) for _,k,s in ec.feature_extractor_layer_descs]
    ec.pos_conv_kernel_size=16;ec.num_pos_conv_groups=4
    arch.quantized_dim=32;arch.final_dim=32;arch.num_codebook_entries=8
    cfg=dict(encoder_dim=64,lora_layers=1,lora_rank=4,lora_alpha=8.,lora_dropout=.05,
        local_evidence=True,window=128,hop=64,window_batch=8,checkpointing=True,
        lora_lr=1e-5,head_lr=3e-5,evidence_lr=1e-4,weight_decay=.01)
    ssl=hub.create_new_model(arch,device=torch.device('cpu'),dtype=torch.float32)
    # Exercise the same memory-mapped safe loader as the public checkpoint path.
    with tempfile.TemporaryDirectory(prefix='v318_interface_') as d:
        path=Path(d)/'tiny.pt';torch.save({'model':ssl.state_dict()},path)
        target_device=torch.device(device)
        dtype=torch.bfloat16 if target_device.type=='cuda' else torch.float32
        ssl=hub.load_custom_model(path,arch,device=target_device,dtype=dtype,mmap=True,restrict=True)
        model=Detector(ssl.encoder_frontend,ssl.encoder,cfg,BatchLayout).to(target_device)
        waves=[np.random.default_rng(i).normal(0,.05,n).astype(np.float32) for i,n in enumerate((16000,20000))]
        target=torch.tensor([0,1],device=target_device);optimizer=optimizer_for(model,cfg)
        for joint in (False,True):
            model.set_phase(joint);model.train();optimizer.zero_grad(set_to_none=True)
            logits=model(waves);torch.nn.functional.cross_entropy(logits,target).backward()
            lora=[p for n,p in model.named_parameters() if n.endswith('.lora_b')]
            if joint and not any(p.grad is not None and p.grad.abs().sum()>0 for p in lora):raise RuntimeError('No LoRA gradient')
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise RuntimeError('Frozen parameter gradient')
            optimizer.step()
        state=partial_state(model);apply_partial(model,state)
        model.eval()
        with torch.no_grad():
            a=model(waves);b=model(waves)
        if not torch.isfinite(a).all() or not torch.allclose(a,b,rtol=1e-5,atol=1e-5):raise RuntimeError('Invalid repeated eval')
    print(f'V318_REAL_FAIRSEQ_INTERFACE_OK=True; device={device}; factory/load/warmup/LoRA/Adam/partial/eval; small random model')


if __name__=='__main__':main()
