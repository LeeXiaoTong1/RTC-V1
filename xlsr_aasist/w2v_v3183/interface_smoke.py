"""Installed fairseq2 hook/checkpoint/fusion acceptance before loading public 3B."""
from copy import deepcopy
import numpy as np
import torch


def main(device='cuda:0'):
    import omnilingual_asr
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    from .model import Detector,optimizer_for
    from .common import partial_state,apply_partial,capture_rng,restore_rng
    rng=capture_rng()
    try:
        hub=get_wav2vec2_model_hub();arch=deepcopy(hub.get_arch_config('3b'));ec=arch.encoder_config
        if (ec.num_encoder_layers,ec.model_dim)!=(60,2048):raise ValueError('Unexpected Omni-3B architecture')
        ec.model_dim=64;ec.num_encoder_layers=4;ec.ffn_inner_dim=128;ec.num_encoder_attn_heads=4
        ec.feature_dim=32;ec.feature_extractor_layer_descs=[(32,k,s) for _,k,s in ec.feature_extractor_layer_descs]
        ec.pos_conv_kernel_size=16;ec.num_pos_conv_groups=4
        arch.quantized_dim=32;arch.final_dim=32;arch.num_codebook_entries=8
        cfg=dict(encoder_dim=64,lora_layers=2,lora_rank=4,lora_alpha=8.,lora_dropout=.05,
            local_evidence=True,window=128,hop=64,window_batch=8,checkpointing=True,
            lora_lr=2e-6,head_lr=6e-6,evidence_lr=1e-5,fusion_lr=1e-4,weight_decay=.01,
            fusion_layers=[1,2,3],fusion_max_residual=.25)
        target_device=torch.device(device);dtype=torch.bfloat16 if target_device.type=='cuda' else torch.float32
        ssl=hub.create_new_model(arch,device=target_device,dtype=dtype)
        model=Detector(ssl.encoder_frontend,ssl.encoder,cfg,BatchLayout).to(target_device)
        del ssl
        waves=[np.random.default_rng(i).normal(0,.05,n).astype(np.float32) for i,n in enumerate((16000,20000))]
        model.set_phase(True);model.eval()
        with torch.no_grad():
            enabled=model(waves);layers=model.fusion.layers;model.fusion.layers=()
            legacy=model(waves);model.fusion.layers=layers
            torch.testing.assert_close(enabled,legacy,rtol=0,atol=0)
        model.train();optimizer=optimizer_for(model,cfg)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True);logits=model(waves)
            torch.nn.functional.cross_entropy(logits,torch.tensor([0,1],device=target_device)).backward()
            if model.fusion.gate.grad is None or model.fusion.gate.grad.abs()==0:raise RuntimeError('No fusion gate gradient')
            if not any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.endswith('.lora_b')):
                raise RuntimeError('No LoRA gradient through the hooked/checkpointed native encoder')
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise RuntimeError('Frozen encoder received gradients')
            if any(layer._forward_hooks for layer in model.encoder.layers):raise RuntimeError('Layer hook leak')
            optimizer.step()
        state=partial_state(model);apply_partial(model,state)
        print('V3183_NATIVE_INTERFACE_OK=True; zero-gate identity / hooks / checkpoint gradients / optimizer / state; small random model',flush=True)
    finally:
        restore_rng(rng)
    if target_device.type=='cuda':
        del model,optimizer,state,logits,enabled,legacy
        torch.cuda.empty_cache()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:0')
    main(p.parse_args().device)
