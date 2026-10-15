"""Installed fairseq2 full-depth reader and phase acceptance on a small model."""
from copy import deepcopy
import numpy as np
import torch


def main(device='cuda:0'):
    import omnilingual_asr
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    from .model import Detector, optimizer_for
    from .objective import phoneme_loss
    from .common import partial_state, apply_partial, capture_rng, restore_rng
    rng = capture_rng()
    target_device = torch.device(device)
    try:
        hub = get_wav2vec2_model_hub()
        arch = deepcopy(hub.get_arch_config('3b')); ec = arch.encoder_config
        if (ec.num_encoder_layers, ec.model_dim) != (60, 2048):
            raise ValueError('Unexpected Omni-3B architecture')
        ec.model_dim = 64; ec.num_encoder_layers = 4
        ec.ffn_inner_dim = 128; ec.num_encoder_attn_heads = 4
        ec.feature_dim = 32
        ec.feature_extractor_layer_descs = [(32,k,s) for _,k,s in ec.feature_extractor_layer_descs]
        ec.pos_conv_kernel_size = 16; ec.num_pos_conv_groups = 4
        arch.quantized_dim = 32; arch.final_dim = 32; arch.num_codebook_entries = 8
        cfg = dict(encoder_dim=64,lora_layers=2,lora_rank=4,lora_alpha=8.,lora_dropout=.05,
            local_evidence=True,window=128,hop=64,window_batch=8,checkpointing=True,
            lora_lr=2e-5,head_lr=1e-4,evidence_lr=1e-4,fusion_lr=3e-4,layer_mix_lr=1e-3,
            weight_decay=.01,classifier_lr=1e-4,pool_lr=1e-4,pooling='asp',whole_max_frames=3000,
            fusion_layers=[1,2,3,4],fusion_dim=128,bn_momentum=.01,
            warm_epochs=1,checkpoint_layers=1,fusion_checkpointing=True)
        dtype = torch.bfloat16 if target_device.type=='cuda' else torch.float32
        ssl = hub.create_new_model(arch,device=target_device,dtype=dtype)
        model = Detector(ssl.encoder_frontend,ssl.encoder,cfg,BatchLayout).to(target_device)
        del ssl
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name.endswith('.lora_b'):p.normal_(0,.01)
            model.head.attention_delta[-1].weight.normal_(0,.01)
            model.head.aasist.out_layer.weight[:,160:].normal_(0,.01)
        waves = [np.random.default_rng(i).normal(0,.05,n).astype(np.float32)
                 for i,n in enumerate((48000,64000,48000))]
        labels = torch.tensor([0,1],device=target_device)
        optimizer = optimizer_for(model,cfg)
        model.set_phase(False);model.train()
        warm_logits=model(waves[:2])
        torch.nn.functional.cross_entropy(warm_logits,labels).backward()
        if any(p.grad is not None for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b'))):
            raise RuntimeError('Head warmup unexpectedly updated LoRA')
        optimizer.zero_grad(set_to_none=True)
        model.set_phase(True); model.train()
        optimizer.zero_grad(set_to_none=True)
        sequences=model.encode(waves)
        # The third received-audio view is an auxiliary clean RTC reference;
        # it shares this single encode but never reaches AASIST/BN/classification.
        logits=model.classify_sequences(sequences[:2])
        if logits.shape!=(2,2) or model.head.aasist.out_layer.in_features!=320:
            raise RuntimeError('Restored ASP classifier geometry failed')
        if any(len(model.head.window_plan(len(seq),seq.device)[0])<2 for seq in sequences[:2]):
            raise RuntimeError('Native window acceptance did not exercise overlapping windows')
        pcl,count=phoneme_loss(sequences[0],sequences[2],[[0,4000,0,4000],[4000,10000,4000,10000]])
        if count!=2:raise RuntimeError('Native PCL feature geometry failed')
        gradient=torch.autograd.grad(pcl,model.fusion.projection.weight,retain_graph=True)[0]
        if not torch.isfinite(gradient).all() or not gradient.abs().sum()>0:
            raise RuntimeError('Missing native PCL reader gradient')
        (torch.nn.functional.cross_entropy(logits,labels)+.15*pcl).backward()
        for name,p in model.fusion.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum()>0:
                raise RuntimeError('Missing/nonfinite full-layer reader gradient: '+name)
        if not any(p.grad is not None and p.grad.abs().sum()>0
                   for n,p in model.named_parameters() if n.endswith('.lora_b')):
            raise RuntimeError('No joint LoRA gradient through native encoder')
        for group_name in ('evidence','pooling','classifier','aasist'):
            gradients=[p.grad for group in optimizer.param_groups if group['name']==group_name
                       for p in group['params'] if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not sum(g.abs().sum() for g in gradients)>0:
                raise RuntimeError('Missing/nonfinite window head gradient: '+group_name)
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen encoder received gradients')
        if any(layer._forward_hooks for layer in model.encoder.layers):
            raise RuntimeError('Layer reader hook leak')
        optimizer.step()
        state = partial_state(model); apply_partial(model,state)
        print('V3187_NATIVE_INTERFACE_OK=True; head warmup / selective checkpointing / all-layer gradients / native PCL backward / overlapping-window ASP / full 320-d classifier / auxiliary reference excluded from CE / joint updates / hook cleanup / state; small random model',flush=True)
    finally:
        restore_rng(rng)
    if target_device.type=='cuda':
        del model,optimizer,state,logits,sequences,pcl,gradient
        torch.cuda.empty_cache()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0')
    main(p.parse_args().device)
