"""Installed fairseq2 all-depth adapter / GN-LN / fixed-scale PCL acceptance."""
from copy import deepcopy
import numpy as np
import torch


def main(device='cuda:0', backend_precision='fp16'):
    import omnilingual_asr
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    from .model import Detector, optimizer_for
    from .precision import scaler_for, unscale_and_clip, commit
    from .objective import phoneme_features, phoneme_loss
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
        cfg = dict(device=device,backend_precision=backend_precision,encoder_dim=64,lora_layers=4,lora_rank=4,lora_alpha=8.,lora_dropout=.05,
            local_evidence=True,window=128,hop=64,window_batch=8,checkpointing=True,
            lora_lr=2e-5,head_lr=1e-4,evidence_lr=1e-4,fusion_lr=3e-4,layer_mix_lr=1e-3,
            weight_decay=.01,classifier_lr=1e-4,pool_lr=1e-4,pooling='asp',whole_max_frames=3000,
            fusion_layers=[1,2,3,4],fusion_dim=128,lora_early_lr=5e-6,lora_middle_lr=1e-5,
            warm_epochs=1,checkpoint_layers=4,fusion_checkpointing=True,temporal_stats=True)
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
        if any(isinstance(module,torch.nn.modules.batchnorm._BatchNorm) for module in model.head.modules()):
            raise RuntimeError('BatchNorm remains in the schema-v2 detector')
        if len(model.lora_inventory)!=4:
            raise RuntimeError('Native all-depth adapter inventory failed')
        optimizer = optimizer_for(model,cfg)
        scaler = scaler_for(optimizer,cfg)
        compute_dtypes=[]
        handle=model.head.aasist.out_layer.register_forward_hook(lambda _m,_a,y:compute_dtypes.append(y.dtype))
        expected_lora={'lora_early':5e-6,'lora_middle':1e-5,'lora_late':2e-5}
        for name,rate in expected_lora.items():
            groups=[g for g in optimizer.param_groups if g['name']==name]
            if not groups or any(g.get('role')!='lora' or g['lr']!=rate for g in groups):
                raise RuntimeError('Native depth-dependent optimizer group failed: '+name)
        model.set_phase(False);model.train()
        warm_sequences=model.encode(waves[:2])
        warm_logits=model.classify_sequences(warm_sequences)
        phone_spans=[[0,4000],[4000,10000],[10000,16000]]
        for index,features in enumerate(warm_sequences):
            model.pcl_scale.observe(phoneme_features(features,phone_spans),f'native_train_{index}')
        scaler.scale(torch.nn.functional.cross_entropy(warm_logits,labels)).backward()
        unscale_and_clip(optimizer,scaler,1.)
        if any(p.grad is not None for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b'))):
            raise RuntimeError('Head warmup unexpectedly updated LoRA')
        if commit(optimizer,scaler)['skipped_update']:raise RuntimeError('Native FP16 warmup gradient overflow')
        optimizer.zero_grad(set_to_none=True)
        model.pcl_scale.freeze()
        if not model.pcl_scale.is_frozen or not bool((model.pcl_scale.scale>0).all()):
            raise RuntimeError('Native PCL Train-scale lifecycle failed')
        del warm_sequences,warm_logits
        model.set_phase(True); model.train()
        from w2v_v3183r2.model import RecomputedLayer
        if sum(isinstance(layer,RecomputedLayer) and layer.enabled for layer in model.encoder.layers)!=4:
            raise RuntimeError('All-depth activation checkpointing is not enabled')
        optimizer.zero_grad(set_to_none=True)
        sequences=model.encode(waves)
        # The third received-audio view is an auxiliary clean RTC reference;
        # it shares this single encode but never reaches AASIST/classification.
        logits=model.classify_sequences(sequences[:2])
        if logits.shape!=(2,2) or model.head.aasist.out_layer.in_features!=384:
            raise RuntimeError('Complete ASP plus temporal-variation classifier geometry failed')
        if any(len(model.head.window_plan(len(seq),seq.device)[0])<2 for seq in sequences[:2]):
            raise RuntimeError('Native window acceptance did not exercise overlapping windows')
        pcl,count=phoneme_loss(sequences[0],sequences[2],[[0,4000,0,4000],[4000,10000,4000,10000]],
            scale=model.pcl_scale.scale)
        if count!=2:raise RuntimeError('Native PCL feature geometry failed')
        gradient=torch.autograd.grad(pcl,model.fusion.projection.weight,retain_graph=True)[0]
        if not torch.isfinite(gradient).all() or not gradient.abs().sum()>0:
            raise RuntimeError('Missing native PCL reader gradient')
        scaler.scale(torch.nn.functional.cross_entropy(logits,labels)+.15*pcl).backward()
        unscale_and_clip(optimizer,scaler,1.)
        for name,p in model.fusion.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum()>0:
                raise RuntimeError('Missing/nonfinite full-layer reader gradient: '+name)
        for index,layer in enumerate(model.encoder.layers):
            adapters=[(name,p) for name,p in layer.named_parameters() if name.endswith(('.lora_a','.lora_b'))]
            if len(adapters)!=8:
                raise RuntimeError('Incomplete native adapter projection inventory: '+str(index))
            for name,p in adapters:
                if p.grad is None or not bool(torch.isfinite(p.grad).all()) or not bool(p.grad.abs().sum()>0):
                    raise RuntimeError(f'Missing/nonfinite native adapter gradient: layer {index+1} {name}')
        for group_name in ('evidence','pooling','classifier','aasist'):
            gradients=[p.grad for group in optimizer.param_groups if group['name']==group_name
                       for p in group['params'] if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not sum(g.abs().sum() for g in gradients)>0:
                raise RuntimeError('Missing/nonfinite window head gradient: '+group_name)
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen encoder received gradients')
        if any(layer._forward_hooks for layer in model.encoder.layers):
            raise RuntimeError('Layer reader hook leak')
        if commit(optimizer,scaler)['skipped_update']:raise RuntimeError('Native FP16 joint gradient overflow')
        handle.remove()
        expected=torch.float16 if target_device.type=='cuda' and backend_precision=='fp16' else torch.float32
        if not compute_dtypes or any(value!=expected for value in compute_dtypes):
            raise RuntimeError('Backend compute dtype differs from requested precision')
        state = partial_state(model)
        if not any(name.startswith('pcl_scale.') for name in state):
            raise RuntimeError('Native checkpoint omitted PCL scale buffers')
        with torch.no_grad():
            model.pcl_scale.frozen.fill_(False);model.pcl_scale.scale.fill_(99.)
        apply_partial(model,state)
        if not model.pcl_scale.is_frozen:
            raise RuntimeError('Native checkpoint did not restore frozen PCL scale')
        torch.testing.assert_close(model.pcl_scale.scale,state['pcl_scale.scale'].to(target_device))
        print('V3189_NATIVE_INTERFACE_OK=True; head warmup / all-depth checkpointing and adapter gradients / GN-LN / fixed Train PCL scale and MSE backward / overlapping-window ASP plus temporal variation / full 384-d classifier / auxiliary reference excluded from CE / scaled joint updates / backend dtype / hook cleanup / scale-state restore; small random model',flush=True)
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
