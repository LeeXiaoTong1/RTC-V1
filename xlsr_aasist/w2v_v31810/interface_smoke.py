"""Native server acceptance for the actual fresh detector and MetaDG update."""
from copy import deepcopy
import numpy as np
import torch


def main(device='cuda:0',backend_precision='fp16'):
    import omnilingual_asr
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    from .model import Detector,optimizer_for
    from .meta import train_step
    from .common import capture_rng,restore_rng,partial_state,apply_partial
    rng=capture_rng()
    try:
        hub=get_wav2vec2_model_hub();arch=deepcopy(hub.get_arch_config('3b'));ec=arch.encoder_config
        if (ec.num_encoder_layers,ec.model_dim)!=(60,2048):raise ValueError('Unexpected official Omni3B architecture')
        ec.model_dim=64;ec.num_encoder_layers=4;ec.ffn_inner_dim=128;ec.num_encoder_attn_heads=4
        ec.feature_dim=32;ec.feature_extractor_layer_descs=[(32,k,s) for _,k,s in ec.feature_extractor_layer_descs]
        ec.pos_conv_kernel_size=16;ec.num_pos_conv_groups=4
        arch.quantized_dim=32;arch.final_dim=32;arch.num_codebook_entries=8
        cfg=dict(device=device,backend_precision=backend_precision,encoder_dim=64,lora_layers=4,lora_rank=4,lora_alpha=8.,lora_dropout=.05,
            local_evidence=True,window=128,hop=64,window_batch=8,checkpointing=True,pooling='asp',whole_max_frames=3000,
            lora_lr=2e-5,lora_early_lr=5e-6,lora_middle_lr=1e-5,head_lr=1e-4,evidence_lr=1e-4,fusion_lr=1e-4,
            layer_mix_lr=1e-4,weight_decay=1e-4,classifier_lr=1e-4,pool_lr=1e-4,fusion_layers=[1,2,3,4],fusion_dim=128,
            warm_epochs=1,checkpoint_layers=4,fusion_checkpointing=True,temporal_stats=True,microbatch=4,frame_budget=2000,
            label_smoothing=.02,max_grad_norm=1.,inner_grad_norm=1.,consistency='none')
        dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32
        ssl=hub.create_new_model(arch,device=torch.device(device),dtype=dtype)
        model=Detector(ssl.encoder_frontend,ssl.encoder,cfg,BatchLayout).to(device);del ssl
        opt=optimizer_for(model,cfg);units=[]
        for side in ('A','B'):
            for role in ('online','noisy'):
                for language,label in (('en',0),('en',1),('zh',0),('zh',1)):
                    identity=f'{side}-{role}-{language}-{label}'
                    units.append([dict(wave=np.random.default_rng(len(units)).normal(0,.05,48000).astype(np.float32),
                        split='train',role=role,language=language,label=label,source_id=identity,group_id=identity,
                        canonical_source=identity,meta_side=side,task_kind='joint')])
        seen=[];hook=model.head.aasist.out_layer.register_forward_hook(lambda _m,_a,y:seen.append(y.dtype))
        def accepted_step(**kwargs):
            for _ in range(4):
                result=train_step(model,opt,units,cfg,**kwargs)
                if not result.get('amp',{}).get('skipped_update'):return result
            raise RuntimeError('Native smoke repeatedly overflowed after scaler backoff')
        warm=accepted_step(alpha=0.,beta=0.,joint=False,audit=False)
        joint=accepted_step(alpha=.003,beta=.5,joint=True,audit=True)
        hook.remove()
        expected=torch.float16 if device.startswith('cuda') and backend_precision=='fp16' else torch.float32
        if not seen or any(v!=expected for v in seen):raise RuntimeError('Backend native precision acceptance failed')
        if any(result.get('amp',{}).get('skipped_update') for result in (warm,joint)):raise RuntimeError('Native smoke gradient overflow')
        if hasattr(model,'pcl_scale') or any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.head.modules()):raise RuntimeError('Unexpected PCL or BatchNorm state')
        state=partial_state(model);apply_partial(model,state)
        if any(layer._forward_hooks for layer in model.encoder.layers):raise RuntimeError('Encoder hook leak')
        print('V31810_NATIVE_INTERFACE_OK=True; official small architecture / FP16 head / fresh CE warmup / source-isolated first-order joint step / checkpoint state; not a 3B memory benchmark',flush=True)
    finally:restore_rng(rng)
    if device.startswith('cuda'):
        del model,opt,state,units
        torch.cuda.empty_cache()
