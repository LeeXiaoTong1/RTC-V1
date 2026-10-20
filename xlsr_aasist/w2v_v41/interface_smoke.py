"""Small real HF model smoke for the deployed precision and dual-PCL update."""
import argparse
import numpy as np
import torch
from .common import capture_rng,restore_rng,partial_state,apply_partial


def main(device='cuda:0',backend_precision='fp16'):
    from transformers import Wav2Vec2BertConfig,Wav2Vec2BertModel,SeamlessM4TFeatureExtractor
    from .model import Detector,optimizer_for,adapter_inventory
    from .step import train_step
    from .calibration import calibrate
    rng=capture_rng();threads=torch.get_num_threads()
    if device.startswith('cpu'):torch.set_num_threads(1)
    try:
        native=Wav2Vec2BertConfig(hidden_size=32,num_hidden_layers=4,num_attention_heads=4,
            intermediate_size=64,feature_projection_input_dim=160,conv_depthwise_kernel_size=3,
            position_embeddings_type='relative_key',left_max_position_embeddings=8,right_max_position_embeddings=4,
            add_adapter=False,apply_spec_augment=False)
        cfg=dict(device=device,backend_precision=backend_precision,encoder_dim=32,encoder_layers=4,lora_layers=4,
            lora_rank=4,lora_alpha=8.,lora_dropout=.05,mlp_dims=[48,32,16],classifier_dropout=.1,warm_epochs=1,
            checkpointing=True,checkpoint_layers=4,fusion_checkpointing=True,whole_max_frames=3000,
            lora_lr=2e-4,head_lr=1e-3,classifier_lr=1e-3,layer_mix_lr=1e-3,weight_decay=.01,
            microbatch=4,frame_budget=2000,label_smoothing=.02,grad_clip=1.,consistency='dual',
            pcl_ramp_updates=1,pcl_gradient_target=.05,pcl_max_weight=1.,gradient_coordination='pcgrad')
        backbone=Wav2Vec2BertModel(native)
        if device.startswith('cuda'):backbone=backbone.to(dtype=torch.bfloat16)
        model=Detector(backbone,cfg,SeamlessM4TFeatureExtractor()).to(device)
        optimizer=optimizer_for(model,cfg)
        frozen={n:p.detach().clone() for n,p in model.backbone.named_parameters() if not n.endswith(('.lora_a','.lora_b'))}
        units=[]
        for role in ('online','noisy'):
            for lang in ('en','zh'):
                for label in (0,1):
                    i=len(units);wave=np.random.default_rng(i+1).normal(0,.05,7000+i*80).astype(np.float32)
                    t=np.arange(len(wave),dtype=np.float32)/16000.
                    source=(wave+.03*np.sin(2*np.pi*330*t)*np.sin(2*np.pi*3*t)).astype(np.float32)
                    row=dict(wave=wave,role=role,language=lang,label=label,split='train',canonical_source=str(i),original_samples=len(wave))
                    row['pcl']=dict(kind='official' if role=='online' else 'noise',reference=dict(row,wave=source,role='offline_ref'),
                        received=dict(row,role='online_ref') if role=='online' else None,reuse_received=True,
                        matches=[[0,len(wave),0,len(wave)]])
                    units.append([row])
        def update(epoch,weights):
            for _ in range(4):
                result=train_step(model,optimizer,units,cfg,weights,epoch,0,2)
                if not result['amp']['skipped_update']:return result
            raise RuntimeError('Native V4 smoke repeatedly overflowed')
        model.set_phase(False);model.train();warm=update(0,dict(official=0.,noise=0.))
        if any(bool(p.count_nonzero()) for n,p in model.named_parameters() if n.endswith('.lora_b')):
            raise RuntimeError('Head warmup changed frozen LoRA')
        model.set_phase(True);weights=calibrate(model,[units],cfg)['weights'];model.train();joint=update(1,weights)
        if joint['pcl_pairs']!={'official':4,'noise':4} or min(joint['pcl_gradient_norms'].values())<=0:
            raise RuntimeError('Both PCL branches must contribute actual gradients')
        if not any(bool(p.count_nonzero()) for n,p in model.named_parameters() if n.endswith('.lora_b')):
            raise RuntimeError('Joint training did not update LoRA')
        for n,p in model.backbone.named_parameters():
            if n in frozen and (not torch.equal(p,frozen[n]) or p.grad is not None):raise RuntimeError('Frozen official weights changed')
        state=partial_state(model);apply_partial(model,state)
        print('V4_NATIVE_INTERFACE_OK: real small HF Conformer; warmup, both PCL branches, PCGrad, LoRA gradients and checkpoint restore. Not a full-model GPU memory benchmark.',flush=True)
        return dict(inventory=adapter_inventory(model),warm=warm,joint=joint)
    finally:
        restore_rng(rng)
        if device.startswith('cpu'):torch.set_num_threads(threads)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--backend-precision',choices=('fp16','fp32'),default='fp16')
    main(**vars(p.parse_args()))
