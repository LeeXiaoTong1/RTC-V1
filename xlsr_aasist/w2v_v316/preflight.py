"""Real fairseq2 API/gradient smoke test before downloading or training 7B."""
import argparse
import copy
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn import functional as F
from w2v_v39.common import read_json,digest
from .model import Detector,LoRALinear,load_model
from .objectives import TFCL
from .state import partial_state,apply_partial


def api_smoke():
    import omnilingual_asr
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    hub=get_wav2vec2_model_hub(); arch=hub.get_arch_config('7b')
    if arch.encoder_config.num_encoder_layers!=128 or arch.encoder_config.model_dim!=2048:
        raise ValueError('7B extension registration differs')
    small=copy.deepcopy(arch); ec=small.encoder_config
    ec.num_encoder_layers=2; ec.model_dim=64; ec.ffn_inner_dim=128; ec.num_encoder_attn_heads=4
    ssl=hub.create_new_model(small,device=torch.device('cpu'),dtype=torch.float32)
    cfg=dict(lora_layers=1,lora_rank=4,lora_alpha=8.,feature_layers=[0,1],amp='none',checkpointing=True)
    model=Detector(ssl.encoder_frontend,ssl.encoder,cfg,BatchLayout,dim=64,
        head_kwargs=dict(projection=16,expansion=64,blocks=2,dropout=0.))
    auxiliary=TFCL(16,4,21)
    rng=np.random.default_rng(316); waves=[rng.normal(0,.1,n).astype(np.float32) for n in (16000,20137)]
    model.eval()
    with torch.no_grad():
        batch=model(waves)[0]
        separate=torch.cat([model([w])[0] for w in waves])
    if not torch.allclose(batch,separate,atol=2e-4,rtol=2e-4):
        raise ValueError('Padding/exact-length frontend invariant failed')
    model.train(); logits,features,valid,lengths=model(waves)
    if lengths!=[(len(w)-400)//320+1 for w in waves]: raise ValueError('Unexpected CNN geometry')
    ta,sd=auxiliary([(features[0,:lengths[0]],features[1,:lengths[1]],valid[0,:lengths[0]],valid[1,:lengths[1]])])
    (F.cross_entropy(logits,torch.tensor([0,1]))+.15*ta.mean()+.045*sd.mean()).backward()
    grads=[m.b.grad for m in model.modules() if isinstance(m,LoRALinear)]
    if not any(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in grads):
        raise ValueError('No finite gradient reaches LoRA')
    if any(p.grad is not None for n,p in model.named_parameters() if not p.requires_grad):
        raise ValueError('Immutable base received gradients')
    state=partial_state(model); apply_partial(model,state)
    print('V316_REAL_FAIRSEQ2_API_OK='+json.dumps(dict(fairseq2=version('fairseq2'),
        omnilingual=version('omnilingual-asr'),variable_length=True,lora_gradient=True,partial_keys=len(state))),flush=True)


def full_smoke(assets):
    value=read_json(assets)
    if digest(value['checkpoint'])!=value['sha256']: raise ValueError('Omni weight hash changed')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 CUDA GPU required')
    cfg=dict(device='cuda:0',omni_checkpoint=value['checkpoint'],amp='bf16',lora_layers=16,lora_rank=16,
        lora_alpha=32.,feature_layers=[15,31,63,95,111,119,123,127],checkpointing=True)
    model=load_model(cfg); auxiliary=TFCL().cuda()
    rng=np.random.default_rng(316)
    waves=[rng.normal(0,.1,n).astype(np.float32) for n in (16000,20320)]
    logits,fused,valid,lengths=model(waves)
    ta,sd=auxiliary([(fused[0,:lengths[0]],fused[1,:lengths[1]],valid[0,:lengths[0]],valid[1,:lengths[1]])])
    loss=F.cross_entropy(logits.float(),torch.tensor([0,1],device='cuda'))+.15*ta.mean()+.045*sd.mean()
    loss.backward()
    if not torch.isfinite(loss): raise ValueError('Nonfinite 7B smoke loss')
    if not any(m.b.grad is not None and bool(m.b.grad.norm()>0) for m in model.modules() if isinstance(m,LoRALinear)):
        raise ValueError('7B LoRA gradient missing')
    print(f'V316_7B_FORWARD_BACKWARD_OK=True peak_GPU_GiB={torch.cuda.max_memory_allocated()/1024**3:.2f}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--omni-assets'); a=p.parse_args()
    torch.set_num_threads(1)
    api_smoke()
    if a.omni_assets: full_smoke(a.omni_assets)
