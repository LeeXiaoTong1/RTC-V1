"""Retain the R2 encoder/reader; add a mean+standard-deviation classifier."""
from pathlib import Path
import torch
from w2v_v3183r2.model import Detector as R2Detector
from .regions import StatisticsHead


class Detector(R2Detector):
    def __init__(self,frontend,encoder,cfg,layout_factory):
        super().__init__(frontend,encoder,cfg,layout_factory)
        if cfg['pooling']=='asp':self.head=StatisticsHead(self.head)
        elif cfg['pooling']!='mean':raise ValueError('Unknown pooling')
        if not cfg['local_evidence']:raise ValueError('Window pooling requires local evidence')
        self.set_phase(False)


def load_model(cfg):
    from w2v_v3184.assets import configured_spec
    from w2v_v3184.runtime import require_version
    item=configured_spec(cfg);require_version('fairseq2','0.6.0')
    import omnilingual_asr
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    hub=get_wav2vec2_model_hub();arch=hub.get_arch_config(item['arch']);ec=arch.encoder_config
    if (ec.model_dim,ec.num_encoder_layers)!=(item['encoder_dim'],item['encoder_layers']):
        raise ValueError('Installed Omni architecture changed')
    device=torch.device(cfg['device']);dtype=torch.bfloat16 if device.type=='cuda' else torch.float32
    ssl=hub.load_custom_model(Path(cfg['omni_checkpoint']),arch,device=device,dtype=dtype,mmap=True,restrict=True)
    if len(ssl.encoder.layers)!=item['encoder_layers']:raise ValueError('Wrong encoder depth')
    model=Detector(ssl.encoder_frontend,ssl.encoder,cfg,BatchLayout);del ssl
    return model.to(device)


def optimizer_for(model,cfg):
    parts=[('lora',[p for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b'))],cfg['lora_lr']),
        ('aasist',[p for n,p in model.head.aasist.named_parameters() if not n.startswith('out_layer.')],cfg['head_lr']),
        ('classifier',list(model.head.aasist.out_layer.parameters()),cfg['classifier_lr']),
        ('evidence',list(model.head.contribution.parameters()),cfg['evidence_lr']),
        ('fusion',[p for n,p in model.fusion.named_parameters() if n!='layer_logits'],cfg['fusion_lr']),
        ('layer_mix',[model.fusion.layer_logits],cfg['layer_mix_lr'])]
    if cfg['pooling']=='asp':parts.append(('pooling',list(model.head.attention_delta.parameters()),cfg['pool_lr']))
    groups=[]
    for name,parameters,rate in parts:
        for decay in (False,True):
            chosen=[p for p in parameters if (p.ndim>=2)==decay]
            if chosen:groups.append(dict(name=name,params=chosen,lr=rate,initial_lr=rate,
                                        weight_decay=cfg['weight_decay'] if decay else 0.))
    actual=[p for g in groups for p in g['params']]
    expected={id(p) for n,p in model.named_parameters() if p.requires_grad or n.endswith(('.lora_a','.lora_b'))}
    if len(actual)!=len({id(p) for p in actual}) or {id(p) for p in actual}!=expected:
        raise ValueError('Optimizer inventory mismatch')
    return torch.optim.AdamW(groups)
