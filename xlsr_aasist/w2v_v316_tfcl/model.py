"""Original pretrained SSL + fresh task head, or explicit V3.15 warm-start control."""
import torch
from w2v_v3.model import Detector,HeadConfig
from w2v_v32.model import Detector as PaddedDetector,install_runtime
from w2v_v312.model import FeatureClassifier
from w2v_v3151.model import load_model as load_reference
from w2v_v39.common import verify_files


class RuntimeDetector(PaddedDetector):
    def train(self,mode=True):
        result=super().train(mode)
        if getattr(self,'head_only',False):self.backbone.eval()
        return result
    def forward(self,features,mask,**kw):
        if not self.training or not getattr(self,'head_only',False):return super().forward(features,mask,**kw)
        with torch.no_grad():
            output=self.backbone(input_features=features.masked_fill(~mask.bool().unsqueeze(-1),0),
                attention_mask=mask,output_hidden_states=True,return_dict=True)
        return self.head(output.hidden_states,mask)


def load_model(cfg,device=None,training=True):
    if cfg['init_mode']=='parent':model=load_reference(cfg,device,training)
    else:
        from transformers import Wav2Vec2BertModel
        verify_files(cfg['pretrained_fingerprints'])
        # Recreate the same fresh task head on resume without perturbing caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(cfg['seed'])
            backbone=Wav2Vec2BertModel.from_pretrained(cfg['pretrained_path'],local_files_only=True,torch_dtype=torch.float32)
            backbone.config.layerdrop=0.;backbone.config.apply_spec_augment=False
            if backbone.config.add_adapter:raise ValueError('Temporal encoder adapter unsupported')
            if cfg['checkpointing']:backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model=Detector(backbone,HeadConfig(input_dim=backbone.config.hidden_size,**cfg.get('new_head_config',{})))
            model.head.classifier=FeatureClassifier(model.head.classifier,cfg['adapter_hidden'],cfg['adapter_max_ratio'])
            model.configure_trainable_layers(cfg['trainable_layers']);model=install_runtime(model)
    if not 0<cfg['trainable_layers']<len(model.backbone.encoder.layers):raise ValueError('Need both frozen prefix and trainable suffix')
    model.__class__=RuntimeDetector;model.head_only=False
    model.configure_trainable_layers(cfg['trainable_layers'])
    return model.to(device or cfg['device']).train(training)


def optimizer_for(model,cfg,auxiliary=None):
    from w2v_v315.model import optimizer_for as original
    # Main method has NO trainable alignment/projection parameters.
    has_aux=auxiliary is not None and any(True for _ in auxiliary.parameters())
    return original(model,cfg,auxiliary if has_aux else None)


def stage(model,cfg,cursor,epoch_steps):
    warmup=min(cfg['head_warmup_updates'],max(1,epoch_steps//4))
    model.head_only=cursor<warmup
    elapsed=max(0,cursor-warmup+1)
    time_strength=min(1.,elapsed/max(1,epoch_steps*cfg['objective_ramp_epochs'])) if not model.head_only else 0.
    delay=epoch_steps*cfg['structure_delay_epochs']
    structure_strength=min(1.,max(0,elapsed-delay)/max(1,epoch_steps*cfg['objective_ramp_epochs'])) if not model.head_only else 0.
    return time_strength,structure_strength
