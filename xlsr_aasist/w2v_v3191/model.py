"""Keep every parent adapter contribution; update only a suffix."""
from pathlib import Path
import torch
from w2v_v3189.model import Detector as ParentDetector, optimizer_for as parent_optimizer
from w2v_v3183r2.model import LoRALinear, RecomputedLayer


class Detector(ParentDetector):
    def __init__(self, frontend, encoder, cfg, layout_factory):
        # Adapter placement and trainability are separate. Removing learned
        # prefix adapters would change the parent detector before any update.
        construction = dict(cfg, lora_layers=cfg["retained_lora_layers"],
                            trainable_lora_layers=cfg["lora_layers"])
        super().__init__(frontend, encoder, construction, layout_factory)
        self.cfg = cfg
        self.set_phase(True)

    def _freeze_prefix(self, *, set_grad=True):
        count = self.cfg.get("trainable_lora_layers", self.cfg["lora_layers"])
        start = len(self.encoder.layers) - count
        for index, layer in enumerate(self.encoder.layers):
            active = self.joint and index >= start
            if isinstance(layer, RecomputedLayer):
                layer.enabled = active and self.cfg.get("checkpointing", True) and (
                    index >= len(self.encoder.layers) - self.cfg["checkpoint_layers"])
            for module in layer.modules():
                if isinstance(module, LoRALinear):
                    module.enabled = True
                    if set_grad:
                        module.lora_a.requires_grad_(active)
                        module.lora_b.requires_grad_(active)
                    module.drop.train(self.training and active)

    def set_phase(self, joint):
        super().set_phase(joint)
        self._freeze_prefix()

    def train(self, mode=True):
        super().train(mode)
        self._freeze_prefix(set_grad=False)
        return self


def adapter_inventory(model):
    depth = len(model.encoder.layers)
    retained, active, checkpointed = [], [], []
    for index, layer in enumerate(model.encoder.layers):
        adapters = [m for m in layer.modules() if isinstance(m, LoRALinear)]
        if not adapters:
            continue
        if len(adapters) != 4:
            raise ValueError("Expected four Q/K/V/O adapters per retained layer")
        retained.append(index + 1)
        flags = [p.requires_grad for m in adapters for p in (m.lora_a, m.lora_b)]
        if any(flags):
            if not all(flags):
                raise ValueError("Partially trainable adapter layer")
            active.append(index + 1)
        if isinstance(layer, RecomputedLayer) and layer.enabled:
            checkpointed.append(index + 1)
    if retained != list(range(depth - model.cfg["retained_lora_layers"] + 1, depth + 1)):
        raise ValueError("Parent adapter coverage changed")
    expected = list(range(depth - model.cfg["lora_layers"] + 1, depth + 1)) if model.joint else []
    if active != expected:
        raise ValueError("Trainable LoRA suffix changed")
    for name, p in model.named_parameters():
        if name.startswith(("frontend.", "encoder.")) and not name.endswith((".lora_a", ".lora_b")) and p.requires_grad:
            raise ValueError("Base encoder must stay frozen")
    return dict(retained_layers=retained, trainable_layers=active, checkpointed_layers=checkpointed,
                frozen_prefix_contribution_preserved=True, prefix_dropout_enabled=False,
                fusion_layers=list(model.fusion.layers))


def optimizer_for(model, cfg):
    model.set_phase(True)
    original = parent_optimizer(model, cfg)
    groups = []
    for group in original.param_groups:
        selected = [p for p in group["params"] if p.requires_grad]
        if selected:
            groups.append(dict(group, params=selected))
    actual = [p for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(expected) or {id(p) for p in actual} != expected:
        raise ValueError("Optimizer must contain exactly the active suffix, fusion and backend")
    return torch.optim.AdamW(groups)


def load_model(cfg):
    from w2v_v3184.assets import configured_spec
    from w2v_v3184.runtime import require_version
    require_version("fairseq2", "0.6.0")
    import omnilingual_asr  # noqa: F401
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    item = configured_spec(cfg)
    hub = get_wav2vec2_model_hub()
    arch = hub.get_arch_config(item["arch"])
    ec = arch.encoder_config
    if (ec.model_dim, ec.num_encoder_layers) != (cfg["encoder_dim"], cfg["encoder_layers"]):
        raise ValueError("Pinned Omni geometry changed")
    device = torch.device(cfg["device"])
    ssl = hub.load_custom_model(Path(cfg["omni_checkpoint"]), arch, device=device,
                               dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                               mmap=True, restrict=True)
    model = Detector(ssl.encoder_frontend, ssl.encoder, cfg, BatchLayout).to(device)
    adapter_inventory(model)
    return model
