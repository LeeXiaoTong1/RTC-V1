"""Utilities for stable w2v-BERT 2.0 fine-tuning.

The 24-layer Conformer backbone is much larger than the original XLS-R front end.
Later RTC adaptation stages should not rewrite the entire 580M-parameter encoder.
"""

def configure_trainable_top_layers(model, top_layers):
    """Freeze all w2v-BERT parameters, then unfreeze only the last N encoder layers.

    Args:
        model: full detector with model.ssl_model.model = Wav2Vec2BertModel.
        top_layers: number of final Conformer layers to fine-tune. 24 means all
            encoder layers, 0 means the entire w2v-BERT backbone is frozen.

    The feature projection stays frozen when top_layers < total_layers. This
    intentionally preserves the generic acoustic front-end during RTC adaptation.
    """
    backbone = model.ssl_model.model
    layers = backbone.encoder.layers
    total_layers = len(layers)
    top_layers = int(top_layers)
    if top_layers < 0 or top_layers > total_layers:
        raise ValueError(f"top_layers must be in [0,{total_layers}], got {top_layers}")

    for p in backbone.parameters():
        p.requires_grad = False

    if top_layers == total_layers:
        for p in backbone.parameters():
            p.requires_grad = True
    elif top_layers > 0:
        for layer in layers[total_layers - top_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    return {
        "total_layers": total_layers,
        "trainable_layers": top_layers,
        "trainable_params": trainable,
        "total_params": total,
    }


def split_trainable_params(model):
    """Return trainable encoder parameters and trainable non-encoder parameters."""
    encoder_params = [p for p in model.ssl_model.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder_params}
    backend_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in encoder_ids
    ]
    if not backend_params:
        raise RuntimeError("No trainable AASIST/backend parameters")
    return encoder_params, backend_params
