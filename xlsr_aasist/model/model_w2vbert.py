"""V2 AASIST model with the XLS-R frontend replaced by Meta w2v-BERT 2.0.

Only the SSL frontend is changed. The AASIST backend is reused from
model/model.py, so V2 data augmentation, pair losses and backend structure
remain unchanged.
"""
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

# The legacy AASIST file imports fairseq at module import time because its old
# XLS-R SSLModel needs it. The w2v-BERT path never instantiates that XLS-R
# class, so fairseq is not a real dependency here. Provide a harmless stub only
# when fairseq is absent, allowing us to reuse the unchanged AASIST classes.
try:
    import fairseq  # noqa: F401
except ImportError:
    sys.modules["fairseq"] = types.ModuleType("fairseq")

from . import model as legacy


class SSLModel(nn.Module):
    """Drop-in replacement for the legacy fairseq XLS-R SSLModel.

    Input: raw 16 kHz waveform, shape [B, L] (or [B, L, 1]).
    Output: w2v-BERT 2.0 hidden sequence, shape [B, T, 1024].
    """

    def __init__(self, device, cp_path=None):
        super().__init__()
        try:
            from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
        except ImportError as exc:
            raise RuntimeError(
                "w2v-BERT 2.0 requires transformers==4.38.2. "
                "Install requirements_w2vbert.txt first."
            ) from exc

        cp_path = cp_path or os.environ.get(
            "W2VBERT_PRETRAINED",
            "/home/ubuntu/LXT/RTC/pretrained/w2v-bert-2.0",
        )
        cp_path = str(Path(cp_path).expanduser().resolve())
        if not Path(cp_path).is_dir():
            raise FileNotFoundError(
                f"w2v-BERT 2.0 directory not found: {cp_path}. "
                "Run python download_w2vbert.py first."
            )

        # local_files_only avoids silently changing the competition experiment
        # because of a future remote model revision.
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            cp_path, local_files_only=True
        )
        self.model = Wav2Vec2BertModel.from_pretrained(
            cp_path, local_files_only=True
        )
        self.device = device
        self.out_dim = int(self.model.config.hidden_size)
        # Later RTC adaptation stages can disable stochastic layerdrop/dropout
        # without disabling gradients. This is intentionally False for Stage 1.
        self.deterministic_finetune = False

        if self.out_dim != 1024:
            raise ValueError(f"Expected hidden_size=1024, got {self.out_dim}")
        if int(getattr(self.model.config, "num_hidden_layers", -1)) != 24:
            raise ValueError(
                "Expected the 24-layer facebook/w2v-bert-2.0 checkpoint"
            )
        if int(getattr(self.feature_extractor, "sampling_rate", 16000)) != 16000:
            raise ValueError("w2v-BERT feature extractor must use 16 kHz audio")

        # Do NOT enable gradient checkpointing here. In our staged fine-tuning
        # recipe the lower Conformer layers are frozen while only final layers are
        # trainable. Re-entrant checkpointing can then receive hidden states with
        # requires_grad=False and silently drop gradients for the trainable top
        # layers ("None of the inputs have requires_grad=True"). With only a few
        # final layers trainable, activation memory is already manageable.
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def _preprocess(self, input_data):
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        if input_data.ndim != 2:
            raise ValueError(
                f"Expected waveform [B, L] or [B, L, 1], got {tuple(input_data.shape)}"
            )

        # The official Wav2Vec2-BERT extractor performs log-mel extraction and
        # frame stacking to the 160-D input expected by the checkpoint.
        wave_cpu = input_data.detach().float().cpu()
        batch = self.feature_extractor(
            wave_cpu.numpy(),
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = batch["input_features"]
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                input_features.shape[:2], dtype=torch.long
            )
        return input_features, attention_mask

    def extract_feat(self, input_data):
        input_features, attention_mask = self._preprocess(input_data)
        if self.deterministic_finetune:
            # eval() disables layerdrop/dropout but does NOT disable autograd.
            # This is important for pair-consistency training.
            self.model.eval()
        target_device = next(self.model.parameters()).device
        input_features = input_features.to(target_device, non_blocking=True)
        attention_mask = attention_mask.to(target_device, non_blocking=True)

        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        emb = outputs.last_hidden_state
        if emb.ndim != 3 or emb.shape[-1] != self.out_dim:
            raise RuntimeError(
                f"Unexpected w2v-BERT output shape: {tuple(emb.shape)}"
            )
        return emb


# The legacy Model resolves SSLModel from model.model at construction time.
# Rebinding only that symbol keeps AASIST unchanged while replacing the frontend.
legacy.SSLModel = SSLModel
Model = legacy.Model
