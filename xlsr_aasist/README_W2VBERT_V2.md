# V2 + w2v-BERT 2.0

This branch is based directly on the original V2 code that produced the 88.7144 weighted submission. It does not include V3 Adapter/Nes2Net-X/WebRTC architecture changes.

## What is unchanged

- AASIST backend in `model/model.py`
- V2 ordinary/real-pair/noisy-pair data composition
- 24 ordinary + 4 real Offline/Online pairs + 4 noisy pairs = 40 waveforms
- RawBoost and MUSAN ordinary-branch augmentation
- FFmpeg V2 noisy cache and 21/6 seen/held-out combinations
- V2 classification weighting and pair losses
- V2 validation/selection logic
- 64600-sample waveform convention and inference score format

## What is changed

Only the generic speech frontend is replaced:

`XLS-R 300M -> facebook/w2v-bert-2.0`

The original AASIST implementation is reused. `model/model_w2vbert.py` only rebinds the legacy `SSLModel`; it does not replace AASIST.

The official w2v-BERT feature extractor converts 16 kHz waveform into stacked log-mel input features before the 24-layer, 1024-dimensional encoder. Gradient checkpointing is enabled to reduce full-fine-tuning memory use.

Old XLS-R checkpoints cannot be loaded into this model. To preserve the original V2 training history, train the w2v-BERT version through the same three logical stages:

1. base w2v-BERT + AASIST training
2. clean real RTC-pair fine-tuning
3. V2 noisy RTC-pair fine-tuning

## Server setup

```bash
conda activate sdd
cd /home/ubuntu/LXT/RTC/xlsr_aasist
pip install -r requirements_w2vbert.txt
python download_w2vbert.py
python check_w2vbert_v2.py --device cuda:0
```

The default checkpoint directory is:

```text
/home/ubuntu/LXT/RTC/pretrained/w2v-bert-2.0
```

## Stage 1: base model

```bash
bash run_train_w2vbert_base.sh
```

Use the resulting `best_model.pth` as Stage 2 input.

## Stage 2: clean RTC pair

```bash
bash run_train_rtc_w2vbert.sh ./exp/<stage1>/ckpt/best_model.pth
```

Use the resulting `best_model.pth` as Stage 3 input.

## Stage 3: the original V2 recipe

```bash
bash run_train_rtc_noisy_v2_w2vbert.sh ./exp/<stage2>/ckpt/best_model.pth --check_data
bash run_train_rtc_noisy_v2_w2vbert.sh ./exp/<stage2>/ckpt/best_model.pth
```

This stage keeps the original V2 defaults: 24 ordinary, 4 real pairs, 4 noisy pairs, noisy CE 0.3, real/noisy pair weights 0.1, LR 1e-6, weight decay 1e-4, BF16, up to 30 epochs, patience 10.

## Evaluation

```bash
bash run_eval_rtc_noisy_w2vbert.sh ./exp/<stage3>/ckpt/best_model.pth
```

## Important

Do not pass an XLS-R `.pth` training checkpoint to any w2v-BERT stage. Stage-to-stage checkpoints must all have been produced by this w2v-BERT model. The generic Meta pretrained directory is supplied through `W2VBERT_PRETRAINED` / `--ssl_path`; that directory is not a Deepfake detector checkpoint.
