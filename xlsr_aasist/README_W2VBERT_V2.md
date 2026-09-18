# Final w2v-BERT 2.0 + AASIST V2 training plan

This branch keeps the successful V2 data idea but redesigns optimization for the much larger w2v-BERT 2.0 backbone.

## Model

```text
waveform
  -> official w2v-BERT 2.0 feature extractor
  -> w2v-BERT 2.0
  -> original AASIST backend
  -> real / fake
```

AASIST is not replaced.

## Why the training recipe is different from XLS-R

w2v-BERT 2.0 is a 24-layer, ~580M-parameter Conformer model. Later RTC adaptation stages therefore become progressively more conservative instead of fine-tuning the whole backbone with the old XLS-R LR.

The Train protocol is also strongly imbalanced. Ordinary examples use square-root inverse-frequency CE weights by default (class_weight_power=0.5), while RTC/noisy pair branches are already 1:1 balanced and therefore use equal CE weights.

For Stage 2/3, the w2v-BERT forward pass is deterministic: dropout/layerdrop are disabled through eval-mode on the SSL backbone while autograd remains enabled for the selected top layers.

## Stage 1: learn Deepfake detection

Purpose: learn the base real/fake detector without immediately rewriting the pretrained speech model.

Default:
- batch = 40 ordinary examples
- w2v-BERT trainable layers = final 8 / 24
- first 2 epochs: encoder LR = 0, AASIST learns on frozen pretrained features
- after warmup: encoder LR = 5e-7
- AASIST LR = 5e-5
- class weight power = 0.5
- gradient clip = 1.0
- selection = Dev Online Macro-F1
- maximum 30 epochs, early-stop patience 6
- validation plateau: 2 bad epochs -> LR x0.5 -> restore best checkpoint -> clear Adam state

Command:

```bash
bash run_train_w2vbert_base.sh
```

## Stage 2: adapt to real RTC

Purpose: keep Stage-1 detection ability while learning verified Offline/Online invariance.

Default:
- 32 ordinary + 4 RTC pairs x2 = 40 waveforms
- only final 4 / 24 w2v-BERT layers trainable
- w2v-BERT deterministic forward
- encoder LR = 5e-8
- AASIST LR = 5e-7
- ordinary CE: square-root inverse-frequency weighting
- RTC-pair CE: equal weight because pair sampling is 1:1 real/fake
- RTC contrastive weight: linearly 0 -> 0.05 over first 3 epochs
- gradient clip = 1.0
- selection = Dev Online Macro-F1
- maximum 10 epochs, early-stop patience 5

Command:

```bash
bash run_train_rtc_w2vbert.sh ./exp/<stage1>/ckpt/best_model.pth
```

## Stage 3: V2 Noisy + RTC adaptation

Purpose: adapt the already-stable detector to noisy communication conditions without destroying its base representation.

Default:
- 16 ordinary + 4 real RTC pairs x2 + 4 noisy RTC pairs x2 = 32 waveforms
- only final 2 / 24 w2v-BERT layers trainable
- w2v-BERT deterministic forward
- encoder LR = 1e-8
- AASIST LR = 2e-7
- ordinary CE: square-root inverse-frequency weighting
- real/noisy pair CE: equal
- real RTC contrastive weight = 0.05
- noisy RTC contrastive weight = 0 -> 0.1 over first 3 epochs
- noisy processed CE coefficient = 0.3
- gradient clip = 1.0
- selection = 0.3 clean Online + 0.35 noisy seen + 0.35 noisy held-out
- maximum 20 epochs, early-stop patience 6

Commands:

```bash
bash run_train_rtc_noisy_v2_w2vbert.sh ./exp/<stage2>/ckpt/best_model.pth --check_data
bash run_train_rtc_noisy_v2_w2vbert.sh ./exp/<stage2>/ckpt/best_model.pth
```

## Evaluation

```bash
bash run_eval_rtc_noisy_w2vbert.sh ./exp/<stage3>/ckpt/best_model.pth
```

## Stage progression

```text
official w2v-BERT 2.0
       |
       v
Stage 1: base Deepfake detector
  top 8 layers, encoder warmup
       |
       v
Stage 2: real RTC adaptation
  top 4 layers, balanced pair CE, RTC warmup
       |
       v
Stage 3: noisy RTC adaptation
  top 2 layers, V2 data recipe
       |
       v
submission model
```

Do not pass an XLS-R detector checkpoint into this branch. Detector checkpoints transferred between stages must all have been produced by the w2v-BERT model.
