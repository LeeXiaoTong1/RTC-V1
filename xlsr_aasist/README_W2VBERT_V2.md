# V2 + w2v-BERT 2.0

This branch is based directly on the original V2 code that produced the 88.7144 weighted submission. It does not include the later V3 Adapter/Nes2Net-X architecture.

## Core design

The detector remains:

`waveform -> w2v-BERT 2.0 -> AASIST -> real/fake`

The V2 data recipe remains intact:

- ordinary Train audio with RawBoost + MUSAN
- verified real RTC Offline/Online pairs
- Offline/noisy-RTC simulated pairs
- V2 21 seen / 6 held-out FFmpeg combinations
- 64600-sample crop/repeat convention
- Stage 3 batch = 24 ordinary + 4 real pairs x2 + 4 noisy pairs x2 = 40

What changed is the optimization strategy required by the much larger 24-layer, ~580M-parameter w2v-BERT 2.0 backbone.

## w2v-BERT-specific stability changes

1. Stage 1 may train all 24 layers, but uses a separate encoder/backend LR and gradient clipping.
2. Stage 2 trains only the final 8 w2v-BERT layers.
3. Stage 3 trains only the final 4 w2v-BERT layers.
4. Stage 2/3 force the w2v-BERT backbone to eval-mode during forward passes. Gradients remain enabled, but internal layerdrop/dropout is disabled so pair-consistency losses are not contaminated by stochastic encoder noise.
5. Stage 2 no longer applies global class weights to the balanced RTC-pair branch. Only the ordinary imbalanced stream uses protocol-derived class weights; RTC pair CE is unweighted.
6. Stage 2 ramps the RTC contrastive weight from 0 to 0.1 during the first 2 epochs.
7. All stages use gradient clipping (default 1.0).
8. Adaptive LR uses 2 consecutive bad validation epochs, then halves LR, restores `best_model.pth`, clears Adam state, and continues from the best representation.

## Default learning rates

| Stage | trainable w2v-BERT layers | encoder LR | AASIST LR | monitored metric |
|---|---:|---:|---:|---|
| Stage 1 | 24 | 1e-6 | 1e-4 | DevLoss |
| Stage 2 | final 8 | 1e-7 | 2e-6 | Dev Online Macro-F1 |
| Stage 3 | final 4 | 5e-8 | 1e-6 | DevRobustProxy |

## Stage 1

```bash
bash run_train_w2vbert_base.sh
```

## Stage 2: real RTC adaptation

```bash
bash run_train_rtc_w2vbert.sh ./exp/<stage1>/ckpt/best_model.pth
```

Default batch:

```text
32 ordinary + 4 Offline/Online pairs x2 = 40 waveforms
```

Classification loss:

```text
ordinary stream: protocol-derived weighted CE
RTC pair stream: unweighted CE because the pair sampler is 1:1 real/fake
combined CE: sample-count weighted
total loss = CE + warmup(rtc_weight) * RTC_pair_loss
```

## Stage 3: V2 noisy RTC adaptation

```bash
bash run_train_rtc_noisy_v2_w2vbert.sh ./exp/<stage2>/ckpt/best_model.pth --check_data
bash run_train_rtc_noisy_v2_w2vbert.sh ./exp/<stage2>/ckpt/best_model.pth
```

Stage 3 preserves the original V2 group logic: ordinary CE uses the Train class weights, while real-pair and noisy-pair CE remain equal-weight because both pair samplers are balanced.

## Evaluation

```bash
bash run_eval_rtc_noisy_w2vbert.sh ./exp/<stage3>/ckpt/best_model.pth
```

Do not load an XLS-R detector checkpoint into this branch. Every detector checkpoint passed between stages must have been produced by the w2v-BERT model.
