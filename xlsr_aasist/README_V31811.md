# V3.18.11: frozen-checkpoint EN-real failure diagnosis

**This is a read-only diagnostic release. It does not change the V3.18.10 detector, objective, checkpoint, or submission.**

The aim is to distinguish (1) representation separability, (2) learned readout and fixed-threshold effects, (3) epoch-dependent score drift, and (4) errors attributable to speech source versus simulated noise/RTC condition.

## Sources and interpretation

- Only the official Train and already-available Dev Select / historical Audit records are read.
- Progress/Eval audio, scores, predictions, and labels are **not accepted**.
- V3.18.10 retains per-epoch Select logits, enabling epoch comparisons *without* epoch-specific model weights.
- The historical Audit has already informed research. V3.18.11 calls it a *descriptive* subset, not a fresh unbiased test. Probes default to Select only.
- The frozen model's best/last weights are available in `last.pt`. The report ZIP has no trained model. Saved logits are **not** sufficient for epoch-6 features; do not infer unrecorded historical embeddings.

## Step 1: CPU score diagnostics (no model inference)

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved/xlsr_aasist
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
RUN=$(cat exp/.latest_v31810_run)
bash run_diagnose_v31811.sh --run "$RUN" --mode scores --out "$RUN/v31811_diagnostics"
```

If only a downloaded V3.18.10 report ZIP is available, use:

```bash
bash run_diagnose_v31811.sh --report-zip /path/to/v31810_report.zip --mode scores --out /tmp/v31811_report
```

Outputs: `score_summary.json`, `epoch_metrics.csv`, `epoch_flips.json`, `paired_summary.json`, `paired_sources.csv`, `audit/paired_summary.json`, and `en_online_margins.png`. Margins are **fake logit minus real logit**; `margin >= 0` is Fake. AUC is computed on both classes, never Real-only. No Dev threshold is optimized or deployed.

## Step 2: GPU frozen-feature probe

```bash
RUN=$(cat exp/.latest_v31810_run)
bash run_diagnose_v31811.sh --run "$RUN" --mode features --checkpoint best --workers 2 --max-train-per-class 64 --max-dev-per-class 64 --out "$RUN/v31811_diagnostics"
```

This loads a single pinned official Omni 3B and the **completed V3.18.10 best**. The model remains `eval()` and fully frozen. From the *same* student forward, it saves:

1. **SSL last-layer**: time mean/std of the actual final native-normalized 2048D output (4096D).
2. **Multi-layer fused**: time mean/std of the 128D feature sequence (256D).
3. **Preclassifier**: exact 384D tensor entering the last AASIST linear layer.

Source-stratified Train subsets include fixed Online and deterministic Train Offline→Noisy→RTC views. Dev Select uses the existing locked three-panel synthesis; no new split is made. Feature extraction is sampled **by original source**, not by filename. Every feature vector receives the original source ID, split, condition panel, language and truth label. The pipeline never crops audio to lower inference cost; only the number of *independent utterances* is limited. All feature files remain local; do not upload raw embeddings to public repositories.

Train-only grouped cross-validation chooses logistic-probe regularization for each embedding, then fits the probe on Train features and evaluates Select. The only post-hoc score calibration is a **single global logit bias fitted on a source-separated Train subset**, checked on a different Train subset and finally described on Select. No language-dependent test-time routing or Dev threshold search is performed.

Optional historical Audit feature probe (descriptive only, not a new test):

```bash
bash run_diagnose_v31811.sh --run "$RUN" --mode features --include-audit-features --out "$RUN/v31811_diagnostics"
```

Outputs: `features/feature_probe_summary.json`, `train_features.npz`, `select_features.npz`, and optional `audit_features.npz`. **No score, probe, epoch, or bias selected by these diagnostics may be tuned against Progress/Eval.** The frozen probe is only a restricted readout: low accuracy does not prove the original sequence lacks nonlinear/temporal information.

## Validation and conclusions

The legitimate comparisons are:
- Strong original AUC but low fixed-threshold Recall with a global *Train-fit* bias transferring to Dev => decision calibration is a likely bottleneck.
- Train-fit probe at a hidden stage surpasses original readout on independent Dev, with stable calibration => later fusion/head may be using evidence poorly.
- All probes generalize poorly => evidence of representation/domain weakness, **not proof that useful evidence is absent**.
- Same canonical source wrong on Online and Noisy => source/recording issue is implicated; newly wrong only after simulated processing => robustness issue is implicated.
- AUC/EER stable across epochs while fixed-threshold Recall changes => score shift, which does not by itself prove a stable deployable offset.

When interpreting Select vs Audit, keep them distinct and report exact denominators. Original V3.18.10 `last.pt` stores best and last, not every prior epoch; score curves are the appropriate historical substitute.

## CPU verification

```bash
python -m unittest w2v_v31811.test_scores w2v_v31811.test_probes -v
```

Production 3B, native RTC synthesis, AMP, and GPU memory/throughput require separate server validation. The diagnostic does **not** train or change V3.18.10 parameters.


## Hotfix: source-index repair and temp.sh exports

The feature-probe Train ticket indexing and class/language-stratified holdout are corrected. Old failed `train_features.npz` outputs were not source-aligned and must be recomputed; the diagnostic uploader excludes unverified old NPZ files. The new `--upload-temp` option posts a ZIP of saved diagnostics to temp.sh, without checkpoint or WAV files.

**See [README_V31811_HOTFIX.md](README_V31811_HOTFIX.md) for the exact pull, rerun, upload/retry instructions.**
