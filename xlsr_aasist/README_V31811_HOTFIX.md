# V3.18.11 Hotfix: source-bound Train probes + temp.sh diagnostic delivery

This is a **read-only** V3.18.11 hotfix. It does NOT train or alter the V3.18.10 detector/checkpoint. It diagnoses only official Train and Dev Select plus already-existing historical Dev Audit predictions. Do not use Progress/Eval to calibrate, tune, or diagnose.

## Why the previous run failed

The previous `probes.py` remapped every selected Train ticket to row indices `0,1,...` **without first subsetting the complete Train manifest**. This caused the wrong waveform to be read under a different source identity; the saved `features/train_features.npz` from that failed run must not be trusted. Separately, the source holdout was hash-split globally without language/class stratification, and could omit a label, causing:

```
ValueError: Train source holdout is not label-stratified; cannot calibrate
```

The hotfix binds every Train ticket to the true original manifest row, verifies source identity and Online/Offline condition, then creates consecutive indices only for the resulting row subset. Train calibration is source-disjoint and stratified by `(language, label)`. The feature run invalidates old feature outputs before extraction, and produces an SHA256-verified feature manifest only after the entire diagnostic succeeds. The uploader refuses to ship unverified legacy features.

## Pull the fix

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git fetch origin
git switch v3.18.11-diagnostics
git pull --ff-only origin v3.18.11-diagnostics
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
python -m unittest w2v_v31811.test_probes w2v_v31811.test_scores w2v_v31811.test_delivery -v
```

## Rerun and upload everything to temp.sh

```bash
RUN=$(cat exp/.latest_v31810_run)
bash run_diagnose_v31811.sh \
  --run "$RUN" \
  --mode all \
  --checkpoint best \
  --workers 2 \
  --max-train-per-class 64 \
  --max-dev-per-class 64 \
  --out "$RUN/v31811_diagnostics" \
  --download-dir /home/ubuntu/LXT/temp \
  --upload-temp
cat "$RUN/v31811_diagnostics/download_links.txt"
```

The displayed `V31811_TEMP_DOWNLOAD_URL` is a real upload result returned by **https://temp.sh**. The archive is also retained in `/home/ubuntu/LXT/temp/<run-name>_v31811_diagnostics.zip`. Send the link or upload the ZIP for analysis.

**Important:** The previous failed Train feature extraction was misindexed, so the Train features need one corrected GPU pass. Score-only diagnostics do not require GPU; the updated probe reuses each student forward to obtain all three feature locations.

## Retry packaging/upload without GPU

When the diagnostic files are already present, or temp.sh times out, retry without inference:

```bash
RUN=$(cat exp/.latest_v31810_run)
bash upload_diagnose_v31811.sh --out "$RUN/v31811_diagnostics" \
  --download-dir /home/ubuntu/LXT/temp
cat "$RUN/v31811_diagnostics/download_links.txt"
```

If the link expired, use `--force` to obtain a new one. If network upload is unavailable, use `--pack-only` and download the retained ZIP through your usual server file-transfer workflow.

```bash
RUN=$(cat exp/.latest_v31810_run)
bash upload_diagnose_v31811.sh --out "$RUN/v31811_diagnostics" \
  --download-dir /home/ubuntu/LXT/temp --force
```

The ZIP contains only allowlisted `scores/` statistics/CSVs/plots, successfully verified `features/` NPZ and summary files, the feature SHA256 manifest, and completion/error metadata. It **does not include** original WAV files, training checkpoints, optimizer weights or Progress/Eval scores. temp.sh links may be accessible to anyone holding the URL; share them selectively.

If feature extraction fails again and `--upload-temp` is used, an available partial report (usually CPU scores) and a `failure.json` traceback can be uploaded. The uploader excludes incomplete or unverified feature NPZs. No successful V3.18.11 completion is claimed after a failed run.
