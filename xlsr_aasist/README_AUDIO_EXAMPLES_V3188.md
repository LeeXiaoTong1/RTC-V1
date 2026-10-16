# V3.18.8 audio examples during training

This standalone CPU exporter reads the active V3.18.8 run and saves paired listening examples. It does not load a detector, read a checkpoint, change training files, acquire the training lock, or modify the training process RNG. Its process hides CUDA, limits numerical-library threads, and runs at lower CPU priority on Linux. It still uses a small amount of CPU, memory and disk I/O alongside training.

Each group contains four **complete** waveforms:

1. `01_offline.wav`: the original Offline source, decoded with the training loader.
2. `02_official_online.wav`: the official Online recording matched by canonical `source_id`, language and class, not by a guessed filename.
3. `03_noisy_before_rtc.wav`: Offline with sampled scene noise/echo and the training common gain, captured immediately before the RTC engine.
4. `04_noisy_online.wav`: that exact input after the unmodified training RTC engine.

The two simulated files use the same draw. No separately redrawn noise, peak normalization, cropping, or reflection padding is introduced for the preview. WAVs use 16 kHz mono IEEE float32 to preserve the exact generated samples. Official Online may have different duration and delay. The common gain in stage 03 and RTC gain processing in stage 04 are recorded; do not assume all stages have identical loudness.

Examples use the training pipeline's **original base branch**, making the effects of noise and RTC easier to hear. They are new diagnostic draws, not audio captured from the currently executing batch. Speed, EQ and amplitude-envelope base variants are not covered by this small preview. These listening examples do not measure model performance or represent the full augmentation distribution.

Defaults select one Train pair from each of EN-Real, EN-Fake, ZH-Real and ZH-Fake, with both recordings between 3 and 20 seconds. Selection uses a deterministic seeded order without model scores. `--per-group 2` produces eight quartets (32 WAVs). There is no truncation to meet the duration filter. `--scene auto` follows the saved scene probabilities; an explicit scene produces a targeted demonstration and is recorded in the manifest.

The ZIP includes an `index.html` listening page, `README.txt` and `manifest.json` with source identities, hashes, scene, measured input SNR, gain, full RTC recipe and sample counts. Extract the ZIP before opening the HTML page. Echo uses delayed-speech gain, not additive SNR. Only this selected audio package is uploaded, not weights, complete protocols or the full training directory.

## Run on the server

Run in a second terminal, with the same environment as training. Installing this update adds standalone files only; existing training modules are unchanged.

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git pull --ff-only origin w2vbert2-balanced-robust-fast
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist
bash export_audio_examples_v3188.sh --per-group 2 --upload
```

The run defaults to `exp/.latest_v3188_run`. Override with `--run /absolute/path/to/run` if needed. An active run is accepted. No model or audio datasets are downloaded.

Output appears under `/home/ubuntu/LXT/temp/audio_examples_v3188_TIMESTAMP_SUFFIX/`, with an adjacent `.zip`. The script prints:

```text
AUDIO_DIR=...
AUDIO_ZIP=...
AUDIO_DOWNLOAD_URL=https://temp.sh/...
LINK_FILE=...
```

The URL is saved in an adjacent `.url.txt`. To keep everything local, omit `--upload`. To choose another destination, use `--output-root`. Do not place output inside the live run directory.

## Optional targeted export and upload retry

```bash
bash export_audio_examples_v3188.sh --per-group 1 --scene rain --seed 318802 --upload
```

Available scenes: `auto`, `office`, `coffee`, `rain`, `footsteps`, `keyboard`, `echo`. A new seed changes the selected sources and draws. A small random export is not guaranteed to contain all six scenes.

On upload failure, the script prints an exact retry command using the retained ZIP. Retry validates its manifest and audio hashes and does not regenerate or rerun RTC:

```bash
bash export_audio_examples_v3188.sh --retry-zip /absolute/path/to/audio_examples_v3188_TIMESTAMP_SUFFIX.zip
```

Tests use synthetic audio, verify exact pre/post capture, canonical pairing, preservation of a live run, deterministic output, upload failure recovery and native bundled FFmpeg/Opus paths. The real server's WebRTC path uses its existing training environment; no synthetic fixture ZIP is presented as real training audio.
