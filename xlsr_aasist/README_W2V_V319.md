# V3.19: return to V3.15, EN6 and received-audio consistency

This is a new experiment starting from the documented Progress Weighted 93.6341 model: `exp/w2v_v315_20261008_075905_5663`, selector `best_guarded`. The historical score is a user-reported result, not a newly reproduced Progress result. The startup pins the actual selected tag, hashes and parent dependencies, then replays fixed Dev. A missing or changed parent fails explicitly; the launcher never guesses another latest run.

## Model and training

The V3.15 detector is retained: w2v-BERT 2.0, shared 1024-to-128 projection and SwiGLU across encoder hidden states, MultiConv, its existing attention mean/standard-deviation pooling, and the bounded classifier adapter. This is not the Omni/AASIST architecture. The last eight encoder layers and existing head are adapted. No new front-end download or random head replacement is required.

Defaults are five additional epochs, no head warmup and no learning-rate warmup. Top encoder LR is `1e-6`, with `.8` layer decay; head LR is `1e-5`; adapter and final output LR are `2e-5`. Cosine decay finishes at 10% of these values. Adam state is fresh for this new recipe; the trained detector is inherited. Normal plateaus do not shorten the five-epoch budget. Nonfinite updates or a severe fixed-Dev regression stop safely and retain committed states.

Each update has 16 distinct sources and 48 views. Quotas in EN-fake, EN-real, ZH-fake, ZH-real order are `4/3/5/4`. Each group has exactly 25% CE and auxiliary source mass; each selected source receives `.25/quota`. Sampling frequency is not the loss weight.

## Targeted augmentation and coverage

EN-fake and EN-real each have **six total fixed versions**, independently for their existing Online and Offline audio: original, mild slowing, mild speeding, frequency response, temporal context and gentle gain envelope. The corresponding Online and Offline use the same temporal geometry. These versions do not create new speakers or independent recordings.

ZH-real retains three fixed versions; ZH-fake retains one fixed version and common dynamic light augmentation. All four groups use the same original exposure probability, one sixth, and shared augmentation families to reduce a label-specific augmentation shortcut. ZH is not expanded sixfold.

Source cycles are deterministic across epoch boundaries and resume. The number of steps per epoch is:

```
max(ceil(number_of_sources / 16),
    ceil(6 * EN_fake_sources / (4 * epochs)),
    ceil(6 * EN_real_sources / (3 * epochs)))
```

The general implementation uses the configured positive quotas. Thus the planned five epochs cover every EN source's original plus five variants in both available streams. Historical source counts give 2,417 steps per epoch; actual counts are checked on the server. Interrupted or safety-stopped runs do not claim completed coverage. `sampling_epoch_*.json` records counts, repeat exposure, realized fixed versions, fallbacks and any additional budget.

Offline versions are only a base for simulated received audio. Each source supplies:

1. Augmented official Online, CE mass .30. If absent, independently simulated clean RTC replaces this view and its official bridge is disabled.
2. Simulated clean reference `R(A(x))`, CE mass .10.
3. Noisy received audio `R(A(x) + n)`, CE mass .60.

No raw Offline classification view is added. V3.18.4's prepared six-scene noise manifest is reused. Additive noise is mixed at 5--25 dB before communication processing; echo uses delayed speech and its own gain rather than pretending to have an additive-noise SNR. Matched endpoints share gain and RTC settings but have fresh processing state. The transparent/moderate/strong RTC profiles expand processing conditions; they are not a full reproduction of unknown competition platforms, AEC or PLC. Train and Dev noise recordings are disjoint; RTC mechanisms are shared.

## Consistency redesign

V3.15 already mixed noise before RTC and constrained two received-end signals. Its sequence was not reversed. V3.19 changes where adaptation happens and adds a weaker official-platform bridge:

- Main edge, 80% of auxiliary source budget: `R(A(x))` versus `R(A(x)+n)` on their common valid timeline.
- Weak edge, 20%: official Online versus simulated clean RTC, with detached local monotonic alignment. Missing Online or actual temporal crops disable this edge without reallocating its budget.
- Temporal coefficient `.15`; channel-structure coefficient `.045`, after CE total coefficient is normalized to one. These are starting values, not equivalent gradient strength to the former learned attention loss.

This is **TFCL-inspired parameter-free consistency**, not an exact reproduction of published TFCL. It uses the pre-MultiConv 128-dimensional frame features, direct masked cosine loss and per-source channel CKA. The official bridge uses at most 192 alignment bins; CKA uses at most 201. Both detector endpoints receive gradients. There is no trainable auxiliary aligner that can absorb the objective by itself, no confidence teacher gate, and no extra detector forward for the losses. Cropped or invalid frames are masked.

`--consistency matched` and `--consistency none` are available for separately initialized controlled experiments. Disabled losses skip their computation. Gradient magnitude/direction diagnostics are measured on the first eligible physical batch each epoch and are explicitly labeled; full-update module norms are also recorded. The report does not misrepresent these as whole-epoch gradient comparisons.

Sources: [TFCL paper](https://arxiv.org/html/2607.17761v2), [author implementation](https://github.com/JunXue-tech/TFCL/blob/main/code/model.py), [RTCFake paper](https://aclanthology.org/2026.findings-acl.285.pdf).

## Runtime, evaluation and disk use

The A100 plan is measured at startup with actual median/long/longest augmented Train recordings. Up to six physical batch/checkpointing combinations are tried. The fastest successful update with at least 4 GiB and 10% GPU reserve wins, instead of maximizing memory occupancy. Profiling restores weights, buffers, Adam, gradients, modes and RNG and does not consume training updates. The recorded execution plan is reused on resume. GPU speed and peak memory need server measurement; CPU tests cannot establish them.

Persistent workers and bounded decoded-wave/noise RAM are used. There is no generated waveform or feature disk cache. Scene validation generates only its required noisy endpoint. One atomic partial checkpoint keeps current state, Adam and at most two distinct best candidates; frozen encoder copies are not saved. Peak additional disk budget is capped at 20 GiB, with a separate 10 GiB reserve. The startup computes the actual requirement. Do not delete V3.15, its V3.12/base dependencies, the prepared noise bank or an active run.

Fixed full Dev and a bounded six-scene selection panel are evaluated once per epoch. Historical Weighted keeps its original definition. Automatic submission defaults to **best_weighted**; `best_guarded` is also saved and is an explicit alternative. Both selectors include a clearly labeled pinned-parent fallback. A final processed audit evaluates the actual requested submission selector without changing selection. Audit processed views are held out, but their original Dev recordings also occur in fixed Dev; this is not unseen-source test evidence. Progress/Eval audio is read only during the requested export, never training or model selection.

## Server deployment and training

Wait for other training/export work to finish. Use the original `sdd` environment for w2v-BERT, with Transformers 4.38.2, not the Omni-only runtime. Existing Torch/CUDA packages are checked and never upgraded by this setup. The noise preparation already completed for V3.18.4 is reused. All commands below are ASCII.

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
git switch w2vbert2-balanced-robust-fast
git pull --ff-only origin w2vbert2-balanced-robust-fast
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sdd
cd xlsr_aasist
bash setup_w2v_v319.sh

bash run_w2v_v319.sh \
  --parent-run exp/w2v_v315_20261008_075905_5663 \
  --parent-checkpoint best_guarded \
  --epochs 5 \
  --submit-best \
  --upload-submission \
  --upload-temp \
  --submission-protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --submission-audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress
```

The launcher runs in the background and opens a fixed-screen viewer. Only progress, epoch metrics and delivery links are displayed; detailed diagnostics remain in the log/report. Ctrl+C exits the viewer only. Startup first verifies/replays the parent, then profiles execution, before epoch 1 begins.

```bash
bash watch_w2v_v319.sh
RUN=$(cat exp/.latest_v319_run)
cat "$RUN/download_links.txt"
```

After a server interruption, remount the data disk at its original path, reactivate the same environment and resume **the new V3.19 run**. The most recent committed epoch, optimizer, sampling/RNG state and numerical plan are restored; an incomplete epoch is replayed. Saved submission/report requests persist. Do not pass changed training settings when resuming.

```bash
RUN=$(cat exp/.latest_v319_run)
bash run_w2v_v319.sh --resume "$RUN"
```

After training, submission and report are uploaded independently to temp.sh. Each URL is recorded in `artifact_links.json` and `download_links.txt`. A failed upload leaves local ZIPs and trained states intact; retry existing archives without training or inference:

```bash
RUN=$(cat exp/.latest_v319_run)
bash upload_w2v_v319.sh --run "$RUN"
```

If an old temp.sh link expired, add `--force`. If inference itself failed before the submission ZIP existed, rerun the completed workflow with `--resume "$RUN"`; it performs the missing export without retraining. The report excludes checkpoints and audio. Neither this repository nor a local code test can generate real competition scores or real temp.sh links before server execution.

## Validation

Tests cover six-version exposure on imbalanced inventories, group loss budgets, missing Online, actual audio transformations, noise-before-RTC, masks, native tiny w2v-BERT/MultiConv forward/backward, physical-batch equivalence, no warmup, exact interrupted resume, best-versus-last export, profiling rollback, upload retries, and the fixed-screen viewer. These are implementation checks, not a GPU accuracy claim.

```bash
python -m unittest discover -s w2v_v319 -t . -p 'test_*.py'
python -m unittest test_v319_progress
```
