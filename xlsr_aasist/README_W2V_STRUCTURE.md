# Local acoustic structure: screen first, then one optional fine-tune

The objective is noisy-condition **fake and real discrimination**, rather than
raising real recall in isolation. This implementation keeps w2v-BERT 2.0 and
AASIST. It tests a candidate mechanism; it does not establish an accuracy gain.

## First server command: diagnostic only

Run in the existing `sdd` environment, after pulling this update into the
`RTC-w2v-improved` checkout:

```bash
cd "$HOME/LXT/RTC-w2v-improved/xlsr_aasist"
bash run_w2v_structure.sh exp/w2v_en_20260928_161650_59ab --upload-temp
```

```bash
tail -n 60 -f "$(cat exp/.latest_structure_log)"
```

The launcher immediately prints the process ID and log location. The default
action is a small, read-only mechanism audit, **not another training epoch**.
The prior English run supplies paths and fingerprints only. Inference loads its
recorded original `w2v_rebuild_20260920_093548/stage3/best_model.pt`, never the
English candidate. The hash is verified; the original checkpoint is preserved.

The diagnostic starts with 64 Train and 32 Dev sources per language/class group,
reusing existing noisy caches. Sources stay on their original split; multiple
processed versions of one recording are not independent examples. It compares
local structure stability and label discrimination with the existing global
representation, including held-out processing conditions. This is a mechanism
screen, not full Dev evaluation and not a platform-test experiment.

Reports are written under a fresh `exp/w2v_structure_audit_*` directory. A ZIP
is copied to `/home/ubuntu/LXT/temp`; `--upload-temp` uploads only that report ZIP
to temp.sh and prints `TEMP_DOWNLOAD_URL=...`. It excludes raw audio, checkpoints
and high-dimensional feature caches. If uploading fails, the local ZIP remains.
Save the link printed by your server; an actual link cannot be generated before
the server completes the diagnostic.

Wait for a successful completion marker and inspect `summary.json` / `report.md`.
`ready_for_review` means the measured mechanism is worth reviewing, **not proof
that fine-tuning will improve noisy F1**. `inconclusive` and `reject` do not permit
the dependent training launcher. No status starts training automatically.

## What the optional training changes

- Ordinary crop and noisy condition coverage reuse the preceding coverage recipe.
- The existing noisy-pair global InfoNCE term is replaced by a small local
  structure objective. The real RTC pair term and supervised CE remain.
- Alignment is restricted to corresponding temporal neighborhoods. Low-confidence
  or invalid regions are masked; local feature relations are compared across
  multiple temporal scales. These are SSL **feature relations**, not literal
  physical frequency bins, phoneme labels or a full reproduction of TFCL/PCL.
- The frame sequence is reused from the same backbone forward. No second encoder,
  external speech model, transcription or extra inference branch is introduced.
  Alignment has additional training cost; actual GPU timing must be measured.
- Start from the protected original best for one epoch. Train the final four
  encoder layers and the head at `1e-7` and `2e-6`. The existing real cost `1.25`,
  English class-conditional budgets `0.35/0.40`, and noisy CE budget are retained.
  The initial local weight `0.02` ramps over 100 optimizer steps, independently
  of the older pair warmup. It is deliberately provisional, not a tuned optimum.
- Training records usable-pair and accepted-bin fractions by language/class.
  If the first 100 steps contain no usable pair, the attempt stops rather than
  spending the full epoch on an inactive objective. Training dropout can change
  matching eligibility relative to the frozen diagnostic.
- Candidate ranking prioritizes mean noisy F1. Promotion additionally requires
  improvement in the existing weighted Dev proxy and no decrease in Online F1.
  Recall slices are reported as diagnostics rather than all being hard floors.
  The original model remains available regardless of promotion.

## Training only after reviewing the diagnostic

The explicit training interface is:

```text
bash run_w2v_structure.sh PREVIOUS_RUN --mode train --reviewed-audit AUDIT_DIRECTORY --upload-temp
```

Do not substitute a ZIP file for `AUDIT_DIRECTORY`; it is the completed server
report directory. The launcher verifies completion, the recommendation, the
original baseline SHA256, source configuration, input fingerprints, code and the
effective proposed training recipe. Changed data, code or settings require a new
screen. An optional `--review-note "..."` is saved with the plan. There is no
automatic transition from screening to training and no unreviewed bypass flag.

`--preview` shows a plan without model loading or creating a run directory.
New outputs live in `exp/w2v_structure_train_*`; no old checkpoint or audio cache
is deleted or regenerated. Reports retain the reviewed audit and execution log.

## Evidence and limits

The research motivation is local temporal/feature stability under communication
processing, supported by [RTCFake/PCL](https://arxiv.org/html/2604.23742v1) and
[TFCL](https://arxiv.org/html/2607.17761v1). Different encoders, data and metrics
mean their reported gains are not predictions for this model. Stable features
can also discard useful fake cues; this is why classification, both classes,
source separation and unseen processing are checked before a full fine-tune.

CPU tests cover the implementation, default read-only dispatch, recipe binding,
report exclusion rules and baseline protection. They do not verify full GPU
runtime, server cache completeness or a gain over the existing noisy score.
