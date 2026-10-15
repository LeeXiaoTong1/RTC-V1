# V3.18.7: fresh detector, dynamic source exposure and efficient full-content input

V3.18.7 loads the official Omni W2V 3B pretrained backbone. LoRA, all-layer fusion, AASIST, window attention/statistics pooling, the complete 320-dimensional classifier and detection BN statistics are initialized afresh. `--data-run` reuses data manifests, fixed Dev partitions and asset locations only: no historical detector or optimizer checkpoint is loaded.

Production defaults are **20 epochs, including one head warmup epoch**. Warmup freezes LoRA and omits both PCL branches and EN-real confidence regularization. Joint training then enables these objectives with a gradual ramp. This release does not claim a measured server GPU speedup or a validated optimal learning rate; the real A100 memory/throughput measurements are still required.

## Training composition and exact budgets

Each ordinary logical update contains 24 official Online recordings and 24 independently sampled Offline sources, each generating Noisy A and Noisy B: **72 classified audio views, up to 48 distinct original sources**. These are independent streams, not necessarily 24 same-source triplets. Auxiliary PCL references are encoded only when valid pairs exist and are never classified. With dual PCL, as many as 36 reference views can be added to an ordinary update; they need not be resident in GPU memory simultaneously.

The following counts are **source draws per stream, per epoch**, not numbers of new independent recordings:

| Group | Online draws | Offline simulation draws | Classified Noisy views |
| --- | ---: | ---: | ---: |
| EN-Fake | 9,668 | 9,668 | 19,336 |
| EN-Real | 8,000 | 8,000 | 16,000 |
| ZH-Fake | 8,000 | 8,000 | 16,000 |
| ZH-Real | 8,000 | 8,000 | 16,000 |

Default total: 33,668 draws per stream, 101,004 classification views per epoch, **1,403 optimizer updates**. The final update uses the exact remaining 20 sources per stream (60 classification views), without silently overfilling the budget. Per-group quotas vary slightly to achieve the exact totals; loss normalization uses each update's actual quotas. Four language/class groups retain 25% classification supervision each, and the Online/Noisy streams retain 50% each. EN-Fake exposure matches the prior 2,417 x 4 budget; ZH-Fake is reduced and continuously rotated through the complete pool.

Source cycles continue across epochs. A source is not repeated within its stream/group update. Complete source cycles precede another repeat, and augmented variants refresh their parameters on subsequent cycles. No expanded waveform dataset is saved to disk. Audits distinguish source recordings, repeated source draws, augmentation identities and generated Noisy views.

## Full-content augmentation and short inputs

The waveform path uses mono 16 kHz input and full original content. The content-dropping temporal crop has been replaced by full-duration envelope variation. Small speed changes retain the entire utterance and record an exact time mapping for PCL. Mild frequency-response changes and dynamic parameter refresh supplement the retained noise/RTC scene simulation. Original visits remain approximately 20% and augmented visits approximately 80% in every language/class group over complete cycles; per-epoch realized fractions are reported.

Official Online receives moderate base perturbations. Offline is used for base augmentation followed by noise and RTC simulation, and for paired reference preparation; raw Offline has no classification branch. Real/Fake use the same label-preserving transformation types and strength distributions. There is no permanent deletion of the ZH-Fake source inventory.

Only inputs shorter than 3,920 samples (0.245 seconds at 16 kHz) are extended by time reflection to satisfy the local 12-frame backend minimum. The original prefix stays unchanged. One-sample input uses constant extension. This minimum is a detector implementation requirement, not a claim about an official Omni minimum duration.

PCL excludes artificial mirrored positions and CNN frames whose complete 400-sample receptive field crosses the original boundary. Waveforms shorter than 400 original samples have no valid PCL frames and can receive classification supervision only. Their phoneme metadata explicitly records the exclusion. Original-position contextual representations can still attend to mirrored context through the encoder; this is not a claim of complete isolation from the artificial context.

The dual PCL branches remain: simulated clean RTC versus same-parameter simulated Noisy RTC, and official Online versus same-source simulated clean RTC. Pairing uses valid mapped phone instances and delay/reliability checks. Mirroring is applied after signal processing, and artificial reversed phone instances are not fabricated.

## Architecture, regularization and throughput

The detector retains full-layer Omni fusion, overlapping AASIST windows (128 feature frames, hop 64), and attention-weighted mean plus standard deviation pooling. Tail windows preserve complete temporal coverage. PCL references bypass the classifier and its BN statistics.

EN-real classification weight is not reduced. The existing C3 classification objective and label smoothing 0.02 remain. A small additional EN-real confidence penalty activates only on a correctly oriented, excessively large real-versus-fake logit gap: the default margin is 3.5 and coefficient is 0.01. It is zero for incorrect and borderline EN-real predictions and disabled in head warmup. This class-conditioned heuristic can change calibration; its activation fraction and loss are recorded, and effectiveness must be judged by AUC/EER and Real recall at matched Fake recall, not only the threshold-0.5 recall. Set `--en-real-confidence-weight 0` only on a new controlled comparison run to disable it.

The default physical encoded-view cap is 16, including active auxiliary references, with a 12,000-frame budget. Length-aware packing keeps each source/reference unit intact. CE and both PCL branches share each encoder result. Classification windows interleave sources, use a window batch cap of 32, and do not include auxiliary references in BN. Eight upper trainable layers use activation checkpointing by default. Gradient audits are sparse; full fixed Dev is evaluated each epoch, while extra mechanism/scene panels run every five epochs and at the end.

Logical batch size 72 does not mean 72 waveforms are forced into one physical forward. The cap is deliberately bounded until the longest recordings and active references have been measured on the server. Increase `--microbatch` or change checkpointing only for a **new run**, after measuring throughput and peak memory. A paused run resumes its exact saved configuration.

## Start training

All commands below contain ASCII characters only. Run them after stopping any other training job and after the existing noise and phoneme preparation has completed. Existing compatible V3.18.6 phoneme caches are reused. V3.18.7 does not redownload the noise library or write a new augmented-audio cache during training.

```bash
cd /home/ubuntu/LXT/RTC-w2v-improved
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ubuntu/LXT/v318-runtime/envs/sdd-v318
cd xlsr_aasist

DATA_RUN=$(python -m w2v_v3187.source)

bash run_w2v_v3187.sh \
  --data-run "$DATA_RUN" \
  --epochs 20 \
  --warm-epochs 1 \
  --submit-best \
  --upload-submission \
  --download-dir /home/ubuntu/LXT/temp \
  --submission-protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --submission-audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress
```

`python -m w2v_v3187.source` resolves the latest compatible data-manifest run. If several runs are found without an unambiguous marker, it lists them and asks for an explicit path. Use the chosen directory with `--data-run`; a completed parent checkpoint is not required. Do not reuse a V3.18.6 training directory with `--resume`.

Defaults for prepared manifests are `assets/noise_v3184_rnnoise/manifest.json` and `assets/phonemes_v3186/manifest.json` beneath the checkout. Pass `--noise-manifest` and `--phoneme-manifest` explicitly if those assets were prepared elsewhere. If the phoneme cache must be prepared again, use the V3.18.7 preparation entry point so ultrashort sources receive the proper PCL exclusion metadata.

```bash
DATA_RUN=$(python -m w2v_v3187.source)
bash prepare_phonemes_v3187.sh --data-run "$DATA_RUN"
```

This preparation command verifies and reuses an existing valid cache, the existing phoneme environment and the existing recognizer weights. It does not create a second large model cache. Run it before starting training, if preparation is needed.

## Progress, pause and resume

The launcher starts training in an isolated background session. Ctrl+C in the watcher closes only the viewer. The current progress line includes loss, CE and ETA; each completed epoch's full Dev table is appended and never cleared by the next epoch. Reopening the watcher replays all committed epoch tables once. The detailed raw log remains available through `exp/.latest_v3187_log`.

```bash
bash watch_w2v_v3187.sh
bash watch_w2v_v3187.sh --once
```

Request a graceful pause and wait for the acknowledgement before shutting down the server:

```bash
RUN=$(cat exp/.latest_v3187_run)
bash stop_w2v_v3187.sh --run "$RUN" --timeout 60
```

`STOP_PENDING` means the safe-boundary request is still active. No force kill is sent. Keep the server on until the watcher reports the saved pause and the process exits. Pausing during a long evaluation may take longer than one update. The saved checkpoint contains model, optimizer, RNG, source position and current-epoch accumulators. It preserves the best completed epoch independently of the unfinished current epoch. Pause and preflight do not trigger automatic network delivery.

Resume with the same code, data and settings:

```bash
RUN=$(cat exp/.latest_v3187_run)
bash run_w2v_v3187.sh --resume "$RUN"
```

Do not append new learning rates, epoch counts, batch caps or augmentation settings to a resume command. Saved submission/report plans are retained. Unexpected termination resumes from the most recently committed checkpoint; only a confirmed graceful pause guarantees preservation through its acknowledged optimizer boundary.

## Best submission and both download links

The start command above exports the best checkpoint automatically after completion. It produces a submission ZIP containing only `scores.txt`, plus a separate report ZIP containing configuration, provenance, metrics and diagnostics. Neither report upload nor submission upload includes audio or model weights. The links are saved in the run's `download_links.txt` and `artifact_links.json` and printed separately.

For a manual export after completion, or after a graceful pause with at least one completed epoch:

```bash
RUN=$(cat exp/.latest_v3187_run)
bash submit_w2v_v3187.sh \
  --run "$RUN" \
  --protocol /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/progress.txt \
  --audio-root /home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav/progress \
  --report-dir /home/ubuntu/LXT/temp
```

The submission script selects `best` and uploads **both** artifacts by default. Paused runs cannot export an unfinished partial-epoch `last`. If either upload fails after export, retry without GPU inference:

```bash
RUN=$(cat exp/.latest_v3187_run)
bash submit_w2v_v3187.sh --retry --run "$RUN"
cat "$RUN/download_links.txt"
```

For expired temp.sh links, repeat the retry with `--force`. Existing scores and report files are verified before reuse. If training resumes and changes the checkpoint, an old exported submission cannot silently be reused for the new checkpoint.

## Audits and validation

The run records `input_audit.json`, `augmentation_budget_audit.json`, per-epoch coverage/loss accounting, performance measurements, PCL validity, confidence regularization activity and checkpoint initialization provenance. `diagnostics/curves.html` and CSV files retain all epoch metrics. Best selection uses the same source-isolated Dev partition; calibration is fitted only on its reserved calibration subset after selecting the checkpoint. Progress/Eval is not used for parameter selection.

CPU tests exercise source-budget exactness, dynamic refresh, reflection masks, matching/normalization, shared encoder features, fresh initialization, checkpoint resume, report boundaries and append-only terminal history. Native A100 runtime, peak memory, full-data convergence and Progress improvement require server execution and are not established by these tests.

Final local acceptance: 88 CPU tests passed across the full suite and the final affected lifecycle regressions; 7 Bash entrypoints passed native Bash syntax checks; all 41 Python files parse under Python 3.10 syntax. The release snapshot preserves all 1,090 files of remote base `08f764e4b9f3a80c05d6c9a56d6cd79d3f1c72fe` and adds only the 49 V3.18.7 source, test, entrypoint and documentation files. These are CPU and interface checks, not an A100 throughput or convergence result.
