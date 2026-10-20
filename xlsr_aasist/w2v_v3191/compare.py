"""Compare matched V3.19.1 arms on already selected final checkpoints.

Usage: python -m w2v_v3191.compare --runs RUN_BASELINE RUN_EXPANDED RUN_HARD
No model is fitted, no threshold is tuned, and no checkpoint is selected here.
"""
import argparse
from pathlib import Path

import numpy as np

from .robust_eval import (_read, _write, _hash, _file_hash, _verify_panel,
                          read_scores, paired_bootstrap, LIMITATION)


def _identity(cfg, final, panel):
    origin = cfg["parent_init"]
    # Match candidate recipes and actual training exposure, not deliberately
    # wasted inference. Scoring cost is reported separately from update budget.
    fields = (
        "seed", "sampling_seed", "epochs", "stream_sources", "source_draws_per_stream",
        "source_epoch_offset", "group_quotas", "view_factors",
        "refresh_base_variants", "lora_layers", "retained_lora_layers",
        "consistency", "backend_precision", "lora_lr", "head_lr",
        "fusion_lr", "layer_mix_lr", "lr_floor", "lora_early_lr",
        "lora_middle_lr", "classifier_lr", "evidence_lr", "pool_lr",
        "microbatch", "frame_budget", "eval_batch", "window_batch",
        "meta_beta", "paired_classification", "label_smoothing",
        "classification_ce_weights", "classification_objective",
        "max_grad_norm", "platform_sources_per_group", "lr_warmup_updates",
        "checkpointing", "checkpoint_layers", "fusion_checkpointing",
    )
    # Coefficients, sampling/candidate controls, validity guards and promotion
    # rules are part of the experiment, including options added in the future.
    controls = {key: value for key, value in cfg.items()
                if key != "robust_arm" and key.startswith(
                    ("robust_", "pcl_", "rtc_", "scene_", "en_real_confidence", "risk_"))}
    return dict(
        parent={key: origin[key] for key in ("checkpoint_sha256", "tag", "config_sha256")},
        panel_hash=panel["hash"],
        source_registry_hash=cfg.get("source_registry", {}).get("registry_hash"),
        training_budget={key: cfg.get(key) for key in fields},
        condition_and_loss_controls=controls,
        compare_budget=final.get("compare_budget"),
    )


def _validate_report(run):
    run = Path(run).resolve()
    cfg, final = _read(run / "config.json"), _read(run / "robust_final.json")
    panel = _read(run / "robust_panel.json")
    _verify_panel(panel)
    if (final.get("selection_split") != "select"
            or final.get("audit_used_for_selection") is not False):
        raise ValueError("Comparison requires a checkpoint frozen before Audit")
    chosen = final["chosen_checkpoint_tag"]
    for split in ("select", "audit"):
        report = final[split]
        if (report["panel_hash"] != panel["hash"] or report["split"] != split
                or report["checkpoint_tag"] != chosen or report.get("threshold") != .5
                or report.get("calibration_fitted") is not False):
            raise ValueError("Final receiver reports mismatch frozen checkpoint/protocol")
    if not final.get("compare_budget"):
        raise ValueError("Actual optimizer/source budget is required for comparison")
    done = _read(run / "completed.json")
    if done.get("status") != "complete" or done.get("best_tag") != chosen:
        raise ValueError("Only final completed, selected checkpoints can be compared")
    if done.get("checkpoint_sha256") and _file_hash(run / "last.pt") != done["checkpoint_sha256"]:
        raise ValueError("Completed checkpoint changed")
    return run, cfg, final, panel


def _pcm_identity(run, result):
    path = Path(run) / result["predictions"]["file"]
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].tolist() for key in
                ("row_ids", "source_ids", "waveform_sha256")}


def compare_runs(runs, *, draws=2000, seed=3191001):
    if len(runs) < 2:
        raise ValueError("Provide at least two matched experimental arms")
    records = [_validate_report(run) for run in runs]
    base_run, base_cfg, base_final, base_panel = records[0]
    expected = _identity(base_cfg, base_final, base_panel)
    arms = [cfg.get("robust_arm") for _, cfg, _, _ in records]
    if len(set(arms)) != len(arms) or any(a not in ("baseline", "expanded", "hard") for a in arms):
        raise ValueError("One run per baseline/expanded/hard arm and training seed is required")
    output = dict(
        schema="rtc_v3191_matched_arm_comparison_v1", reference_run=str(base_run),
        reference_arm=arms[0], protocol_hash=base_panel["hash"],
        matching_identity=expected, comparisons=[], limitation=LIMITATION,
        scope="Equal source/outer-update budgets and candidate recipes; unused scoring/DSP may be omitted; compare actual compute costs separately",
        selection="Already fixed on Select; this command never chooses a checkpoint or arm",
        confidence_note="Marginal source-cluster intervals; not a 95% benchmark-success probability")
    for run, cfg, final, panel in records[1:]:
        actual = _identity(cfg, final, panel)
        if actual != expected:
            unequal = [key for key in expected if actual[key] != expected[key]]
            raise ValueError("Runs are not matched on " + ", ".join(unequal))
    pairs = [(records[0], record) for record in records[1:]]
    by_arm = {record[1]["robust_arm"]: record for record in records}
    if len(records) == 3:
        existing = {(left[1]["robust_arm"], right[1]["robust_arm"]) for left, right in pairs}
        if ("expanded", "hard") not in existing and ("hard", "expanded") not in existing:
            pairs.append((by_arm["expanded"], by_arm["hard"]))
    for reference, candidate in pairs:
        reference_run, reference_cfg, reference_final, _ = reference
        run, cfg, final, panel = candidate
        comparisons = {}
        for split in ("select", "audit"):
            left, right = reference_final[split], final[split]
            a, b = read_scores(reference_run, left), read_scores(run, right)
            if _pcm_identity(reference_run, left) != _pcm_identity(run, right):
                raise ValueError("Receiver PCM/order differ across experimental arms")
            rows = [panel["rows"][i] for i in panel["split_indices"][split]]
            comparison = paired_bootstrap(rows, a, b, draws=draws, seed=seed)
            comparison["reference_score"] = left["score"]
            comparison["candidate_score"] = right["score"]
            comparison["interpretation"] = (
                "Descriptive selection-set comparison; selection optimism applies"
                if split == "select" else
                "Final historical-Dev Audit comparison; do not select a new checkpoint on this result")
            comparisons[split] = comparison
        output["comparisons"].append(dict(
            run=str(run), arm=cfg["robust_arm"],
            reference_run=str(reference_run), reference_arm=reference_cfg["robust_arm"],
            chosen_checkpoint_tag=final["chosen_checkpoint_tag"],
            reference_checkpoint_tag=reference_final["chosen_checkpoint_tag"],
            results=comparisons))
    output["identity_sha256"] = _hash(expected)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3191001)
    args = parser.parse_args()
    output = compare_runs(args.runs, draws=args.draws, seed=args.seed)
    _write(args.out, output)
    for item in output["comparisons"]:
        interval = item["results"]["audit"]["metrics"]["score"]
        print(f'{item["arm"]} vs {item["reference_arm"]}: Audit score delta={interval["delta"]*100:+.3f} pp; '
              f'95% source interval=[{interval["low"]*100:+.3f}, '
              f'{interval["high"]*100:+.3f}] pp')
    print("Historical Dev only; this does not estimate benchmark-success probability.")
    print("COMPARISON=" + str(Path(args.out).resolve()))


if __name__ == "__main__":
    main()
