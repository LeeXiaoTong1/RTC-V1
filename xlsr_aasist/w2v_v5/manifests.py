"""Verified official speech for screening and the separate Train+Dev refit.

Historical simulated Dev files are evaluation views, never extra training
recordings. Final refitting reads genuine Official Dev Offline/Online instead.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path

from w2v_v31810.identity import build_registry, registry_config, verify_registry
from w2v_v31810.validation import _AudioIdentity, _speech_partition
from w2v_v3185.common import GROUPS

SCHEMA = "rtc_v5_official_refit_manifest_v1"
FORBIDDEN_SPLITS = {"progress", "eval", "evaluation", "test"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _check_scope(rows, expected, *, official=False):
    if not rows:
        raise ValueError("V5 requires nonempty official " + expected + " speech")
    for row in rows:
        split = str(row.get("split", expected)).lower()
        if split != expected:
            raise ValueError("Wrong speech split in V5 " + expected + ": " + str(row.get("id")))
        for key in ("id", "audio", "source_audio"):
            parts = set(str(row.get(key, "")).replace("\\", "/").lower().split("/"))
            if parts & FORBIDDEN_SPLITS:
                raise ValueError("Progress/Eval/Test speech cannot enter V5 preparation")
        if (row.get("language"), row.get("label")) not in GROUPS:
            raise ValueError("Official speech lacks an EN/ZH Real/Fake label")
        if official and (row.get("domain") not in ("offline", "online") or row.get("noisy")):
            raise ValueError("Final refit requires original Official Offline/Online, not simulated views")
    if official and len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Repeated official speech ID")


def _summary(rows):
    cells = {}
    for language, label in GROUPS:
        members = [r for r in rows if (r["language"], r["label"]) == (language, label)]
        cells[f"{language}/{'fake' if label == 0 else 'real'}"] = dict(
            rows=len(members),
            unique_sources=len({r["canonical_source"] for r in members}),
            conditions=dict(Counter(r["condition"] for r in members)),
            condition_sources={kind: len({r["canonical_source"] for r in members
                                         if r["condition"] == kind})
                               for kind in ("offline", "online")})
    return cells


def parent_phoneme_rows(rows):
    """Keep the original phoneme inventory/order and its portable lineage."""
    return [row for row in rows if row.get("v5_origin_split") == "train"]


def bind_parent_phonemes(cfg, rows, *, verify_files=True):
    """Fail visibly if inherited PCL assets cannot cover the declared scope.

    A final refit does not silently fabricate Dev CTC metadata. Its Dev samples
    receive the same classification objective, while PCL stays on parent Train.
    """
    scope = cfg.get("v5_pcl_scope")
    if cfg.get("consistency") != "dual":
        if scope != "disabled":
            raise ValueError("V5 disabled PCL must be explicit in its saved scope")
        return None
    if scope != "parent_train":
        raise ValueError("V5 currently supports inherited PCL only on parent Train")
    manifest = cfg.get("phoneme_manifest")
    if not manifest:
        raise ValueError("V5 parent-Train PCL needs the verified parent phoneme manifest")
    from w2v_v3189.phonemes import bind_manifest
    binding = bind_manifest(manifest, parent_phoneme_rows(rows), verify_files=verify_files)
    expected = cfg.get("phoneme_provenance")
    if expected is not None and binding != expected:
        raise ValueError("V5 inherited phoneme provenance changed")
    return binding


def _training_dev(official):
    groups = defaultdict(list)
    for row in official:
        groups[row["canonical_source"]].append(row)
    anchors = {}
    for source, members in groups.items():
        offline = sorted((r for r in members if r["domain"] == "offline"), key=lambda r: r["id"])
        if not offline:
            raise ValueError("Official Dev Online lacks a verified Offline source")
        anchors[source] = offline[0]
    output = []
    for raw in official:
        row = dict(raw)
        anchor = anchors[row["canonical_source"]]
        import soundfile as sf
        info = sf.info(row["audio"])
        # read_wave() resample_poly returns ceil(frames * target_sr / source_sr).
        samples = (int(info.frames) * 16000 + int(info.samplerate) - 1) // int(info.samplerate)
        if samples < 1:
            raise ValueError("Empty genuine Official Dev waveform")
        row.update(split="train", v5_origin_split="dev", v5_pcl_eligible=False,
                   source_id=anchor["id"], group_id=anchor["audio_sha256"],
                   source_sha256=anchor["audio_sha256"], output_samples=samples,
                   full_length=True, view="full", condition=row["domain"], noisy=False, band=-1)
        # Generated cache lineage is never inherited by this original waveform.
        for key in ("bank", "processing_family", "source_audio"):
            row.pop(key, None)
        output.append(row)
    return output


def prepare_manifests(cfg, train, historical_dev, *, phase="screen", official=None, identity=None):
    """Return (training_rows, evaluation_rows, source_registry, audit).

    Updates cfg's registry/provenance bindings. In screen, evaluation_rows are
    copies of the unchanged historical Dev views, preserving score ordering.
    In final, evaluation_rows is empty: trained Dev is not a validation set.
    A caller may reuse ``identity`` between phases to avoid decoding PCM twice.
    """
    if phase not in ("screen", "final"):
        raise ValueError("V5 manifest phase must be screen or final")
    _check_scope(train, "train", official=True)
    _check_scope(historical_dev, "dev")
    identity = identity or _AudioIdentity()
    if official is None:
        from w2v_aasist.data import read_protocol
        official = read_protocol(cfg["official_dev_protocol"], cfg["official_dev_root"])
    official = copy.deepcopy(official)
    _check_scope(official, "dev", official=True)
    if set(r["id"] for r in train) & set(r["id"] for r in official):
        raise ValueError("Official Train and Dev IDs overlap")
    if cfg.get("official_dev_protocol"):
        identity.file(cfg["official_dev_protocol"])
    partitions, identity_audit, prepared_train, old_registry = _speech_partition(
        cfg, train, historical_dev, official, identity)
    prepared_by_id = {r["id"]: r for subset in partitions.values() for r in subset}
    prepared_official = [prepared_by_id[row["id"]] for row in official]
    eligible = cfg.get("consistency") == "dual"
    for row in prepared_train:
        row.update(v5_origin_split="train", v5_pcl_eligible=eligible)
    if phase == "final":
        added = _training_dev(prepared_official)
        output = prepared_train + added
        evaluation = []
        registry = build_registry(output, [], old_registry["official_pair_records"],
                                  old_registry.get("pairing_audit"))
    else:
        added = []
        output = prepared_train
        evaluation = copy.deepcopy(historical_dev)
        registry = old_registry
    cfg["source_registry"] = registry_config(registry)
    verify_registry(output, registry, cfg, {"source_registry_hash": registry["registry_hash"]})
    cfg["identity_input_files"] = dict(identity.input_files)
    cfg["v5_pcl_scope"] = "parent_train" if eligible else "disabled"
    cfg["v5_parent_train_count"] = len(prepared_train)
    cfg["v5_parent_train_ids_sha256"] = _digest([r["id"] for r in prepared_train])
    cfg["v5_manifest_phase"] = phase
    audit = dict(
        schema=SCHEMA, phase=phase, identity=identity_audit,
        train=_summary(prepared_train), official_dev=_summary(prepared_official),
        added=_summary(added), final_training=_summary(output),
        historical_dev_views=len(historical_dev), official_dev_rows=len(prepared_official),
        final_dev_rows_added=len(added), historical_noisy_views_added=0,
        parent_train_rows=len(prepared_train), training_rows=len(output),
        evaluation_rows=len(evaluation), source_registry_hash=registry["registry_hash"],
        final_selection_policy=("No validation/checkpoint selection on trained Dev; fixed screened recipe"
                                if phase == "final" else "Original Train only; Dev remains held out"),
        pcl=dict(scope=cfg["v5_pcl_scope"], eligible_rows=sum(r["v5_pcl_eligible"] for r in output),
                 dev_rows_with_classification_only=len(added),
                 reason="Preserve verified inherited Train phonemes; no fabricated Dev phoneme labels"),
        missing_online_policy="Retain genuine Offline sources; never synthesize an official Online partner")
    cfg["v5_manifest_audit"] = audit
    return output, evaluation, registry, audit


def apply_coverage_budgets(cfg, rows):
    """Final refit visits every canonical source per stream at least once/epoch.

    Parent exposure is a minimum, not a truncation. Losses remain balanced by
    language/class. Screening deliberately retains the original exposure budget.
    """
    if cfg.get("v5_manifest_phase") != "final":
        raise ValueError("Full-source exposure expansion is only for the final Train+Dev refit")
    batch = cfg["stream_sources"]
    if type(batch) is not int or batch < 8 or batch % 4:
        raise ValueError("V5 refit requires stream_sources divisible by four and >= 8")
    previous = cfg["source_draws_per_stream"]
    if len(previous) != 4 or any(type(n) is not int or n < 1 for n in previous):
        raise ValueError("Four positive parent exposure budgets are required")
    counts = []
    for group in GROUPS:
        counts.append({kind: len({r["canonical_source"] for r in rows
                                  if (r["language"], r["label"]) == group and r["condition"] == kind})
                       for kind in ("offline", "online")})
    if any(min(cell.values()) < 1 for cell in counts):
        raise ValueError("Every classification group needs Offline and Online sources")
    unit = batch // 4
    budgets = [math.ceil(max(old, *cell.values()) / unit) * unit
               for old, cell in zip(previous, counts)]
    from w2v_v31810.sampling import quota_schedule
    schedule = quota_schedule(tuple(budgets), batch)
    cfg["source_draws_per_stream"] = budgets
    report = dict(
        previous_draws=list(previous), draws=budgets, steps_per_epoch=len(schedule),
        canonical_sources_by_group={f"{language}/{label}": cell
                                    for (language, label), cell in zip(GROUPS, counts)},
        policy="At least one full canonical-source cycle per stream/group per final epoch; inherited draws are minima",
        classification_loss_weights_unchanged=True)
    cfg["v5_full_source_coverage"] = report
    return report
