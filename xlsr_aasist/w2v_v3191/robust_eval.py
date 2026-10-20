"""Frozen source-clustered Dev receiver panel; never reads evaluation audio.

This panel postprocesses ONE frozen inherited synthetic Noisy view per Dev
source. It is not new clean/noisy RTC capture and is not a blind benchmark.
The inherited calibration partition is used only as a final audit here.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

SCHEMA = "rtc_v3191_receiver_panel_v1"
PROTOCOL_SEED = 3191001
LIMITATION = (
    "Inherited Dev sources have previously informed model development. These "
    "are fixed synthetic Noisy waveforms followed by withheld local receiver "
    "postprocessors, not unseen real RTC platforms. Source-cluster confidence "
    "intervals describe this Dev panel, not a probability of benchmark success."
)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _seed(*values):
    return int(_hash(values)[:16], 16) % (2**32)


def _wave_hash(wave):
    return hashlib.sha256(np.asarray(wave, dtype="<f4").tobytes()).hexdigest()


def _train_rows(cfg, run):
    explicit = cfg.get("robust_train_manifest")
    choices = [Path(explicit)] if explicit else [Path(run) / "train_rows.json"]
    if cfg.get("parent_init", {}).get("run"):
        choices.append(Path(cfg["parent_init"]["run"]) / "train_rows.json")
    for path in choices:
        if path.is_file():
            return _read(path)
    raise FileNotFoundError("The verified Train manifest is required for receiver-panel isolation")


def _validate_isolation(dev, partition, train):
    n = len(dev)
    if not n or len(partition.get("source_keys", [])) != n:
        raise ValueError("Receiver panel requires inherited complete Dev source keys")
    select = partition.get("select", [])
    audit = partition.get("audit", partition.get("calibration", []))
    if (len(set(select)) != len(select) or len(set(audit)) != len(audit)
            or set(select) & set(audit) or set(select) | set(audit) != set(range(n))):
        raise ValueError("Select/Audit must be disjoint and cover the full Dev manifest")
    if any(r.get("split") != "dev" for r in dev):
        raise ValueError("Receiver validation accepts only official-development-derived rows")
    if any(r.get("split") != "train" for r in train):
        raise ValueError("Train isolation manifest contains non-Train rows")
    aliases = ("canonical_source", "source_id", "group_id",
               "source_sha256", "audio_sha256", "decoded_pcm_sha256")
    for field in aliases:
        a = {str(r[field]) for r in train if r.get(field)}
        b = {str(r[field]) for r in dev if r.get(field)}
        if a & b:
            raise ValueError("Train/Dev identity overlap in " + field)
    select_indices = set(select)
    selected = {partition["source_keys"][i] for i in select}
    audited = {partition["source_keys"][i] for i in audit}
    if selected & audited:
        raise ValueError("One complete source crosses Select/Audit")
    return {i: "select" if i in select_indices else "audit" for i in range(n)}


def _canonical_dev_sources(cfg, run, dev, partition):
    """Bind old cached variants to the SAME verified PCM graph used by Train."""
    registry_path = Path(run) / "source_registry.json"
    expected = cfg.get("source_registry", {}).get("registry_hash")
    if not registry_path.is_file():
        if expected:
            raise FileNotFoundError("Verified PCM registry is missing from receiver evaluation")
        return [str(key) for key in partition["source_keys"]], "inherited_pairing_only"
    registry = _read(registry_path)
    actual = _hash({k: v for k, v in registry.items() if k != "registry_hash"})
    if actual != registry.get("registry_hash") or (expected and expected != actual):
        raise ValueError("Receiver source PCM registry changed")
    aliases, basenames = defaultdict(set), defaultdict(set)
    for row in registry["dev"]:
        source = row.get("canonical_source")
        if not source:
            raise ValueError("Dev PCM registry lacks a canonical source")
        for field in ("id", "source_id"):
            if row.get(field):
                value = str(row[field]).replace("\\", "/")
                aliases[("id", value)].add(source)
                basenames[Path(value).name].add(source)
        for field in ("audio_sha256", "decoded_pcm_sha256", "source_sha256"):
            if row.get(field):
                aliases[(field, str(row[field]))].add(source)
    result = []
    for row in dev:
        candidates = set()
        for field in ("id", "source_id"):
            if row.get(field):
                candidates.update(aliases.get(("id", str(row[field]).replace("\\", "/")), ()))
        for field in ("audio_sha256", "decoded_pcm_sha256", "source_sha256"):
            if row.get(field):
                candidates.update(aliases.get((field, str(row[field])), ()))
        if not candidates:
            # Canonical historical source_id may be a relative/absolute path
            # while the official protocol ID is relative. Basename matching is
            # allowed only when the PCM graph makes that reference unambiguous.
            name = Path(str(row.get("source_id", "")).replace("\\", "/")).name
            candidates.update(basenames.get(name, ()))
        if len(candidates) != 1:
            raise ValueError("Cannot unambiguously bind historical Dev view to PCM registry: " + row["id"])
        result.append(str(next(iter(candidates))))
    train_sources = {str(row["canonical_source"]) for row in registry["train"]}
    if train_sources & set(result):
        raise ValueError("Train and receiver-panel canonical PCM sources overlap")
    select_sources = {result[i] for i in partition["select"]}
    audit_sources = {result[i] for i in partition.get("audit", partition.get("calibration", []))}
    if select_sources & audit_sources:
        raise ValueError("PCM-identical Dev source crosses inherited Select/Audit; rebuild split explicitly")
    return result, "verified_shared_Train_Dev_PCM_registry"


def prepare_panel(cfg, run, dev, partition):
    """Persist a common manifest for all arms; audio stays in its parent cache.

    Each complete source contributes two views (one per heldout mechanism).
    Which inherited noisy lane supplies that source is frozen independently of
    training seed/arm. Output PCM hashes are checked on every evaluation.
    """
    from .mechanisms import mechanism_catalog

    catalog = mechanism_catalog(cfg)
    if not catalog["heldout"] or set(catalog["heldout"]) & set(catalog["train"]):
        raise ValueError("Heldout receiver implementations overlap training")
    run = Path(run)
    assignment = _validate_isolation(dev, partition, _train_rows(cfg, run))
    source_keys, identity_policy = _canonical_dev_sources(cfg, run, dev, partition)
    grouped = defaultdict(list)
    for i, source in enumerate(source_keys):
        grouped[str(source)].append(i)
    rows = []
    seed = int(cfg.get("robust_protocol_seed", PROTOCOL_SEED))
    files = {}
    source_meta = []
    for source, indices in sorted(grouped.items()):
        labels = {(dev[i]["language"], dev[i]["label"]) for i in indices}
        if len(labels) != 1:
            raise ValueError("Source has conflicting language/class metadata")
        noisy = [i for i in indices if dev[i].get("condition") in ("seen", "heldout")]
        if not noisy:
            raise ValueError("Full-source receiver panel lacks a frozen noisy view: " + source)
        base = min(noisy, key=lambda i: (_seed(seed, "base-lane", source, dev[i]["id"]),
                                        dev[i]["id"]))
        online = [i for i in indices if dev[i].get("condition") == "online"]
        chosen_online = min(online, key=lambda i: dev[i]["id"]) if online else None
        r = dev[base]
        if r["language"] not in ("en", "zh") or r["label"] not in (0, 1):
            raise ValueError("Unsupported Dev stratum")
        path = str(Path(r["audio"]).resolve())
        if path not in files:
            files[path] = _file_hash(path)
        source_meta.append(dict(source_id=source, split=assignment[base],
                                base_index=base, online_index=chosen_online))
        for mechanism in catalog["heldout"]:
            rows.append(dict(
                id=_hash((source, mechanism)), audio=path, split="dev",
                panel_split=assignment[base], source_id=source, group_id=source,
                language=r["language"], label=r["label"], full_length=True,
                condition="receiver_" + mechanism, mechanism=mechanism,
                mechanism_seed=_seed(seed, "receiver", source, mechanism),
                base_condition=r["condition"], base_id=r["id"], base_index=base,
                online_index=chosen_online, audio_sha256=files[path]))
    splits = {part: [i for i, r in enumerate(rows) if r["panel_split"] == part]
              for part in ("select", "audit")}
    for part, indices in splits.items():
        strata = {(rows[i]["language"], rows[i]["label"]) for i in indices}
        if strata != {("en", 0), ("en", 1), ("zh", 0), ("zh", 1)}:
            raise ValueError("Receiver partition lacks one of four strata: " + part)
    body = dict(
        schema=SCHEMA, protocol_seed=seed, rows=rows, split_indices=splits,
        source_manifest=source_meta, inherited_partition_sha256=_hash(partition),
        parent_checkpoint_sha256=cfg["parent_init"]["checkpoint_sha256"],
        parent_tag=cfg["parent_init"]["tag"], input_files=files, catalog=catalog,
        provenance=dict(
            input="One deterministic frozen inherited synthetic Noisy lane per complete Dev source",
            processing="Heldout postprocessing implementation and heldout composition",
            retained_all_sources=True, no_progress_data=True, waveform_cache=False,
            source_identity=identity_policy,
            audit_policy="Inherited calibration sources; final chosen checkpoint only; no fitting",
            score="Unweighted mean of Macro-F1 over heldout mechanism x language",
            threshold=.5, limitation=LIMITATION))
    body["hash"] = _hash(body)
    manifest_path = run / "robust_panel.json"
    if manifest_path.is_file() and _read(manifest_path) != body:
        raise ValueError("Frozen receiver panel changed; refuse resume/comparison")
    _write(manifest_path, body)
    return body


def _verify_panel(panel):
    body = {k: v for k, v in panel.items() if k != "hash"}
    if panel.get("schema") != SCHEMA or panel.get("hash") != _hash(body):
        raise ValueError("Receiver panel identity changed")


class _PanelWaves:
    def __init__(self, rows, cfg):
        self.rows, self.cfg = rows, cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from w2v_v3184.data import read_wave
        from w2v_v3189.data import ready
        from .augment import apply_view
        row = self.rows[index]
        wave = read_wave(row)
        output, metadata = apply_view(wave, row["mechanism"],
                                      row["mechanism_seed"], self.cfg)
        output = np.asarray(output, dtype=np.float32)
        if output.ndim != 1 or not len(output) or not np.isfinite(output).all():
            raise ValueError("Invalid receiver panel PCM")
        return [dict(row, wave=ready(output), row_index=index,
                     waveform_sha256=_wave_hash(output), augmentation=metadata)]


def infer_panel(model, rows, cfg):
    """Transform in CPU workers and use the same recording inference policy."""
    from .common import observation
    from .data import loader, close, microbatches
    local = dict(cfg, workers=cfg.get("eval_workers", cfg["workers"]))
    batches = loader(_PanelWaves(rows, local), local, batch_size=local["eval_batch"])
    logits, hashes = [], []
    last_report = 0
    try:
        with observation(model):
            for units in batches:
                expected = [r for unit in units for r in unit]
                positions = {id(row): i for i, row in enumerate(expected)}
                values = [None] * len(expected)
                for micro in microbatches(units, local["microbatch"], local["frame_budget"]):
                    examples = [r for unit in micro for r in unit]
                    z = model([r["wave"] for r in examples]).float().cpu().numpy()
                    for row, prediction in zip(examples, z):
                        values[positions[id(row)]] = prediction
                if any(z is None for z in values):
                    raise ValueError("Receiver inference lost a row")
                logits.extend(values)
                hashes.extend(r["waveform_sha256"] for r in expected)
                if (len(logits) - last_report >= int(local.get("robust_eval_log_interval", 128))
                        or len(logits) == len(rows)):
                    last_report = len(logits)
                    print(f"Receiver {rows[0]['panel_split']} evaluation "
                          f"{len(logits)}/{len(rows)}", flush=True)
    finally:
        close(batches)
    return np.asarray(logits), hashes


def _predictions(logits, count):
    z = np.asarray(logits)
    if z.shape != (count, 2) or not np.isfinite(z).all():
        raise ValueError("Expected finite [N, 2] fake/real logits")
    # P(fake) >= .5 is fake, including exact ties, as in exported scoring.
    return (z[:, 1] > z[:, 0]).astype(np.int64)


def _binary_metrics(labels, predictions, weights=None):
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights)
    counts = np.bincount(np.asarray(labels) * 2 + np.asarray(predictions),
                         weights=weights, minlength=4).reshape(2, 2)
    support = counts.sum(1)
    recall = np.divide(np.diag(counts), support, out=np.zeros(2), where=support > 0)
    denominator = counts.sum(1) + counts.sum(0)
    f1 = np.divide(2 * np.diag(counts), denominator, out=np.zeros(2),
                   where=denominator > 0)
    return dict(macro_f1=float(f1.mean()), recall=recall.tolist(),
                class_counts=support.tolist(), confusion=counts.tolist())


def summarize_panel(rows, logits, weights=None):
    y = np.asarray([r["label"] for r in rows])
    p = _predictions(logits, len(rows))
    w = np.ones(len(rows)) if weights is None else np.asarray(weights)
    mechanisms, groups, balanced_scores = {}, {}, []
    names = sorted({r["mechanism"] for r in rows})
    for name in names:
        ix = np.asarray([i for i, r in enumerate(rows) if r["mechanism"] == name])
        result = _binary_metrics(y[ix], p[ix], w[ix])
        result["by_language"] = {}
        for lang in ("en", "zh"):
            sub = np.asarray([i for i in ix if rows[i]["language"] == lang], dtype=int)
            item = _binary_metrics(y[sub], p[sub], w[sub])
            result["by_language"][lang] = item
            balanced_scores.append(item["macro_f1"])
        mechanisms[name] = result
    for lang in ("en", "zh"):
        for label, name in ((0, "fake"), (1, "real")):
            ix = np.asarray([i for i, r in enumerate(rows)
                             if r["language"] == lang and r["label"] == label], dtype=int)
            count = float(w[ix].sum())
            errors = float(w[ix][p[ix] != label].sum())
            groups[lang + "/" + name] = dict(count=count, errors=errors,
                                             recall=1. - errors / count if count else 0.)
    values = [m["macro_f1"] for m in mechanisms.values()]
    return dict(count=len(rows), sources=len({r["source_id"] for r in rows}),
                score=float(np.mean(balanced_scores)),
                average_macro_f1=float(np.mean(values)),
                worst_mechanism_f1=float(min(values)),
                groups=groups, mechanisms=mechanisms)


def transitions(rows, reference_logits, logits, reference_kind="base_index"):
    """Matched flips are descriptive; extra receiver processing is not pure noise."""
    z = np.asarray(reference_logits)
    p = _predictions(logits, len(rows))
    result = {}
    for lang in ("en", "zh"):
        for label, name in ((0, "fake"), (1, "real")):
            indices = [i for i, r in enumerate(rows)
                       if r["language"] == lang and r["label"] == label
                       and r.get(reference_kind) is not None]
            if not indices:
                continue
            prior = _predictions(z[[rows[i][reference_kind] for i in indices]], len(indices))
            current = p[indices]
            a, b = prior == label, current == label
            result[lang + "/" + name] = dict(
                matched_views=len(indices), correct_to_wrong=int(np.sum(a & ~b)),
                wrong_to_correct=int(np.sum(~a & b)), persistent_wrong=int(np.sum(~a & ~b)),
                correct_to_wrong_rate=float(np.mean(~b[a])) if a.any() else None)
    return result


def paired_bootstrap(rows, baseline_logits, candidate_logits, draws=1000, seed=3191001):
    """Stratified complete-source bootstrap, retaining all mechanism views."""
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    a = _predictions(baseline_logits, len(rows))
    b = _predictions(candidate_logits, len(rows))
    source_rows, strata = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        source_rows[r["source_id"]].append(i)
    for source, indices in source_rows.items():
        tags = {(rows[i]["language"], rows[i]["label"]) for i in indices}
        if len(tags) != 1:
            raise ValueError("Bootstrap source has conflicting labels")
        strata[next(iter(tags))].append(source)
    if set(strata) != {("en", 0), ("en", 1), ("zh", 0), ("zh", 1)}:
        raise ValueError("Bootstrap requires all four source strata")
    # Aggregated source confusion matrices make 1000 draws independent of
    # waveform duration and avoid rebuilding row masks inside every replicate.
    sources = sorted(source_rows)
    source_position = {s: i for i, s in enumerate(sources)}
    cells = sorted({(r["mechanism"], r["language"]) for r in rows})
    cell_position = {key: i for i, key in enumerate(cells)}
    counts_a = np.zeros((len(sources), len(cells), 4), dtype=np.float64)
    counts_b = np.zeros_like(counts_a)
    for i, r in enumerate(rows):
        s, c = source_position[r["source_id"]], cell_position[r["mechanism"], r["language"]]
        counts_a[s, c, r["label"] * 2 + a[i]] += 1
        counts_b[s, c, r["label"] * 2 + b[i]] += 1

    def scores(counts):
        matrix = counts.reshape(-1, 2, 2)
        denominator = matrix.sum(1) + matrix.sum(2)
        f1 = np.divide(2 * np.diagonal(matrix, axis1=1, axis2=2), denominator,
                       out=np.zeros_like(denominator), where=denominator > 0).mean()
        values = [f1]
        for lang in ("en", "zh"):
            cm = matrix[[j for j, c in enumerate(cells) if c[1] == lang]].sum(0)
            for label in (0, 1):
                support = cm[label].sum()
                values.append(cm[label, label] / support if support else 0.)
        return np.asarray(values)

    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        multiplicity = np.zeros(len(sources), dtype=np.float64)
        for values in strata.values():
            positions = [source_position[s] for s in values]
            multiplicity[positions] = rng.multinomial(len(values),
                                                      np.full(len(values), 1 / len(values)))
        ca = np.einsum("s,scv->cv", multiplicity, counts_a)
        cb = np.einsum("s,scv->cv", multiplicity, counts_b)
        differences.append(scores(cb) - scores(ca))
    differences = np.asarray(differences)
    delta = scores(counts_b.sum(0)) - scores(counts_a.sum(0))
    interval = np.quantile(differences, [.025, .975], axis=0)
    names = ["score", "en/fake_recall", "en/real_recall", "zh/fake_recall", "zh/real_recall"]
    return dict(
        unit="Complete source, all receiver views together; stratified language/class",
        draws=draws, seed=seed, confidence=.95,
        metrics={name: dict(delta=float(delta[i]),
                            low=float(interval[0, i]), high=float(interval[1, i]))
                 for i, name in enumerate(names)},
        minimum_stratum_sources=min(map(len, strata.values())),
        interval_scope=LIMITATION, simultaneous_confidence=False,
        note="Intervals are marginal, not corrected for multiple comparisons or repeated model selection.")


def evaluate_panel(model, cfg, run, panel, baseline_logits=None, infer_fn=None,
                   *, split="select", checkpoint_tag="parent", reference_logits=None,
                   bootstrap=False):
    """Evaluate Select during training, Audit only after the checkpoint is fixed."""
    _verify_panel(panel)
    if split not in ("select", "audit"):
        raise ValueError("Choose receiver Select or final Audit")
    if split == "audit" and not cfg.get("robust_final_audit", False):
        raise ValueError("Audit is locked until final checkpoint selection")
    rows = [panel["rows"][i] for i in panel["split_indices"][split]]
    for path in {r["audio"] for r in rows}:
        if _file_hash(path) != panel["input_files"][path]:
            raise ValueError("Frozen parent Dev audio changed")
    output = (infer_fn or infer_panel)(model, rows, cfg)
    if isinstance(output, tuple):
        logits, hashes = output
    else:
        logits, hashes = output, None
    logits = np.asarray(logits)
    _predictions(logits, len(rows))
    if hashes is None:
        # An injected inference function is a testing hook; production always
        # binds generated PCM. Never imply unobserved hashes were verified.
        if infer_fn is None:
            raise ValueError("Production receiver inference must return PCM hashes")
        hashes = ["unverified_injected_inference"] * len(rows)
    if len(hashes) != len(rows):
        raise ValueError("Receiver PCM hash count differs")
    run = Path(run)
    pcm_path = run / f"robust_pcm_{split}.json"
    pcm_identity = dict(panel_hash=panel["hash"], row_ids=[r["id"] for r in rows], hashes=list(hashes))
    if pcm_path.is_file() and _read(pcm_path) != pcm_identity:
        raise ValueError("Deterministic receiver waveform identity changed")
    _write(pcm_path, pcm_identity)
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", str(checkpoint_tag))
    scores_path = run / f"robust_scores_{split}_{safe_tag}.npz"
    np.savez_compressed(scores_path, logits=logits,
                        row_ids=np.asarray([r["id"] for r in rows]),
                        source_ids=np.asarray([r["source_id"] for r in rows]),
                        waveform_sha256=np.asarray(hashes))
    result = dict(schema=SCHEMA, panel_hash=panel["hash"], split=split,
                  checkpoint_tag=checkpoint_tag, threshold=.5, calibration_fitted=False,
                  limitation=LIMITATION, **summarize_panel(rows, logits),
                  predictions=dict(file=scores_path.name, sha256=_file_hash(scores_path)))
    if reference_logits is not None:
        result["matched_transitions"] = {
            "inherited_noisy_to_receiver": transitions(rows, reference_logits, logits),
            "official_online_to_receiver": transitions(rows, reference_logits, logits, "online_index")}
    if baseline_logits is not None:
        prior = summarize_panel(rows, baseline_logits)
        result["delta_vs_parent"] = dict(
            score=result["score"]-prior["score"],
            average_macro_f1=result["average_macro_f1"]-prior["average_macro_f1"],
            worst_mechanism_f1=result["worst_mechanism_f1"]-prior["worst_mechanism_f1"],
            group_recall={k: result["groups"][k]["recall"]-prior["groups"][k]["recall"]
                          for k in prior["groups"]})
        if bootstrap:
            result["paired_comparison"] = paired_bootstrap(
                rows, baseline_logits, logits, int(cfg.get("robust_bootstrap_draws", 1000)),
                int(cfg.get("robust_protocol_seed", PROTOCOL_SEED)))
    _write(run / f"robust_metrics_{split}_{safe_tag}.json", result)
    return result


def read_scores(run, result):
    """Read only the scored artifact cryptographically bound to a report."""
    meta = result["predictions"]
    path = Path(run) / meta["file"]
    if Path(meta["file"]).name != meta["file"] or not meta["file"].startswith("robust_scores_"):
        raise ValueError("Unexpected receiver score artifact")
    if _file_hash(path) != meta["sha256"]:
        raise ValueError("Receiver score file changed")
    with np.load(path, allow_pickle=False) as saved:
        return saved["logits"].copy()
