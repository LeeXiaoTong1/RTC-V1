"""Validate explicit, official Offline/Online correspondence.

No filename numbering or label/identity similarity is used to infer pairs.
Only paths named in the supplied CSV and training protocol are eligible.
"""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


_LABELS = {"spoof": 0, "fake": 0, "bonafide": 1, "bona-fide": 1, "real": 1}
_ALIASES = {
    side: {
        side, f"{side}_path", f"{side}_file", f"{side}_filepath",
        f"{side}_filename", f"{side}_audio", f"{side}_audio_path",
        f"{side}_wav", f"{side}_wav_path", f"path_{side}",
    }
    for side in ("offline", "online")
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Audio path must be a nonempty string")
    value = value.strip().replace("\\", "/")
    if "\x00" in value or ".." in PurePosixPath(value).parts:
        raise ValueError(f"Path traversal is forbidden: {value!r}")
    return str(PurePosixPath(value))


def _protocol_path(value):
    value = _path_text(value)
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"Protocol paths must be relative to the training root: {value}")
    return value


def _check_file(data_root, utt_id):
    root = Path(data_root).resolve()
    path = (root / utt_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Audio path escapes the training root: {utt_id}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Training audio does not exist: {path}")


def read_train_protocol(protocol_path, data_root):
    """Read the same labeled protocol formats as V1; reject ambiguous entries."""
    labels = {}
    with open(protocol_path, encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) == 2:
                utt_id, label = fields
            elif len(fields) >= 5:
                utt_id, label = fields[1], fields[4]
            else:
                raise ValueError(f"Invalid labeled protocol line {protocol_path}:{line_no}")
            utt_id = _protocol_path(utt_id)
            if utt_id in labels:
                raise ValueError(f"Duplicate protocol path at line {line_no}: {utt_id}")
            if label.lower() not in _LABELS:
                raise ValueError(f"Unsupported label at line {line_no}: {label}")
            _check_file(data_root, utt_id)
            labels[utt_id] = _LABELS[label.lower()]
    if not labels:
        raise ValueError("Training protocol is empty")
    return labels


def _select_column(headers, side, explicit=None):
    if explicit is not None:
        if explicit not in headers:
            raise ValueError(f"Column {explicit!r} is absent. Available headers: {headers}")
        return explicit
    matches = [
        header for header in headers
        if header is not None and re.sub(r"[\s-]+", "_", header.strip().lower()) in _ALIASES[side]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely determine the {side} column (matches={matches}). "
            f"Available headers: {headers}. Pass --{side}_column with the exact header."
        )
    return matches[0]


class _PathResolver:
    def __init__(self, labels):
        self.labels = labels
        self.by_basename = defaultdict(list)
        self.by_suffix = defaultdict(list)
        for utt_id in labels:
            parts = PurePosixPath(utt_id).parts
            self.by_basename[parts[-1]].append(utt_id)
            for position in range(len(parts) - 1):
                self.by_suffix["/".join(parts[position:])].append(utt_id)

    def resolve(self, value, side):
        value = _path_text(value)
        candidates = []
        parts = PurePosixPath(value).parts
        for first, second in zip(parts, parts[1:]):
            if first.lower() in {"dev", "progress", "eval", "test", "development", "evaluation"} and second.lower() in {"clean", "noisy", "offline", "online"}:
                raise ValueError(f"An explicit non-training split cannot map into training data: {value}")
        # Strip only a leading directory prefix. Exact protocol suffixes take
        # priority; do not change basenames, extensions, IDs, or numbering.
        for position in range(len(parts)):
            suffix = "/".join(parts[position:])
            if suffix in self.labels:
                candidates.append(suffix)
        candidates = list(dict.fromkeys(candidates))
        if not candidates and not value.startswith("/"):
            candidates = self.by_suffix.get(value, [])
        if not candidates and len(parts) == 1:
            # A bare filename carries no split or directory claim. Resolve
            # it only inside the explicitly specified CSV side. A nonmatching
            # full path must never silently degrade into basename matching.
            candidates = [
                utt_id for utt_id in self.by_basename.get(parts[0], [])
                if side in {part.lower() for part in PurePosixPath(utt_id).parts[:-1]}
            ]
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot uniquely map {side} path {value!r} to the training protocol; "
                f"candidates={candidates[:8]} (count={len(candidates)}). "
                "Supply a path that identifies an exact protocol entry."
            )
        resolved = candidates[0]
        directories = {part.lower() for part in PurePosixPath(resolved).parts[:-1]}
        other = "online" if side == "offline" else "offline"
        if side not in directories or other in directories:
            raise ValueError(f"Expected an {side} directory, got: {resolved}")
        return resolved


def _validate_pair(offline, online, label, labels, seen_pairs, online_sources):
    if labels[offline] != labels[online]:
        raise ValueError(f"Pair has different truth labels: {offline} / {online}")
    expected_label = labels[offline]
    if label is not None and (type(label) is not int or label != expected_label):
        raise ValueError(f"Manifest label {label!r} disagrees with protocol label {expected_label}")
    key = (offline, online)
    if key in seen_pairs:
        raise ValueError(f"Duplicate pair: {offline} / {online}")
    if online in online_sources and online_sources[online] != offline:
        raise ValueError(f"One Online audio maps to multiple Offline sources: {online}")
    seen_pairs.add(key)
    online_sources[online] = offline
    return {"offline": offline, "online": online, "label": expected_label}


def prepare_pairs(pairs_csv, train_protocol, train_data_path, output,
                  offline_column=None, online_column=None):
    """Build a JSONL manifest from an explicit CSV, plus provenance metadata."""
    labels = read_train_protocol(train_protocol, train_data_path)
    resolver = _PathResolver(labels)
    pairs, seen_pairs, online_sources = [], set(), {}
    with open(pairs_csv, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError(f"Duplicate CSV headers are ambiguous: {headers}")
        off_col = _select_column(headers, "offline", offline_column)
        on_col = _select_column(headers, "online", online_column)
        if off_col == on_col:
            raise ValueError("Offline and Online columns must differ")
        for row_number, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"Extra CSV values on row {row_number}; check the CSV delimiter")
            try:
                offline = resolver.resolve(row.get(off_col), "offline")
                online = resolver.resolve(row.get(on_col), "online")
                pairs.append(_validate_pair(offline, online, None, labels, seen_pairs, online_sources))
            except (ValueError, FileNotFoundError) as exc:
                raise ValueError(f"Invalid pair at CSV row {row_number}: {exc}") from exc
    if not pairs:
        raise ValueError("Pair CSV contains no pairs")

    covered = {pair[side] for pair in pairs for side in ("offline", "online")}
    class_counts = Counter(pair["label"] for pair in pairs)
    metadata = {
        "format_version": 1,
        "pairs_csv": str(Path(pairs_csv).resolve()),
        "pairs_csv_sha256": sha256_file(pairs_csv),
        "train_protocol": str(Path(train_protocol).resolve()),
        "train_protocol_sha256": sha256_file(train_protocol),
        "train_data_path": str(Path(train_data_path).resolve()),
        "offline_column": off_col,
        "online_column": on_col,
        "pair_count": len(pairs),
        "pairs_by_label": {"fake_0": class_counts[0], "real_1": class_counts[1]},
        "unique_offline": len({pair["offline"] for pair in pairs}),
        "unique_online": len({pair["online"] for pair in pairs}),
        "protocol_utterances": len(labels),
        "covered_utterances": len(covered),
        "coverage_fraction": len(covered) / len(labels),
        "unpaired_utterances": len(labels) - len(covered),
        "unpaired_by_label": {
            "fake_0": sum(label == 0 for utt_id, label in labels.items() if utt_id not in covered),
            "real_1": sum(label == 1 for utt_id, label in labels.items() if utt_id not in covered),
        },
        "unpaired_policy": "Keep every protocol utterance in the main V1 classification loader.",
    }
    output = Path(output)
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    inputs = {Path(pairs_csv).resolve(), Path(train_protocol).resolve()}
    if output.resolve() in inputs or meta_path.resolve() in inputs:
        raise ValueError("Output must not overwrite the source CSV or protocol")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
    metadata["manifest_sha256"] = sha256_file(output)
    with open(meta_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return pairs, metadata


def load_pairs(manifest_path, protocol_path, data_root):
    """Revalidate every manifest record; a prebuilt manifest is not trusted."""
    labels = read_train_protocol(protocol_path, data_root)
    resolver = _PathResolver(labels)
    pairs, seen_pairs, online_sources = [], set(), {}
    with open(manifest_path, encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or not {"offline", "online", "label"}.issubset(record):
                    raise ValueError("Each record needs offline, online, label fields")
                # Generated manifests contain exact protocol IDs. Unlike the
                # input CSV, manifests may not use basename/prefix fallbacks.
                offline = _protocol_path(record["offline"])
                online = _protocol_path(record["online"])
                if offline not in labels or online not in labels:
                    raise ValueError("Manifest audio is not in the supplied training protocol")
                resolver.resolve(offline, "offline")
                resolver.resolve(online, "online")
                pairs.append(_validate_pair(
                    offline, online, record["label"], labels, seen_pairs, online_sources,
                ))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid manifest record {manifest_path}:{line_number}: {exc}") from exc
    if not pairs:
        raise ValueError("Pair manifest is empty")
    metadata_path = Path(manifest_path).with_suffix(Path(manifest_path).suffix + ".meta.json")
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as stream:
            metadata = json.load(stream)
        for key, actual in (
            ("manifest_sha256", sha256_file(manifest_path)),
            ("train_protocol_sha256", sha256_file(protocol_path)),
        ):
            if metadata.get(key) != actual:
                raise ValueError(f"Pair metadata {key} differs from the current file; rebuild the manifest")
    return pairs
