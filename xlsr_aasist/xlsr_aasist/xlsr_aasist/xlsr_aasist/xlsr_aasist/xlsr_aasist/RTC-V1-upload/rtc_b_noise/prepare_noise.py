"""Split screened non-speech WAV recordings; duplicates stay in one group."""
import argparse
import hashlib
import json
from pathlib import Path
import re


HUMAN = re.compile(r"crowd|speech|voice|talk|babbl|conversation|chatter|people|scream|laugh|cheer|chant|singing|children|baby|human|vocal", re.I)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--exclude-list", type=Path, help="One relative WAV path per line, for manually excluded files")
    a = p.parse_args()
    root, out = a.noise_root.expanduser().resolve(), a.out.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit("Noise directory missing: " + str(root))
    if (root / "speech").exists() or (root / "music").exists() or root.name.lower() in {"speech", "music"}:
        raise SystemExit("Point --noise-root at the screened noise subset, not a speech/music/full corpus folder")
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output directory is not empty; choose a new --out to preserve the old split")
    excluded = set()
    if a.exclude_list:
        excluded = {line.strip() for line in a.exclude_list.read_text().splitlines() if line.strip() and not line.startswith("#")}
    files = sorted(root.rglob("*.wav"))
    if not files:
        raise SystemExit("No WAV files found")
    meta_cache, rows, removed = {}, [], []
    for path in files:
        relative = str(path.relative_to(root))
        if path.parent not in meta_cache:
            texts = []
            for name in ("ANNOTATIONS", "LICENSE"):
                meta = path.parent / name
                if meta.exists():
                    texts.extend(meta.read_text(errors="replace").splitlines())
            meta_cache[path.parent] = texts
        description = " ".join(line for line in meta_cache[path.parent] if path.stem in line)
        if relative in excluded or HUMAN.search(relative + " " + description):
            removed.append({"path": relative, "reason": "manual_or_human_keyword", "metadata": description})
            continue
        rows.append({"path": str(path), "original_recording": relative, "sha256": digest(path)})
    # Byte-identical recordings cannot cross the train/dev noise split.
    hashes = sorted({row["sha256"] for row in rows},
                    key=lambda h: hashlib.sha256(f"{a.seed}:{h}".encode()).hexdigest())
    if len(hashes) < 2:
        raise SystemExit("Need at least two distinct non-speech recordings")
    ndev = max(1, round(len(hashes) * 0.2))
    held_out = set(hashes[:ndev])
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "dev"):
        selected = [dict(row, split=split) for row in rows if (row["sha256"] in held_out) == (split == "dev")]
        (out / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
        counts[split] = len(selected)
    report = {"root": str(root), "seed": a.seed, "counts": counts,
              "removed": removed,
              "note": "Keyword filtering is not speech detection. Check remaining ambiguous recordings. If multiple files derive from one recording, group them before splitting."}
    (out / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "counts": counts, "excluded": len(removed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
