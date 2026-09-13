#!/usr/bin/env python3
"""Remove the delivered RTC V3/V4 additions and keep the existing V2 system.

Run in the activated sdd environment:
    python rollback_to_v2.py --apply
Without --apply, print the exact deletion list without changing anything.
This script does not retrain, regenerate audio, reset Git, or uninstall packages.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

DEFAULT_PROJECT = "/home/ubuntu/LXT/RTC/xlsr_aasist"
V2_CHECKPOINT = "exp/xlsr_aasist_RTC_noisy_v2_epoch30_bs40_20260912010625167834/ckpt/best_model.pth"
REQUIRED = (
    "model/model.py", "main_train.py", "main_train_rtc_noisy.py",
    "main_train_rtc_noisy_v2.py", "main_eval_rtc_noisy.py",
    "run_train_rtc_noisy_v2.sh", "run_eval_rtc_noisy.sh",
    "utils/data_utils.py", "utils/env_noise.py", "utils/rtc_pairs.py",
    "utils/rtc_data.py", "utils/rtc_loss.py",
    "rtc_noisy/common.py", "rtc_noisy/data.py", "rtc_noisy/training.py",
    "rtc_noisy_v2/cache.py", "rtc_noisy_v2/plan.py",
    "rtc_noisy_v2/losses.py", "rtc_noisy_v2/sampling.py", "rtc_noisy_v2/training.py",
)
NEW_FILES = (
    "build_webrtc_v3.py", "check_rtc_noisy_v3.py", "main_train_rtc_noisy_v3.py",
    "prepare_rtc_noisy_v3.py", "run_eval_rtc_noisy_v3.sh", "run_prepare_rtc_noisy_v3.sh",
    "run_train_rtc_noisy_v3.sh", "setup_rtc_noisy_v3.sh", "README_RTC_NOISY_V3.md", "TEST_REPORT_V3.md",
    "check_rtc_noisy_v4.py", "install_v4.py", "main_train_rtc_noisy_v4.py", "main_eval_rtc_noisy_v4.py",
    "run_train_rtc_noisy_v4.sh", "run_eval_rtc_noisy_v4.sh", "README_RTC_NOISY_V4.md", "TEST_REPORT_V4.md",
    "tests/test_rtc_noisy_v3_cache.py", "tests/test_rtc_noisy_v3_native.py", "tests/test_rtc_noisy_v3_training.py",
    "tests/test_rtc_noisy_v4_core.py", "tests/test_rtc_noisy_v4_integration.py", "tests/v4_fixtures.py",
    "third_party/WebRTC_LICENSE.txt", "third_party/WebRTC_PATENTS.txt",
    "third_party/webrtc_audio_processing-0.1.3.tar.gz",
)
DIRECTORIES = ("rtc_noisy_v3", "rtc_noisy_v4", "v4_backups", "v3_backups")
HASH_ONLY = {
    "THIRD_PARTY_NOTICES.md": "58b396d1166048a353dca54a3bd07e335869344efb49c60caa7866da8fd769b6",
    "PACKAGE_FILES.json": "58039f36c6fd0d0bb9bc2407a334d589ae68ee4fdf5061511156dbb5ebe828ee",
}
VERSION_NAME = re.compile(r"^(?:(?:xlsr_aasist_)?rtc_noisy_v[34](?:[_.-]|$)|v4_dual_(?:only|fusion|local)(?:_|$))", re.I)
PACKAGE_NAME = re.compile(r"^rtc_noisy_pair_v[34]_manual(?:$|[ ._(-])", re.I)
CACHE_NAME = re.compile(r"^rtc_noisy_cache_v[34](?:$|[_.-])", re.I)
SHORT_LOG = re.compile(r"^(?:v[34]|(?:train|eval)_v[34])(?:$|[_.-])", re.I)


def exists(path):
    return os.path.lexists(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover(project):
    candidates, retained = set(), []
    for name in NEW_FILES:
        for suffix in ("", ".bak", ".orig", ".rej", "~"):
            candidates.add(project/(name+suffix))
        if name.endswith(".py"):
            source = Path(name)
            candidates.update((project/source.parent/"__pycache__").glob(source.stem+".*.pyc"))
    candidates.update(project/name for name in DIRECTORIES)
    for name, expected in HASH_ONLY.items():
        path = project/name
        if exists(path):
            if path.is_file() and digest(path) == expected:
                candidates.add(path)
            else:
                retained.append(f"{name}: content differs from the supplied V3/V4 file; kept")
    for path in project.iterdir():
        if PACKAGE_NAME.match(path.name):
            candidates.add(path)
        if path.is_file() and path.suffix.lower() in (".patch", ".diff"):
            if VERSION_NAME.match(path.stem) or SHORT_LOG.match(path.stem):
                candidates.add(path)
            elif path.stat().st_size < 5_000_000:
                text = path.read_text(encoding="utf-8", errors="replace")
                pairs = re.findall(r"^diff --git a/(\S+) b/(\S+)$", text, re.M)
                def owned(name):
                    if name.startswith("xlsr_aasist/"):
                        name = name[len("xlsr_aasist/"):]
                    return name in NEW_FILES or name.startswith(("rtc_noisy_v3/", "rtc_noisy_v4/"))
                if pairs and all(owned(a) and owned(b) for a, b in pairs):
                    candidates.add(path)
    for subfolder, matcher in (("exp", VERSION_NAME), ("exp/eval", VERSION_NAME), ("logs", VERSION_NAME), ("dataset", CACHE_NAME)):
        parent = project/subfolder
        if parent.is_dir():
            for path in parent.iterdir():
                if matcher.match(path.name) or (subfolder == "logs" and SHORT_LOG.match(path.name)):
                    candidates.add(path)
    existing = sorted((p for p in candidates if exists(p)), key=lambda p: (len(p.parts), str(p)))
    result = []
    for path in existing:
        if not any(is_within(path, prior) for prior in result):
            result.append(path)
    return result, retained


def preflight(project, checkpoint, targets):
    missing = [name for name in REQUIRED if not (project/name).is_file()]
    if missing:
        raise RuntimeError("V2 dependencies are missing; no files deleted: " + ", ".join(missing))
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise RuntimeError("The V2 best checkpoint is missing/empty; no files deleted: " + str(checkpoint))
    for name in REQUIRED:
        if not name.endswith(".py"):
            continue
        tree = ast.parse((project/name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            if any(m.split(".")[0] in {"rtc_noisy_v3", "rtc_noisy_v4", "main_train_rtc_noisy_v3", "main_train_rtc_noisy_v4"} for m in modules):
                raise RuntimeError(f"Shared V2 file {name} has been changed to depend on V3/V4. No files deleted; this change needs to be reverted first.")
    protected = [project/name for name in REQUIRED]
    protected += [checkpoint, project/"dataset/wav", project/"dataset/rtc_noisy_cache_v2",
                  project/"dataset/train_label.txt", project/"dataset/dev_label.txt", project/"dataset/progress.txt",
                  project/"dataset/train_rtc_pairs.jsonl", project.parent/"pretrained", project.parent/"external_noise"]
    for key in ("RTC_V2_CACHE_ROOT", "RTC_V2_TRAIN_CACHE", "RTC_V2_DEV_SEEN_CACHE", "RTC_V2_DEV_HELDOUT_CACHE"):
        if os.environ.get(key):
            protected.append((project/Path(os.environ[key]).expanduser()).absolute())
    for target in targets:
        if not is_within(target, project) or target == project:
            raise RuntimeError("Deletion escaped the selected project: " + str(target))
        for keep in protected:
            # Removing a symlink only unlinks it; never traverse its target.
            if is_within(keep.absolute(), target) or (not target.is_symlink() and is_within(keep.resolve(), target.resolve())):
                raise RuntimeError("Deletion would affect a protected V2 file: " + str(keep))
    return {name: digest(project/name) for name in REQUIRED}


def active_jobs(project):
    names = {Path(n).name for n in NEW_FILES if n.endswith(".py")}
    found = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            if int(process.name) == os.getpid():
                continue
            args = (process/"cmdline").read_bytes().decode(errors="replace").split("\0")
            if not any(Path(a).name in names for a in args if a):
                continue
            cwd = (process/"cwd").resolve(strict=True)
            if is_within(cwd, project) or any(a.startswith(str(project)+"/") for a in args):
                found.append(int(process.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return sorted(found)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--checkpoint", default=V2_CHECKPOINT, help="V2 checkpoint to protect; relative to project or absolute")
    parser.add_argument("--apply", action="store_true", help="Delete the listed V3/V4 files, including their checkpoints/caches/logs")
    args = parser.parse_args(argv)
    project = Path(args.project).expanduser().resolve(strict=True)
    checkpoint = (project/Path(args.checkpoint).expanduser()).absolute()
    targets, retained = discover(project)
    before = preflight(project, checkpoint, targets)
    print("V2 project:", project)
    print("KEEP V2 checkpoint:", checkpoint)
    print("V3/V4 deletion list (directories include all their contents):", flush=True)
    for target in targets:
        print("  DELETE", target.relative_to(project), "[symlink only]" if target.is_symlink() else "", flush=True)
    for note in retained:
        print("KEPT:", note, flush=True)
    if not args.apply:
        print("Preview only; nothing changed. Use --apply to perform the listed cleanup.")
        return
    jobs = active_jobs(project)
    if jobs:
        raise RuntimeError("V3/V4 jobs are still active (PIDs " + ", ".join(map(str, jobs)) + "). Stop these jobs in their terminals, then rerun. No files deleted.")
    record = {"time": datetime.now().isoformat(), "v2_checkpoint": str(checkpoint),
              "planned": [str(p.relative_to(project)) for p in targets], "deleted": [], "retained": retained}
    log = project/f"rollback_to_v2_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    def report():
        temp = log.with_suffix(".json.tmp")
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        temp.replace(log)
    report()
    try:
        for target in targets:
            print("Removing:", target.relative_to(project), flush=True)
            if target.is_symlink() or not target.is_dir():
                target.unlink()
            else:
                shutil.rmtree(target)
            record["deleted"].append(str(target.relative_to(project)))
            report()
        for name, checksum in before.items():
            if digest(project/name) != checksum:
                raise RuntimeError("V2 file changed during cleanup: "+name)
        if not checkpoint.is_file():
            raise RuntimeError("V2 checkpoint disappeared during cleanup")
        record["status"] = "completed"
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        report()
    print("V2 rollback complete. V2 code, weights and its original dependencies are retained.")
    print("Deletion record:", log)
    print("V2 evaluation entry: bash run_eval_rtc_noisy.sh", str(checkpoint))
    print("V2 training entry: bash run_train_rtc_noisy_v2.sh", str(checkpoint))
    print("No training or evaluation has been started.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, SyntaxError, ValueError) as error:
        print("Rollback stopped:", error, file=sys.stderr)
        sys.exit(1)
