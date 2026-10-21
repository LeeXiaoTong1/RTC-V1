"""Frozen V5.3 single-model scoring and retryable small artifact delivery."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse
import zipfile

from w2v_aasist.data import read_protocol
from w2v_aasist.evaluate import package_scores
from w2v_aasist.launch import run_lock, upload_archive
from .common import (ROOT, RECIPE, atomic_json, read_json, digest, fingerprint,
                     schema_for, apply_partial, seed_all)

REPORT_EXCLUDED = {
    "submission_status.json", "submission_request.json", "report_status.json",
    "report_upload_receipt.json", "upload_receipt.json", "upload_status.json",
    "artifact_links.json", "REPORT_MANIFEST.json",
}
MAX_REPORT_FILE = 2 * 1024 * 1024
MAX_REPORT_TOTAL = 16 * 1024 * 1024


def valid_url(url):
    value = urlparse(url or "")
    return (value.scheme == "https" and value.netloc == "temp.sh"
            and value.path not in ("", "/", "/upload") and not value.fragment)


def final_identity(run):
    """Verify final freezing without torch/model/audio loading, including retries."""
    run = Path(run).expanduser().resolve()
    done = read_json(run / "completed.json")
    cfg = read_json(run / "final_config.json")
    if (done.get("version") != "5.3" or done.get("status") != "complete"
            or cfg.get("v5_stage") != "final"):
        raise ValueError("Only a completed, frozen final V5.3 model may score submission audio")
    schema_for(cfg)
    if digest(run / "final.pt") != done.get("checkpoint_sha256"):
        raise ValueError("Final checkpoint changed after completion")
    if digest(run / "final_config.json") != done.get("final_config_sha256"):
        raise ValueError("Final configuration changed after completion")
    recipe_hash = digest(run / "recipe_lock.json")
    if (recipe_hash != done.get("recipe_sha256")
            or recipe_hash != cfg.get("v5_recipe_lock_sha256")):
        raise ValueError("Frozen recipe changed after completion")
    if not isinstance(done.get("selected"), str) or not done["selected"]:
        raise ValueError("Completed run has no selected final checkpoint")
    return cfg, dict(done, run=str(run), config_identity=fingerprint(cfg))


def selected(run):
    import torch
    run = Path(run).expanduser().resolve()
    cfg, meta = final_identity(run)
    state = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    if (state.get("schema") != schema_for(cfg)
            or state.get("identity") != fingerprint(cfg)
            or state.get("tag") != meta["selected"]
            or state.get("recipe") != read_json(run / "recipe_lock.json")):
        raise ValueError("Final checkpoint/config/locked recipe identity mismatch")
    return cfg, state["model"], dict(meta, locked_recipe=state["recipe"])


def export_identity(run, meta):
    cfg, current = final_identity(run)
    if (Path(meta.get("run", "")).resolve() != Path(run).resolve()
            or any(meta.get(key) != current[key]
                   for key in ("selected", "checkpoint_sha256", "config_identity"))):
        raise ValueError("Submission belongs to a different frozen final model")
    if meta.get("input_policy") != cfg.get("v5_input"):
        raise ValueError("Submission input processing differs from the final model")
    protocol = Path(meta["protocol"])
    if digest(protocol) != meta.get("protocol_sha256"):
        raise ValueError("Submission protocol changed since export")
    return current


def check_archive(out):
    out = Path(out).expanduser().resolve()
    meta = read_json(out / "submission_meta.json")
    archive = out / "submission.zip"
    if (meta.get("version") != "5.3" or meta.get("recipe") != RECIPE
            or meta.get("score") != "P(fake=0)" or meta.get("threshold") != .5
            or meta.get("calibration") != "none"):
        raise ValueError("Expected an uncalibrated single-model V5.3 export")
    if digest(archive) != meta.get("zip_sha256"):
        raise ValueError("Submission ZIP changed")
    with zipfile.ZipFile(archive) as zipped:
        if zipped.namelist() != ["scores.txt"]:
            raise ValueError("Submission ZIP must contain only scores.txt")
        contents = zipped.read("scores.txt")
    ids = []
    for line in contents.decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError("Invalid submission score line")
        value = float(fields[1])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid fake probability")
        ids.append(fields[0])
    if (not ids or len(ids) != meta.get("count") or len(set(ids)) != len(ids)
            or fingerprint(ids) != meta.get("id_order_sha256")):
        raise ValueError("Submission count or ID order changed")
    if (out / "scores.txt").is_file() and (out / "scores.txt").read_bytes() != contents:
        raise ValueError("Plain scores differ from the submission ZIP")
    return archive, meta


def export(run, out, *, protocol, audio_root, device=None, workers=None,
           model_factory=None, inference=None, verify=True):
    """Read an unlabeled protocol only after the final detector is frozen."""
    import torch
    run, out = Path(run).expanduser().resolve(), Path(out).expanduser().resolve()
    if out == run or run in out.parents:
        raise ValueError("Keep submission output outside the training run")
    original, weights, meta = selected(run)
    protocol = Path(protocol).expanduser().resolve()
    audio_root = Path(audio_root).expanduser().resolve()
    protocol_hash = digest(protocol)
    # This is deliberately after selected(): Progress/Eval are not consulted
    # during screening, recipe selection, or full Train+Dev fitting.
    rows = read_protocol(protocol, audio_root, labeled=False)
    ids = [row["id"] for row in rows]
    if workers is not None and workers < 0:
        raise ValueError("Inference worker count must be nonnegative")
    if out.exists():
        archive, existing = check_archive(out)
        export_identity(run, existing)
        if (existing["protocol_sha256"] != protocol_hash
                or existing["id_order_sha256"] != fingerprint(ids)
                or Path(existing["audio_root"]).resolve() != audio_root):
            raise ValueError("Existing export uses another evaluation input")
        print("SUBMISSION_ZIP=" + str(archive), flush=True)
        return archive
    cfg = dict(original)
    if device is not None:
        cfg["device"] = device
    if workers is not None:
        cfg.update(workers=workers, eval_workers=workers)
    if verify:
        from .config import verify_inputs
        verify_inputs(original, training=False)
    if model_factory is None:
        from .model import load_model
        model_factory = load_model
    if inference is None:
        from w2v_v5.inference import infer
        inference = infer
    seed_all(cfg["seed"])
    model = model_factory(cfg)
    apply_partial(model, weights)
    del weights
    model.requires_grad_(False).eval()
    with torch.inference_mode():
        logits, returned = inference(model, rows, cfg)
    logits = torch.as_tensor(logits, dtype=torch.float32, device="cpu")
    if (tuple(logits.shape) != (len(rows), 2) or not bool(torch.isfinite(logits).all())
            or [row["id"] for row in returned] != ids):
        raise ValueError("Submission inference lost/reordered rows or produced invalid logits")
    probabilities = logits.softmax(-1)[:, 0].tolist()
    metadata = dict(meta, version="5.3", recipe=RECIPE, score="P(fake=0)", threshold=.5,
                    calibration="none", protocol=str(protocol), protocol_sha256=protocol_hash,
                    audio_root=str(audio_root), count=len(rows), id_order_sha256=fingerprint(ids),
                    input_policy=original.get("v5_input"), independent_prediction_ensemble=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with run_lock(out.parent / ("." + out.name + ".export.lock")):
        if out.exists():
            raise ValueError("Submission output appeared during inference; retain it and retry")
        with tempfile.TemporaryDirectory(prefix=out.name + ".pending-", dir=out.parent) as folder:
            stage = Path(folder) / "export"
            archive = package_scores(ids, probabilities, stage)
            atomic_json(stage / "submission_meta.json", dict(metadata, zip_sha256=digest(archive)))
            check_archive(stage)
            export_identity(run, metadata)
            stage.replace(out)
    archive = out / "submission.zip"
    print("SUBMISSION_ZIP=" + str(archive), flush=True)
    return archive


def upload_submission(out, force=False):
    out = Path(out).expanduser().resolve()
    if not out.is_dir():
        raise FileNotFoundError("Submission export directory is missing: " + str(out))
    with run_lock(out / ".upload.lock"):
        archive, meta = check_archive(out)
        export_identity(meta["run"], meta)
        receipt = out / "upload_receipt.json"
        previous = read_json(receipt) if receipt.is_file() else {}
        identity = {key: meta[key] for key in
                    ("zip_sha256", "selected", "checkpoint_sha256", "config_identity")}
        try:
            if (not force and all(previous.get(key) == value for key, value in identity.items())
                    and valid_url(previous.get("url"))):
                url = previous["url"]
            else:
                url = upload_archive(archive)
                if not valid_url(url) or digest(archive) != meta["zip_sha256"]:
                    raise ValueError("Upload response or archive identity changed")
                export_identity(meta["run"], meta)
                atomic_json(receipt, dict(identity, url=url,
                    uploaded_at=datetime.now(timezone.utc).isoformat()))
            (out / "temp_download_url.txt").write_text(url + "\n", encoding="utf-8")
            atomic_json(out / "upload_status.json", dict(status="complete", url=url))
            print("SUBMISSION_TEMP_URL=" + url, flush=True)
            return url
        except Exception as error:
            atomic_json(out / "upload_status.json",
                        dict(status="failed", error=str(error), archive=str(archive)))
            raise RuntimeError("Upload failed; existing scores are retained. Retry "
                               "upload_w2v_v53.sh --out with this export directory.") from error


def _report_files(run):
    included, skipped, total = [], [], 0
    priority = {'run_summary.json', 'fusion_diagnostics.jsonl', 'parameter_inventory.json'}
    for path in sorted(run.rglob("*"), key=lambda p: (p.name not in priority, str(p))):
        relative = path.relative_to(run)
        if (not path.is_file() or path.suffix not in (".json", ".jsonl", ".txt", ".log")
                or path.name in REPORT_EXCLUDED or any(part.startswith(".") for part in relative.parts)
                or path.name.endswith("_rows.json") or "registry" in path.name
                or path.name.endswith("_download_url.txt")):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(run):
            raise ValueError("Report inputs may not escape the run: " + str(relative))
        size = path.stat().st_size
        if size > MAX_REPORT_FILE or total + size > MAX_REPORT_TOTAL:
            skipped.append(dict(file=relative.as_posix(), bytes=size, reason="bounded report size"))
            continue
        included.append(dict(file=relative.as_posix(), bytes=size, sha256=digest(path)))
        total += size
    return included, skipped


def report(run, destination, upload=False, force=False):
    """Package bounded metrics/config text; never audio, tensors, or model weights."""
    run, destination = Path(run).expanduser().resolve(), Path(destination).expanduser().resolve()
    cfg_file = run / "final_config.json"
    cfg = read_json(cfg_file if cfg_file.is_file() else run / "config.json")
    if cfg.get("version") != "5.3" or cfg.get("recipe") != RECIPE:
        raise ValueError("Expected a V5.3 run report")
    if destination == run or run in destination.parents:
        raise ValueError("Keep the report destination outside the training run")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / (run.name + "_report.zip")
    with run_lock(run / ".report.lock"):
        rows, skipped = _report_files(run)
        identity = fingerprint(dict(files=rows, skipped=skipped))
        status_path = run / "report_status.json"
        old = read_json(status_path) if status_path.is_file() else {}
        if not (archive.is_file() and old.get("content_identity") == identity
                and old.get("zip_sha256") == digest(archive)):
            temporary = archive.with_suffix(".zip.tmp")
            try:
                with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as zipped:
                    for row in rows:
                        path = run / row["file"]
                        zipped.write(path, row["file"])
                        if digest(path) != row["sha256"]:
                            raise ValueError("Report input changed while packaging: " + row["file"])
                    zipped.writestr("REPORT_MANIFEST.json", json.dumps(dict(
                        version="5.3", recipe=RECIPE, files=rows, skipped=skipped,
                        content_identity=identity), ensure_ascii=False, indent=2))
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
        archive_hash = digest(archive)
        status = dict(status="exported", archive=str(archive), zip_sha256=archive_hash,
                      content_identity=identity, url=None)
        atomic_json(status_path, status)
        print("REPORT_ZIP=" + str(archive), flush=True)
        if upload:
            receipt = run / "report_upload_receipt.json"
            previous = read_json(receipt) if receipt.is_file() else {}
            try:
                if (not force and previous.get("zip_sha256") == archive_hash
                        and valid_url(previous.get("url"))):
                    url = previous["url"]
                else:
                    url = upload_archive(archive)
                    if not valid_url(url) or digest(archive) != archive_hash:
                        raise ValueError("Report upload response or archive changed")
                    atomic_json(receipt, dict(url=url, zip_sha256=archive_hash))
                status.update(status="complete", url=url)
                (run / "report_temp_download_url.txt").write_text(url + "\n", encoding="utf-8")
                print("REPORT_TEMP_URL=" + url, flush=True)
            except Exception as error:
                status.update(status="failed", error=str(error))
                atomic_json(status_path, status)
                raise
            atomic_json(status_path, status)
    return archive


def finish(run, args):
    """Called after freezing; never inspect Progress for an unfinished stage."""
    run = Path(run).expanduser().resolve()
    destination = Path(args.download_dir).expanduser().resolve()
    errors, submission = [], None
    if getattr(args, "submit_best", False):
        try:
            final_identity(run)
            base = "/home/ubuntu/LXT/RTC/xlsr_aasist/dataset"
            protocol = args.submission_protocol or os.environ.get("EVAL_PROTOCOL", base + "/progress.txt")
            audio = args.submission_audio_root or os.environ.get("EVAL_AUDIO_ROOT", base + "/wav/progress")
            out = destination / (run.name + "_submission_final")
            atomic_json(run / "submission_request.json", dict(out=str(out), protocol=str(protocol),
                        audio_root=str(audio), upload=bool(args.upload_submission)))
            archive = export(run, out, protocol=protocol, audio_root=audio)
            url = upload_submission(out) if args.upload_submission else None
            submission = dict(status="complete", out=str(out), archive=str(archive), url=url)
            atomic_json(run / "submission_status.json", submission)
        except Exception as error:
            errors.append("submission: " + str(error))
            atomic_json(run / "submission_status.json", dict(status="failed", error=str(error)))
    try:
        archive = report(run, destination, upload=bool(getattr(args, "upload_temp", False)))
    except Exception as error:
        errors.append("report: " + str(error))
        archive = None
    if errors:
        raise RuntimeError("V5.3 model files retained; artifact delivery incomplete: " + "; ".join(errors))
    return dict(submission=submission, report=str(archive))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--run", required=True)
    export_parser.add_argument("--out", required=True)
    export_parser.add_argument("--protocol", required=True)
    export_parser.add_argument("--audio-root", required=True)
    export_parser.add_argument("--device")
    export_parser.add_argument("--workers", type=int)
    export_parser.add_argument("--upload-temp", action="store_true")
    upload_parser = commands.add_parser("upload")
    targets = upload_parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--out")
    targets.add_argument("--run")
    upload_parser.add_argument("--force", action="store_true")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--run", required=True)
    report_parser.add_argument("--destination", required=True)
    report_parser.add_argument("--upload-temp", action="store_true")
    report_parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        with run_lock(ROOT / "exp" / ".aasist-launch.lock"):
            archive = export(args.run, args.out, protocol=args.protocol, audio_root=args.audio_root,
                             device=args.device, workers=args.workers)
        run = Path(args.run).expanduser().resolve()
        atomic_json(run / "submission_request.json",
                    dict(out=str(archive.parent), protocol=str(Path(args.protocol).resolve()),
                         audio_root=str(Path(args.audio_root).resolve()),
                         upload=bool(args.upload_temp)))
        if args.upload_temp:
            upload_submission(archive.parent)
    elif args.command == "report":
        report(args.run, args.destination, args.upload_temp, args.force)
    elif args.out:
        upload_submission(args.out, args.force)
    else:
        run = Path(args.run).expanduser().resolve()
        request_path = run / "submission_request.json"
        request = read_json(request_path) if request_path.is_file() else {}
        status_path = run / "report_status.json"
        previous = read_json(status_path) if status_path.is_file() else {}
        destination = (Path(previous["archive"]).parent if previous.get("archive")
                       else Path(request["out"]).parent if request
                       else Path("/home/ubuntu/LXT/temp"))
        errors = []
        if request:
            try:
                upload_submission(request["out"], args.force)
            except Exception as error:
                errors.append("submission: " + str(error))
        try:
            report(run, destination, True, args.force)
        except Exception as error:
            errors.append("report: " + str(error))
        if errors:
            raise RuntimeError("Artifact delivery incomplete: " + "; ".join(errors))


if __name__ == "__main__":
    main()
