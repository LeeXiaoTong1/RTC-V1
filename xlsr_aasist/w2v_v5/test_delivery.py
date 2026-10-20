"""Submission identity, no early test access, and CPU-only delivery retries."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch import nn

from .common import RECIPE, atomic_json, read_json, digest, fingerprint, schema_for
from . import delivery


def fixture(root):
    run = root / "run"
    run.mkdir()
    cfg = dict(version="5", recipe=RECIPE, v5_stage="final", lora_layers=16,
               retained_lora_layers=60, experiment="E0", robust_arm="baseline",
               device="cpu", workers=0, seed=510,
               v5_input=dict(policy="identity", short_seconds=0.))
    recipe = dict(input=cfg["v5_input"])
    atomic_json(run / "recipe_lock.json", recipe)
    cfg["v5_recipe_lock_sha256"] = digest(run / "recipe_lock.json")
    atomic_json(run / "config.json", cfg)
    atomic_json(run / "final_config.json", cfg)
    weights = {"head.weight": torch.eye(2), "head.bias": torch.zeros(2)}
    payload = dict(schema=schema_for(cfg), identity=fingerprint(cfg), model=weights,
                   tag="final_locked_recipe", recipe=recipe)
    torch.save(payload, run / "final.pt")
    atomic_json(run / "completed.json", dict(status="complete", version="5",
                checkpoint_sha256=digest(run / "final.pt"), selected=payload["tag"],
                final_config_sha256=digest(run / "final_config.json"),
                recipe_sha256=digest(run / "recipe_lock.json")))
    audio = root / "audio"
    audio.mkdir()
    for name in ("one.wav", "two.wav"):
        (audio / name).write_bytes(b"mock audio is never decoded")
    protocol = root / "progress.txt"
    protocol.write_text("one.wav\ntwo.wav\n", encoding="utf-8")
    return run, protocol, audio


def model_factory(cfg):
    model = nn.Module()
    model.head = nn.Linear(2, 2)
    return model


def inference(model, rows, cfg):
    value = model.head(torch.tensor([[3., 1.], [1., 4.]])).detach().numpy()
    return value, copy.deepcopy(rows)


class DeliveryTests(unittest.TestCase):
    def export(self, run, out, protocol, audio, **kwargs):
        return delivery.export(run, out, protocol=protocol, audio_root=audio,
                               model_factory=model_factory, inference=inference,
                               verify=False, **kwargs)

    def test_single_model_scores_order_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            run, protocol, audio = fixture(root)
            out = root / "submission"
            archive = self.export(run, out, protocol, audio)
            self.assertEqual(archive, out / "submission.zip")
            _, meta = delivery.check_archive(out)
            self.assertEqual(meta["input_policy"], read_json(run / "final_config.json")["v5_input"])
            self.assertEqual(meta["calibration"], "none")
            self.assertFalse(meta["independent_prediction_ensemble"])
            lines = (out / "scores.txt").read_text().splitlines()
            self.assertEqual([line.split()[0] for line in lines], ["one.wav", "two.wav"])
            actual = [float(line.split()[1]) for line in lines]
            expected = torch.tensor([[3., 1.], [1., 4.]]).softmax(-1)[:, 0].tolist()
            np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-11)
            with patch.object(delivery, "apply_partial", side_effect=AssertionError("No re-inference")):
                self.assertEqual(self.export(run, out, protocol, audio), archive)

    def test_no_protocol_read_before_final_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, protocol, audio = fixture(root)
            (run / "completed.json").unlink()
            with patch.object(delivery, "read_protocol", side_effect=AssertionError("Test set touched")) as read:
                with self.assertRaises(FileNotFoundError):
                    self.export(run, root / "submission", protocol, audio)
                read.assert_not_called()

    def test_upload_missing_export_does_not_create_an_empty_output(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(FileNotFoundError):
                delivery.upload_submission(missing)
            self.assertFalse(missing.exists())

    def test_output_is_not_published_on_invalid_inference_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, protocol, audio = fixture(root)
            out = root / "submission"
            def bad(model, rows, cfg):
                return np.ones((2, 2), np.float32), list(reversed(rows))
            with self.assertRaisesRegex(ValueError, "reordered"):
                delivery.export(run, out, protocol=protocol, audio_root=audio,
                                model_factory=model_factory, inference=bad, verify=False)
            self.assertFalse(out.exists())
            cfg = read_json(run / "final_config.json")
            cfg["seed"] += 1
            atomic_json(run / "final_config.json", cfg)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                self.export(run, out, protocol, audio)
            self.assertFalse(out.exists())

    def test_cpu_only_upload_reuse_force_and_checkpoint_binding(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            run, protocol, audio = fixture(root)
            out = root / "submission"
            self.export(run, out, protocol, audio)
            with patch("torch.load", side_effect=AssertionError("Upload must not load a model")), \
                 patch.object(delivery, "upload_archive", return_value="https://temp.sh/v5/submission.zip") as upload:
                self.assertEqual(delivery.upload_submission(out), "https://temp.sh/v5/submission.zip")
                delivery.upload_submission(out)
                self.assertEqual(upload.call_count, 1)
                delivery.upload_submission(out, force=True)
                self.assertEqual(upload.call_count, 2)
                with (run / "final.pt").open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    delivery.upload_submission(out)
                self.assertEqual(upload.call_count, 2)

    def test_failed_upload_retains_export_and_rejects_protocol_changes(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            run, protocol, audio = fixture(root)
            out = root / "submission"
            self.export(run, out, protocol, audio)
            before = digest(out / "submission.zip")
            with patch.object(delivery, "upload_archive", side_effect=RuntimeError("network")):
                with self.assertRaisesRegex(RuntimeError, "scores are retained"):
                    delivery.upload_submission(out)
            self.assertEqual(digest(out / "submission.zip"), before)
            protocol.write_text("two.wav\none.wav\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                delivery.upload_submission(out)

    def test_recipe_and_partial_checkpoint_identity_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _, _ = fixture(root)
            payload = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
            payload["identity"] = "another configuration"
            torch.save(payload, run / "final.pt")
            done = read_json(run / "completed.json")
            done["checkpoint_sha256"] = digest(run / "final.pt")
            atomic_json(run / "completed.json", done)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                delivery.selected(run)
            atomic_json(run / "recipe_lock.json", {"input": "changed"})
            with self.assertRaisesRegex(ValueError, "Frozen recipe changed"):
                delivery.final_identity(run)

    def test_report_is_bounded_contains_no_audio_or_checkpoint_and_is_retryable(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            run, _, _ = fixture(root)
            atomic_json(run / "metrics.json", dict(weighted_f1=.98))
            atomic_json(run / "train_rows.json", [{"audio": "not included"}])
            (run / "wave.wav").write_bytes(b"audio")
            (run / "huge.log").write_bytes(b"x" * (delivery.MAX_REPORT_FILE + 1))
            destination = root / "downloads"
            with patch("torch.load", side_effect=AssertionError("Report must not load a model")), \
                 patch.object(delivery, "upload_archive", return_value="https://temp.sh/v5/report.zip") as upload:
                archive = delivery.report(run, destination, upload=True)
                before = digest(archive)
                again = delivery.report(run, destination, upload=True)
                self.assertEqual(before, digest(again))
                self.assertEqual(upload.call_count, 1)
            with zipfile.ZipFile(archive) as zipped:
                names = zipped.namelist()
                self.assertIn("metrics.json", names)
                self.assertIn("REPORT_MANIFEST.json", names)
                for name in ("final.pt", "wave.wav", "train_rows.json", "huge.log"):
                    self.assertNotIn(name, names)

    def test_report_only_run_can_retry_upload_without_submission_request(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            run, _, _ = fixture(root)
            destination = root / "downloads"
            delivery.report(run, destination)
            with patch("sys.argv", ["delivery", "upload", "--run", str(run)]), \
                 patch.object(delivery, "upload_archive", return_value="https://temp.sh/v5/report.zip") as upload, \
                 patch("torch.load", side_effect=AssertionError("No GPU/model loading")):
                delivery.main()
            self.assertEqual(upload.call_count, 1)
            self.assertEqual(read_json(run / "report_status.json")["status"], "complete")

    def test_tampered_zip_or_plain_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            run, protocol, audio = fixture(root)
            out = root / "submission"
            self.export(run, out, protocol, audio)
            (out / "scores.txt").write_text("one.wav 0.2\ntwo.wav 0.3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Plain scores"):
                delivery.check_archive(out)
            with (out / "submission.zip").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "ZIP changed"):
                delivery.check_archive(out)


if __name__ == "__main__":
    unittest.main()
