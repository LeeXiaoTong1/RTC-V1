"""Checkpoint continuity, immutable resume and pinned processing asset checks."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from . import assets
from .config import parser, training_configuration, validate_locked_cli
from .common import RECIPE, schema_for
from .resources import audit_training
from .processes import is_suite_parent
from w2v_v3189.test_training import fixture


class ConfigurationTests(unittest.TestCase):
    def test_repeat_seed_does_not_repartition_historical_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            old, _, _ = fixture(Path(directory))
            old.update(encoder_layers=60, lora_layers=60, backend_precision="fp16", epochs=10)
            origin = dict(epoch=6, steps=1403, run=directory, manifest_files={}, tag="epoch_6_step_1403")
            args = parser().parse_args(["--seed", "319102", "--workers", "0"])
            with patch("w2v_v3191.config.runtime_versions", return_value={"fixture":"cpu"}):
                cfg = training_configuration(old, origin, args)
            self.assertEqual(cfg["seed"], old["seed"])
            self.assertEqual(cfg["sampling_seed"], 319102)
            self.assertEqual(cfg["lora_layers"], 16)
            self.assertEqual(cfg["retained_lora_layers"], 60)
            self.assertEqual(cfg["epochs"], 2)
            self.assertEqual(cfg["experiment"], "E0")
            self.assertEqual(cfg["warm_epochs"], 0)
            self.assertEqual(cfg["meta_beta"], 1.)
            self.assertEqual(cfg["recipe"], RECIPE)
            self.assertEqual(cfg["source_epoch_offset"], old.get("source_epoch_offset", 0)+6)
            self.assertTrue(all(cfg[k]>0 for k in ("lora_lr", "head_lr", "fusion_lr")))
            schema_for(cfg)
            changed = parser().parse_args(["--resume", "existing", "--arm", "expanded"])
            with self.assertRaisesRegex(ValueError, "overrides"):
                validate_locked_cli(changed, ["--resume", "existing", "--arm", "expanded"])

    def test_asset_atomic_hash_validation_and_reuse(self):
        payload = b"test pinned model"
        correct = hashlib.sha256(payload).hexdigest()
        calls = []
        def opener(request, timeout):
            calls.append((request.full_url, timeout))
            return io.BytesIO(payload)
        with tempfile.TemporaryDirectory() as directory, patch.object(assets, "SHA256", correct):
            first = assets.prepare(directory, opener=opener)
            second = assets.prepare(directory, opener=opener)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            Path(first["rnnoise_model"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "registered"):
                assets.prepare(directory, opener=opener)
            self.assertEqual(len(calls), 1)
        with tempfile.TemporaryDirectory() as directory, patch.object(assets, "SHA256", correct):
            def wrong(request, timeout):
                return io.BytesIO(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "preparation failed"):
                assets.prepare(directory, opener=wrong)
            self.assertFalse((Path(directory)/"sh.rnnn").exists())
            self.assertFalse(list(Path(directory).glob("*.part")))

    def test_paired_clean_ce_in_resource_budget_without_pcl(self):
        rows=[dict(id="a", source_id="source", condition="offline", split="train", output_samples=16000)]
        cfg=dict(consistency="none", paired_classification=True,
                 whole_max_frames=100, frame_budget=300, microbatch=12)
        result=audit_training(rows, [], cfg)
        self.assertEqual(result["maxima"]["atomic_views"], 2)
        with self.assertRaisesRegex(ValueError, "resource preflight"):
            audit_training(rows, [], dict(cfg, frame_budget=60))

    def test_parent_process_exception_is_narrow(self):
        argv=["python", "-X", "faulthandler", "-u", "-m", "w2v_v3191.suite"]
        self.assertTrue(is_suite_parent(argv, 50, 50, "50"))
        self.assertFalse(is_suite_parent(argv, 50, 51, "50"))
        self.assertFalse(is_suite_parent(argv, 50, 50, "51"))
        self.assertFalse(is_suite_parent(["python", "another.py", "-m", "w2v_v3191.suite"], 50, 50, "50"))


if __name__ == "__main__":
    unittest.main()
