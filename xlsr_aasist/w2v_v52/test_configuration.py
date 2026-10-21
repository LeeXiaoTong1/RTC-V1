"""Parent best continuity and non-overridable resume contracts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .config import parser, validate, training_configuration
from .common import schema_for
from w2v_v5.processing import validate_policy
from w2v_v3189.test_training import fixture
from .ce import classification_weights


class ConfigurationTests(unittest.TestCase):
    def test_parent_best_rates_and_geometry_without_new_teacher_or_meta(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as directory:
            old, _, _ = fixture(Path(directory))
            old.update(encoder_layers=60, lora_layers=60, backend_precision="fp16", epochs=10)
            origin = dict(epoch=6, steps=1403, run=directory, manifest_files={}, tag="epoch_6_step_1403")
            args = parser().parse_args(["--workers", "0"])
            with patch("w2v_v52.config.runtime_versions", return_value={"fixture": "cpu"}):
                cfg = training_configuration(old, origin, args)
            self.assertEqual(cfg["seed"], old["seed"])
            self.assertEqual((cfg["lora_layers"], cfg["retained_lora_layers"]), (16, 60))
            self.assertEqual(cfg["epochs"], 1)
            self.assertEqual(cfg["experiment"], "E0")
            self.assertEqual(cfg["warm_epochs"], 0)
            self.assertEqual(cfg["backend_precision"], "fp16")
            self.assertEqual(cfg["consistency"], old["consistency"])
            self.assertFalse(cfg["_candidate_diagnostics"])
            self.assertEqual(classification_weights(cfg), dict(online=.6, clean_rtc=.2, noisy=.2))
            self.assertEqual(cfg["source_draws_per_stream"], old["source_draws_per_stream"])
            for key in ("lora_lr", "fusion_lr", "head_lr"):
                self.assertGreater(cfg[key], 0)
            self.assertEqual(cfg['pcl_noise_weight'], old['pcl_noise_weight']*.2)
            self.assertEqual(cfg['pcl_platform_weight'], old['pcl_platform_weight'])
            schema_for(cfg)
            validate_policy(cfg["v5_input"])

    def test_resume_cannot_change_recipe_and_final_needs_existing_screen(self):
        valid = ["--resume", "run", "--phase", "final", "--submit-best"]
        validate(parser().parse_args(valid), valid)
        for bad in (["--resume", "run", "--epochs", "2"],
                    ["--resume", "run", "--context-seconds", "0"],
                    ["--upload-submission"],
                    ["--context-seconds", "nan"], ["--microbatch", "1"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate(parser().parse_args(bad), bad)

    def test_resource_guard_counts_repetition_and_canonical_duplicates(self):
        from .resources import audit_training
        base = dict(source_id="s", canonical_source="s", split="train")
        rows = [dict(base, id="off1", condition="offline", output_samples=16000),
                dict(base, id="off2", condition="offline", output_samples=32000),
                dict(base, id="on", condition="online", output_samples=16000)]
        cfg = dict(consistency="dual", paired_classification=True, whole_max_frames=3000,
                   microbatch=12, frame_budget=9000, window=128, hop=64, window_batch=16)
        report = audit_training(rows, [], cfg)
        self.assertEqual(report["units"], 2)
        self.assertEqual(report["maxima"]["atomic_views"], 3)
        self.assertGreaterEqual(report["maxima"]["classification_frames"], 148)
        with self.assertRaisesRegex(ValueError, "resource preflight"):
            audit_training(rows, [], dict(cfg, frame_budget=200))


if __name__ == "__main__":
    unittest.main()
