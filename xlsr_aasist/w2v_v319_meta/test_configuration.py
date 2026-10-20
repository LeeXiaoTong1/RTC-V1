import copy
import math
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from w2v_v3189.test_model import tiny_cfg
from .config import parser, training_configuration, validate_locked_cli
from .parent import resolve


class ConfigurationTests(unittest.TestCase):
    def test_one_epoch_inherits_best_endpoint_and_full_adapter_history(self):
        old = tiny_cfg(60)
        old.update(encoder_layers=60, consistency="dual", epochs=10, warm_epochs=0,
                   lr_warmup_updates=200, lr_floor=.1, backend_precision="fp16",
                   rtc_activity_policy="mixed", source_epoch_offset=9,
                   classifier_lr=.004, evidence_lr=.005, pool_lr=.006,
                   head_lr=.002, fusion_lr=.003, layer_mix_lr=.007)
        before = copy.deepcopy(old)
        args = parser().parse_args([])
        origin = dict(epoch=6, steps=1403, run="/frozen-parent", tag="epoch_6_step_1403",
                      manifest_files={"/frozen-parent/config.json": "verified"})
        with patch("w2v_v319_meta.config.runtime_versions", return_value={}), \
             patch("w2v_v319_meta.workers.settings", return_value={"workers": 0, "eval_workers": 0}):
            cfg = training_configuration(old, origin, args)
        self.assertEqual(old, before)
        self.assertEqual((cfg["epochs"], cfg["warm_epochs"]), (1, 0))
        self.assertEqual((cfg["retained_lora_layers"], cfg["lora_layers"], cfg["checkpoint_layers"]), (60, 16, 16))
        self.assertEqual(cfg["source_epoch_offset"], 15)
        self.assertEqual(cfg["consistency"], "dual")
        self.assertEqual(cfg["pooling"], old["pooling"])
        multiplier = .1 + .9 * .5 * (1 + math.cos(math.pi * (6 * 1403 - 1) / (10 * 1403 - 1)))
        for name in ("lora_lr", "lora_early_lr", "lora_middle_lr", "head_lr", "fusion_lr",
                     "layer_mix_lr", "classifier_lr", "evidence_lr", "pool_lr"):
            self.assertAlmostEqual(cfg[name], old[name] * multiplier, places=12)
        self.assertEqual(cfg["lr_floor"], .5)
        self.assertFalse(cfg["initialization"]["old_optimizer_loaded"])

    def test_default_parent_never_uses_a_newer_latest_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exp = root / "exp"
            exp.mkdir()
            (exp / ".latest_v3189_run").write_text("/unrelated-new-run", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "w2v_v3189_20261016_183410_a437"):
                resolve(root=root)
            parent = exp / "w2v_v3189_20261016_183410_a437"
            parent.mkdir()
            for name in ("config.json", "last.pt", "train_rows.json", "dev_rows.json", "dev_partition.json"):
                (parent / name).touch()
            self.assertEqual(resolve(root=root), parent.resolve())

    def test_resume_rejects_changed_epoch_or_meta_contract(self):
        args = parser().parse_args(["--resume", "/run", "--epochs", "2"])
        with self.assertRaisesRegex(ValueError, "saved settings"):
            validate_locked_cli(args, ["--resume", "/run", "--epochs", "2"])
        args = parser().parse_args(["--resume", "/run"])
        validate_locked_cli(args, ["--resume", "/run"])


if __name__ == "__main__":
    unittest.main()
