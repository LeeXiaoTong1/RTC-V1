"""The actual submitted final is the anchor; trained Dev cannot select it again."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import config, parent, train, workflow, delivery
from .common import atomic_json, digest, fingerprint, partial_state, apply_partial, read_json
from .test_lifecycle import TinyDetector, initialize_tiny
from w2v_v5.common import RECIPE as PARENT_RECIPE, schema_for as parent_schema
from w2v_v5.test_integration import inventory
from w2v_v31810.identity import build_registry, registry_config

LR_KEYS = ("lora_early_lr", "lora_middle_lr", "lora_lr", "head_lr", "fusion_lr",
           "layer_mix_lr", "classifier_lr", "evidence_lr", "pool_lr")


def fixture(root):
    run = root / "submitted"
    (run / "final").mkdir(parents=True)
    rows, cfg = inventory()
    registry = build_registry(rows, [], {})
    cfg.update(version="5", recipe=PARENT_RECIPE, v5_stage="final", epochs=1,
        lora_layers=16, retained_lora_layers=60, encoder_layers=60, robust_arm="baseline",
        source_registry=registry_config(registry), lr_floor=.37, source_epoch_offset=6,
        v5_tail_snapshots=4, v5_tail_start=.75, v5_context_seconds=1.5,
        data_files={}, runtime_versions={}, pcl_platform_weight=.045,
        classification_ce_weights=dict(online=.25, clean_rtc=.25, noisy=.5), pcl_noise_weight=.15,
        **{key: (index + 1) * 1e-5 for index, key in enumerate(LR_KEYS)})
    lock = dict(input=cfg["v5_input"],
                weights=dict(name="anchor_tail_50", source="tail", parent_weight=.5))
    atomic_json(run / "recipe_lock.json", lock)
    cfg["v5_recipe_lock_sha256"] = digest(run / "recipe_lock.json")
    files = {"train_rows.json": rows, "source_registry.json": registry}
    for name, value in files.items():
        atomic_json(run / "final" / name, value)
    cfg["saved_manifests"] = {name: digest(run / "final" / name) for name in files}
    for name in ("config.json", "final/config.json", "final_config.json"):
        atomic_json(run / name, cfg)
    model = TinyDetector()
    initialize_tiny(model, cfg)
    model.head.weight.data.add_(3.)  # Easily distinguish the submitted model.
    weights = {k:v for k,v in partial_state(model).items() if not k.startswith('fusion.conditioner.')}
    tag = "final_anchor_tail_50"
    torch.save(dict(schema=parent_schema(cfg), identity=fingerprint(cfg),
                    model=weights, tag=tag, recipe=lock), run / "final.pt")
    atomic_json(run / "completed.json", dict(status="complete", version="5", selected=tag,
        checkpoint_sha256=digest(run / "final.pt"), final_config_sha256=digest(run / "final_config.json"),
        recipe_sha256=digest(run / "recipe_lock.json")))
    # These must never be used as parent weights or as a failure/completion gate.
    (run / "final" / "last.pt").write_bytes(b"NOT the submitted weights")
    (run / "failure.log").write_text("artifact delivery incomplete: curl exit 22")
    atomic_json(run / "submission_status.json", dict(status="failed", error="upload only"))
    return run, cfg, weights, rows


def child_configuration(old, origin, *arguments):
    args = config.parser().parse_args(["--workers", "0", *arguments])
    config.validate(args)
    with patch.object(config, "runtime_versions", return_value={}):
        return config.training_configuration(old, origin, args)


class SubmittedParentTests(unittest.TestCase):
    def test_exact_final_weights_and_predictions_survive_failed_upload(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            run, _, expected, rows = fixture(Path(folder))
            old, origin, actual_rows, dev = parent.metadata(run)
            self.assertEqual(dev, [])
            self.assertEqual(actual_rows, rows)
            self.assertEqual(origin["checkpoint"], str(run / "final.pt"))
            child = child_configuration(old, origin)
            model = TinyDetector()
            # This fixture has no phoneme buffers; production verifies frozen scales.
            train.initialize(model, dict(child, consistency="none"))
            reference = TinyDetector()
            apply_partial(reference, train.parent_anchor(reference, expected))
            x = torch.tensor([[.1, .7], [-.4, .3]])
            torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
            for name, value in expected.items():
                self.assertTrue(torch.equal(value, partial_state(model)[name]), name)
            self.assertFalse(hasattr(model, "_v52_contrastive_bank"))

    def test_inherits_full_adapter_geometry_policy_context_and_correct_end_rates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            run, old, _, _ = fixture(Path(folder))
            _, origin, _, _ = parent.metadata(run)
            child = child_configuration(old, origin)
            self.assertEqual((child["retained_lora_layers"], child["lora_layers"]), (60, 16))
            self.assertEqual((child["v5_stage"], child["epochs"]), ("final", 1))
            self.assertEqual(child["v5_input"], old["v5_input"])
            self.assertEqual(child["v5_context_seconds"], 1.5)
            self.assertEqual(child["source_epoch_offset"], 7)
            self.assertEqual(child["pcl_noise_weight"], old["pcl_noise_weight"])
            self.assertEqual(child["pcl_platform_weight"], .045)
            self.assertEqual(child['v53_parent_pcl_weights'], dict(noise=.15, platform=.045))
            self.assertEqual(child['pcl_noise_weight'], .15)
            self.assertEqual(child['contrastive_weight'], 0.)
            self.assertEqual(child["classification_ce_weights"], old['classification_ce_weights'])
            for key in LR_KEYS:
                self.assertAlmostEqual(child[key], old[key] * .37, places=14)
            self.assertFalse(child["initialization"]["old_optimizer_loaded"])
            self.assertNotIn("v5_recipe_lock_sha256", child)

    def test_default_pins_v51_without_latest_marker_or_incomplete_fallback(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            root = Path(folder)
            (root / "exp").mkdir()
            run, *_ = fixture(root)
            target = root / "exp" / parent.DEFAULT_PARENT
            run.rename(target)
            (root/'exp'/'.latest_v52_run').write_text('must-never-be-used', encoding='utf-8')
            # Fixture manifest hashes are relative; its empty data_files need no relocation.
            with patch.object(parent, "ROOT", root):
                _, origin, _, _ = parent.metadata()
            self.assertEqual(Path(origin["run"]), target)
            (target/'completed.json').unlink()
            with patch.object(parent, 'ROOT', root), self.assertRaises(FileNotFoundError):
                parent.metadata()
            target.rename(root/'moved_parent')
            with patch.object(parent, 'ROOT', root), self.assertRaisesRegex(FileNotFoundError, 'parent-run'):
                parent.metadata()

    def test_bad_checkpoint_manifest_or_recipe_is_rejected(self):
        for name in ("final.pt", "final/train_rows.json", "recipe_lock.json", "final/config.json"):
            with self.subTest(file=name), tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
                run, *_ = fixture(Path(folder))
                _, origin, _, _ = parent.metadata(run)
                path = run / name
                if name == "final.pt":
                    with path.open("ab") as stream:
                        stream.write(b"changed")
                else:
                    atomic_json(path, {})
                with self.assertRaises((ValueError, KeyError)):
                    parent.load(run, origin)

    def test_incomplete_parent_never_falls_back_to_last_or_ancestor(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            run, *_ = fixture(Path(folder))
            (run / "completed.json").unlink()
            with patch.object(parent.legacy, "load", side_effect=AssertionError("ancestor fallback")):
                with self.assertRaises(FileNotFoundError):
                    parent.metadata(run)
            with self.assertRaisesRegex(ValueError, "run directory"):
                parent.metadata(run / "final")

    def test_rejects_dev_screening_but_accepts_direct_final(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            run, old, _, _ = fixture(Path(folder))
            _, origin, _, _ = parent.metadata(run)
            with self.assertRaisesRegex(ValueError, "trained on Dev"):
                child_configuration(old, origin, "--phase", "screen")
            cfg = child_configuration(old, origin, "--phase", "final")
            with patch.object(workflow, "screen", side_effect=AssertionError("screen called")), \
                 patch.object(workflow, "final_fit", return_value=True) as fit:
                self.assertTrue(workflow.run_training(Path(folder), cfg, [], [], {}, "all"))
                self.assertEqual(fit.call_count, 1)
            with self.assertRaisesRegex(ValueError, "trained by submitted parent"):
                workflow.checked_eval(None, [], cfg, Path(folder), "forbidden", "id")

    def test_phoneme_verification_uses_original_train_lineage_in_final_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            run, old, _, rows = fixture(Path(folder))
            _, origin, _, _ = parent.metadata(run)
            cfg = child_configuration(old, origin)
            cfg.update(code_fingerprints={}, omni_checkpoint="unused", omni_sha256="unused",
                augmentation_files={}, ffmpeg="unused", ffmpeg_sha256="unused", scene_noise_files={},
                phoneme_manifest="phones.json", phoneme_provenance={"expected": True})
            with patch("w2v_v3184.assets.configured_spec"), \
                 patch("w2v_v3185.config.verify_unique") as verify, \
                 patch.object(config, "runtime_versions", return_value={}), \
                 patch("w2v_v3189.phonemes.bind_manifest", return_value={"expected": True}) as phones:
                config.verify_inputs(cfg)
            expected = [r for r in rows if r["v5_origin_split"] == "train"]
            self.assertEqual(phones.call_args.args[1], expected)
            self.assertIn({origin["checkpoint"]: origin["checkpoint_sha256"]}, verify.call_args.args)
            self.assertTrue(all(r["v5_origin_split"] == "dev" for r in rows if r not in expected))

    def test_one_final_stage_reuses_manifest_and_anchors_to_submitted_weights(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            base = Path(folder)
            run, old, weights, rows = fixture(base)
            _, origin, _, _ = parent.metadata(run)
            cfg = child_configuration(old, origin)
            root = base / "child"
            root.mkdir()
            model = TinyDetector()
            apply_partial(model, train.parent_anchor(model, weights))
            last, tail = partial_state(model), partial_state(model)
            last["head.weight"] += 8.
            tail["head.weight"] += .4
            state = dict(status="complete", model=last, tail_model=tail)
            with patch.object(workflow, "verify_inputs"), \
                 patch.object(workflow, "bind_parent_phonemes"), \
                 patch.object(workflow, "audit_training", return_value={}), \
                 patch.object(workflow, "make_plan", return_value=SimpleNamespace(steps=10)), \
                 patch.object(workflow, "run_stage", return_value=(model, state)) as runner, \
                 patch.object(workflow, "prepare_manifests", side_effect=AssertionError("data duplicated")), \
                 patch.object(workflow, "checked_eval", side_effect=AssertionError("Dev evaluated")), \
                 patch.object(workflow, "screen_input", side_effect=AssertionError("input selected")), \
                 patch.object(workflow, "choose_candidate", side_effect=AssertionError("weights selected")):
                self.assertTrue(workflow.run_training(root, cfg, rows, [], {}, "all"))
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.args[0]["epochs"], 1)
            self.assertEqual(runner.call_args.args[2], rows)
            self.assertFalse((root / "screen").exists())
            actual_cfg, actual, _ = delivery.selected(root)
            torch.testing.assert_close(actual["head.weight"], weights["head.weight"] + .2)
            self.assertEqual(actual_cfg["v5_input"], old["v5_input"])
            self.assertIsNone(read_json(root / "completed.json")["evaluation"])
            workflow.continuation_lock(root, cfg)  # A restart cannot reselect the rule.
            lock = read_json(root / "recipe_lock.json")
            lock["weights"]["parent_weight"] = 0.
            atomic_json(root / "recipe_lock.json", lock)
            with self.assertRaisesRegex(ValueError, "recipe changed"):
                workflow.continuation_lock(root, cfg)


if __name__ == "__main__":
    unittest.main()
