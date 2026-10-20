"""Numerical continuation, saved-boundary resume and actual score export."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayer, TinyLayout
from w2v_v31810.test_meta import assert_state_equal
from w2v_v319_meta.test_lifecycle import fixture as parent_fixture
from .common import RECIPE, atomic_json, read_json, digest
from .model import Detector
from .test_step import fixture as task_fixture
from .train import run_experiment, train_step, promotion_check
from .evaluate import selected, export
from .delivery import check_archive
from .calibration import validate_identity
from .reports import report_files


class Plan:
    def __init__(self, *args, **kwargs):
        self.steps = 2

    def batches(self, epoch):
        for step in range(self.steps):
            yield [dict(epoch=epoch, step=step, index=i) for i in range(16)]

    def coverage(self, epoch):
        return dict(epoch=epoch, fixture=True)


class Waves:
    def __init__(self, *args, **kwargs):
        self.cache = {}

    def __getitem__(self, ticket):
        key = ticket["epoch"], ticket["step"]
        if key not in self.cache:
            self.cache[key] = task_fixture(copies=1)[1]
        return self.cache[key][ticket["index"]]


class Loader:
    def __init__(self, dataset, sampler):
        self.dataset, self.sampler = dataset, sampler

    def __iter__(self):
        for batch in self.sampler.batches:
            yield [self.dataset[t] for t in batch]


def detector(cfg):
    encoder = TinyEncoder()
    encoder.layers = nn.ModuleList([TinyLayer() for _ in range(cfg["encoder_layers"])])
    return Detector(TinyFrontend(), encoder, cfg, TinyLayout)


def fixture(root):
    cfg, rows, dev = parent_fixture(root)
    cfg.update(version="3.19.1", recipe=RECIPE, experiment="E0",
               robust_arm="baseline", robust_candidate_count=4, robust_hard_fraction=.5,
               paired_classification=True, consistency="none", sampling_seed=cfg["seed"],
               meta_beta=1., lr_floor=.5)
    return cfg, rows, dev


def receiver(model, cfg, run, panel, *, split, checkpoint_tag,
             baseline_logits=None, reference_logits=None, bootstrap=False):
    if split == "audit":
        assert cfg["robust_final_audit"]
        assert (Path(run)/"robust_selection_locked.json").is_file()
    path = Path(run)/f"robust_scores_{split}_{checkpoint_tag}.npz"
    z = np.array([[2., -2.], [-2., 2.]], np.float32)
    np.savez_compressed(path, logits=z)
    # Parent retained: an unsafe/worse candidate must not be promoted.
    is_parent = checkpoint_tag.startswith("parent")
    return dict(score=.95 if is_parent else .90, worst_mechanism_f1=.95 if is_parent else .90,
                split=split, checkpoint_tag=checkpoint_tag, panel_hash=panel["hash"],
                groups={f"{lang}/{label}":dict(recall=.95, count=10)
                        for lang in ("en", "zh") for label in ("fake", "real")},
                predictions=dict(file=path.name, sha256=digest(path)))


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pause_resume_parent_retention_and_fixed_export(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(directory)
            cfg, rows, dev = fixture(root)
            full, resumed = root/"full", root/"resumed"
            for run in (full, resumed):
                run.mkdir()
                atomic_json(run/"config.json", cfg)
            stack.enter_context(patch("w2v_v3191.train.Plan", Plan))
            stack.enter_context(patch("w2v_v3191.train.Waves", Waves))
            stack.enter_context(patch("w2v_v3191.train.loader",
                lambda dataset, cfg, *, batch_sampler: Loader(dataset, batch_sampler)))
            stack.enter_context(patch("w2v_v3191.train.close"))
            stack.enter_context(patch("w2v_v3191.train.prepare_panel", return_value={"hash":"test-fixed-panel"}))
            evaluate = stack.enter_context(patch("w2v_v3191.train.evaluate_panel", side_effect=receiver))

            def scores(model, rows, cfg):
                return np.asarray([[-3., 3.] if r["label"] == 0 else [3., -3.]
                                   for r in rows], dtype=np.float32), rows

            done = run_experiment(cfg, full, rows, dev, detector, infer_fn=scores)
            self.assertTrue(done["baseline_fallback"])

            def stop_after_update(*args, **kwargs):
                result = train_step(*args, **kwargs)
                atomic_json(resumed/"stop_requested.json", dict(reason="test boundary"))
                return result

            with patch("w2v_v3191.train.train_step", stop_after_update):
                paused = run_experiment(cfg, resumed, rows, dev, detector, infer_fn=scores)
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["next_step"], 1)
            self.assertFalse((resumed/"robust_final.json").exists())
            done = run_experiment(cfg, resumed, rows, dev, detector, infer_fn=scores)
            self.assertEqual(done["completed_epochs"], 1)
            a = torch.load(full/"last.pt", weights_only=True)
            b = torch.load(resumed/"last.pt", weights_only=True)
            for field in ("model", "optimizer", "rng", "best_model"):
                assert_state_equal(self, a[field], b[field])
            for field in ("last_logits", "best_logits"):
                torch.testing.assert_close(a[field], b[field], rtol=0, atol=0, equal_nan=True)
            partition = read_json(resumed/"dev_partition.json")
            self.assertTrue(torch.isnan(b["last_logits"][partition["calibration"]]).all())
            self.assertEqual(b["cursor"], 2)
            self.assertEqual(len((resumed/"steps.jsonl").read_text().splitlines()), 2)
            _, _, best = selected(resumed, "best")
            _, _, last = selected(resumed, "last")
            self.assertEqual(best["best_origin"], "v3189_parent")
            self.assertEqual(last["best_origin"], "v3191_trained")
            final = read_json(resumed/"robust_final.json")
            self.assertEqual(final["select"]["checkpoint_tag"], final["chosen_checkpoint_tag"])
            calibration = read_json(resumed/"calibration_best.json")
            validate_identity(calibration)
            for call in evaluate.call_args_list:
                if call.kwargs["split"] == "audit":
                    self.assertTrue(call.args[1]["robust_final_audit"])

            protocol = root/"submission.txt"
            protocol.write_text("\n".join(r["id"] for r in dev[:2])+"\n", encoding="utf-8")
            out = root/"export"
            artifact = export(resumed, out, device="cpu", workers=0, protocol=protocol,
                              audio_root=cfg["official_dev_root"], model_factory=detector,
                              verify=False, announce=False)
            checked, metadata = check_archive(out)
            self.assertEqual(artifact, checked)
            self.assertEqual(metadata["best_origin"], "v3189_parent")
            self.assertFalse(metadata["calibration"]["fitted"])
            files = {r["file"] for r in report_files(resumed)}
            self.assertTrue(any(f.startswith("robust_scores_") for f in files))
            self.assertNotIn("last.pt", files)

    def test_promoted_model_sees_audit_only_after_checkpoint_lock(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(directory)
            cfg, rows, dev = fixture(root)
            run = root/"promoted"
            run.mkdir()
            atomic_json(run/"config.json", cfg)
            stack.enter_context(patch("w2v_v3191.train.Plan", Plan))
            stack.enter_context(patch("w2v_v3191.train.Waves", Waves))
            stack.enter_context(patch("w2v_v3191.train.loader",
                lambda dataset, cfg, *, batch_sampler: Loader(dataset, batch_sampler)))
            stack.enter_context(patch("w2v_v3191.train.close"))
            stack.enter_context(patch("w2v_v3191.train.prepare_panel", return_value={"hash":"test-fixed-panel"}))
            def improving(*args, **kwargs):
                result = receiver(*args, **kwargs)
                if not kwargs["checkpoint_tag"].startswith("parent"):
                    result.update(score=.99, worst_mechanism_f1=.99)
                return result
            stack.enter_context(patch("w2v_v3191.train.evaluate_panel", side_effect=improving))
            part = read_json(Path(cfg["parent_init"]["run"])/"dev_partition.json")
            calls = []
            def scores(model, current, config):
                indices = [next(i for i,r in enumerate(dev) if r["id"] == row["id"]) for row in current]
                calls.append(indices)
                if set(indices) & set(part["calibration"]):
                    self.assertTrue((run/"robust_selection_locked.json").is_file())
                return np.asarray([[3., -3.] if r["label"] == 0 else [-3., 3.]
                                   for r in current], np.float32), current
            result = run_experiment(cfg, run, rows, dev, detector, infer_fn=scores)
            self.assertFalse(result["baseline_fallback"])
            self.assertEqual(calls, [part["select"], part["calibration"]])
            state = torch.load(run/"last.pt", weights_only=True)
            self.assertTrue(torch.isfinite(state["best_logits"]).all())
            self.assertTrue(torch.isfinite(state["last_logits"]).all())
            self.assertEqual(state["cursor"], 2)
            selected(run, "best")  # Final checkpoint hash includes completed Audit scores.

    def test_promotion_requires_both_classes_and_select_only(self):
        parent = dict(split="select", score=.9, worst_mechanism_f1=.89,
                      groups={"en/real":dict(recall=.8), "en/fake":dict(recall=.99)})
        current = copy.deepcopy(parent)
        current["score"] = .94
        current["groups"]["en/fake"]["recall"] = .97
        metrics = dict(clean_f1=.98)
        result = promotion_check(current, parent, .90, metrics, metrics, {})
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["recall_en/fake"])
        current["groups"]["en/fake"]["recall"] = .99
        self.assertTrue(promotion_check(current, parent, .90, metrics, metrics, {})["passed"])
        current["split"] = "audit"
        with self.assertRaisesRegex(ValueError, "Audit"):
            promotion_check(current, parent, .90, metrics, metrics, {})

    def test_export_forbids_fitted_or_affine_protocol(self):
        for record in (dict(scale=1., bias=0., fitted=True, selected_candidate="fixed_identity"),
                       dict(scale=1., bias=.1, fitted=False, selected_candidate="fixed_identity")):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                validate_identity(record)


if __name__ == "__main__":
    unittest.main()
