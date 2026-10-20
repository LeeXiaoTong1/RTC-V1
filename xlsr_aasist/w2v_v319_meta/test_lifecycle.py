"""Real checkpoint loading and numerical pause/resume with tiny detector tasks."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from w2v_v3189.test_training import fixture as source_fixture
from w2v_v3189.test_model import tiny_model as parent_model
from w2v_v3189.common import schema_for as parent_schema
from w2v_v3189.monitor import measure
from w2v_v31810.test_meta import assert_state_equal
from .common import RECIPE, atomic_json, read_json, fingerprint, digest, partial_state
from .test_continuation import detector, tasks
from .train import run_experiment, train_step
from .evaluate import selected, export
from .delivery import check_archive
from .reports import report_files


class Plan:
    def __init__(self, *args, **kwargs):
        self.steps = 2

    def batches(self, epoch):
        for step in range(self.steps):
            yield [dict(epoch=epoch, step=step, index=i) for i in range(16)]

    def calibration_batches(self):
        return []

    def coverage(self, epoch):
        return dict(epoch=epoch, fixture=True)


class Waves:
    def __init__(self, *args, **kwargs):
        self.cache = {}

    def __getitem__(self, ticket):
        key = ticket["epoch"], ticket["step"]
        if key not in self.cache:
            self.cache[key] = tasks()[1]
        return self.cache[key][ticket["index"]]


class Loader:
    def __init__(self, dataset, sampler):
        self.dataset, self.sampler = dataset, sampler

    def __iter__(self):
        for batch in self.sampler.batches:
            yield [self.dataset[t] for t in batch]


def fixture(root):
    old, rows, dev = source_fixture(root)
    old.update(warm_epochs=0, backend_precision="fp32", encoder_layers=2,
               free_reserve_bytes=0, disk_margin_bytes=0)
    source = Path(old["data_source"]["run"])
    atomic_json(source / "config.json", old)
    torch.manual_seed(old["seed"])
    model = parent_model(old)
    model.pcl_scale.observe(torch.stack((torch.zeros(128), torch.ones(128))), "calibration-1")
    model.pcl_scale.observe(torch.stack((torch.zeros(128), torch.ones(128))), "calibration-2")
    model.pcl_scale.freeze()
    z = np.asarray([[3., -3.] if r["label"] == 0 else [-3., 3.] for r in dev], dtype=np.float32)
    partition = read_json(source / "dev_partition.json")
    ix = partition["select"]
    metrics = measure([dev[i] for i in ix], z[ix])
    tag = "epoch_1_step_2"
    torch.save(dict(schema=parent_schema(old), identity=fingerprint(old), best_tag=tag,
                    best_model=partial_state(model), best_logits=torch.from_numpy(z),
                    best_metrics=metrics, history=[dict(tag=tag, epoch=1)]), source / "last.pt")
    origin = dict(run=str(source), tag=tag, epoch=1, steps=2,
                  checkpoint_sha256=digest(source / "last.pt"), config_sha256=digest(source / "config.json"))
    cfg = dict(old, version="3.19", recipe=RECIPE, lora_layers=1, retained_lora_layers=2,
               checkpoint_layers=1, epochs=1, warm_epochs=0, experiment="E2",
               pcl_ramp_updates=1, pcl_variance_weight=0., en_real_confidence_weight=0.,
               meta_beta=.5, meta_ramp_updates=1, parent_init=origin,
               initialization=dict(mode="v3189_best", old_detector_loaded=True),
               selection="parent select", saved_manifests={}, runtime_versions={},
               omni_provenance={"fixture": "tiny frozen pretrained encoder"},
               classification_objective="source_isolated_first_order_metadg_with_parent_PCL_on_A")
    return cfg, rows, dev


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_completed_parent_best_pause_resume_and_export_identity(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(directory)
            cfg, rows, dev = fixture(root)
            full, resumed = root / "full", root / "resume"
            for run in (full, resumed):
                run.mkdir()
                atomic_json(run / "config.json", cfg)
            stack.enter_context(patch("w2v_v319_meta.train.Plan", Plan))
            stack.enter_context(patch("w2v_v319_meta.train.Waves", Waves))
            stack.enter_context(patch("w2v_v319_meta.train.loader",
                                      lambda dataset, cfg, *, batch_sampler: Loader(dataset, batch_sampler)))
            stack.enter_context(patch("w2v_v319_meta.train.close"))
            calibration = stack.enter_context(patch("w2v_v319_meta.calibration.calibrate",
                                                     return_value=dict(selected_alpha=.001, status="complete")))

            def scores(model, rows, cfg):
                # Deliberately worse scores exercise explicit parent retention.
                return np.asarray([[-3., 3.] if r["label"] == 0 else [3., -3.] for r in rows],
                                  dtype=np.float32), rows

            done = run_experiment(cfg, full, rows, dev, detector, infer_fn=scores)
            self.assertTrue(done["baseline_fallback"])

            def stop_after_update(*args, **kwargs):
                result = train_step(*args, **kwargs)
                atomic_json(resumed / "stop_requested.json", dict(reason="test boundary"))
                return result

            with patch("w2v_v319_meta.train.train_step", stop_after_update):
                paused = run_experiment(cfg, resumed, rows, dev, detector, infer_fn=scores)
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["next_step"], 1)
            calls = calibration.call_count
            done = run_experiment(cfg, resumed, rows, dev, detector, infer_fn=scores)
            self.assertEqual(calibration.call_count, calls)
            self.assertEqual(done["completed_epochs"], 1)
            a = torch.load(full / "last.pt", weights_only=True)
            b = torch.load(resumed / "last.pt", weights_only=True)
            for field in ("model", "optimizer", "rng", "best_model", "last_logits", "best_logits"):
                assert_state_equal(self, a[field], b[field])
            self.assertEqual(b["cursor"], 2)
            self.assertEqual(len((resumed / "steps.jsonl").read_text().splitlines()), 2)
            _, _, best = selected(resumed, "best")
            _, _, last = selected(resumed, "last")
            self.assertEqual(best["best_origin"], "v3189_parent")
            self.assertEqual(last["best_origin"], "v319_meta_trained")
            self.assertNotEqual(best["selected"], last["selected"])
            self.assertTrue((resumed / "diagnostics" / "curves.html").is_file())
            protocol = root / "submission.txt"
            protocol.write_text("\n".join(r["id"] for r in dev[:2]) + "\n", encoding="utf-8")
            out = root / "export"
            artifact = export(resumed, out, device="cpu", workers=0, protocol=protocol,
                              audio_root=cfg["official_dev_root"], model_factory=detector,
                              verify=False, announce=False)
            checked, metadata = check_archive(out)
            self.assertEqual(artifact, checked)
            self.assertEqual(metadata["count"], 2)
            self.assertEqual(metadata["best_origin"], "v3189_parent")
            self.assertEqual(metadata["selected"], best["selected"])
            files = {r["file"] for r in report_files(resumed)}
            self.assertIn("parent_baseline_scores.npz", files)
            self.assertIn("dev_scores_epoch_1_step_2.npz", files)
            self.assertNotIn("last.pt", files)


if __name__ == "__main__":
    unittest.main()
