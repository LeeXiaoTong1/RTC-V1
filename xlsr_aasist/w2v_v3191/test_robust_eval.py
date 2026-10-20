"""CPU tests for frozen receiver evaluation, source clustering and arm matching."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .robust_eval import (
    prepare_panel, evaluate_panel, summarize_panel, paired_bootstrap,
    transitions, read_scores, infer_panel, _write, _file_hash, _hash,
)
from .compare import compare_runs


class ReceiverPanelTest(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parent
        self.temp = tempfile.TemporaryDirectory(dir=self.workspace, prefix=".test_receiver_")
        self.root = Path(self.temp.name)
        self.parent = self.root / "parent"
        self.parent.mkdir()
        _write(self.parent / "train_rows.json", [
            dict(id="train", source_id="train", group_id="train", split="train")])
        self.cfg = dict(
            seed=42, sampling_seed=100, robust_protocol_seed=3191001, robust_arm="baseline",
            parent_init=dict(run=str(self.parent), checkpoint_sha256="f" * 64,
                             config_sha256="e" * 64, tag="epoch_6"),
            epochs=1, stream_sources=24, source_draws_per_stream=[8000] * 4,
            source_epoch_offset=6, lora_layers=16, retained_lora_layers=60,
            consistency="dual", backend_precision="fp16")
        self.dev = []
        self.partition = dict(select=[], calibration=[], source_keys=[])
        # Two complete sources per stratum in each split. Every source has an
        # Online view plus both inherited Noisy lanes.
        for split in ("select", "calibration"):
            for lang in ("en", "zh"):
                for label in (0, 1):
                    for instance in range(2):
                        source = f"{split}_{lang}_{label}_{instance}"
                        for condition in ("online", "seen", "heldout"):
                            i = len(self.dev)
                            path = self.root / f"{source}_{condition}.wav"
                            path.write_bytes(f"frozen PCM placeholder {source} {condition}".encode())
                            self.dev.append(dict(
                                id=path.name, audio=str(path), source_id=source,
                                group_id=source, split="dev", condition=condition,
                                language=lang, label=label, full_length=True))
                            self.partition[split].append(i)
                            self.partition["source_keys"].append(source)
        self.run = self.root / "run"
        self.panel = prepare_panel(self.cfg, self.run, self.dev, self.partition)

    def tearDown(self):
        if not self.root.resolve().is_relative_to(self.workspace):
            raise RuntimeError("Refusing cleanup outside the test workspace")
        self.temp.cleanup()

    @staticmethod
    def infer(model, rows, cfg):
        z = np.full((len(rows), 2), -2., dtype=np.float32)
        for i, row in enumerate(rows):
            label = row["label"]
            if model == "wrong" and row["source_id"].endswith("_0"):
                label = 1 - label
            z[i, label] = 2.
        # In production this is derived from processed PCM; tests provide
        # deterministic per-row hashes to exercise artifact verification.
        return z, [_hash((r["id"], r["mechanism_seed"])) for r in rows]

    def test_all_sources_frozen_across_training_seeds_and_arms(self):
        second = prepare_panel(dict(self.cfg, seed=999, robust_arm="hard"),
                               self.root / "second", self.dev, self.partition)
        self.assertEqual(self.panel["hash"], second["hash"])
        self.assertEqual(len(self.panel["rows"]), 32)
        self.assertEqual(len(self.panel["source_manifest"]), 16)
        self.assertTrue(self.panel["provenance"]["retained_all_sources"])
        self.assertIn("not unseen real RTC", self.panel["provenance"]["limitation"])
        for source in self.panel["source_manifest"]:
            rows = [r for r in self.panel["rows"] if r["source_id"] == source["source_id"]]
            self.assertEqual(len({r["base_index"] for r in rows}), 1)
            self.assertEqual(len({r["panel_split"] for r in rows}), 1)

    def test_complete_source_cannot_cross_select_audit(self):
        invalid = copy.deepcopy(self.partition)
        invalid["select"].remove(0)
        invalid["calibration"].append(0)
        with self.assertRaisesRegex(ValueError, "complete source"):
            prepare_panel(self.cfg, self.root / "bad", self.dev, invalid)

    def test_train_overlap_is_rejected(self):
        _write(self.parent / "train_rows.json", [
            dict(source_id=self.dev[0]["source_id"], split="train")])
        with self.assertRaisesRegex(ValueError, "identity overlap"):
            prepare_panel(self.cfg, self.root / "bad", self.dev, self.partition)

    def test_pcm_registry_merges_independent_ids_before_split_check(self):
        run = self.root / "pcm"
        run.mkdir()
        registry = dict(
            train=[dict(canonical_source="train_pcm")],
            dev=[dict(id=r["id"], source_id=r["source_id"],
                      canonical_source="pcm_" + r["source_id"]) for r in self.dev])
        # A second ID assigned to Audit contains the exact same original PCM.
        first = "pcm_" + self.dev[0]["source_id"]
        other = self.dev[self.partition["calibration"][0]]["source_id"]
        for row in registry["dev"]:
            if row["source_id"] == other:
                row["canonical_source"] = first
        registry["registry_hash"] = _hash(registry)
        _write(run / "source_registry.json", registry)
        cfg = dict(self.cfg, source_registry=dict(registry_hash=registry["registry_hash"]))
        with self.assertRaisesRegex(ValueError, "PCM-identical"):
            prepare_panel(cfg, run, self.dev, self.partition)

    def test_audit_locked_and_input_pcm_integrity(self):
        with self.assertRaisesRegex(ValueError, "locked"):
            evaluate_panel(None, self.cfg, self.run, self.panel,
                           split="audit", infer_fn=self.infer)
        used = self.panel["rows"][self.panel["split_indices"]["select"][0]]["audio"]
        Path(used).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "audio changed"):
            evaluate_panel(None, self.cfg, self.run, self.panel, infer_fn=self.infer)

    def test_saved_scores_and_deterministic_transforms(self):
        parent = evaluate_panel("wrong", self.cfg, self.run, self.panel,
                                checkpoint_tag="parent", infer_fn=self.infer)
        baseline = read_scores(self.run, parent)
        selected = evaluate_panel("right", self.cfg, self.run, self.panel,
                                  baseline_logits=baseline, checkpoint_tag="epoch_1",
                                  infer_fn=self.infer, bootstrap=True)
        self.assertEqual(selected["score"], 1.)
        self.assertGreater(selected["delta_vs_parent"]["score"], 0.)
        self.assertGreaterEqual(selected["paired_comparison"]["metrics"]["score"]["low"], 0.)

        def changed(model, rows, cfg):
            logits, hashes = self.infer(model, rows, cfg)
            return logits, ["different"] + hashes[1:]

        with self.assertRaisesRegex(ValueError, "waveform identity"):
            evaluate_panel("right", self.cfg, self.run, self.panel,
                           checkpoint_tag="epoch_2", infer_fn=changed)

    def test_fixed_threshold_and_class_error_transitions(self):
        rows = [r for r in self.panel["rows"] if r["panel_split"] == "select"]
        tie = np.zeros((len(rows), 2))
        summary = summarize_panel(rows, tie)
        self.assertEqual(summary["groups"]["en/fake"]["recall"], 1.)
        self.assertEqual(summary["groups"]["en/real"]["recall"], 0.)
        reference = np.asarray([[2., -2.] if r["label"] == 0 else [-2., 2.]
                                for r in self.dev])
        changed = transitions(rows, reference, tie)
        self.assertEqual(changed["en/real"]["correct_to_wrong"], 4)
        self.assertEqual(changed["en/fake"]["correct_to_wrong"], 0)

    def test_bootstrap_keeps_all_views_per_source(self):
        rows = [r for r in self.panel["rows"] if r["panel_split"] == "select"]
        base, _ = self.infer("wrong", rows, self.cfg)
        target, _ = self.infer("right", rows, self.cfg)
        first = paired_bootstrap(rows, base, target, draws=100, seed=7)
        # Duplicating every view must not create more independent observations.
        repeat = paired_bootstrap(rows * 3, np.tile(base, (3, 1)),
                                  np.tile(target, (3, 1)), draws=100, seed=7)
        for name in first["metrics"]:
            for field in ("delta", "low", "high"):
                self.assertAlmostEqual(first["metrics"][name][field],
                                       repeat["metrics"][name][field])
        self.assertFalse(first["simultaneous_confidence"])

    def test_production_receiver_loader_keeps_order_modes_and_rng(self):
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("Production soundfile dependency is unavailable in this CPU test environment")
        import torch

        class Detector(torch.nn.Module):
            def forward(self, waves):
                means = torch.tensor([float(np.mean(w)) for w in waves])
                return torch.stack((means, -means), dim=1)

        rows = copy.deepcopy(self.panel["rows"][:4])
        for row in rows:
            signal = (.05 * np.sin(np.arange(8000) * .08)).astype(np.float32)
            sf.write(row["audio"], signal, 16000, subtype="FLOAT")
        cfg = dict(self.cfg, workers=0, eval_workers=0, eval_batch=3,
                   microbatch=2, frame_budget=1000)
        model = Detector().train()
        torch.manual_seed(17)
        rng = torch.random.get_rng_state().clone()
        first, hashes = infer_panel(model, rows, cfg)
        second, second_hashes = infer_panel(model, rows, cfg)
        self.assertEqual(first.shape, (len(rows), 2))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(hashes, second_hashes)
        self.assertTrue(all(len(value) == 64 for value in hashes))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def _final_run(self, arm, model):
        path = self.root / arm
        cfg = dict(self.cfg, robust_arm=arm)
        panel = prepare_panel(cfg, path, self.dev, self.partition)
        reports = {}
        for split in ("select", "audit"):
            reports[split] = evaluate_panel(
                model, dict(cfg, robust_final_audit=True), path, panel, split=split,
                checkpoint_tag="epoch_1", infer_fn=self.infer)
        _write(path / "config.json", cfg)
        _write(path / "robust_final.json", dict(
            chosen_checkpoint_tag="epoch_1", selection_split="select",
            audit_used_for_selection=False, compare_budget=dict(optimizer_updates=10),
            **reports))
        (path / "last.pt").write_bytes(b"frozen selected checkpoint")
        _write(path / "completed.json", dict(
            status="complete", best_tag="epoch_1",
            checkpoint_sha256=_file_hash(path / "last.pt")))
        return path

    def test_comparison_requires_same_protocol_budget_and_no_audit_selection(self):
        baseline = self._final_run("baseline", "wrong")
        hard = self._final_run("hard", "right")
        result = compare_runs([baseline, hard], draws=100)
        self.assertGreater(result["comparisons"][0]["results"]["audit"]["metrics"]["score"]["delta"], 0.)
        modified = dict(self.cfg, robust_arm="hard", epochs=2)
        _write(hard / "config.json", modified)
        with self.assertRaisesRegex(ValueError, "not matched"):
            compare_runs([baseline, hard], draws=100)
        modified = dict(self.cfg, robust_arm="hard", sampling_seed=101)
        _write(hard / "config.json", modified)
        with self.assertRaisesRegex(ValueError, "not matched"):
            compare_runs([baseline, hard], draws=100)
        for change in (dict(microbatch=4), dict(pcl_noise_weight=.2),
                       dict(robust_hard_fraction=.9), dict(robust_min_gain=.002),
                       dict(robust_score_microbatch=8), dict(label_smoothing=.1)):
            _write(hard / "config.json", dict(self.cfg, robust_arm="hard", **change))
            with self.assertRaisesRegex(ValueError, "not matched"):
                compare_runs([baseline, hard], draws=100)

    def test_three_arm_comparison_isolates_hard_from_expanded(self):
        baseline = self._final_run("baseline", "wrong")
        expanded = self._final_run("expanded", "wrong")
        hard = self._final_run("hard", "right")
        result = compare_runs([baseline, expanded, hard], draws=100)
        contrasts = {(r["reference_arm"], r["arm"]) for r in result["comparisons"]}
        self.assertEqual(contrasts, {("baseline", "expanded"), ("baseline", "hard"),
                                    ("expanded", "hard")})

    def test_select_score_indexing_marks_audit_unobserved(self):
        from .train import indexed_scores
        logits = np.asarray([[2., -2.], [-2., 2.]], dtype=np.float32)
        frame = indexed_scores(4, [1, 3], logits)
        np.testing.assert_array_equal(frame[[1, 3]], logits)
        self.assertTrue(np.isnan(frame[[0, 2]]).all())


if __name__ == "__main__":
    unittest.main()
