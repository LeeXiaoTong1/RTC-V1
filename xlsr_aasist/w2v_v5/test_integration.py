"""CPU contracts across V5 manifests, sampler, data, loss layout and resume."""
from collections import Counter
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from w2v_v31810.identity import build_registry, registry_config
from w2v_v3185.common import GROUPS
from w2v_v3191.side import _layout
from .common import RECIPE
from .data import Waves
from .manifests import apply_coverage_budgets
from .processing import policy
from .sampling import Plan


def inventory():
    rows = []
    for language, label in GROUPS:
        for source_index in range(10):
            source = f"{language}/{label}/{source_index}"
            canonical = hashlib.sha256(source.encode()).hexdigest()
            origin = "train" if source_index < 5 else "dev"
            offline_id = source + "/offline"
            for condition in ("offline", "online"):
                key = source + "/" + condition
                rows.append(dict(id=key, audio=key, source_id=offline_id,
                    group_id=canonical, source_sha256=canonical,
                    audio_sha256=hashlib.sha256(("bytes/" + key).encode()).hexdigest(),
                    decoded_pcm_sha256=hashlib.sha256(("pcm/" + key).encode()).hexdigest(),
                    canonical_source=canonical, split="train", condition=condition,
                    domain=condition, label=label, language=language, full_length=True,
                    v5_origin_split=origin, v5_pcl_eligible=origin == "train"))
    registry = build_registry(rows, [], {})
    cfg = dict(source_registry=registry_config(registry), seed=31801, sampling_seed=50001,
        consistency="dual", v5_pcl_scope="parent_train", source_draws_per_stream=[8]*4,
        stream_sources=8, group_quotas=(2,)*4, view_factors=(5, 5, 1, 3),
        refresh_base_variants=True, experiment="E0", pcl_noise_weight=.15,
        pcl_platform_weight=.15, platform_sources_per_group=1,
        v5_manifest_phase="final", v5_input=policy("lowpass", 3.),
        v5_context_seconds=2., pcl_min_frames=1, warm_epochs=0,
        raw_audio_cache_mib=0)
    apply_coverage_budgets(cfg, rows)
    return rows, cfg


def ticket(row_index, *, paired=False):
    value = dict(index=row_index, kind="offline", epoch=0, step=0,
        base_variant=0, base_view_id="original", base_family="original",
        base_generation=0, base_dynamic=False, occurrence="test-source",
        pcl_enabled=paired, pcl_noise=paired, meta_side="A",
        batch_group_quotas=(2,)*4)
    if paired:
        value["pcl_online_index"] = row_index+1
    return value


def fake_augment(wave, rec):
    return wave.copy(), dict(recipe={"family": "original"},
        time_map=dict(scale=1., offset=0., input_samples=len(wave), output_samples=len(wave)))


def fake_matched(wave, rec, bank, engines, *, context_seconds):
    return wave.copy(), wave*.95, dict(pair_eligible=True,
        clean_activity={"suppressed_spans": []}, rtc_activity={"suppressed_spans": []},
        receiver_context={"context_samples": int(context_seconds*16000)})


class TinyModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)
        self.encoder.weight.requires_grad_(False)
        self.encoder.weight.data.fill_(2.)
        self.head = nn.Linear(1, 1, bias=False)
        self.head.weight.data.fill_(.5)


class FakePlan:
    steps = 4

    def batches(self, epoch):
        return [[{"index": i}] for i in range(self.steps)]

    def coverage(self, epoch):
        return {"epoch": epoch, "steps": self.steps}


class FakeLoader:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        yield from self.sampler.batches


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_final_sources_enter_both_streams_and_dev_never_requests_phones(self):
        rows, cfg = inventory()
        plan = Plan(rows, cfg["stream_sources"], cfg["sampling_seed"],
                    cfg["group_quotas"], cfg["view_factors"], cfg=cfg)
        seen = {"offline": set(), "online": set()}
        pcl = Counter()
        for batch in plan.batches(0):
            self.assertEqual(len({t["canonical_source"] for t in batch}), len(batch))
            for item in batch:
                row = rows[item["index"]]
                seen[item["kind"]].add(row["canonical_source"])
                if row["v5_origin_split"] == "dev":
                    self.assertFalse(item["pcl_enabled"])
                    self.assertFalse(item["pcl_noise"])
                    self.assertNotIn("pcl_online_index", item)
                if item["meta_side"] == "B":
                    self.assertFalse(item["pcl_enabled"])
                if item["pcl_enabled"]:
                    self.assertEqual(row["v5_origin_split"], "train")
                    pcl["parent"] += 1
        expected = {r["canonical_source"] for r in rows}
        self.assertEqual(seen["offline"], expected)
        self.assertEqual(seen["online"], expected)
        self.assertGreater(pcl["parent"], 0)
        self.assertEqual(plan.steps, cfg["v5_full_source_coverage"]["steps_per_epoch"])
        for left, right in plan.reference_pairs.items():
            self.assertEqual(rows[left]["v5_origin_split"], "train")
            self.assertEqual(rows[right]["v5_origin_split"], "train")
            self.assertEqual(rows[left]["canonical_source"], rows[right]["canonical_source"])

    def test_final_dev_keeps_both_ce_views_with_empty_pcl_keys(self):
        rows, cfg = inventory()
        index = next(i for i, row in enumerate(rows)
                     if row["v5_origin_split"] == "dev" and row["condition"] == "offline")
        data = Waves(rows, cfg)
        wave = np.random.default_rng(5).normal(0, .1, 3200).astype(np.float32)
        data.original = lambda row: wave.copy()
        data.bank, data.engines = object(), object()
        data._phones = lambda row: self.fail("Dev must not request unavailable phone records")
        with patch("w2v_v5.data.make_recipe", return_value={}), \
             patch("w2v_v5.data.augment", side_effect=fake_augment), \
             patch("w2v_v5.data.generate_matched_context", side_effect=fake_matched):
            unit = data[ticket(index)]
        row = unit[0]
        self.assertEqual(set(row["pcl_pairs"]), {"noise", "platform"})
        self.assertEqual(row["pcl_pairs"], {"noise": [], "platform": []})
        self.assertEqual(set(row["pcl_refs"]), {"clean"})
        self.assertEqual(len(row["wave"]), 48000)
        self.assertEqual(len(row["pcl_refs"]["clean"]["wave"]), 48000)
        self.assertEqual(row["original_samples"], len(wave))
        self.assertEqual(row["pcl_valid_frames"], 9)
        self.assertEqual(row["augmentation"]["receiver_context"]["context_samples"], 32000)
        views, slots = _layout(unit, {"noise": .15, "platform": .15})
        self.assertEqual(len(views), 2)
        self.assertIn("clean", slots)
        self.assertNotIn("noise", slots)
        self.assertNotIn("platform", slots)
        self.assertTrue(all(r["split"] == "train" for r in views))

    def test_parent_pcl_uses_original_prefix_and_removes_filter_boundaries(self):
        rows, cfg = inventory()
        data = Waves(rows, cfg)
        wave = np.random.default_rng(5).normal(0, .1, 3200).astype(np.float32)
        data.original = lambda row: wave.copy()
        data.bank, data.engines = object(), object()
        phones = [dict(instance=i, token=i+1, start_sample=start, end_sample=end)
                  for i, (start, end) in enumerate(((0, 640), (800, 1600), (2800, 3200)))]
        data._phones = lambda row: copy.deepcopy(phones)
        with patch("w2v_v5.data.make_recipe", return_value={}), \
             patch("w2v_v5.data.augment", side_effect=fake_augment), \
             patch("w2v_v5.data.generate_matched_context", side_effect=fake_matched), \
             patch("w2v_v5.data.estimate_delay", return_value={"valid": True, "offset": 0.}), \
             patch("w2v_v5.data.reliable_spans", side_effect=lambda a, b, spans, *rest: [True]*len(spans)):
            unit = data[ticket(0, paired=True)]
        row = unit[0]
        # The 32-sample FIR support plus 200-sample receptive-field guard
        # trims the first phone. Its remaining safe support is still useful.
        self.assertEqual(row["pcl_pairs"]["noise"],
                         [[232., 640., 232., 640.], [800, 1600, 800., 1600.]])
        self.assertEqual(row["pcl_pairs"]["platform"],
                         [[232., 640., 232., 640.], [800, 1600, 800, 1600]])
        views, slots = _layout(unit, {"noise": .15, "platform": .15})
        self.assertEqual(len(views), 3)
        self.assertIn("noise", slots)
        self.assertIn("platform", slots)
        for value in views:
            self.assertEqual(value["input_processing"]["pcl_valid_samples"], 3200)
            self.assertEqual(value["input_processing"]["repeated_samples"], 44800)
        self.assertTrue(all(max(pair) <= 3200 for pairs in row["pcl_pairs"].values() for pair in pairs))

    def test_parent_phoneme_cache_is_bound_only_to_parent_inventory(self):
        rows, cfg = inventory()
        cfg.update(phoneme_manifest="verified.json", phoneme_provenance={"test": True})
        data = Waves(rows, cfg)
        cache = SimpleNamespace(get=lambda row: {"segments": []})
        with patch("w2v_v3189.phonemes.PhonemeCache", return_value=cache) as create:
            data._phones(rows[0])
        bound_rows = create.call_args.args[1]
        self.assertEqual([r["id"] for r in bound_rows],
                         [r["id"] for r in rows if r["v5_origin_split"] == "train"])
        self.assertTrue(all(r["v5_pcl_eligible"] for r in bound_rows))

    def test_committed_resume_reconstructs_tail_average_with_frozen_base_excluded(self):
        from . import train as training
        cfg = dict(version="5", recipe=RECIPE, v5_stage="final", lora_layers=16,
            retained_lora_layers=60, experiment="E0", robust_arm="baseline",
            seed=31801, sampling_seed=50001, epochs=1, device="cpu",
            backend_precision="fp32", lr_floor=.5, checkpoint_interval=1,
            v5_tail_snapshots=4, v5_tail_start=0.)
        with tempfile.TemporaryDirectory(prefix=".v5_integration_", dir=Path.cwd()) as folder:
            root = Path(folder)
            snapshots = []
            def step(model, optimizer, units, runtime, audit=False):
                optimizer.zero_grad(set_to_none=True)
                loss = (model.head.weight - 1.).square().sum()
                loss.backward()
                optimizer.step()
                snapshots.append(model.head.weight.detach().clone())
                if len(snapshots) == 2:
                    (root / "STOP_REQUESTED").write_text("safe update", encoding="utf-8")
                return dict(loss=float(loss.detach()), classification_loss=float(loss.detach()),
                            amp={"skipped_update": False})
            def optimizer(model, config):
                return torch.optim.AdamW([dict(params=[model.head.weight], lr=.01,
                                               initial_lr=.01, name="head")], weight_decay=0.)
            patches = [
                patch.object(training, "make_plan", return_value=FakePlan()),
                patch.object(training, "Waves", return_value=object()),
                patch.object(training, "loader",
                             side_effect=lambda data, config, batch_sampler: FakeLoader(batch_sampler)),
                patch.object(training, "close"),
                patch.object(training, "optimizer_for", side_effect=optimizer),
                patch.object(training, "adapter_inventory", return_value={"test": True}),
            ]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for item in patches:
                    stack.enter_context(item)
                kwargs = dict(model_factory=TinyModel, step_fn=step, initializer=lambda m, c: {})
                _, paused = training.run_stage(cfg, root / "final", [], **kwargs)
                self.assertEqual(paused["status"], "paused")
                self.assertEqual(paused["cursor"], 2)
                self.assertEqual(paused["tail_count"], 2)
                (root / "STOP_REQUESTED").unlink()
                model, finished = training.run_stage(cfg, root / "final", [], **kwargs)
                self.assertEqual(finished["status"], "complete")
                self.assertEqual(finished["cursor"], 4)
                self.assertEqual(finished["tail_count"], 4)
                self.assertEqual(len(snapshots), 4)
                torch.testing.assert_close(finished["tail_model"]["head.weight"],
                                           torch.stack(snapshots).mean(0))
                self.assertNotIn("encoder.weight", finished["tail_model"])
                torch.testing.assert_close(model.encoder.weight, torch.full((1, 1), 2.))
                training.run_stage(cfg, root / "final", [], **kwargs)
                self.assertEqual(len(snapshots), 4, "Completed resume must consume no new update")
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    training.run_stage(dict(cfg, sampling_seed=50002), root / "final", [], **kwargs)


if __name__ == "__main__":
    unittest.main()
