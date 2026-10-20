"""Real PCM fixtures protect the final refit's data and PCL boundaries."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from w2v_v31810.identity import registered_sources, verify_registry
from w2v_v31810.validation import _AudioIdentity
from w2v_v3185.common import GROUPS
from .manifests import (
    apply_coverage_budgets, bind_parent_phonemes, parent_phoneme_rows, prepare_manifests,
)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix=".v5_manifest_", dir=Path.cwd())
        self.root = Path(self.folder.name)
        self.counter = 0
        self.train, self.official, self.dev = [], [], []
        self.pairs = []
        for language, label in GROUPS:
            for index in range(2):
                self._source("train", language, label, index)
                self._source("dev", language, label, index)
        # Missing Online is legal and must survive final refitting.
        self._source("dev", "en", 1, 2, paired=False)
        self.pair_file = self.root / "pairs.csv"
        self.pair_file.write_text("offline_id,online_id\n" + "\n".join(
            f"{off},{on}" for off, on in self.pairs), encoding="utf-8")
        self.protocol = self.root / "dev.txt"
        self.protocol.write_text("\n".join(f"{r['id']} {r['label']}" for r in self.official),
                                 encoding="utf-8")
        self.cfg = dict(
            seed=31801, consistency="dual", official_dev_protocol=str(self.protocol),
            official_dev_root=str(self.root), dev_pairs=str(self.pair_file),
            source_draws_per_stream=[12, 12, 12, 12], stream_sources=24,
        )

    def tearDown(self):
        self.folder.cleanup()

    def _wave(self, name):
        self.counter += 1
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        wave = np.random.default_rng(self.counter).normal(0, .1, 640).astype(np.float32)
        sf.write(path, wave, 16000, subtype="PCM_16")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def _source(self, split, language, label, index, paired=True):
        identifier = f"{language}_{label}_{index}"
        off_id = f"{split}/offline/{language}/{identifier}.wav"
        off_path, off_hash = self._wave(off_id)
        rows = []
        for kind in ("offline", "online") if paired else ("offline",):
            name = off_id if kind == "offline" else f"{split}/online/{language}/{identifier}.wav"
            path, sha = (off_path, off_hash) if kind == "offline" else self._wave(name)
            row = dict(id=name, audio=str(path), audio_sha256=sha, language=language,
                       label=label, domain=kind, condition=kind, noisy=False, band=-1,
                       source_id=off_id, group_id=off_hash, source_sha256=off_hash,
                       output_samples=640, full_length=True, view="full", split=split)
            rows.append(row)
        if split == "train":
            self.train.extend(rows)
            return
        self.official.extend(rows)
        self.pairs.append((off_id, rows[-1]["id"] if paired else ""))
        for row in rows:
            if row["domain"] == "online":
                self.dev.append(copy.deepcopy(row))
            else:
                for condition in ("seen", "heldout"):
                    # These paths deliberately do not exist. The shared graph
                    # must use their original lineage, not decode stale caches.
                    self.dev.append(dict(row, condition=condition, noisy=True,
                        audio=str(self.root / "old_generated" / (condition + identifier + ".wav")),
                        source_audio=row["audio"]))

    def test_screen_preserves_dev_order_and_parent_phoneme_lineage(self):
        train, dev, registry, audit = prepare_manifests(
            self.cfg, self.train, self.dev, official=self.official)
        self.assertEqual(dev, self.dev)
        self.assertEqual(len(train), len(self.train))
        self.assertEqual(audit["final_dev_rows_added"], 0)
        self.assertEqual(self.cfg["v5_pcl_scope"], "parent_train")
        for old, new in zip(self.train, parent_phoneme_rows(train)):
            for field in old:
                self.assertEqual(old[field], new[field])
            self.assertTrue(new["v5_pcl_eligible"])
        verify_registry(train, registry, self.cfg,
                        {"source_registry_hash": registry["registry_hash"]})

    def test_final_adds_each_official_wave_once_without_synthetic_dev(self):
        original_train = copy.deepcopy(self.train)
        original_dev = copy.deepcopy(self.dev)
        train, dev, registry, audit = prepare_manifests(
            self.cfg, self.train, self.dev, phase="final", official=self.official)
        self.assertEqual(dev, [])
        added = [r for r in train if r["v5_origin_split"] == "dev"]
        self.assertEqual({r["id"] for r in added}, {r["id"] for r in self.official})
        self.assertEqual(len(added), len(self.official))
        self.assertLess(len(added), len(self.dev))
        self.assertEqual(audit["historical_noisy_views_added"], 0)
        self.assertEqual(audit["pcl"]["dev_rows_with_classification_only"], len(added))
        self.assertTrue(all(r["condition"] in ("offline", "online") for r in added))
        self.assertTrue(all(r["output_samples"] == 640 and Path(r["audio"]).is_file() for r in added))
        self.assertTrue(all(r["split"] == "train" and not r["v5_pcl_eligible"] for r in added))
        _, identity = registered_sources(train, self.cfg)
        self.assertEqual(identity["offline_only_sources"], 1)
        self.assertEqual(len(parent_phoneme_rows(train)), len(self.train))
        self.assertEqual(registry["dev"], [])
        self.assertEqual(self.train, original_train)
        self.assertEqual(self.dev, original_dev)
        for group in audit["added"].values():
            self.assertGreater(group["unique_sources"], 0)

    def test_pcm_duplicate_across_train_dev_is_rejected_before_refit(self):
        # Different container bytes, exactly the same decoded audio.
        source = self.train[0]
        target = self.official[0]
        data, rate = sf.read(source["audio"], dtype="float32")
        sf.write(target["audio"], data, rate, format="WAVEX", subtype="PCM_16")
        self.assertNotEqual(Path(source["audio"]).read_bytes(), Path(target["audio"]).read_bytes())
        target.pop("audio_sha256")
        with self.assertRaisesRegex(ValueError, "Train/Dev speech source or exact audio content overlaps"):
            prepare_manifests(self.cfg, self.train, self.dev, phase="final", official=self.official)

    def test_final_binds_registry_to_exact_pcm_and_row_order(self):
        rows, _, registry, _ = prepare_manifests(
            self.cfg, self.train, self.dev, phase="final", official=self.official)
        with self.assertRaisesRegex(ValueError, "identity/order"):
            registered_sources(list(reversed(rows)), self.cfg)
        changed = copy.deepcopy(rows)
        changed[-1]["decoded_pcm_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "identity/order"):
            registered_sources(changed, self.cfg)
        self.assertNotEqual(registry["registry_hash"], "")

    def test_progress_and_simulated_official_rows_rejected(self):
        bad = copy.deepcopy(self.official)
        bad[0]["audio"] = str(self.root / "progress" / "sample.wav")
        with self.assertRaisesRegex(ValueError, "Progress/Eval"):
            prepare_manifests(self.cfg, self.train, self.dev, official=bad)
        bad = copy.deepcopy(self.official)
        bad[0]["noisy"] = True
        with self.assertRaisesRegex(ValueError, "not simulated"):
            prepare_manifests(self.cfg, self.train, self.dev, official=bad)

    def test_missing_phoneme_assets_fail_instead_of_silent_disabled_loss(self):
        rows, _, _, _ = prepare_manifests(
            self.cfg, self.train, self.dev, phase="final", official=self.official)
        with self.assertRaisesRegex(ValueError, "phoneme manifest"):
            bind_parent_phonemes(self.cfg, rows)
        self.cfg["consistency"] = "none"
        with self.assertRaisesRegex(ValueError, "explicit"):
            bind_parent_phonemes(self.cfg, rows)
        self.cfg["v5_pcl_scope"] = "disabled"
        self.assertIsNone(bind_parent_phonemes(self.cfg, rows))

    def test_reused_identity_preserves_registry(self):
        identity = _AudioIdentity()
        screen_cfg, final_cfg = copy.deepcopy(self.cfg), copy.deepcopy(self.cfg)
        prepare_manifests(screen_cfg, self.train, self.dev, official=self.official, identity=identity)
        cached = len(identity.cache)
        _, _, registry, _ = prepare_manifests(
            final_cfg, self.train, self.dev, official=self.official, identity=identity, phase="final")
        self.assertEqual(cached, len(identity.cache))
        fresh_cfg = copy.deepcopy(self.cfg)
        _, _, fresh_registry, _ = prepare_manifests(
            fresh_cfg, self.train, self.dev, official=self.official, phase="final")
        self.assertEqual(registry, fresh_registry)

    def test_coverage_expands_final_but_cannot_change_screen_budget(self):
        with self.assertRaisesRegex(ValueError, "only for the final"):
            apply_coverage_budgets(self.cfg, [])
        self.cfg["v5_manifest_phase"] = "final"
        rows = []
        for group_index, (language, label) in enumerate(GROUPS):
            for index in range(15 if group_index == 0 else 4):
                for condition in ("offline", "online"):
                    rows.append(dict(canonical_source=f"{group_index}/{index}",
                                     language=language, label=label, condition=condition))
        report = apply_coverage_budgets(self.cfg, rows)
        self.assertEqual(report["draws"], [18, 12, 12, 12])
        self.assertEqual(report["steps_per_epoch"], 3)
        self.assertTrue(report["classification_loss_weights_unchanged"])


if __name__ == "__main__":
    unittest.main()
