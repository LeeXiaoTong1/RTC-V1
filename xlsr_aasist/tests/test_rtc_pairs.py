import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from utils.rtc_data import BalancedPairBatchSampler, RTCPairDataset
from utils.rtc_pairs import load_pairs, prepare_pairs


class PairFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.audio = self.root / "wav" / "train"
        self.audio.mkdir(parents=True)
        self.protocol = self.root / "train_label.txt"
        self.csv_path = self.root / "official_pairs.csv"
        self.manifest = self.root / "train_pairs.jsonl"
        self.entries = {}
        self.waveform = np.sin(np.arange(800, dtype=np.float32) * 0.05) * 0.2

    def add_audio(self, name, label):
        path = self.audio / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, self.waveform, 16000, subtype="FLOAT")
        self.entries[name] = label
        self.protocol.write_text(
            "".join(f"{utt_id} {'fake' if target == 0 else 'real'}\n"
                    for utt_id, target in self.entries.items()), encoding="utf-8",
        )
        return name

    def add_pair(self, suffix="fake", label=0):
        offline = self.add_audio(f"clean/offline/en/{suffix}_off.wav", label)
        online = self.add_audio(f"clean/online/en/{suffix}_on.wav", label)
        return offline, online

    def write_csv(self, rows, headers=("offline_path", "online_path")):
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)

    def prepare(self, **kwargs):
        return prepare_pairs(self.csv_path, self.protocol, self.audio, self.manifest, **kwargs)

    def test_explicit_pairs_one_to_many_and_coverage(self):
        offline, online = self.add_pair()
        second_online = self.add_audio("clean/online/zh/fake_second_on.wav", 0)
        real = self.add_pair("real", 1)
        self.add_audio("clean/offline/en/unpaired.wav", 1)
        # Directory prefixes are removable, while a unique complete basename
        # is resolvable without guessing any utterance identifier.
        self.write_csv([
            ("/download/RTCFake/train/" + offline, online),
            (offline, Path(second_online).name),
            real,
        ])
        pairs, metadata = self.prepare()
        self.assertEqual(len(pairs), 3)
        self.assertEqual(metadata["pairs_by_label"], {"fake_0": 2, "real_1": 1})
        self.assertEqual(metadata["covered_utterances"], 5)
        self.assertEqual(metadata["unpaired_utterances"], 1)
        self.assertEqual(pairs, load_pairs(self.manifest, self.protocol, self.audio))
        self.assertTrue(Path(str(self.manifest) + ".meta.json").is_file())

    def test_different_labels_rejected(self):
        offline = self.add_audio("offline/x.wav", 0)
        online = self.add_audio("online/y.wav", 1)
        self.write_csv([(offline, online)])
        with self.assertRaisesRegex(ValueError, "different truth labels"):
            self.prepare()

    def test_official_id_headers_and_missing_transmissions(self):
        pair = self.add_pair()
        orphan_offline = self.add_audio("offline/en/failed_transmission.wav", 1)
        orphan_online = self.add_audio("online/en/received_only.wav", 0)
        before = self.protocol.read_bytes()
        self.write_csv([pair, (orphan_offline, ""), ("  ", orphan_online), ("", "")],
                       headers=("offline_id", "online_id"))
        pairs, metadata = self.prepare()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(metadata["csv_rows"], 4)
        self.assertEqual(metadata["skipped_unpaired_rows"], 3)
        self.assertEqual(metadata["skipped_unpaired_by_reason"],
                         {"missing_online": 1, "missing_offline": 1, "both_empty": 1})
        self.assertEqual(metadata["unpaired_utterances"], 2)
        self.assertEqual(self.protocol.read_bytes(), before)
        self.assertEqual(load_pairs(self.manifest, self.protocol, self.audio), pairs)

    def test_missing_counterpart_does_not_hide_invalid_present_path(self):
        pair = self.add_pair()
        self.write_csv([pair, ("offline/en/unknown.wav", "")])
        with self.assertRaisesRegex(ValueError, "Cannot uniquely map"):
            self.prepare()

    def test_truncated_csv_row_is_not_a_missing_transmission(self):
        pair = self.add_pair()
        self.write_csv([pair, (pair[0],)])
        with self.assertRaisesRegex(ValueError, "Missing CSV field"):
            self.prepare()

    def test_ambiguous_basename_rejected(self):
        offline, online = self.add_pair()
        self.add_audio("clean/online/zh/fake_on.wav", 0)
        self.write_csv([(offline, "fake_on.wav")])
        with self.assertRaisesRegex(ValueError, "Cannot uniquely map"):
            self.prepare()
        # An explicit relative path disambiguates the official mapping.
        self.write_csv([(offline, online)])
        self.assertEqual(self.prepare()[0][0]["online"], online)

    def test_no_number_or_extension_guessing(self):
        offline, online = self.add_pair()
        self.write_csv([(offline, Path(online).stem)])
        with self.assertRaisesRegex(ValueError, "Cannot uniquely map"):
            self.prepare()

    def test_nontraining_split_cannot_be_stripped(self):
        offline, online = self.add_pair()
        for other in ("dev", "progress", "eval"):
            with self.subTest(split=other):
                self.write_csv([(f"/download/{other}/" + offline, online)])
                with self.assertRaisesRegex(ValueError, "non-training split"):
                    self.prepare()

    def test_nonmatching_full_path_never_falls_back_to_basename(self):
        offline, online = self.add_pair()
        self.write_csv([(offline, "unknown/location/fake_on.wav")])
        with self.assertRaisesRegex(ValueError, "Cannot uniquely map"):
            self.prepare()

    def test_bare_basename_uses_column_direction(self):
        offline = self.add_audio("clean/offline/en/shared.wav", 0)
        online = self.add_audio("clean/online/en/shared.wav", 0)
        self.write_csv([("shared.wav", "shared.wav")])
        pairs, _ = self.prepare()
        self.assertEqual(pairs[0], {"offline": offline, "online": online, "label": 0})

    def test_relative_path_can_supply_missing_clean_prefix(self):
        offline, online = self.add_pair()
        self.write_csv([(offline.removeprefix("clean/"), online.removeprefix("clean/"))])
        pairs, _ = self.prepare()
        self.assertEqual(pairs[0]["offline"], offline)

    def test_path_traversal_rejected_even_if_basename_exists(self):
        offline, online = self.add_pair()
        self.write_csv([(offline, "../" + online)])
        with self.assertRaisesRegex(ValueError, "traversal"):
            self.prepare()

    def test_symlink_outside_training_root_rejected(self):
        offline, online = self.add_pair()
        outside = self.root / "outside.wav"
        sf.write(outside, self.waveform, 16000)
        (self.audio / online).unlink()
        (self.audio / online).symlink_to(outside)
        self.write_csv([(offline, online)])
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.prepare()

    def test_duplicate_pair_rejected(self):
        pair = self.add_pair()
        self.write_csv([pair, pair])
        with self.assertRaisesRegex(ValueError, "Duplicate pair"):
            self.prepare()

    def test_online_multiple_sources_rejected(self):
        offline, online = self.add_pair()
        second_offline = self.add_audio("offline/second.wav", 0)
        self.write_csv([(offline, online), (second_offline, online)])
        with self.assertRaisesRegex(ValueError, "multiple Offline"):
            self.prepare()

    def test_unknown_columns_need_explicit_choice(self):
        pair = self.add_pair()
        self.write_csv([pair], headers=("original", "transmitted"))
        with self.assertRaisesRegex(ValueError, "Available headers.*original.*transmitted"):
            self.prepare()
        self.assertEqual(len(self.prepare(offline_column="original", online_column="transmitted")[0]), 1)

    def test_ambiguous_column_aliases_rejected(self):
        offline, online = self.add_pair()
        self.write_csv([(offline, offline, online)], headers=("offline", "offline_path", "online"))
        with self.assertRaisesRegex(ValueError, "Cannot uniquely determine the offline column"):
            self.prepare()

    def test_direction_and_missing_audio_rejected(self):
        offline, online = self.add_pair()
        self.write_csv([(online, offline)])
        with self.assertRaisesRegex(ValueError, "Expected an offline"):
            self.prepare()
        self.write_csv([(offline, online)])
        (self.audio / online).unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()

    def test_loader_rechecks_labels_membership_and_paths(self):
        offline, online = self.add_pair()
        for record, pattern in (
            ({"offline": offline, "online": online, "label": 1}, "disagrees"),
            ({"offline": offline, "online": online, "label": "0"}, "disagrees"),
            ({"offline": offline, "online": "online/not_in_train.wav", "label": 0}, "not in"),
            ({"offline": offline, "online": "../" + online, "label": 0}, "traversal"),
        ):
            with self.subTest(record=record):
                self.manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, pattern):
                    load_pairs(self.manifest, self.protocol, self.audio)

    def test_loader_detects_modified_manifest_and_protocol(self):
        pair = self.add_pair()
        self.write_csv([pair])
        self.prepare()
        with self.manifest.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            load_pairs(self.manifest, self.protocol, self.audio)
        self.prepare()
        with self.protocol.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "train_protocol_sha256"):
            load_pairs(self.manifest, self.protocol, self.audio)

    def test_loader_detects_missing_files_after_preparation(self):
        pair = self.add_pair()
        self.write_csv([pair])
        self.prepare()
        (self.audio / pair[1]).unlink()
        with self.assertRaises(FileNotFoundError):
            load_pairs(self.manifest, self.protocol, self.audio)

    def test_pair_audio_repeat_crop_and_no_environment_augmentation(self):
        pair = self.add_pair()
        self.write_csv([pair])
        pairs, _ = self.prepare()
        with patch.dict(os.environ, {"RTC_B_NOISE_MANIFEST": "/missing/should_not_be_read.jsonl",
                                     "RTC_B_NOISE_PROB": "1.0"}):
            dataset = RTCPairDataset(pairs, self.audio, cut=2000)
            offline, online, label = dataset[0]
            expected = torch.from_numpy(np.tile(self.waveform, 3)[:2000])
            torch.testing.assert_close(offline, expected, rtol=0, atol=1e-7)
            torch.testing.assert_close(online, expected, rtol=0, atol=1e-7)
            self.assertEqual(label, 0)
            self.assertEqual(offline.dtype, torch.float32)
            cropped = RTCPairDataset(pairs, self.audio, cut=200)[0][0]
            torch.testing.assert_close(cropped, torch.from_numpy(self.waveform[:200]))


class BalancedSamplerTests(unittest.TestCase):
    def test_balanced_deterministic_epochs_and_no_batch_duplicates(self):
        pairs = [{"label": label} for label in (0, 1) for _ in range(5)]
        sampler = BalancedPairBatchSampler(pairs, pairs_per_batch=8, steps_per_epoch=9, seed=9)
        first = list(sampler)
        self.assertEqual(len(first), 9)
        self.assertEqual(first, list(sampler))
        for batch in first:
            self.assertEqual(len(batch), 8)
            self.assertEqual(len(set(batch)), 8)
            self.assertEqual(sum(pairs[index]["label"] for index in batch), 4)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        sampler.set_epoch(0)
        self.assertEqual(first, list(sampler))

    def test_small_class_cycles_without_losing_balance(self):
        pairs = [{"label": 0}, {"label": 1}, {"label": 1}]
        sampler = BalancedPairBatchSampler(pairs, pairs_per_batch=8, steps_per_epoch=3)
        for batch in sampler:
            self.assertEqual(sum(pairs[index]["label"] for index in batch), 4)

    def test_requires_both_classes_and_even_pair_count(self):
        with self.assertRaisesRegex(ValueError, "both real and fake"):
            BalancedPairBatchSampler([{"label": 0}])
        with self.assertRaisesRegex(ValueError, "even"):
            BalancedPairBatchSampler([{"label": 0}, {"label": 1}], pairs_per_batch=3)


if __name__ == "__main__":
    unittest.main()
