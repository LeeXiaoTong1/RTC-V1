import copy
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from rtc_noisy_v2.cache import check_metadata, check_suite, inspect_initial_history
from rtc_noisy_v2.plan import (ALL_SETTINGS, TRAIN_SETTINGS, HELDOUT_SETTINGS, PLAN_ID, SCHEMA,
                                group_coefficients, plan_definition, settings_for)
from rtc_noisy_v2.sampling import RotatingViewBatchSampler


class SourceSchedule:
    """Predetermined, deliberately uneven visits; source selection is not under test."""
    def __init__(self):
        self.epoch = 0
    def __len__(self):
        return 3
    def set_epoch(self, epoch):
        self.epoch = epoch
    def __iter__(self):
        yield [0, 2]
        yield [1, 3]
        yield [self.epoch % 2, 2 + (self.epoch % 2)]


SOURCES = [{"offline": f"offline/source_{i}.wav", "label": i//2} for i in range(4)]


def make_sampler(banks=1):
    return RotatingViewBatchSampler(SourceSchedule(), SOURCES, 1729, banks)


class LogicTests(unittest.TestCase):
    def test_rtc_split_disjoint_and_marginally_balanced(self):
        self.assertEqual(len(TRAIN_SETTINGS), 21)
        self.assertEqual(len(HELDOUT_SETTINGS), 6)
        self.assertFalse(set(TRAIN_SETTINGS) & set(HELDOUT_SETTINGS))
        self.assertEqual(set(TRAIN_SETTINGS) | set(HELDOUT_SETTINGS), set(ALL_SETTINGS))
        for column in range(3):
            self.assertEqual(set(Counter(x[column] for x in TRAIN_SETTINGS).values()), {7})
            self.assertEqual(set(Counter(x[column] for x in HELDOUT_SETTINGS).values()), {2})

    def test_view_cycles_cross_epoch_boundaries_and_cache_banks(self):
        sampler, collected = make_sampler(2), defaultdict(list)
        for epoch in range(1, 13):
            sampler.set_epoch(epoch)
            for batch in sampler:
                for source, bank, band in batch:
                    collected[source].append((bank, band))
            sampler.commit_epoch(len(sampler))
        for views in collected.values():
            for start in range(0, len(views)-3, 4):
                block = views[start:start+4]
                self.assertEqual({band for bank, band in block}, {0, 1, 2, 3})
                self.assertEqual({bank for bank, band in block}, {(start//4) % 2})

    def test_prefetch_does_not_commit_visits_and_plans_repeat(self):
        sampler = make_sampler()
        sampler.set_epoch(1)
        first = list(sampler)
        self.assertEqual(list(sampler), first)
        self.assertEqual(sampler.state_dict()["visits"], [0]*4)
        with self.assertRaises(ValueError):
            sampler.commit_epoch(1)
        sampler.discard_epoch()
        sampler.set_epoch(1)
        self.assertEqual(list(sampler), first)
        sampler.commit_epoch(3)
        self.assertEqual(sum(sampler.visits), 6)

    def test_rotation_state_restores_next_plan_exactly(self):
        original = make_sampler()
        for epoch in (1, 2):
            original.set_epoch(epoch)
            list(original)
            original.commit_epoch(3)
        restored = make_sampler()
        restored.load_state_dict(json.loads(json.dumps(original.state_dict())))
        original.set_epoch(3)
        restored.set_epoch(3)
        self.assertEqual(list(original), list(restored))
        invalid = make_sampler(2)
        with self.assertRaises(ValueError):
            invalid.load_state_dict(original.state_dict())

    def test_group_coefficients_have_explicit_noisy_share(self):
        coeff = group_coefficients(24, 8, 4, .3)
        for a, b in zip(coeff, (7/15, 7/45, 7/90, .3)):
            self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(sum(coeff), 1.)
        self.assertEqual(group_coefficients(16, 8, 8, .3)[-1], .3)
        with self.assertRaises(ValueError):
            group_coefficients(24, 8, 4, float("nan"))

    def test_cache_cannot_relabel_legacy_or_forbidden_settings(self):
        role = "train"
        config = {"optimization_schema": SCHEMA, "role": role, "plan_id": PLAN_ID,
                  "plan":plan_definition(), "allowed_settings":[list(x) for x in settings_for(role)],
                  "generation":0}
        row = {"role":role,"generation":0,"mix_id":"a"*64,
               "rtc":dict(zip(("noise_reduction","max_gain","bitrate"), TRAIN_SETTINGS[0]))}
        check_metadata(config,[row],role)
        bad = copy.deepcopy(row)
        bad["rtc"] = dict(zip(("noise_reduction","max_gain","bitrate"), HELDOUT_SETTINGS[0]))
        with self.assertRaises(ValueError):
            check_metadata(config,[bad],role)
        with self.assertRaises(ValueError):
            check_metadata({"format":"rtc_noisy_pair_cache_v1"},[row],role)

    def test_dev_matches_noise_and_rejects_recording_leakage(self):
        row = {"source":"offline/a.wav","band":0,"source_sha256":"source","label":0,"mix_id":"mix"}
        tc = {"ffmpeg_version":"6.1", "seed":1234,"generation":0,
              "noise":{"recording_ids":["train"],"file_sha256":["train_hash"]}}
        dc = {"ffmpeg_version":"6.1", "noise":{"recording_ids":["dev"],"file_sha256":["dev_hash"]}}
        check_suite([([row],tc)],([row],dc),([row],dc))
        changed = dict(row, mix_id="different noise")
        with self.assertRaises(ValueError):
            check_suite([([row],tc)],([row],dc),([changed],dc))
        dc2 = copy.deepcopy(dc)
        dc2["noise"]["file_sha256"] = ["train_hash"]
        with self.assertRaises(ValueError):
            check_suite([([row],tc)],([row],dc2),([row],dc2))

    def test_initialization_exposure_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/"ckpt").mkdir()
            ckpt=root/"ckpt"/"best_model.pth"
            self.assertEqual(inspect_initial_history(ckpt)["status"],"unknown")
            (root/"config.json").write_text(json.dumps({"train_cache_config":{}}))
            self.assertEqual(inspect_initial_history(ckpt)["status"],"previous_simulation")
            (root/"config.json").write_text(json.dumps({"rtc_pairs":"pairs","train_protocol_sha256":"h"}))
            self.assertEqual(inspect_initial_history(ckpt)["status"],"rtc_pair_before_simulation")


if __name__ == "__main__":
    unittest.main()
