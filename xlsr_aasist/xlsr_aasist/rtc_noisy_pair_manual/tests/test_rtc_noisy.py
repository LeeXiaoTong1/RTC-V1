"""Meaningful CPU integration tests; FFmpeg is real, XLS-R is substituted."""
import copy
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import utils.data_utils as original_data
from utils.rtc_pairs import prepare_pairs
from rtc_noisy.common import CUT, SNR_BANDS, assert_noise_disjoint, noise_catalog
from rtc_noisy.data import NoisyPairDataset, load_cache
from rtc_noisy.simulator import LocalRTC, RTCSettings
from rtc_noisy.training import evaluate_noisy, linear_weight, paired_objective, selection_key, validate
import main_train_rtc_noisy
import main_eval_rtc_noisy
import prepare_rtc_noisy


@contextmanager
def use_test_model(module):
    # Patch only this key: restoring the whole sys.modules dictionary could
    # unload newly imported torch modules while retaining their C++ registries.
    missing = object()
    previous = sys.modules.get("model.model", missing)
    sys.modules["model.model"] = module
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop("model.model", None)
        else:
            sys.modules["model.model"] = previous


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(16, 160)
        self.classifier = nn.Linear(160, 2)
        self.calls = 0

    def forward(self, waveforms, return_features=False):
        self.calls += 1
        small = F.adaptive_avg_pool1d(waveforms.unsqueeze(1), 16).squeeze(1)
        features = self.encoder(small).tanh()
        self.last_features = features
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


class StubSSL(nn.Module):
    def __init__(self, device, cp_path=None):
        super().__init__()
        self.projection = nn.Linear(1, 1024)
        self.out_dim = 1024

    def extract_feat(self, waveforms):
        small = F.adaptive_avg_pool1d(waveforms.unsqueeze(1), 12).transpose(1, 2)
        return self.projection(small)


def toy_batches(length=64):
    labels = torch.tensor([0, 1])
    ordinary = (torch.randn(2, length), labels, ["a", "b"])
    real = (torch.randn(2, length), torch.randn(2, length), labels)
    noisy = (torch.randn(2, length), torch.randn(2, length), labels, torch.tensor([0, 3]))
    return ordinary, real, noisy


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_noisy_loss_uses_correct_slices_and_backpropagates_to_both_versions(self):
        torch.manual_seed(7)
        model = TinyDetector()
        batches = toy_batches()
        zero_ce = lambda logits, labels: logits.sum() * 0
        loss, stats = paired_objective(model, *batches, zero_ce, torch.device("cpu"),
                                       rtc_weight=0., noisy_weight=.1, temperature=.1, amp=False)
        grad = torch.autograd.grad(loss, model.last_features, retain_graph=True)[0]
        self.assertEqual(model.calls, 1)
        self.assertEqual(stats["examples"], 10)
        self.assertEqual(stats["valid_noisy_anchors"], 4)
        self.assertEqual(float(grad[:6].abs().sum()), 0.)
        self.assertGreater(float(grad[6:8].abs().sum()), 0.)
        self.assertGreater(float(grad[8:].abs().sum()), 0.)
        loss.backward()
        self.assertGreater(float(model.encoder.weight.grad.abs().sum()), 0.)

    def test_noisy_zero_weight_changes_only_loss_not_inputs_or_classifier(self):
        torch.manual_seed(8)
        model = TinyDetector()
        batches = toy_batches()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([.1, .9]))
        base, first = paired_objective(model, *batches, criterion, torch.device("cpu"), .1, 0., .1, False)
        logits1 = model.last_features.detach().clone()
        full, second = paired_objective(model, *batches, criterion, torch.device("cpu"), .1, .1, .1, False)
        self.assertTrue(torch.equal(logits1, model.last_features.detach()))
        self.assertAlmostEqual(float(full - base), .1 * second["rtc_noisy"], places=6)
        self.assertAlmostEqual(first["rtc_real"], second["rtc_real"], places=6)
        self.assertAlmostEqual(linear_weight(.1, 1, 0, 10, 2), .005)
        self.assertEqual(linear_weight(.1, 2, 9, 10, 2), .1)
        self.assertEqual(linear_weight(.1, 4, 9, 10, 2), .1)

    def test_full_aasist_backend_receives_both_pair_losses_and_preserves_checkpoint_keys(self):
        project = Path(original_data.__file__).resolve().parents[1]
        module = types.ModuleType("test_rtc_noisy_aasist")
        with patch.dict(sys.modules, {"fairseq": types.ModuleType("fairseq")}):
            exec(compile((project / "model/model.py").read_text(), "model/model.py", "exec"), module.__dict__)
        module.SSLModel = StubSSL
        model = module.Model(types.SimpleNamespace(ssl_path=None), torch.device("cpu"))
        keys = list(model.state_dict())
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([.1, .9]))
        loss, stats = paired_objective(model, *toy_batches(), criterion, torch.device("cpu"), .1, .1, .1, False)
        loss.backward()
        for parameter in (model.ssl_model.projection.weight, model.LL.weight,
                          model.GAT_layer_S.att_proj.weight, model.out_layer.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        clone = module.Model(types.SimpleNamespace(ssl_path=None), torch.device("cpu"))
        clone.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(keys, list(clone.state_dict()))
        model.eval()
        clone.eval()
        waveforms = torch.randn(2, 64)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(waveforms), clone(waveforms)))

    def test_validation_averages_four_f1s_and_keeps_fixed_fake_threshold(self):
        scores = torch.tensor([.9, .8, .2, .1, .9, .8, .9, .8, .1, .2, .1, .2, .5, .5, .5, .5])
        logits = torch.stack([torch.logit(scores), torch.zeros_like(scores)], 1)
        labels = torch.tensor([0, 0, 1, 1] * 4)
        bands = torch.arange(4).repeat_interleave(4)
        loader = DataLoader(TensorDataset(logits, labels, bands), batch_size=3)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([.1, .9]))
        result = evaluate_noisy(loader, nn.Identity(), torch.device("cpu"), criterion)
        self.assertAlmostEqual(result["macro_f1"], .5)
        self.assertNotAlmostEqual(result["macro_f1"], result["pooled"]["macro_f1"])
        self.assertEqual(result["bands"]["20_25dB"]["confusion"], [[2., 0.], [2., 0.]])
        clean = [(logits[i], labels[i], f"online/en/{i}.wav") for i in range(4)]
        dev = validate(DataLoader(clean, batch_size=4), loader, nn.Identity(), torch.device("cpu"), criterion)
        self.assertAlmostEqual(dev["proxy"], .65)
        lower = copy.deepcopy(dev)
        lower["proxy"] = .6
        lower["clean"]["loss"] = 0.
        self.assertGreater(selection_key(dev), selection_key(lower))


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required for real processing tests")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.dataset = cls.root / "dataset"
        cls.ids = ["offline/en/f0.wav", "offline/en/r0.wav", "offline/en/f1.wav", "offline/en/r1.wav",
                   "online/en/f0.wav", "online/en/r0.wav"]
        for split in ("train", "dev"):
            for i, utt in enumerate(cls.ids):
                path = cls.dataset / "wav" / split / utt
                path.parent.mkdir(parents=True, exist_ok=True)
                time = np.arange(70321) / 16000
                signal = .1 * np.sin(2 * np.pi * (160 + 31*i + (11 if split == "dev" else 0)) * time)
                sf.write(path, signal, 16000, subtype="FLOAT")
            (cls.dataset / f"{split}_label.txt").write_text("".join(
                f"{utt} {'fake' if Path(utt).stem.startswith('f') else 'real'}\n" for utt in cls.ids))
            noise = cls.root / f"{split}_noise.wav"
            sf.write(noise, np.random.RandomState(len(split)).randn(85000) * .05, 16000, subtype="FLOAT")
            row = {"path": str(noise), "split": split, "original_recording": f"{split}_recording",
                   "sha256": hashlib.sha256(noise.read_bytes()).hexdigest()}
            manifest = cls.root / f"{split}_noise.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            args = types.SimpleNamespace(dataset_root=str(cls.dataset), split=split, noise_manifest=str(manifest),
                                         output=str(cls.root / f"cache_{split}"), workers=2, seed=1234,
                                         ffmpeg="ffmpeg", limit=0)
            prepare_rtc_noisy.prepare(args)
        csv_path = cls.root / "pairs.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["offline_id", "online_id"])
            writer.writerow(["train/offline/en/f0.wav", "train/online/en/f0.wav"])
            writer.writerow(["train/offline/en/r0.wav", "train/online/en/r0.wav"])
            writer.writerow(["train/offline/en/f1.wav", ""])
            writer.writerow(["train/offline/en/r1.wav", ""])
        cls.pairs = cls.root / "pairs.jsonl"
        prepare_pairs(csv_path, cls.dataset / "train_label.txt", cls.dataset / "wav/train", cls.pairs)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.old_threads)

    def test_cache_keeps_unpaired_sources_labels_bands_and_full_processing(self):
        rows, config = load_cache(self.root / "cache_train", "train", self.dataset / "train_label.txt", self.dataset / "wav/train")
        self.assertEqual(len(rows), 16)
        self.assertEqual({row["source"] for row in rows}, set(self.ids[:4]))
        for row in rows:
            self.assertGreater(row["source_samples"], CUT)
            self.assertEqual(row["source_samples"], row["output_samples_before_crop"])
            self.assertEqual(row["label"], int(Path(row["source"]).stem.startswith("r")))
            low, high = SNR_BANDS[row["band"]]
            self.assertTrue(low <= row["snr_db"] <= high)
        data = NoisyPairDataset(rows, self.dataset / "wav/train")
        self.assertEqual(len(data), 4)
        for i in range(4):
            original, noisy, label, _ = data[i]
            self.assertEqual(tuple(original.shape), (CUT,))
            self.assertEqual(tuple(noisy.shape), (CUT,))
            self.assertGreater(float((original-noisy).abs().mean()), 1e-4)
        # Exact rerun reuses the same source/variant mapping.
        before = (self.root / "cache_train/manifest.jsonl").read_bytes()
        args = types.SimpleNamespace(dataset_root=str(self.dataset), split="train", noise_manifest=str(self.root / "train_noise.jsonl"),
                                     output=str(self.root / "cache_train"), workers=2, seed=1234, ffmpeg="ffmpeg", limit=0)
        prepare_rtc_noisy.prepare(args)
        self.assertEqual(before, (self.root / "cache_train/manifest.jsonl").read_bytes())

    def test_noise_leakage_and_incomplete_dev_are_rejected(self):
        _, train = load_cache(self.root / "cache_train", "train", self.dataset / "train_label.txt", self.dataset / "wav/train")
        _, dev = load_cache(self.root / "cache_dev", "dev", self.dataset / "dev_label.txt", self.dataset / "wav/dev")
        assert_noise_disjoint(train, dev)
        leaked = copy.deepcopy(dev)
        leaked["noise"]["recording_ids"] = train["noise"]["recording_ids"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_noise_disjoint(train, leaked)
        leaked = copy.deepcopy(dev)
        leaked["noise"]["file_sha256"] = train["noise"]["file_sha256"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_noise_disjoint(train, leaked)
        with self.assertRaisesRegex(ValueError, "split=train"):
            noise_catalog(self.root / "dev_noise.jsonl", "train")
        manifest = self.root / "cache_dev/manifest.jsonl"
        backup = manifest.read_bytes()
        try:
            manifest.write_bytes(b"\n".join(backup.splitlines()[:-1]) + b"\n")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                load_cache(self.root / "cache_dev", "dev", self.dataset / "dev_label.txt", self.dataset / "wav/dev")
        finally:
            manifest.write_bytes(backup)

    def test_real_processor_resets_state_and_noise_order_is_noncommutative(self):
        rng = np.random.RandomState(8)
        signal = np.sin(np.arange(16037) * .09).astype(np.float32) * .1
        noise = rng.randn(len(signal)).astype(np.float32) * .015
        rtc = LocalRTC()
        first, second = rtc(signal + noise), rtc(signal + noise)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), len(signal))
        self.assertGreater(float(np.mean(np.abs(first - (rtc(signal) + noise)))), 1e-4)

    def test_full_training_save_old_style_inference_and_submission(self):
        checkpoint = self.root / "start.pth"
        initial = TinyDetector().state_dict()
        torch.save(initial, checkpoint)
        module = types.ModuleType("model.model")
        module.Model = lambda _args, _device: TinyDetector()
        argv = ["main_train_rtc_noisy.py", "--model_path", str(checkpoint),
                "--train_data_path", str(self.dataset / "wav/train"), "--dev_data_path", str(self.dataset / "wav/dev"),
                "--train_protocol", str(self.dataset / "train_label.txt"), "--dev_protocol", str(self.dataset / "dev_label.txt"),
                "--rtc_pairs", str(self.pairs), "--train_noisy_cache", str(self.root / "cache_train"),
                "--dev_noisy_cache", str(self.root / "cache_dev"), "--train_noise_manifest", str(self.root / "train_noise.jsonl"),
                "--rtc_pairs_per_batch", "2", "--noisy_pairs_per_batch", "2", "--batch_size", "2",
                "--num_workers", "0", "--num_epochs", "1", "--device", "cpu", "--amp", "none", "--algo", "0",
                "--out_path", str(self.root / "exp")]
        with use_test_model(module), patch.dict(os.environ, {"RTC_B_NOISE_PROB": "0.5"}), patch.object(sys, "argv", argv):
            main_train_rtc_noisy.main()
        experiment = next((self.root / "exp").iterdir())
        config = json.loads((experiment / "config.json").read_text())
        self.assertEqual(config["unpaired_offline_sources"], 2)
        self.assertEqual(config["effective_audio_batch"], 10)
        metrics = [json.loads(line) for line in (experiment / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(metrics[1]["train"]["ordinary_examples"], 6)
        self.assertEqual(metrics[1]["train"]["examples"], 30)
        self.assertEqual(metrics[1]["train"]["valid_noisy_anchors"], 12)
        state = torch.load(experiment / "ckpt/last_model.pth", weights_only=True)
        model = TinyDetector()
        model.load_state_dict(state, strict=True)
        self.assertEqual(list(initial), list(state))
        self.assertTrue(any(not torch.equal(initial[k], state[k]) for k in state))
        protocol = self.root / "progress.txt"
        protocol.write_text("\n".join(self.ids) + "\n")
        scores, archive = self.root / "scores.txt", self.root / "submission.zip"
        argv = ["main_eval_rtc_noisy.py", "--model_path", str(experiment / "ckpt/last_model.pth"),
                "--eval_data_path", str(self.dataset / "wav/dev"), "--protocol_path", str(protocol),
                "--score_path", str(scores), "--submission_zip", str(archive), "--device", "cpu", "--num_workers", "0"]
        with use_test_model(module), patch.object(sys, "argv", argv):
            main_eval_rtc_noisy.main()
        rows = [line.split() for line in scores.read_text().splitlines()]
        self.assertEqual([row[0] for row in rows], self.ids)
        self.assertTrue(all(0 <= float(row[1]) <= 1 for row in rows))
        model.eval()
        with torch.no_grad():
            dataset = original_data.SpoofAudioDataset(self.ids, self.dataset / "wav/dev")
            expected = model(torch.stack([dataset[i][0] for i in range(len(dataset))])).softmax(1)[:, 0]
        self.assertTrue(np.allclose(expected.numpy(), [float(row[1]) for row in rows], atol=1e-8))
        with zipfile.ZipFile(archive) as zipped:
            self.assertEqual(zipped.namelist(), ["scores.txt"])
            self.assertEqual(zipped.read("scores.txt"), scores.read_bytes())


if __name__ == "__main__":
    unittest.main()
