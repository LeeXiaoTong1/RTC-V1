"""CPU integration tests, without fairseq or real XLS-R checkpoint downloads.

The compatibility test reads the original model from the pinned V1 commit. That
part is skipped when running an exported source archive without Git history.
"""

import copy
import csv
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


PROJECT = Path(__file__).resolve().parents[1]
V1_COMMIT = "a447ce13fccea5ad688e583a881b2ff02ecdd952"
sys.path.insert(0, str(PROJECT))

from utils.rtc_loss import rtc_pair_contrastive_loss
from utils.rtc_pairs import prepare_pairs
import main_train_rtc as trainer


class StubSSL(nn.Module):
    """Preserve XLS-R's interface, with a small differentiable 12-frame output."""

    def __init__(self, device, cp_path=None):
        super().__init__()
        self.projection = nn.Linear(1, 1024)
        self.out_dim = 1024

    def extract_feat(self, audio):
        pooled = F.adaptive_avg_pool1d(audio.unsqueeze(1), 12).transpose(1, 2)
        return self.projection(pooled)


def load_model_module(source, name):
    module = types.ModuleType(name)
    with mock.patch.dict(sys.modules, {"fairseq": types.ModuleType("fairseq")}):
        exec(compile(source, str(PROJECT / "model/model.py"), "exec"), module.__dict__)
    module.SSLModel = StubSSL
    return module


class ModelCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.current_module = load_model_module(
            (PROJECT / "model/model.py").read_text(), "rtc_current_model_test"
        )

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(91)
        self.device = torch.device("cpu")
        self.args = types.SimpleNamespace(ssl_path=None)
        self.model = self.current_module.Model(self.args, self.device)
        self.audio = torch.randn(4, 72)

    def test_original_checkpoint_loads_strictly_and_eval_logits_are_identical(self):
        try:
            result = subprocess.run(
                ["git", "show", f"{V1_COMMIT}:xlsr_aasist/model/model.py"],
                cwd=PROJECT.parent, check=True, capture_output=True, text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            self.skipTest("Pinned original V1 model is unavailable in this source archive")
        original_module = load_model_module(result.stdout, "rtc_original_model_test")
        original = original_module.Model(self.args, self.device)
        original_state = original.state_dict()
        current_state = self.model.state_dict()
        self.assertEqual(list(original_state), list(current_state))
        for key, tensor in original_state.items():
            self.assertEqual(tensor.shape, current_state[key].shape, key)
        report = self.model.load_state_dict(original_state, strict=True)
        self.assertEqual(report.missing_keys, [])
        self.assertEqual(report.unexpected_keys, [])
        original.eval()
        self.model.eval()
        with torch.no_grad():
            expected = original(self.audio)
            actual = self.model(self.audio)
            feature_logits, features = self.model(self.audio, return_features=True)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected, feature_logits))
        self.assertEqual(features.shape, (4, 160))

    def test_returned_features_are_pre_dropout_and_survive_inplace_dropout(self):
        self.model.train()
        self.model.drop.p = 1.0
        before_dropout, classifier_inputs = [], []
        hooks = [
            self.model.drop.register_forward_pre_hook(
                lambda _module, args: before_dropout.append(args[0].detach().clone())
            ),
            self.model.out_layer.register_forward_pre_hook(
                lambda _module, args: classifier_inputs.append(args[0].detach().clone())
            ),
        ]
        try:
            logits, features = self.model(self.audio, return_features=True)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(features.shape, (4, 160))
        self.assertTrue(torch.equal(features.detach(), before_dropout[0]))
        self.assertGreater(features.detach().abs().sum().item(), 0)
        self.assertEqual(classifier_inputs[0].abs().sum().item(), 0)
        self.assertTrue(torch.equal(logits, self.model.out_layer.bias.expand(4, -1)))

    def test_ce_and_rtc_backpropagate_through_shared_ssl_and_aasist(self):
        self.model.train()
        logits, features = self.model(self.audio, return_features=True)
        pair_labels = torch.tensor([0, 1])
        rtc, stats = rtc_pair_contrastive_loss(features[:2], features[2:], pair_labels)
        ce = F.cross_entropy(logits, torch.cat([pair_labels, pair_labels]))
        (ce + 0.1 * rtc).backward()
        self.assertEqual(stats["valid_anchors"], 4)
        for parameter in (
            self.model.ssl_model.projection.weight,
            self.model.LL.weight,
            self.model.GAT_layer_S.att_proj.weight,
            self.model.out_layer.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


class OrdinaryWaveDataset(Dataset):
    def __init__(self, waves, labels, ids):
        self.waves, self.labels, self.ids = waves, labels, ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.waves[index], self.labels[index], self.ids[index]


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(24, 160)
        self.classifier = nn.Linear(160, 2)
        self.seen_inputs = []

    def forward(self, audio, return_features=False):
        self.seen_inputs.append(audio.detach().clone())
        features = self.encoder(audio).tanh()
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


class RTCTrainerTests(unittest.TestCase):
    def setUp(self):
        self.progress_patch = mock.patch.object(trainer, "tqdm", lambda iterable, **kwargs: iterable)
        self.progress_patch.start()
        self.addCleanup(self.progress_patch.stop)

    def test_one_forward_per_step_and_weight_zero_keeps_identical_sample_stream(self):
        torch.manual_seed(43)
        waves = torch.randn(4, 24)
        offline = torch.randn(4, 24)
        online = offline + 0.2 * torch.randn(4, 24)
        labels = torch.tensor([0, 1, 0, 1])
        ordinary = DataLoader(
            OrdinaryWaveDataset(waves, labels, [f"ordinary_{i}.wav" for i in range(4)]),
            batch_size=2, shuffle=False,
        )
        paired = DataLoader(TensorDataset(offline, online, labels), batch_size=2, shuffle=False)
        control = TinyDetector()
        treatment = copy.deepcopy(control)
        initial_keys = list(control.state_dict())
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]))
        records = []
        for model, weight in ((control, 0.0), (treatment, 0.2)):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            records.append(trainer.train_epoch(
                ordinary, paired, model, optimizer, torch.device("cpu"), criterion,
                rtc_weight=weight, temperature=0.1, amp=False,
            ))
            self.assertEqual(list(model.state_dict()), initial_keys)
            self.assertEqual(len(model.seen_inputs), 2)
            for step, actual in enumerate(model.seen_inputs):
                sl = slice(2 * step, 2 * step + 2)
                self.assertTrue(torch.equal(actual, torch.cat([waves[sl], offline[sl], online[sl]])))
            self.assertEqual(records[-1]["steps"], 2)
            self.assertEqual(records[-1]["examples"], 12)
            self.assertEqual(records[-1]["pairs"], 4)
            self.assertEqual(records[-1]["valid_anchors"], 8)
            self.assertGreater(records[-1]["rtc"], 0)
        for first, second in zip(control.seen_inputs, treatment.seen_inputs):
            self.assertTrue(torch.equal(first, second))
        self.assertAlmostEqual(records[0]["loss"], records[0]["ce"], places=6)
        self.assertAlmostEqual(records[1]["loss"], records[1]["ce"] + 0.2 * records[1]["rtc"], places=6)
        self.assertFalse(torch.equal(control.encoder.weight, treatment.encoder.weight))

    def test_online_macro_f1_uses_spoof_score_greater_equal_half_including_ties(self):
        # A tied score is classified as spoof (class 0). There is one tied real
        # utterance, so making the threshold strict would change the confusion.
        logits = torch.tensor([[0., 0.], [0., 0.], [-0.2, 0.2], [0.2, -0.2]])
        labels = torch.tensor([0, 1, 1, 0])
        ids = ["clean/online/en/a.wav", "clean/online/zh/b.wav",
               "clean/online/en/c.wav", "clean/offline/zh/d.wav"]
        loader = DataLoader(OrdinaryWaveDataset(logits, labels, ids), batch_size=2)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]))
        result = trainer.evaluate_dev(loader, nn.Identity(), torch.device("cpu"), criterion)
        self.assertEqual(result["online"]["confusion"], [[1., 0.], [1., 1.]])
        self.assertEqual(result["online"]["count"], 3)
        self.assertAlmostEqual(result["online"]["macro_f1"], 2 / 3)
        self.assertEqual(result["offline"]["count"], 1)
        self.assertAlmostEqual(result["all"]["macro_f1"], (0.8 + 2 / 3) / 2)
        self.assertAlmostEqual(result["loss"], criterion(logits, labels).item(), places=6)

    def test_prepare_shell_preflight_and_full_main_save_compatible_checkpoint(self):
        import numpy as np
        import soundfile as sf

        class WaveDetector(TinyDetector):
            def forward(self, audio, return_features=False):
                pooled = F.adaptive_avg_pool1d(audio.unsqueeze(1), 24).squeeze(1)
                return super().forward(pooled, return_features)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            ids = [f"clean/{side}/en/{label}.wav" for side in ("offline", "online")
                   for label in ("fake", "real")]
            for split in ("train", "dev"):
                for index, utt in enumerate(ids):
                    path = dataset / "wav" / split / utt
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, 0.1 * np.sin(np.arange(800) * (0.05 + index * 0.01)), 16000)
                (dataset / f"{split}_label.txt").write_text(
                    "".join(f"{utt} {Path(utt).stem}\n" for utt in ids))
            csv_path = root / "official_test_pairs.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["offline_path", "online_path"])
                writer.writerows([(ids[0], ids[2]), (ids[1], ids[3])])
            manifest = dataset / "train_rtc_pairs.jsonl"
            prepare_pairs(csv_path, dataset / "train_label.txt", dataset / "wav/train", manifest)
            checkpoint = root / "initial.pth"
            torch.save(WaveDetector().state_dict(), checkpoint)
            env = os.environ.copy()
            env.update(DATA_ROOT=str(dataset / "wav"), DATASET_ROOT=str(dataset),
                       RTC_B_NOISE_PROB="0", NUM_WORKERS="0",
                       PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""))
            process = subprocess.run(
                ["bash", str(PROJECT / "run_train_rtc.sh"), str(checkpoint), str(manifest), "--check_data"],
                env=env, text=True, capture_output=True, timeout=45)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertIn("Data checks passed", process.stdout)
            model_module = types.ModuleType("model.model")
            model_module.Model = lambda _args, _device: WaveDetector()
            argv = ["main_train_rtc.py", "--model_path", str(checkpoint), "--rtc_pairs", str(manifest),
                    "--train_protocol", str(dataset / "train_label.txt"),
                    "--dev_protocol", str(dataset / "dev_label.txt"),
                    "--train_data_path", str(dataset / "wav/train"),
                    "--dev_data_path", str(dataset / "wav/dev"),
                    "--out_path", str(root / "exp"), "--device", "cpu", "--amp", "none",
                    "--num_workers", "0", "--algo", "0", "--batch_size", "4",
                    "--rtc_pairs_per_batch", "2", "--num_epochs", "1"]
            with mock.patch.dict(sys.modules, {"model.model": model_module}), \
                 mock.patch.dict(os.environ, env), mock.patch.object(sys, "argv", argv):
                trainer.main()
            experiment = next((root / "exp").iterdir())
            config = json.loads((experiment / "config.json").read_text())
            self.assertEqual(config["effective_audio_batch"], 8)
            metrics = [json.loads(line) for line in (experiment / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in metrics], [0, 1])
            self.assertEqual(metrics[1]["train"]["valid_anchors"], 4)
            for name in ("best_model.pth", "last_model.pth"):
                state = torch.load(experiment / "ckpt" / name, weights_only=True)
                WaveDetector().load_state_dict(state, strict=True)


if __name__ == "__main__":
    unittest.main()
