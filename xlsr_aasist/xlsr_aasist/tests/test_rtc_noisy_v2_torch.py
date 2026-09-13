"""CPU checks against the real PyTorch loss and DataLoader interfaces.

Run in the existing sdd project environment; no XLS-R checkpoint/GPU needed.
"""
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rtc_noisy_v2.losses import grouped_classification
from rtc_noisy_v2.cache import RotatingNoisyDataset
from rtc_noisy_v2.sampling import RotatingViewBatchSampler
from rtc_noisy_v2.training import (combine_validation, error_rates, paired_objective,
                                    selection_key, train_epoch)
from utils.rtc_data import BalancedPairBatchSampler


class TicketDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, ticket):
        return torch.tensor(ticket, dtype=torch.int64)


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 6)
        self.classifier = nn.Linear(6, 2)
        self.calls = []

    def forward(self, waveforms, return_features=False):
        self.calls.append(len(waveforms))
        features = self.encoder(waveforms).tanh()
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


def batches():
    generator = torch.Generator().manual_seed(21)
    wave = lambda count: torch.randn(count, 8, generator=generator)
    pair_labels = torch.tensor([0, 0, 1, 1])
    ordinary = (wave(24), torch.tensor([0]*20 + [1]*4), ["unused"]*24)
    real = (wave(4), wave(4), pair_labels)
    noisy = (wave(4), wave(4), pair_labels, torch.tensor([0, 1, 2, 3]), torch.zeros(4, dtype=torch.long))
    return ordinary, real, noisy


def arguments():
    return SimpleNamespace(amp="none", noisy_ce_weight=.3, rtc_weight=.1,
                           noisy_weight=.1, rtc_temperature=.1, cache_banks=1,
                           grad_clip=0., noisy_warmup_epochs=2.)


class TorchTests(unittest.TestCase):
    def test_audio_dataset_reads_exactly_the_ticketed_source_and_view(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "offline").mkdir()
            banks = [[], []]
            for source_index in range(2):
                source = f"offline/{source_index}.wav"
                sf.write(root/source, np.full(100, .01*(source_index+1), np.float32), 16000, subtype="FLOAT")
                for bank in range(2):
                    for band in range(4):
                        path = root / f"view_{source_index}_{bank}_{band}.wav"
                        value = .1*bank + .01*band + .001*source_index
                        sf.write(path, np.full(64600, value, np.float32), 16000, subtype="FLOAT")
                        banks[bank].append({"source":source,"label":source_index,"band":band,"audio":str(path)})
            dataset = RotatingNoisyDataset(banks, root)
            for ticket in ((0, 0, 2), (1, 1, 3)):
                source, bank, band = ticket
                original, processed, label, returned_band, returned_bank = dataset[ticket]
                self.assertEqual(original.shape, (64600,))
                self.assertEqual(processed.shape, (64600,))
                self.assertTrue(torch.allclose(original, torch.full_like(original, .01*(source+1))))
                self.assertTrue(torch.allclose(processed, torch.full_like(processed, .1*bank+.01*band+.001*source)))
                self.assertEqual((label, returned_bank, returned_band), (source, bank, band))

    def test_class_gradient_ratio_is_nine_only_for_ordinary(self):
        # 2 ordinary + 2 real pairs x2 + 2 noisy pairs x2 = 10 waveforms.
        logits = torch.zeros(10, 2, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        loss, parts, coefficients = grouped_classification(logits, labels, 2, 2, 2, .3)
        loss.backward()
        sizes = logits.grad.abs().sum(1)
        self.assertAlmostEqual(float(sizes[1]/sizes[0]), 9., places=5)
        for a, b in ((2, 3), (4, 5), (6, 7), (8, 9)):
            self.assertAlmostEqual(float(sizes[a]/sizes[b]), 1., places=6)
        self.assertAlmostEqual(coefficients["noisy_processed"], .3)
        self.assertEqual(len(parts), 4)

    def test_noisy_classification_has_fixed_share_and_can_be_disabled(self):
        for n, r, s in ((2, 2, 2), (24, 4, 4), (24, 4, 8)):
            labels = torch.arange(n+2*r+2*s) % 2
            logits = torch.zeros(len(labels), 2, requires_grad=True)
            value, _, _ = grouped_classification(logits, labels, n, r, s, .3)
            value.backward()
            # At zero logits, each row's L1 gradient equals its CE coefficient.
            self.assertAlmostEqual(float(logits.grad[-s:].abs().sum()), .3, places=6)
            self.assertAlmostEqual(float(logits.grad.abs().sum()), 1., places=6)
            zero_logits = torch.zeros(len(labels), 2, requires_grad=True)
            value, _, _ = grouped_classification(zero_logits, labels, n, r, s, 0.)
            value.backward()
            self.assertEqual(float(zero_logits.grad[-s:].abs().sum()), 0.)

    def test_objective_is_one_forward_and_both_detector_parts_get_gradients(self):
        torch.manual_seed(11)
        model = TinyDetector()
        loss, stats = paired_objective(model, *batches(), torch.device("cpu"), arguments(), .1)
        loss.backward()
        self.assertEqual(model.calls, [40])
        self.assertEqual(stats["examples"], 40)
        self.assertEqual(stats["noisy_pairs"], 4)
        self.assertEqual(stats["snr_counts"], [1, 1, 1, 1])
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_train_epoch_updates_and_plain_checkpoint_stays_loadable(self):
        torch.manual_seed(12)
        model = TinyDetector()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = batches()
        loaders = tuple([part, part] for part in batch)
        stats = train_epoch(loaders, model, optimizer, torch.device("cpu"), arguments(), 1)
        self.assertEqual(model.calls, [40, 40])
        self.assertEqual(stats["steps"], 2)
        self.assertEqual(stats["snr_counts"], [2, 2, 2, 2])
        self.assertEqual(sum(map(sum, stats["noisy_confusion"])), 8)
        self.assertTrue(any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = TinyDetector()
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        for name, value in restored.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))

    def test_persistent_workers_receive_identical_rotation_tickets(self):
        sources = [{"offline": f"offline/{i}.wav", "label": i % 2} for i in range(12)]
        def sampler():
            base = BalancedPairBatchSampler(sources, pairs_per_batch=4, steps_per_epoch=7, seed=31)
            return RotatingViewBatchSampler(base, sources, seed=32)
        synchronous, asynchronous = sampler(), sampler()
        loader = DataLoader(TicketDataset(), batch_sampler=asynchronous, num_workers=2,
                            persistent_workers=True, prefetch_factor=3, multiprocessing_context="spawn")
        for epoch in range(1, 5):
            synchronous.set_epoch(epoch)
            asynchronous.set_epoch(epoch)
            expected = [[list(ticket) for ticket in batch] for batch in synchronous]
            received = [batch.tolist() for batch in loader]
            self.assertEqual(received, expected)
            # Prefetch must not advance committed history ahead of the trainer.
            self.assertEqual(asynchronous.completed_epoch, epoch-1)
            for current in (synchronous, asynchronous):
                current.commit_epoch(len(current))
            self.assertEqual(synchronous.state_dict(), asynchronous.state_dict())
        del loader

    def test_dev_errors_and_selection_use_the_documented_groups(self):
        rates = error_rates([[8, 2], [3, 27]])
        self.assertAlmostEqual(rates["fake_as_real_rate"], .2)
        self.assertAlmostEqual(rates["real_as_fake_rate"], .1)
        clean = {"online": {"macro_f1": .9}, "loss": .2}
        metrics = combine_validation(clean, {"macro_f1": .8}, {"macro_f1": .6})
        self.assertAlmostEqual(metrics["proxy_v1"], .83)
        self.assertAlmostEqual(metrics["robust_proxy"], .76)
        self.assertEqual(selection_key(metrics), (metrics["robust_proxy"], .9, -.2))


if __name__ == "__main__":
    unittest.main()
