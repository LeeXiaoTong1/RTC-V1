"""Synthetic audio -> cache manifests -> balanced loaders -> exact feature reuse.

Uses actual FFmpeg and the official feature extractor. WebRTC is a test double;
run the server smoke script to check the native WebRTC binding.
"""
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from transformers import SeamlessM4TFeatureExtractor
import prepare_rtc_noisy_v2 as preparation
from rtc_noisy.common import sha256
from rtc_noisy import diverse
from utils.rtc_pairs import prepare_pairs
from .train import parser
from .data import DataBundle, combine_batches


@unittest.skipUnless(os.environ.get('RTC_TEST_FFMPEG'), 'Set RTC_TEST_FFMPEG to run audio integration')
class AudioIntegration(unittest.TestCase):
    def test_generation_loading_balancing_and_cached_feature_equivalence(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = []
            for split in ('train', 'dev'):
                lines = []
                for i in range(4):
                    wave = (.1*np.sin(np.arange(66001)*(0.03+i*.005))).astype(np.float32)
                    for domain in ('offline', 'online'):
                        name = f'{domain}/source_{i}.wav'
                        path = root/'wav'/split/name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(path, wave, 16000, subtype='FLOAT')
                        lines.append(f'{name} {"fake" if i < 2 else "real"}\n')
                    if split == 'train': pairs.append((f'offline/source_{i}.wav', f'online/source_{i}.wav'))
                (root/f'{split}_label.txt').write_text(''.join(lines), encoding='utf-8')
                noise = root/f'{split}_noise.wav'
                sf.write(noise, np.random.RandomState(10 if split == 'train' else 11).normal(size=100000)*.03,
                         16000, subtype='FLOAT')
                (root/f'{split}_noise.jsonl').write_text(json.dumps({'path': str(noise), 'split': split,
                    'original_recording': split, 'sha256': sha256(noise)}), encoding='utf-8')
            with (root/'pairs.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream); writer.writerow(['offline_path', 'online_path']); writer.writerows(pairs)
            prepare_pairs(root/'pairs.csv', root/'train_label.txt', root/'wav'/'train', root/'pairs.jsonl')
            original_version = importlib.metadata.version
            def version(name):
                return '0.1.3' if name == 'webrtc-audio-processing' else original_version(name)
            with patch.object(diverse.importlib.metadata, 'version', side_effect=version), \
                 patch.object(diverse, 'webrtc_process', side_effect=lambda x, *a: x.copy()):
                for name, role, profile, generation in [('old', 'train', 'legacy', 0), ('new', 'train', 'diverse', 1),
                                                       ('seen', 'dev_seen', 'diverse', 0), ('held', 'dev_heldout', 'unseen', 0)]:
                    args = SimpleNamespace(dataset_root=str(root), role=role, processing_profile=profile,
                        noise_manifest=str(root/('train_noise.jsonl' if role == 'train' else 'dev_noise.jsonl')),
                        output=str(root/name), ffmpeg=os.environ['RTC_TEST_FFMPEG'], generation=generation,
                        seed=19, limit=0, workers=2)
                    preparation.prepare(args)
            SeamlessM4TFeatureExtractor().save_pretrained(root/'model')
            (root/'model'/'config.json').write_text('{}', encoding='utf-8')
            argv = ['--stage','3','--device','cpu','--amp','none','--num_workers','0','--algo','0',
                    '--ordinary_sampling','balanced','--noisy_bank_policy','mixed']
            for name, value in {'train_data_path':root/'wav'/'train', 'dev_data_path':root/'wav'/'dev',
                               'train_protocol':root/'train_label.txt', 'dev_protocol':root/'dev_label.txt',
                               'rtc_pairs':root/'pairs.jsonl','train_noise_manifest':root/'train_noise.jsonl',
                               'train_noisy_cache':root/'old','extra_train_noisy_cache':root/'new',
                               'dev_noisy_cache':root/'seen','dev_heldout_cache':root/'held',
                               'ssl_path':root/'model','out':root/'run','feature_cache':root/'features'}.items():
                argv += ['--'+name, str(value)]
            args = parser().parse_args(argv)
            with patch.dict(os.environ, {'RTC_B_NOISE_PROB':'0.5','RTC_B_SNR_MIN':'10','RTC_B_SNR_MAX':'30'}):
                bundle = DataBundle(args)
                bundle.begin(1)
                batches = [next(iter(loader)) for loader in bundle.train]
                merged, layout = combine_batches(batches)
                self.assertEqual(layout, (24, 4, 4))
                self.assertEqual(batches[0]['labels'].bincount().tolist(), [12, 12])
                self.assertEqual(bundle.weights.tolist(), [1., 1.])
                self.assertTrue(torch.isfinite(merged['features']).all())
                self.assertTrue(merged['mask'].all())
                dev = next(iter(bundle.dev['seen']))
                cached_files = list((root/'features').rglob('*.npy'))
                self.assertTrue(cached_files)
                bundle.end(bundle.steps)
                args.feature_cache = None
                uncached = DataBundle(args)
                expected = next(iter(uncached.dev['seen']))
                torch.testing.assert_close(dev['features'], expected['features'], rtol=0, atol=0)
                self.assertEqual(bundle.fingerprints, uncached.fingerprints)


if __name__ == '__main__': unittest.main(verbosity=2)
