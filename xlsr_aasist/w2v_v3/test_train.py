"""Real tiny w2v-BERT integration: phase transition and half-epoch exact resume."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from w2v_aasist.runtime import atomic_save, sha256
from .data import FullEpochPlan
from .model import HeadConfig
from . import train as training
from .control import Controller
from .test_control import config as control_config, dev

torch.set_num_threads(1)


class TrainingIntegrationTests(unittest.TestCase):
    def test_interrupted_promotion_recovers_last_committed_winners(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            c = Controller(control_config()); c.observe(dev(), 'old')
            names = [*training.WINNERS.values(), 'control_best.pt']
            for name in names:
                atomic_save(run/name, {'schema':training.SCHEMA, 'tag':'old', 'model':{'x':torch.ones(1)}})
            training.backup_winners(run, names)
            # Simulate interruption after three files changed, before control/last commit.
            for name in names[:3]:
                atomic_save(run/name, {'schema':training.SCHEMA, 'tag':'future', 'model':{'x':torch.zeros(1)}})
            training.recover_winners(run, c)
            for name in names:
                self.assertEqual(training._checkpoint_tag(run/name), 'old')
                self.assertFalse((run/(name+'.previous')).exists())
            # If last committed first, recovery retains new files and discards backups.
            training.backup_winners(run, names)
            for name in names:
                atomic_save(run/name, {'schema':training.SCHEMA, 'tag':'new', 'model':{'x':torch.zeros(1)}})
            c.observe(dev(.96, .95), 'new')
            training.recover_winners(run, c)
            for name in names:
                self.assertEqual(training._checkpoint_tag(run/name), 'new')
                self.assertFalse((run/(name+'.previous')).exists())

    def test_real_encoder_full_wave_training_and_half_epoch_exact_resume(self):
        import soundfile as sf
        from transformers import SeamlessM4TFeatureExtractor
        from w2v_aasist.tests import tiny_detector
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); ssl = root/'ssl'
            SeamlessM4TFeatureExtractor().save_pretrained(ssl)
            original = tiny_detector()
            baseline = root/'original.pt'
            fingerprints = {str(ssl/'preprocessor_config.json'):sha256(ssl/'preprocessor_config.json')}
            atomic_save(baseline, {'schema':'rtc_w2v_rebuild_v1', 'model':original.state_dict(),
                                  'model_config':original.backbone.config.to_dict(),
                                  'data_fingerprints':fingerprints, 'dev':{}})
            ordinary, noisy, dev_noisy = [], [], []
            for i in range(8):
                wave = np.random.default_rng(i).normal(0, .03, 9000+i*40).astype(np.float32)
                path = root/f'{i}.wav'; sf.write(path, wave, 16000, subtype='FLOAT')
                row = dict(id=f'{i}.wav', audio=str(path), label=i%2, language='en',
                           domain='online' if i<4 else 'offline', noisy=False, band=-1)
                ordinary.append(row)
                if i >= 4:
                    for version in (0,1):
                        noisy.append(dict(row, noisy=True, band=version*2, version=version,
                                          full_length=True, output_samples=len(wave),
                                          source_audio=str(path), source_sha256=sha256(path)))
                    for band in range(4):
                        dev_noisy.append(dict(row, noisy=True, band=band, full_length=True,
                                              output_samples=len(wave)))
            plan = FullEpochPlan(ordinary, [noisy], ordinary_batch=4, noisy_batch=4, seed=7)
            validation = {'clean':ordinary, 'seen':dev_noisy, 'heldout':dev_noisy}
            head = HeadConfig(input_dim=16, projection=8, expansion=32, blocks=4,
                              kernels=(3,7), merge_kernel=3, dropout=.1)
            cfg = dict(device='cpu', amp='none', eval_amp='none', seed=71,
                       baseline=str(baseline), baseline_sha256=sha256(baseline),
                       ordinary_batch=4, noisy_batch=4, head_epochs=1, joint_epochs=1,
                       evals_per_epoch=2, trainable_layers=1, head_lr=.001, joint_head_lr=.0001,
                       encoder_lr=.00001, weight_decay=.0001, workers=0, ssl_path=str(ssl),
                       max_seconds=0., rawboost=0, raw_config={}, head_warmup_steps=1,
                       joint_warmup_steps=1, noisy_weight_start=.3, noisy_weight=.5,
                       cka_weight=.01, cka_warmup_steps=1, grad_clip=1.,
                       checkpointing=False, microbatch=2, frame_budget=100,
                       production_layout=False, eval_batch=8, head_config=asdict(head),
                       offload_activations=False, full_noisy=True,
                       en_real_tolerance=.01, noisy_fake_tolerance=.005)
            discovery = lambda _:(plan, validation, torch.ones(2), torch.tensor([4,4]), fingerprints)
            first, resumed, epoch_resumed = root/'first', root/'resumed', root/'epoch_resumed'
            with patch.object(training, 'build_data', discovery):
                training.train(cfg, first)
                actual_validate = training.validate
                def interrupt(model, validation, cfg, device, path):
                    if path.name == 'epoch_1_scores.jsonl':
                        raise RuntimeError('simulated interruption before second validation')
                    return actual_validate(model, validation, cfg, device, path)
                with patch.object(training, 'validate', side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        training.train(cfg, resumed)
                boundary = training.read_state(resumed/'last.pt')
                self.assertEqual(boundary['cursor'], 1)
                self.assertEqual(boundary['epoch'], 1)
                self.assertEqual(boundary['controller']['phase'], 'head')
                training.train(cfg, resumed, resumed/'last.pt')
                def interrupt_joint(model, validation, cfg, device, path):
                    if path.name == 'epoch_2_step_1_scores.jsonl':
                        raise RuntimeError('simulated epoch-boundary interruption')
                    return actual_validate(model, validation, cfg, device, path)
                with patch.object(training, 'validate', side_effect=interrupt_joint):
                    with self.assertRaisesRegex(RuntimeError, 'epoch-boundary interruption'):
                        training.train(cfg, epoch_resumed)
                boundary = training.read_state(epoch_resumed/'last.pt')
                self.assertEqual(boundary['cursor'], 2)
                self.assertEqual(boundary['controller']['phase'], 'joint')
                training.train(cfg, epoch_resumed, epoch_resumed/'last.pt')
            a, b = training.read_state(first/'last.pt'), training.read_state(resumed/'last.pt')
            self.assertTrue(a['complete'])
            self.assertEqual(a['global_steps'], 4)
            self.assertEqual(a['controller'], b['controller'])
            self.assertEqual(a['history'], b['history'])
            for k in a['model']:
                torch.testing.assert_close(a['model'][k], b['model'][k], rtol=0, atol=0)
            for index, state in a['optimizer']['state'].items():
                for k, value in state.items():
                    torch.testing.assert_close(value, b['optimizer']['state'][index][k], rtol=0, atol=0)
            c = training.read_state(epoch_resumed/'last.pt')
            self.assertEqual(a['controller'], c['controller'])
            self.assertEqual(a['history'], c['history'])
            for k in a['model']:
                torch.testing.assert_close(a['model'][k], c['model'][k], rtol=0, atol=0)
            for index, state in a['optimizer']['state'].items():
                for k, value in state.items():
                    torch.testing.assert_close(value, c['optimizer']['state'][index][k], rtol=0, atol=0)
            for k, value in original.state_dict().items():
                if k.startswith('backbone.encoder.layers.0.'):
                    torch.testing.assert_close(a['model'][k], value, rtol=0, atol=0)
            self.assertEqual(sha256(baseline), cfg['baseline_sha256'])
            for name in ('epoch_1.json','epoch_2.json','best_model.pt','best_weighted.pt',
                         'best_noisy.pt','control_best.pt','completed.json'):
                self.assertTrue((first/name).is_file(), name)
            self.assertEqual(json.loads((first/'planned_coverage.json').read_text())['noisy_rows_per_epoch'], 8)
            from . import evaluate
            from .model import Detector
            from .step import predict
            from .data import loader, read_protocol
            protocol = root/'eval.txt'; protocol.write_text('1.wav\n0.wav\n', encoding='utf-8')
            argv = ['evaluate','--checkpoint',str(first/'best_model.pt'), '--protocol',str(protocol),
                    '--audio-root',str(root),'--out',str(root/'submission'),'--device','cpu','--workers','0']
            with patch('sys.argv', argv): evaluate.main()
            with zipfile.ZipFile(root/'submission'/'submission.zip') as z:
                lines = [line.split() for line in z.read('scores.txt').decode().splitlines()]
            self.assertEqual([r[0] for r in lines], ['1.wav','0.wav'])
            selected = Detector.from_checkpoint(training.read_state(first/'best_model.pt'), checkpointing=False).eval()
            examples = next(iter(loader(read_protocol(protocol,root,labeled=False),cfg)))
            expected = predict(selected, examples, 'cpu', 'none', cfg['microbatch'], cfg['frame_budget']).softmax(1)[:,0]
            for row, value in zip(lines, expected):
                self.assertAlmostEqual(float(row[1]), float(value), places=8)
            metadata = json.loads((root/'submission'/'submission_meta.json').read_text())
            self.assertEqual(metadata['score'], 'P(fake)')
            self.assertEqual(metadata['input_policy'], 'full utterance')


if __name__ == '__main__':
    unittest.main()
