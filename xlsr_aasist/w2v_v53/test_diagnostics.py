"""Train-only fixed weights, identity binding, RNG/mode and real export wiring."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import diagnostics, delivery
from .common import atomic_json, read_json, partial_state, fingerprint, digest, schema_for
from .test_fusion import configuration, detector
from .test_delivery import fixture as delivery_fixture


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_train_only_fixed_mixture_and_state_restoration(self):
        torch.manual_seed(713)
        cfg = dict(configuration(), seed=731, sampling_seed=731, microbatch=12, frame_budget=9000)
        model = detector(cfg).train()
        model.fusion.conditioner[-1].weight.data.normal_(0., .1)
        lengths = [4240, 5200, 6800]
        waves = [torch.randn(n).numpy() for n in lengths]
        units = [[dict(id=str(i), source_id=str(i), role='online', wave=w,
                       label=i%2, language='en')] for i,w in enumerate(waves)]
        model.eval()
        model.fusion.reset_observations(enabled=True)
        with torch.no_grad():
            model.encode(waves)
        expected = model.fusion.observations()['mean_weights']
        model.fusion.reset_observations(); model.train()
        rng = torch.get_rng_state()
        class Plan:
            def batches(self, epoch):
                yield ['one balanced plan batch']
                raise AssertionError('Should use only one calibration batch')
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            root = Path(folder); (root/'screen').mkdir()
            atomic_json(root/'screen'/'train_rows.json', [dict(id='train-a')])
            with patch.object(diagnostics, 'make_plan', return_value=Plan()), \
                 patch.object(diagnostics, 'Waves', return_value=object()), \
                 patch.object(diagnostics, 'loader', return_value=[units]), \
                 patch.object(diagnostics, 'close'):
                weights = diagnostics.fixed_train_weights(model, cfg, root, 'trained-checkpoint')
            np.testing.assert_allclose(weights, expected, rtol=1e-5, atol=1e-7)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(model.fusion._collect)
            with patch.object(diagnostics, 'loader', side_effect=AssertionError('cached values rebuilt')):
                self.assertEqual(diagnostics.fixed_train_weights(model, cfg, root, 'trained-checkpoint'), weights)
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                diagnostics.fixed_train_weights(model, cfg, root, 'other-checkpoint')
            with self.assertRaisesRegex(ValueError, 'trained Dev'):
                diagnostics.fixed_train_weights(model, dict(cfg, parent_init={'kind':'v5_submitted_final'}), root, 'id')

    def test_export_loads_actual_adaptive_model_and_requires_gate_tensors(self):
        torch.manual_seed(53)
        cfg = configuration()
        model = detector(cfg).eval()
        model.fusion.conditioner[-1].weight.data.normal_(0., .5)
        waves = [torch.randn(4240), torch.randn(5200)]
        with torch.inference_mode():
            expected = model(waves).softmax(-1)[:,0].numpy()
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent) as folder:
            root = Path(folder)
            run, protocol, audio = delivery_fixture(root)
            saved = read_json(run/'final_config.json')
            saved.update(cfg, version='5.3', v5_stage='final', lora_layers=16, retained_lora_layers=60)
            # Tiny encoder has 3 layers; schema validates production suffix, while
            # factory uses matching tiny geometry to test the real export path.
            from .common import RECIPE
            saved['recipe'] = RECIPE
            saved.update(experiment='E0', robust_arm='baseline')
            atomic_json(run/'final_config.json', saved)
            payload = torch.load(run/'final.pt', weights_only=True)
            payload.update(schema=schema_for(saved), identity=fingerprint(saved), model=partial_state(model))
            torch.save(payload, run/'final.pt')
            done = read_json(run/'completed.json')
            done.update(checkpoint_sha256=digest(run/'final.pt'), final_config_sha256=digest(run/'final_config.json'))
            atomic_json(run/'completed.json', done)
            def factory(_):
                torch.manual_seed(53)
                return detector(cfg)
            def infer(m, rows, _):
                return m(waves).detach().numpy(), rows
            out = root/'submission'
            delivery.export(run, out, protocol=protocol, audio_root=audio,
                            model_factory=factory, inference=infer, verify=False)
            actual = [float(line.split()[1]) for line in (out/'scores.txt').read_text().splitlines()]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-10)
            payload['model'].pop('fusion.conditioner.3.weight')
            torch.save(payload, run/'final.pt')
            done['checkpoint_sha256'] = digest(run/'final.pt'); atomic_json(run/'completed.json', done)
            with self.assertRaisesRegex(ValueError, 'partial keys'):
                delivery.export(run, root/'missing_gate', protocol=protocol, audio_root=audio,
                    model_factory=factory, inference=infer, verify=False)


if __name__ == '__main__':
    unittest.main()
