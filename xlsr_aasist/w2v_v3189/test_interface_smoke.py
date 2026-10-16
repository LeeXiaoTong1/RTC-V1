"""Exercise smoke-test lifecycle locally; this fixture is not native fairseq2."""
import contextlib
import io
import importlib
import sys
import types
import unittest
import torch
from torch import nn
from w2v_v318.test_core import TinyEncoder, TinyFrontend, TinyLayer, TinyLayout
from .interface_smoke import main


class InterfaceSmokeTests(unittest.TestCase):
    def test_smoke_contract_with_four_layer_cpu_fixture(self):
        torch.set_num_threads(1)
        seen = {}
        class Hub:
            def get_arch_config(self, name):
                return types.SimpleNamespace(encoder_config=types.SimpleNamespace(
                    num_encoder_layers=60, model_dim=2048,
                    feature_extractor_layer_descs=[(32,400,320)]))

            def create_new_model(self, arch, device, dtype):
                ec = arch.encoder_config
                seen.update(depth=ec.num_encoder_layers, dim=ec.model_dim)
                encoder = TinyEncoder()
                encoder.layers = nn.ModuleList([TinyLayer(ec.model_dim) for _ in range(ec.num_encoder_layers)])
                return types.SimpleNamespace(
                    encoder_frontend=TinyFrontend(ec.model_dim).to(device=device,dtype=dtype),
                    encoder=encoder.to(device=device,dtype=dtype))

        names = ('omnilingual_asr','fairseq2','fairseq2.models',
                 'fairseq2.models.wav2vec2','fairseq2.nn','fairseq2.nn.batch_layout')
        modules = {name:types.ModuleType(name) for name in names}
        modules['fairseq2.models.wav2vec2'].get_wav2vec2_model_hub = lambda: Hub()
        modules['fairseq2.nn.batch_layout'].BatchLayout = TinyLayout
        output = io.StringIO()
        rng = torch.random.get_rng_state().clone()
        # Restoring the entire sys.modules dictionary would remove lazily
        # imported torch._dynamo/_inductor modules while leaving their C++
        # registrations alive. Subsequent tests would then re-register them.
        # Restore only the explicit fake dependency keys owned by this test.
        missing = object()
        original = {name:sys.modules.get(name,missing) for name in modules}
        try:
            sys.modules.update(modules)
            with contextlib.redirect_stdout(output):
                main('cpu')
        finally:
            for name,value in original.items():
                if value is missing:
                    sys.modules.pop(name,None)
                else:
                    sys.modules[name]=value
        self.assertEqual(seen,dict(depth=4,dim=64))
        self.assertIn('full 384-d classifier',output.getvalue())
        self.assertIn('all-depth checkpointing and adapter gradients',output.getvalue())
        self.assertIn('scale-state restore',output.getvalue())
        torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0,atol=0)
        # Explicit regression: the next Torch consumer must not re-register
        # compiled operator namespaces after the fake native runtime is gone.
        importlib.import_module('torch._dynamo')
        for name,value in original.items():
            if value is missing:
                self.assertNotIn(name,sys.modules)
            else:
                self.assertIs(sys.modules[name],value)


if __name__=='__main__':
    unittest.main()
