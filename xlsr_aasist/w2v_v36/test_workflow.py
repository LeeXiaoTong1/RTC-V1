"""Small real-model pipeline through feature fitting and deployable submission."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import torch
from transformers import SeamlessM4TFeatureExtractor

from w2v_aasist.runtime import sha256, atomic_json
from w2v_v3.test_model_step import tiny_detector
from w2v_v3.step import predict
from .patch import load_base, final_layer, save_patch, load_selected, apply_patch
from .workflow import run_experiment, report, export_report
from .features import inference_batches
from .evaluate import export
from . import SCHEMA

torch.set_num_threads(1)


def fixture(root):
    processor = root/'processor'
    SeamlessM4TFeatureExtractor().save_pretrained(processor)
    torch.manual_seed(64)
    model = tiny_detector(checkpointing=False).eval()
    base = root/'original_best.pt'
    torch.save(dict(kind='weights', tag='baseline', model=model.state_dict(), **model.architecture(),
                    data_fingerprints={str(processor/'preprocessor_config.json'):sha256(processor/'preprocessor_config.json')}), base)
    cfg = dict(version='3.6', base_checkpoint=str(base), base_checkpoint_sha256=sha256(base),
        base_tag='baseline', ssl_path=str(processor), device='cpu', workers=0, eval_batch=4,
        microbatch=2, frame_budget=2400, seed=3601, code_fingerprints={}, source_config={}, dev_config={},
        lambda_grid=[.1,1.], max_iterations=200, min_gain=2., feature_free_margin_bytes=0,
        metric_definition='fixture full online .3 + mean noisy .7')
    train, dev = [], []
    for split, target, count, conditions in [('train', train, 8, ('offline','online','noisy_a','noisy_b')),
                                             ('dev', dev, 4, ('online','seen','heldout'))]:
        for i in range(count):
            language, label = ('en' if i % 4 < 2 else 'zh'), i % 2
            identity = f'online/{language}/{split}_{i}.wav'
            path = root/'audio'/identity; path.parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(i+(0 if split=='train' else 100))
            sf.write(path, rng.normal(0,.1,8000+i*80).astype(np.float32),16000,subtype='FLOAT')
            for condition in conditions:
                target.append(dict(id=identity, source_id=f'{split}_{i}', group_id=f'{split}_{i}',
                    audio=str(path), condition=condition, label=label, language=language,
                    noisy=condition not in ('offline','online'), view='full', full_length=True,
                    output_samples=8000+i*80))
    return cfg, model, dict(train=train, dev=dev, coverage={'fixture':True}, fingerprints={})


class WorkflowTests(unittest.TestCase):
    def test_real_model_fit_fallback_patch_and_protocol_ordered_export(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder); cfg, original, records = fixture(root)
            run = root/'run'; run.mkdir(); atomic_json(run/'config.json', cfg)
            base_digest = sha256(cfg['base_checkpoint'])
            with patch('w2v_v36.data.build_records', return_value=records):
                done = run_experiment(cfg, run)
            self.assertTrue(done['baseline_fallback'])
            self.assertEqual(base_digest, sha256(cfg['base_checkpoint']))
            state, _ = load_selected(run)
            loaded = apply_patch(load_base(cfg), state)
            for a,b in zip(loaded.state_dict().values(), original.state_dict().values()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertLess((run/'best_patch.pt').stat().st_size, 100000)
            rows = [records['dev'][i] for i in (9,0,6,3)]
            protocol = root/'progress.txt'; protocol.write_text('\n'.join(r['id'] for r in rows)+'\n')
            expected = []
            for examples in inference_batches(rows,cfg):
                with torch.inference_mode():
                    expected.extend(predict(original,examples,torch.device('cpu'),'none',2,2400).softmax(-1)[:,0].tolist())
            archive = export(run, protocol, root/'audio', root/'submission', 'cpu', 0)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.namelist(), ['scores.txt'])
                scored = [line.split() for line in z.read('scores.txt').decode().splitlines()]
            self.assertEqual([r[0] for r in scored], [r['id'] for r in rows])
            np.testing.assert_allclose([float(r[1]) for r in scored], expected,rtol=0,atol=1e-9)
            metadata = json.loads((root/'submission'/'submission_meta.json').read_text())
            self.assertTrue(metadata['baseline_fallback'])
            diagnostics = export_report(run,root/'download')
            with zipfile.ZipFile(diagnostics) as z:
                names = z.namelist()
                self.assertIn('fit_report.json',names)
                self.assertIn('dev_scores_baseline.npz',names)
                self.assertFalse(any('features/' in n or n.endswith(('.pt','.wav')) for n in names))
            # Candidate deployment changes only the final decision layer and the
            # ZIP follows that patch rather than silently exporting the old base.
            layer=final_layer(original)
            fitted=dict(selected='group_balanced',selected_patch=dict(
                weight=layer.weight.detach().tolist(),
                bias=(layer.bias.detach()+torch.tensor([1.,-1.])).tolist()))
            patched=save_patch(run/'best_patch.pt',cfg,fitted)
            done.update(selected='group_balanced',baseline_fallback=False,patch_sha256=sha256(run/'best_patch.pt'))
            atomic_json(run/'completed.json',done)
            apply_patch(original,patched)
            expected=[]
            for examples in inference_batches(rows,cfg):
                with torch.inference_mode():
                    expected.extend(predict(original,examples,torch.device('cpu'),'none',2,2400).softmax(-1)[:,0].tolist())
            candidate_zip=export(run,protocol,root/'audio',root/'candidate_submission','cpu',0)
            with zipfile.ZipFile(candidate_zip) as z:
                candidate=[float(line.split()[1]) for line in z.read('scores.txt').decode().splitlines()]
            np.testing.assert_allclose(candidate,expected,rtol=0,atol=1e-9)
            self.assertGreater(max(abs(a-float(b[1])) for a,b in zip(candidate,scored)),.01)

    def test_changed_patch_and_base_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg,model,_=fixture(root);run=root/'run';run.mkdir()
            layer=final_layer(model)
            result=dict(selected='group_balanced',selected_patch=dict(
                weight=layer.weight.detach().tolist(),bias=layer.bias.detach().tolist()))
            save_patch(run/'best_patch.pt',cfg,result)
            atomic_json(run/'completed.json',dict(version='3.6',selected='group_balanced',
                base_checkpoint_sha256=cfg['base_checkpoint_sha256'],patch_sha256=sha256(run/'best_patch.pt')))
            loaded,_=load_selected(run)
            self.assertEqual(loaded['schema'],SCHEMA)
            (run/'best_patch.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'matching selected patch'):load_selected(run)
            with open(cfg['base_checkpoint'],'ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA256'):load_base(cfg)

    def test_patch_changes_only_last_layer_and_nonfinite_rejected(self):
        model=tiny_detector(checkpointing=False).eval()
        before={k:v.clone() for k,v in model.state_dict().items()}
        layer=final_layer(model)
        state=dict(schema=SCHEMA,kind='classifier_patch',weight=layer.weight.detach().clone()+.01,
                   bias=layer.bias.detach().clone()+.2)
        apply_patch(model,state)
        for name,value in model.state_dict().items():
            if not name.startswith('head.classifier.2.'):
                torch.testing.assert_close(before[name],value,rtol=0,atol=0)
        state['weight'][0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):apply_patch(model,state)

    def test_report_handles_rejected_candidates(self):
        from .metrics import evaluate
        rows=[dict(condition=c,language='en',label=y) for c in ('online','seen','heldout') for y in (0,1)]
        metrics=evaluate(rows,np.tile(np.eye(2),(3,1)))
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            report(Path(d),dict(baseline=metrics,candidates=[dict(tag='class_balanced',metrics=None)],selected='baseline'))
            self.assertIn('Selected: baseline',(Path(d)/'report.md').read_text())


if __name__=='__main__':
    unittest.main()
