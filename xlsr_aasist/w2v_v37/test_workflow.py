"""Real small w2v-BERT/MultiConv extraction, fit, patch, and submission contract."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from w2v_aasist.runtime import atomic_json,sha256
from w2v_v36.test_workflow import fixture as small_fixture
from w2v_v36.features import inference_batches,load_cache
from w2v_v3.step import predict
from .workflow import run_experiment,export_report
from .patch import load_base,final_layer,save_patch,load_selected,apply_patch,LanguageCorrectedClassifier
from .student import fit_student
from .fit import fit_language_state,logits_from_state
from .evaluate import export

torch.set_num_threads(1)


def fixture(root):
    from w2v_v3.model import HeadConfig,MultiConvHead
    cfg,model,records=small_fixture(root)
    # Real checkpoint-compatible head width (512), small encoder/conv expansion.
    model.head=MultiConvHead(HeadConfig(input_dim=16,projection=128,expansion=32,
                                      kernels=(3,5,7,9),merge_kernel=5,dropout=0.)).eval()
    base=Path(cfg['base_checkpoint'])
    processor=Path(cfg['ssl_path'])/'preprocessor_config.json'
    torch.save(dict(kind='weights',tag='baseline',model=model.state_dict(),**model.architecture(),
                    data_fingerprints={str(processor):sha256(processor)}),base)
    cfg['base_checkpoint_sha256']=sha256(base)
    return cfg,model,records


class FakeTeacher:
    def __init__(self,*args,**kwargs):pass


def fake_language_cache(records,cfg,out,identity,encoder=None):
    assert identity['split']=='train'
    rng=np.random.default_rng(9)
    g=rng.normal(size=(len(records),256)).astype(np.float32)
    g/=np.linalg.norm(g,axis=1,keepdims=True)
    return dict(lid=g,rows=records,manifest={})


class WorkflowTests(unittest.TestCase):
    def test_full_pipeline_fallback_and_language_patch_export_without_teacher(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);cfg,original,records=fixture(root)
            cfg.update(version='3.7',alpha_grid=[.25],ridge_grid=[1.],lambda_grid=[1.],
                       student_epochs=2,fit_threads=1,student_hidden=8,feature_run=None)
            for split in ('train','dev'):
                for row in records[split]:row['split']=split
            run=root/'run';run.mkdir();atomic_json(run/'config.json',cfg)
            with patch('w2v_v37.workflow.build_records',return_value=records), \
                 patch('w2v_v37.language.ensure_language_assets',return_value={'test_teacher':True}), \
                 patch('w2v_v37.language.LanguageEncoder',FakeTeacher), \
                 patch('w2v_v37.language_cache.extract_language_cache',side_effect=fake_language_cache) as teacher:
                done=run_experiment(cfg,run)
            self.assertEqual(teacher.call_count,1)
            self.assertEqual(len(teacher.call_args.args[0]),len(records['train']))
            self.assertTrue(done['baseline_fallback'])
            self.assertFalse(done['language_debias_applied'])
            state,_=load_selected(run)
            loaded=apply_patch(load_base(cfg),state)
            for a,b in zip(original.state_dict().values(),loaded.state_dict().values()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            selected_rows=[records['dev'][i] for i in (9,0,6,3)]
            protocol=root/'progress.txt';protocol.write_text('\n'.join(r['id'] for r in selected_rows)+'\n')
            expected=[]
            for examples in inference_batches(selected_rows,cfg):
                with torch.inference_mode():expected.extend(predict(original,examples,torch.device('cpu'),'none',2,2400).softmax(-1)[:,0].tolist())
            with patch('w2v_v37.language.LanguageEncoder',side_effect=AssertionError('Teacher must never run on Progress')):
                archive=export(run,protocol,root/'audio',root/'submission','cpu',0)
            with zipfile.ZipFile(archive) as z:
                scores=[r.split() for r in z.read('scores.txt').decode().splitlines()]
            self.assertEqual([r[0] for r in scores],[r['id'] for r in selected_rows])
            np.testing.assert_allclose([float(r[1]) for r in scores],expected,rtol=0,atol=1e-9)
            # Force a valid nonzero branch to verify the selected deployment path.
            bundle=load_cache(run/'features'/'train')
            g=fake_language_cache(records['train'],cfg,None,{'split':'train'})['lid']
            student=fit_student(bundle['x'],g,records['train'],cfg)
            from .student import predict_student
            predicted=predict_student(bundle['x'],student)
            language=fit_language_state(bundle['x'],predicted,records['train'],1.,.25)
            final=final_layer(original)
            chosen=dict(weight=final.weight.detach().tolist(),bias=(final.bias.detach()+torch.tensor([.5,-.5])).tolist(),
                        language_state=language,student_state=student)
            state=save_patch(run/'best_patch.pt',cfg,dict(selected='language_debias',selected_patch=chosen))
            done.update(selected='language_debias',baseline_fallback=False,language_debias_applied=True,
                        patch_sha256=sha256(run/'best_patch.pt'))
            atomic_json(run/'completed.json',done)
            expected_cached=logits_from_state(bundle['x'],state=chosen)
            with torch.inference_mode():
                deployed=LanguageCorrectedClassifier(state,512)(torch.from_numpy(np.array(bundle['x']))).numpy()
            np.testing.assert_allclose(deployed,expected_cached,rtol=3e-5,atol=1e-4)
            expected_model=apply_patch(load_base(cfg),state)
            expected=[]
            for examples in inference_batches(selected_rows,cfg):
                with torch.inference_mode():expected.extend(predict(expected_model,examples,torch.device('cpu'),'none',2,2400).softmax(-1)[:,0].tolist())
            with patch('w2v_v37.language.LanguageEncoder',side_effect=AssertionError('No teacher at inference')):
                archive=export(run,protocol,root/'audio',root/'language_submission','cpu',0)
            with zipfile.ZipFile(archive) as z:
                actual=[float(r.split()[1]) for r in z.read('scores.txt').decode().splitlines()]
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-9)
            meta=json.loads((root/'language_submission'/'submission_meta.json').read_text())
            self.assertTrue(meta['language_debias_applied'])
            self.assertFalse(meta['external_teacher_at_inference'])
            self.assertLess((run/'best_patch.pt').stat().st_size,8*1024**2)
            report=export_report(run,root/'download')
            with zipfile.ZipFile(report) as z:
                self.assertFalse(any(n.endswith(('.wav','.pt','.npy')) for n in z.namelist()))
                diagnostics=json.loads(z.read('fit_report.json'))
                self.assertNotIn('selected_patch',diagnostics)
                self.assertTrue(all('patch' not in c for c in diagnostics['candidates']))
                self.assertNotIn('original_classifier.json',z.namelist())
            self.assertEqual(sha256(cfg['base_checkpoint']),cfg['base_checkpoint_sha256'])
            bundle['x']._mmap.close();bundle['logits']._mmap.close()

    def test_patch_sha_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,model,_=fixture(root);cfg['version']='3.7'
            final=final_layer(model)
            result=dict(selected='baseline',selected_patch=dict(weight=final.weight.detach().tolist(),bias=final.bias.detach().tolist()))
            save_patch(root/'best_patch.pt',cfg,result)
            atomic_json(root/'completed.json',dict(version='3.7',selected='baseline',
                base_checkpoint_sha256=cfg['base_checkpoint_sha256'],patch_sha256=sha256(root/'best_patch.pt')))
            load_selected(root)
            with (root/'best_patch.pt').open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'matching selected patch'):load_selected(root)


if __name__=='__main__':unittest.main()
