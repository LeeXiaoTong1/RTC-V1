"""Actual tiny HF inference, protected fallback, and official ZIP contract."""
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
from w2v_aasist.tests import tiny_detector
from w2v_aasist.runtime import atomic_save, atomic_json, sha256
from w2v_v3.model import HeadConfig
from w2v_v33.model import Detector
from . import SCHEMA, evaluate, select_checkpoint


class ExportTests(unittest.TestCase):
    def test_actual_full_utterance_export_reference_and_candidate(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); ssl=root/'ssl'
            SeamlessM4TFeatureExtractor().save_pretrained(ssl)
            model=Detector(tiny_detector().backbone, HeadConfig(input_dim=16,projection=8,
                expansion=32,kernels=(3,7),merge_kernel=3)).eval()
            cfg=dict(version='3.5',ssl_path=str(ssl),device='cpu',workers=0,seed=7,
                microbatch=2,frame_budget=1600,max_seconds=0.,rawboost=0,raw_config={},
                eval_batch=2,reference_checkpoint_sha256='protected')
            fingerprints={str(ssl/'preprocessor_config.json'):sha256(ssl/'preprocessor_config.json')}
            protocol=root/'progress.txt'
            ids=['noisy/online/en/z.wav','clean/online/zh/a.wav']
            for i, name in enumerate(ids):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                # The first recording exceeds the former 4s cutoff.
                sf.write(path,np.random.default_rng(i).normal(0,.03,82000 if i==0 else 13300).astype('float32'),16000,subtype='FLOAT')
            protocol.write_text('\n'.join(ids)+'\n')
            atomic_json(root/'config.json',cfg)
            for tag in ('reference','epoch_2'):
                state=dict(schema=SCHEMA,kind='weights',tag=tag,config=cfg,
                    model=model.state_dict(),**model.architecture(),data_fingerprints=fingerprints)
                checkpoint=root/'best_model.pt';atomic_save(checkpoint,state)
                original_hash=sha256(checkpoint)
                atomic_json(root/'completed.json',dict(selected_tag=tag))
                self.assertEqual(select_checkpoint.select(root),checkpoint.resolve())
                out=root/tag
                seen=[]; real_predict=evaluate.predict
                def tracking(model,examples,*args,**kwargs):
                    seen.extend((e['id'],e['features'].shape[1]) for e in examples)
                    return real_predict(model,examples,*args,**kwargs)
                argv=['export','--checkpoint',str(checkpoint),'--protocol',str(protocol),
                    '--audio-root',str(root),'--out',str(out),'--device','cpu','--workers','0']
                with patch('sys.argv',argv),patch.object(evaluate,'predict',side_effect=tracking):
                    evaluate.main()
                self.assertGreater(seen[0][1],200)
                self.assertEqual([name for name,_ in seen],ids)
                self.assertEqual(sha256(checkpoint),original_hash)
                with zipfile.ZipFile(out/'submission.zip') as archive:
                    self.assertEqual(archive.namelist(),['scores.txt'])
                    rows=[line.split() for line in archive.read('scores.txt').decode().splitlines()]
                    self.assertEqual([row[0] for row in rows],ids)
                    self.assertTrue(all(0<=float(row[1])<=1 for row in rows))
                metadata=json.loads((out/'submission_meta.json').read_text())
                self.assertEqual(metadata['reference_fallback'],tag=='reference')
                self.assertEqual(metadata['checkpoint_sha256'],original_hash)
            atomic_json(root/'completed.json',dict(selected_tag='epoch_999'))
            with self.assertRaisesRegex(ValueError,'differs'):select_checkpoint.select(root)
            state['kind']='training';atomic_save(checkpoint,state)
            with patch('sys.argv',argv),self.assertRaisesRegex(ValueError,'weights checkpoint'):evaluate.main()


if __name__=='__main__':unittest.main()
