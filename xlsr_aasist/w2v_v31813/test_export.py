from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from w2v_v31810.common import atomic_json,atomic_save,partial_state,fingerprint,digest,read_json
from .test_core import Tiny,cfg
from .evaluate import export
from . import VERSION,SCHEMA


class ExportTests(unittest.TestCase):
    def test_single_detector_probabilities_and_upload_retry_without_teacher(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory);run=root/'run';run.mkdir();out=root/'scores';protocol=root/'protocol.txt'
            protocol.write_text('a\nb\n');settings=cfg(version=VERSION,recipe=SCHEMA)
            atomic_json(run/'config.json',settings);model=Tiny()
            atomic_save(run/'last.pt',dict(schema=SCHEMA,identity=fingerprint(settings),best_tag='epoch_1',best_model=partial_state(model)))
            atomic_json(run/'completed.json',dict(status='complete',version=VERSION,best_tag='epoch_1',checkpoint_sha256=digest(run/'last.pt')))
            stack.enter_context(patch('w2v_v31813.evaluate.verify'))
            stack.enter_context(patch('w2v_v31813.evaluate.read_protocol',return_value=[dict(id='a'),dict(id='b')]))
            inference=stack.enter_context(patch('w2v_v31813.evaluate.infer',return_value=(np.array([[2,0],[0,2]],np.float32),[])))
            upload=stack.enter_context(patch('w2v_v31813.evaluate.upload_archive',return_value='https://temp.sh/fixture/submission.zip'))
            path=export(run,protocol,root,out,'cpu',0,True,model_factory=Tiny)
            with zipfile.ZipFile(path) as z:
                self.assertEqual(z.namelist(),['scores.txt']);lines=z.read('scores.txt').decode().splitlines()
            self.assertEqual([line.split()[0] for line in lines],['a','b'])
            self.assertAlmostEqual(float(lines[0].split()[1]),float(torch.tensor([2.,0.]).softmax(-1)[0]),places=6)
            meta=read_json(out/'submission_meta.json');self.assertFalse(meta['teacher_at_inference']);self.assertFalse(meta['pcl_at_inference'])
            export(run,protocol,root,out,'cpu',0,True,model_factory=Tiny)
            self.assertEqual(inference.call_count,1);self.assertEqual(upload.call_count,2)

if __name__=='__main__':unittest.main()
