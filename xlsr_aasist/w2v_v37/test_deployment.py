"""A numerical candidate failure must retain a serializable baseline result."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .deployment import validate_deployment
from .test_fit import bundle,dev_rows
from w2v_v36.metrics import evaluate


class DeploymentTests(unittest.TestCase):
    def test_nonfinite_torch_candidate_falls_back_with_valid_report(self):
        weight=np.zeros((2,512),dtype=np.float32);weight[0,0]=1.
        bias=np.zeros(2,dtype=np.float32)
        dev=bundle(dev_rows(),weight,bias,teacher=False)
        state=dict(weight=weight.tolist(),bias=bias.tolist(),language_state=None,student_state=None)
        for invalid in (float('nan'),float('inf')):
            with self.subTest(invalid=invalid),tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
                result=dict(baseline=evaluate(dev['rows'],dev['logits']),dev_score_files={},
                    candidates=[dict(name='head_only_control',tag='head_only_control',status='converged',patch=state)])
                def broken(module,hidden):
                    return torch.full((len(hidden),2),invalid,dtype=torch.float32,device=hidden.device)
                with patch('w2v_v37.deployment.LanguageCorrectedClassifier.forward',broken):
                    result=validate_deployment(result,dev,weight,bias,{'device':'cpu'},directory)
                self.assertEqual(result['selected'],'baseline')
                self.assertEqual(result['candidates'][0]['status'],'rejected')
                saved=json.loads((Path(directory)/'fit_report.json').read_text())
                audit=saved['candidates'][0]['deployment_replay']
                self.assertFalse(audit['finite_logits'])
                self.assertIsNone(audit['max_absolute_logit_difference'])
                self.assertIsNone(audit['decision_differences'])
                self.assertNotIn('selected_patch',saved)


if __name__=='__main__':unittest.main()
