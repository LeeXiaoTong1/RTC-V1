from contextlib import redirect_stdout
import io
import unittest

import numpy as np
import torch

from .student import fit_student,predict_student,LanguageStudent


def fixture():
    rng=np.random.default_rng(16)
    rows=[];x=[];g=[]
    teacher=rng.normal(size=(5,12)).astype(np.float32)
    for i in range(32):
        vector=rng.normal(size=8).astype(np.float32)
        for c in ('offline','online','noisy_a','noisy_b'):
            point=vector+rng.normal(0,.03,size=8).astype(np.float32)
            target=np.tanh(point[:5])@teacher
            rows.append(dict(source_id=f's{i}',group_id=f's{i}',condition=c,split='train',
                             language='en' if i%4<2 else 'zh',label=i%2))
            x.append(point);g.append(target/np.linalg.norm(target))
    return np.asarray(x,dtype=np.float32),np.asarray(g,dtype=np.float32),rows


class StudentTests(unittest.TestCase):
    def test_fits_nonlinear_teacher_reproducibly_and_replays_deployment(self):
        x,g,rows=fixture()
        cfg=dict(student_epochs=30,student_hidden=16,student_batch_rows=64,student_learning_rate=.02)
        with redirect_stdout(io.StringIO()):
            state=fit_student(x,g,rows,cfg)
            again=fit_student(x,g,rows,cfg)
        self.assertEqual(state,again)
        self.assertGreater(state['diagnostics']['train_weighted_cosine'],.95)
        self.assertLess(state['diagnostics']['loss_trace'][-1],state['diagnostics']['loss_trace'][0]*.1)
        predicted=predict_student(x,state,chunk_rows=17)
        with torch.inference_mode():expected=LanguageStudent(state)(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(predicted,expected,rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(np.linalg.norm(predicted,axis=1),1.,atol=1e-6)

    def test_rejects_dev_nonfinite_and_zero_teacher(self):
        x,g,rows=fixture()
        bad=[dict(r,split='dev') for r in rows]
        with self.assertRaisesRegex(ValueError,'Train only'):fit_student(x,g,bad,{})
        g[0]=0
        with self.assertRaises(ValueError):fit_student(x,g,rows,{})
        g[0,0]=float('nan')
        with self.assertRaises(ValueError):fit_student(x,g,rows,{})


if __name__=='__main__':unittest.main()
