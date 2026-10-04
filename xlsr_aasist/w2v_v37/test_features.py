from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from w2v_v36.test_features import fixture
from w2v_v36.features import extract_cache
from w2v_v36.config import code_fingerprints
from w2v_v3.test_model_step import tiny_detector
from w2v_aasist.runtime import sha256
from .features import reusable_cache


class ReuseTests(unittest.TestCase):
    def test_v36_vectors_borrowed_read_only_and_wrong_source_or_order_rejected(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            root=Path(folder);cfg,rows=fixture(root)
            cfg['base_checkpoint_sha256']='a'*64
            identity=dict(base_checkpoint_sha256='a'*64,split='train',data_fingerprints={},
                          code_fingerprints=code_fingerprints())
            model=tiny_detector(checkpointing=False)
            bundle=extract_cache(model,rows,cfg,root/'features'/'train',identity)
            expected=np.array(bundle['x'])
            bundle['x']._mmap.close();bundle['logits']._mmap.close()
            before={str(p):sha256(p) for p in (root/'features').rglob('*') if p.is_file()}
            with patch.object(model,'forward',side_effect=AssertionError('No recomputation')):
                reused=reusable_cache(root,'train',rows,cfg,{})
            np.testing.assert_array_equal(reused['x'],expected)
            self.assertTrue(reused['reused_path'])
            reused['x']._mmap.close();reused['logits']._mmap.close()
            self.assertEqual(before,{str(p):sha256(p) for p in (root/'features').rglob('*') if p.is_file()})
            with self.assertRaisesRegex(ValueError,'different model'):
                reusable_cache(root,'train',rows,dict(cfg,base_checkpoint_sha256='b'*64),{})
            with self.assertRaisesRegex(ValueError,'identities/order'):
                reusable_cache(root,'train',rows[::-1],cfg,{})
            self.assertIsNone(reusable_cache(root,'dev',rows,cfg,{}))


if __name__=='__main__':unittest.main()
