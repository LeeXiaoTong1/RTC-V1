import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from . import validation


class ValidationTests(unittest.TestCase):
    def test_online_only_pooled_conditions_and_saved_scores(self):
        records=[dict(id=f'{language}/{label}/{band}',domain='online',language=language,label=label,band=band,
                      prediction=label) for language in ('en','zh') for label in (0,1) for band in range(4)]
        suite={'clean':[dict(r) for r in records], 'seen':[dict(r) for r in records], 'heldout':[dict(r) for r in records]}
        suite['heldout'][0]['prediction']=1
        class Model:
            def eval(self):pass
        def loader(rows,cfg):return [rows]
        def predict(model,examples,*args):
            return torch.tensor([[3.,-3.] if row['prediction']==0 else [-3.,3.] for row in examples])
        with tempfile.TemporaryDirectory() as d,patch.object(validation,'loader',loader),patch.object(validation,'predict',predict):
            report=validation.validate(Model(),suite,dict(microbatch=4,frame_budget=2400),'cpu',Path(d)/'scores.jsonl')
            self.assertEqual(report['clean_f1'],1.)
            self.assertEqual(report['seen_f1'],1.)
            self.assertEqual(report['heldout_f1'],report['groups']['heldout']['macro_f1'])
            self.assertAlmostEqual(report['weighted_f1'],.3+.7*.5*(1+report['heldout_f1']))
            self.assertNotIn('offline',report['groups'])
            self.assertEqual(len((Path(d)/'scores.jsonl').read_text().splitlines()),48)
            suite['clean'][0]['domain']='offline'
            with self.assertRaisesRegex(ValueError,'Online only'):
                validation.validate(Model(),suite,dict(microbatch=4,frame_budget=2400),'cpu',Path(d)/'bad.jsonl')


if __name__=='__main__':unittest.main()
