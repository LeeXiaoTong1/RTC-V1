"""CPU-only checks of analysis math and source-level controls."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from .scores import _load, get_margin, metrics, analyze, paired_conditions
from .probes import _balanced_intercept, _group_holdout, subset_tickets, train_tickets


def fixture(epoch=1,shift=0.,split='select'):
    sources=np.array(['S1','S2','S3','S4']*3)
    panels=np.array(['official_online']*4 + ['regular_noisy']*4 + ['heldout_noisy']*4)
    labels=np.array([1,0,1,0]*3)
    z=np.column_stack((np.tile([-1.,1.,-.4,.8],3)+shift,np.zeros(12)))
    arrays=dict(logits=z.astype(np.float32), ids=np.array([f'u{i}' for i in range(12)]),
        canonical_source=sources,language=np.array(['en']*12),label=labels,panel=panels,
        row_index=np.arange(12),metadata=json.dumps(dict(split=split,protocol_hash='same',checkpoint_tag=f'epoch_{epoch}_step_4')))
    mem=io.BytesIO();np.savez_compressed(mem,**arrays)
    return mem.getvalue()


class DiagnosticsTests(unittest.TestCase):
    def test_sign_and_metrics(self):
        d=_load(fixture())
        met=metrics(d,np.ones(len(d['ids']),dtype=bool))
        self.assertEqual(met['n'],12)
        self.assertAlmostEqual(met['auc'],1.)
        self.assertEqual(met['real_false_positives'],0)
        self.assertEqual(met['fake_false_negatives'],0)
        self.assertTrue(np.array_equal(get_margin(d),d['logits'][:,0]-d['logits'][:,1]))

    def test_reject_corrupt_and_same_source_overlap(self):
        broken=np.load(io.BytesIO(fixture()),allow_pickle=False)
        contents={x:broken[x] for x in broken.files}
        contents['logits'][0,0]=np.nan
        obj=io.BytesIO();np.savez_compressed(obj,**contents)
        with self.assertRaises(ValueError):_load(obj.getvalue())

    def test_epoch_drift_and_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            zip_path=Path(temp)/'report.zip'
            with zipfile.ZipFile(zip_path,'w') as z:
                z.writestr('selection_scores_epoch_1_step_4.npz',fixture(1))
                z.writestr('selection_scores_epoch_6_step_4.npz',fixture(6,.25))
                z.writestr('selection_scores_epoch_15_step_4.npz',fixture(15,.5))
            result=analyze(zip_path,Path(temp)/'result')
            self.assertEqual(result['epochs'],3)
            self.assertTrue((Path(temp)/'result'/'epoch_flips.json').is_file())
            pairs=json.loads((Path(temp)/'result'/'paired_summary.json').read_text())
            self.assertEqual(pairs['en/1/regular_noisy']['paired_sources'],2)
            with zipfile.ZipFile(zip_path,'a') as z:
                z.writestr('locked_audit_scores.npz',fixture(15,0,'audit'))
            with self.assertRaisesRegex(ValueError,'overlap'):analyze(zip_path,Path(temp)/'result2')

    def test_train_source_only_bias_and_group_partition(self):
        x=np.array([-1.,-.2,1.,2.,-2.,.3,1.4,3.])
        label=np.array([1,1,0,0,1,1,0,0])
        panel=np.array(['official_online']*8)
        language=np.array(['en']*8)
        bias=_balanced_intercept(x,label,language,panel,np.ones(8,bool))
        self.assertTrue(np.isfinite(bias))
        groups=np.array(['a','a','b','b','c','c','d','d'])
        mask=_group_holdout(groups,42)
        for s in set(groups):self.assertEqual(len(set(mask[groups==s])),1)

    def test_source_based_subset(self):
        rows=[dict(canonical_source=f'k{i//2}',language='en',label=i//4,id=f'u{i}',validation_panel='official_online') for i in range(8)]
        t=[dict(index=i,kind='online') for i in range(8)]
        a,b=subset_tickets(rows,t,limit=1,seed=1,split='select')
        self.assertEqual(len({r['canonical_source'] for r in a}),2)
        self.assertEqual(len(a),4)
        self.assertEqual(len(a),len(b))


if __name__=='__main__':unittest.main()
