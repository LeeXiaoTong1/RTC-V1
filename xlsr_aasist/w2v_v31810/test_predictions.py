"""Ranking ties, row-bound pickle-free diagnostics and report delivery."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

from .common import RECIPE, atomic_json, digest, read_json
from .predictions import ranking_metrics, save_predictions
from .validation import PANEL_WEIGHTS, binary_metrics, evaluate_protocol
from .reports import report
from .train import _evaluate


def protocol_fixture():
    rows=[]; panels={}
    for panel in PANEL_WEIGHTS:
        panels[panel]=[]
        for lang in ('en', 'zh'):
            for label in (0, 1):
                index=len(rows); panels[panel].append(index)
                rows.append(dict(id=f'{panel}-{lang}-{label}', canonical_source=f'{lang}-{label}',
                                 language=lang, label=label, validation_panel=panel))
    part=dict(rows=rows, panels=panels, scene_noise_records=[], tickets=list(range(len(rows))))
    return dict(select=part, audit=part, protocol_hash='locked-inner-protocol')


class PredictionTests(unittest.TestCase):
    def test_ranking_ties_use_half_credit_and_roc_interpolation(self):
        # There are four fake/real pairs: one tie, three correctly ranked.
        result=ranking_metrics([0, 0, 1, 1], [3., 2., 2., 0.])
        self.assertEqual(result['auc'], .875)
        self.assertEqual(result['eer'], .25)
        for labels in ([0, 0, 1, 1], [1, 0, 1, 0]):
            self.assertEqual(ranking_metrics(labels, [1., 1., 1., 1.]), dict(auc=.5, eer=.5))

    def test_ranking_extremes_single_class_and_fake_positive_direction(self):
        self.assertEqual(ranking_metrics([0, 1], [1., 0.]), dict(auc=1., eer=0.))
        self.assertEqual(ranking_metrics([0, 1], [0., 1.]), dict(auc=0., eer=1.))
        self.assertEqual(ranking_metrics([0, 0], [1., 0.]), dict(auc=None, eer=None))
        result=binary_metrics([dict(label=label) for label in (0, 0, 1, 1)],
                              [[2., 0.], [0., 2.], [1., 1.], [0., 2.]])
        self.assertEqual(result['confusion']['matrix'], [[1, 1], [1, 1]])
        self.assertEqual(result['auc'], .625)
        # Fixed-threshold recalls are unchanged by the new ranking diagnostics.
        self.assertEqual(result['recall'], [.5, .5])

    def test_auc_matches_all_pair_tie_definition(self):
        rng=np.random.default_rng(31810)
        for _ in range(30):
            labels=np.array([0]*7+[1]*9); scores=rng.integers(-3, 4, len(labels))
            comparisons=scores[labels==0,None]-scores[labels==1][None,:]
            expected=float(np.mean((comparisons>0)+.5*(comparisons==0)))
            self.assertAlmostEqual(ranking_metrics(labels,scores)['auc'],expected)

    def test_actual_validation_saves_logits_once_and_binds_order_metadata(self):
        protocol=protocol_fixture(); rows=protocol['select']['rows']; calls=[]
        z=np.asarray([[3., -2.] if r['label']==0 else [-3., 2.] for r in rows], dtype=np.float32)
        def infer(model, observed, cfg, tickets, split):
            calls.append(split); return z, observed
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'selection_scores_epoch_2_step_1403.npz'
            cfg=dict(validation_protocol_hash='full-protocol-file-hash')
            result=evaluate_protocol(object(), protocol, cfg, infer_fn=infer,
                predictions_path=path, checkpoint_tag='epoch_2_step_1403')
            self.assertEqual(calls, ['dev']); self.assertEqual(result['score'],1.)
            self.assertEqual(result['predictions']['sha256'],digest(path))
            with np.load(path,allow_pickle=False) as archive:
                self.assertEqual(archive['logits'].dtype,np.float32)
                np.testing.assert_array_equal(archive['logits'],z)
                self.assertEqual(archive['ids'].tolist(),[r['id'] for r in rows])
                self.assertEqual(archive['canonical_source'].tolist(),[r['canonical_source'] for r in rows])
                self.assertEqual(archive['language'].tolist(),[r['language'] for r in rows])
                self.assertEqual(archive['label'].tolist(),[r['label'] for r in rows])
                self.assertEqual(archive['panel'].tolist(),[r['validation_panel'] for r in rows])
                meta=json.loads(str(archive['metadata']))
                self.assertEqual(meta['checkpoint_tag'],'epoch_2_step_1403')
                self.assertEqual(meta['protocol_hash'],protocol['protocol_hash'])
                self.assertEqual(meta['protocol_content_sha256'],cfg['validation_protocol_hash'])
                self.assertTrue(all(not archive[name].dtype.hasobject for name in archive.files))
            self.assertNotIn('logits',result)

    def test_bad_inference_order_never_publishes_prediction_file(self):
        protocol=protocol_fixture()
        def infer(model, rows, cfg, tickets, split):
            return np.zeros((len(rows),2)),list(reversed(rows))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'selection_scores_epoch_1_step_1403.npz'
            with self.assertRaisesRegex(ValueError,'row order'):
                evaluate_protocol(object(),protocol,{},infer_fn=infer,predictions_path=path,checkpoint_tag='epoch_1')
            self.assertFalse(path.exists())

    def test_train_routes_existing_evaluation_to_named_outputs_and_report_includes_them(self):
        protocol=protocol_fixture()
        def infer(model, rows, cfg, tickets, split):
            return np.asarray([[2.,0.] if r['label']==0 else [0.,2.] for r in rows]),rows
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); run=root/'run'; run.mkdir()
            cfg=dict(version='3.18.10',recipe=RECIPE,validation_protocol_hash='full-protocol')
            atomic_json(run/'config.json',cfg)
            with patch('w2v_v31810.validation_data.infer_fixed',side_effect=infer) as inference:
                selected=_evaluate(evaluate_protocol,object(),protocol,cfg,run,'select','epoch_1_step_1403')
                audited=_evaluate(evaluate_protocol,object(),protocol,cfg,run,'audit','epoch_1_step_1403')
                self.assertEqual(inference.call_count,2)
            self.assertEqual(audited['selected_checkpoint'],'epoch_1_step_1403')
            atomic_json(run/'selection_epoch_1_step_1403.json',selected)
            atomic_json(run/'locked_audit.json',dict(metrics=audited))
            np.savez(run/'unrelated.npz',waveform=np.ones(3))
            with redirect_stdout(io.StringIO()): archive=report(run,root/'downloads')
            with zipfile.ZipFile(archive) as stream:
                names=set(stream.namelist())
                self.assertIn('selection_scores_epoch_1_step_1403.npz',names)
                self.assertIn('locked_audit_scores.npz',names)
                self.assertNotIn('unrelated.npz',names)
                manifest=json.loads(stream.read('REPORT_MANIFEST.json'))
                row=next(r for r in manifest['files'] if r['file']=='locked_audit_scores.npz')
                self.assertEqual(row['sha256'],audited['predictions']['sha256'])

    def test_saved_audit_prediction_export_uses_no_inference(self):
        from .test_delivery import fixture
        from .evaluate import export
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run,_,_,_,_=fixture(directory)
            rows=protocol_fixture()['audit']['rows']
            info=save_predictions(run/'locked_audit_scores.npz',rows,np.zeros((len(rows),2)),
                protocol_hash='inner',protocol_content_hash='outer',checkpoint_tag='epoch_9_step_1403',split='audit')
            audit=read_json(run/'locked_audit.json');audit['metrics']['predictions']=info
            atomic_json(run/'locked_audit.json',audit)
            out=Path(directory)/'audit_export'
            with patch('w2v_v31810.evaluate.infer',side_effect=AssertionError('No inference')):
                export(run,out,audit=True,model_factory=lambda cfg:self.fail('No model load'))
            self.assertEqual(digest(out/'locked_audit_scores.npz'),info['sha256'])


if __name__=='__main__': unittest.main()
