"""Observation correctness and isolation from model optimization."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from w2v_v39.common import atomic_json,read_json,digest
from w2v_v313.test_workflow import assert_nested_equal
from w2v_v3161.test_release import ReleaseFixture,NativeFixtureEngine
from w2v_v317.arguments import parser
from w2v_v317.config import configuration
from w2v_v317.model import load_model
from w2v_v317.state import capture_rng,load_resume
from w2v_v317.train import run_experiment
from .metrics import summarize,fit_state
from .history import read_steps,collect,console_summary
from .render import write_artifacts,chart
from .panel import evaluate_panel,observation_mode,definition
from .hooks import attach
from .audit import audit
from .archive import export_report
from w2v_v316_tfcl.data import Triplets


class WorkerFixtureTriplets(Triplets):
    def __getitem__(self,ticket):
        with patch('w2v_v316_tfcl.data.Engines',NativeFixtureEngine):
            return super().__getitem__(ticket)


def tiny_config(source):
    cfg=configuration(parser().parse_args(['--data-run',str(source),'--epochs','2','--workers','0',
        '--device','cpu','--no-autotune','--lora-layers','1','--lora-rank','2']))
    cfg.update(feature_dim=8,head_expansion=32,head_blocks=2,tfcl_heads=2,tfcl_bins=21,
               train_probe_per_group=4,disk_margin_bytes=0,free_reserve_bytes=0)
    return cfg


class MetricTests(unittest.TestCase):
    def test_group_polarity_ties_and_balancing(self):
        rows=[dict(language=l,label=y,condition='online') for l in ('en','zh') for y in (0,1)]
        value=summarize(rows,np.zeros((4,2)))
        self.assertEqual(value['groups']['online/en']['recall'],[1.,0.])
        self.assertEqual(value['groups']['online/en']['auc'],.5)
        self.assertAlmostEqual(value['conditions']['online']['balanced_ce'],np.log(2))
        duplicate=summarize(rows+[rows[0]]*12,np.zeros((16,2)))
        self.assertEqual(duplicate['conditions']['online'],value['conditions']['online'])
        missing=summarize(rows[:1],np.zeros((1,2)))
        self.assertIsNone(missing['groups']['online/en']['auc'])
        self.assertIsNone(missing['conditions']['online']['balanced_ce'])

    def test_fit_status_requires_a_trend_and_not_just_high_train_recall(self):
        def epoch(train,dev):
            return {s:dict(conditions={'online':dict(balanced_ce=v,balanced_recall=.99 if s=='train' else .89)})
                    for s,v in [('train',train),('dev',dev)]}
        self.assertEqual(fit_state([epoch(.01,.6)])['code'],'one_epoch')
        self.assertEqual(fit_state([epoch(.01,.6),epoch(.005,.7)])['code'],'overfitting_risk')
        self.assertEqual(fit_state([epoch(.01,.6),epoch(.009,.5)])['code'],'dev_ce_improving')

    def test_retry_discards_stale_uncommitted_suffix_and_partial_line(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'steps.jsonl'
            rows=[{'cursor':i,'value':v} for i,v in [(1,1),(2,2),(3,3),(4,4),(2,20)]]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"cursor":3',encoding='utf-8')
            self.assertEqual(read_steps(path),[rows[0],rows[-1]])

    def test_chart_handles_zero_missing_and_one_point_without_fake_values(self):
        text=chart('<loss>',[('zero',[(1,0)]),('one',[(1,1.)]),('missing',[(1,None)])],log=True)
        self.assertNotIn('nan',text.lower());self.assertIn('&lt;loss&gt;',text)
        self.assertNotIn('>zero<',text);self.assertIn('y=1',text)


class FullReplayTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_full_train_replay_counts_missing_online_determinism_and_rng(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()), \
             patch('w2v_v317.config.runtime',return_value={'fixture_native_apm':True}), \
             patch('w2v_v316_tfcl.data.Engines',NativeFixtureEngine):
            f=ReleaseFixture(directory);cfg=tiny_config(f.source);run=f.root/'observe';run.mkdir()
            rows=list(f.train)
            # Missing Online never becomes an invented Online example.
            rows.remove(next(r for r in rows if r['condition']=='online'))
            model=load_model(cfg).train();before=copy.deepcopy(model.state_dict());rng=capture_rng()
            modes=[m.training for m in model.modules()]
            result=evaluate_panel(model,rows,cfg,run,'epoch_1_step_1','container-v1')
            self.assertEqual(result['sources'],16);self.assertEqual(result['metrics']['count'],47)
            self.assertEqual(sum(g['count'] for n,g in result['metrics']['groups'].items() if n.startswith('noisy_train/')),16)
            self.assertEqual(sum(g['count'] for n,g in result['metrics']['groups'].items() if n.startswith('online/')),15)
            self.assertFalse(any(n.startswith(('seen/','heldout/')) for n in result['metrics']['groups']))
            assert_nested_equal(self,before,model.state_dict());assert_nested_equal(self,rng,capture_rng())
            self.assertEqual(modes,[m.training for m in model.modules()])
            # A later container can carry the same best weights; use tensor identity.
            self.assertEqual(evaluate_panel(model,rows,cfg,run,'epoch_1_step_1','container-v2'),result)
            self.assertFalse(list(run.rglob('*.wav')));self.assertFalse(list(run.rglob('*.pt')))
            second=evaluate_panel(model,rows,cfg,run,'epoch_2_step_1','container-v2')
            self.assertEqual(second['generated_recipe_sha256'],result['generated_recipe_sha256'])
            a=np.load(run/'diagnostics/panel_scores_epoch_1_step_1.npz',allow_pickle=False)
            b=np.load(run/'diagnostics/panel_scores_epoch_2_step_1.npz',allow_pickle=False)
            try:np.testing.assert_array_equal(a['logits'],b['logits'])
            finally:a.close();b.close()

    def test_observation_exception_restores_modes_and_random_state(self):
        from w2v_v317.test_core import model_fixture
        model=model_fixture().train();before=capture_rng();modes=[m.training for m in model.modules()]
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with observation_mode(model):
                torch.rand(3);np.random.rand();raise RuntimeError('fixture')
        assert_nested_equal(self,before,capture_rng());self.assertEqual(modes,[m.training for m in model.modules()])

    def test_full_replay_spawn_workers_match_single_process(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()), \
             patch('w2v_v317.config.runtime',return_value={'fixture_native_apm':True}), \
             patch('w2v_v317_monitor.panel.Triplets',WorkerFixtureTriplets):
            f=ReleaseFixture(directory);cfg=tiny_config(f.source);model=load_model(cfg)
            first=f.root/'single';first.mkdir();second=f.root/'spawn';second.mkdir()
            a=evaluate_panel(model,f.train,cfg,first,'epoch_1_step_1')
            b=evaluate_panel(model,f.train,dict(cfg,workers=2,feature_workers=2),second,'epoch_1_step_1')
            self.assertEqual(a['metrics'],b['metrics'])
            self.assertEqual(a['panel_signature'],b['panel_signature'])
            self.assertEqual(a['generated_recipe_sha256'],b['generated_recipe_sha256'])

    def test_monitored_training_resume_is_identical_and_exports_full_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()), \
             patch('w2v_v317.config.runtime',return_value={'fixture_native_apm':True}), \
             patch('w2v_v316_tfcl.data.Engines',NativeFixtureEngine):
            f=ReleaseFixture(directory);cfg=tiny_config(f.source)
            baseline=f.root/'plain';baseline.mkdir();atomic_json(baseline/'config.json',cfg)
            run_experiment(cfg,baseline);expected=load_resume(baseline,cfg)
            observed=f.root/'observed';observed.mkdir();atomic_json(observed/'config.json',cfg)
            from w2v_v317 import train
            original=train.save
            def interrupted(*args,**kwargs):
                original(*args,**kwargs)
                if args[2]['cursor']==1:raise RuntimeError('stop after saved epoch')
            with patch.object(train,'save',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'stop after saved epoch'):run_experiment(cfg,observed)
            self.assertFalse((observed/'diagnostics').exists())
            # Resume a pre-monitor checkpoint: backfill epoch 1 before epoch 2.
            with attach():run_experiment(cfg,observed)
            actual=load_resume(observed,cfg)
            for key in ('model','optimizer','auxiliary','rng'):
                assert_nested_equal(self,expected[key],actual[key])
            self.assertEqual(expected['best_tag'],actual['best_tag'])
            self.assertEqual([x['metrics'] for x in expected['history']],[x['metrics'] for x in actual['history']])
            self.assertFalse(list((observed/'diagnostics').glob('failure_*')))
            panels=sorted((observed/'diagnostics').glob('panel_epoch_*.json'))
            self.assertEqual(len(panels),2)
            self.assertTrue(all(read_json(p)['metrics']['count']==48 for p in panels))
            report=collect(observed);self.assertIn('full Train Online',report['fit']['scope'])
            page=write_artifacts(report,observed/'diagnostics')
            self.assertIn('全量 Train',page.read_text(encoding='utf-8'))
            self.assertIn('[Train FULL',console_summary(report))
            # Re-evaluating a selected best after completion can reuse identical weights.
            saved=digest(observed/'last.pt');audit(observed,'best','cpu',0)
            self.assertEqual(saved,digest(observed/'last.pt'))
            archive=export_report(observed,f.root/'download')
            with zipfile.ZipFile(archive) as z:
                self.assertIn('diagnostics/curves.html',z.namelist())
                self.assertIn('diagnostics/group_metrics.csv',z.namelist())
                self.assertFalse(any(n.endswith(('.pt','.wav','.npz')) for n in z.namelist()))
            # Legacy runs remain readable and do not pretend to have Train Noisy.
            legacy=collect(baseline);self.assertIn('not measured',console_summary(legacy))


if __name__=='__main__':unittest.main()
