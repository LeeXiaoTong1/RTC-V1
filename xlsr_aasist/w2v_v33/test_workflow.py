"""Operational tests: exact starting tag, bounded retirement, arm isolation and export."""
from argparse import Namespace
import copy
from contextlib import nullcontext
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from w2v_aasist.runtime import atomic_json, sha256
from . import config, workflow
from .maintenance import retire_replaced_cache
from .select_checkpoint import select


class ConfigurationTests(unittest.TestCase):
    def base(self, root):
        return dict(train_caches=[str(root/'old')], warm_checkpoint=str(root/'best_model.pt'),
                    source_run=str(root/'v3'), baseline=str(root/'original.pt'))

    def test_default_requires_the_named_v3_winner_and_separate_cache(self):
        args=config.parser().parse_args([])
        self.assertEqual(args.epochs,1); self.assertEqual(args.arm,'both')
        self.assertEqual(args.source_run,str(config.DEFAULT_SOURCE))
        with tempfile.TemporaryDirectory() as td:
            base=self.base(Path(td))
            good=dict(schema='rtc_w2v_multiconv_v3',kind='weights',tag=config.EXPECTED_TAG)
            with patch.object(config,'previous_configuration',return_value=copy.deepcopy(base)), \
                 patch('w2v_v3.train.read_state',return_value=good):
                cfg=config.configuration(args)
            self.assertEqual(cfg['legacy_full_cache'],base['train_caches'][0])
            self.assertNotEqual(cfg['train_caches'][0],base['train_caches'][0])
            self.assertEqual(cfg['pair_weight'],.02)
            self.assertEqual(cfg['source_batch'],16)
            for change in ({'tag':'latest'}, {'kind':'training'}, {'schema':'rtc_w2v_multiconv_v32'}):
                with self.subTest(change=change), patch.object(config,'previous_configuration',return_value=copy.deepcopy(base)), \
                     patch('w2v_v3.train.read_state',return_value={**good,**change}):
                    with self.assertRaises(ValueError):config.configuration(args)

    def test_invalid_recipe_rejected_before_checkpoint_io(self):
        for flags in (['--pair-weight','nan'],['--pair-weight','-.1'],['--pair-weight','.2'],
                      ['--source-batch','1'],['--cache-workers','0'],['--aux-max-tokens','1'],
                      ['--pair-warmup-fraction','0']):
            with self.subTest(flags=flags),patch.object(config,'previous_configuration',side_effect=AssertionError('too late')):
                with self.assertRaises(ValueError):config.configuration(config.parser().parse_args(flags))


class RetirementTests(unittest.TestCase):
    def fixture(self, root):
        old=root/'data'/'rtc_noisy_full2_v1'/'train'; (old/'audio').mkdir(parents=True)
        new=root/'data'/'rtc_noisy_v33'/'train';new.mkdir(parents=True)
        protocol=root/'train.txt';protocol.write_text('official train')
        wave=old/'audio'/'owned.wav';wave.write_bytes(b'owned cached wave')
        unrelated=old/'audio'/'unrelated.wav';unrelated.write_bytes(b'not owned')
        atomic_json(old/'config.json',dict(format='rtc_noisy_full2_cache_v1',role='train',protocol_sha256=sha256(protocol)))
        (old/'manifest.jsonl').write_text(json.dumps({'audio':'audio/owned.wav','source':'offline/en/0.wav'})+'\n')
        atomic_json(old/'complete.json',{})
        for name in ('config.json','manifest.jsonl','complete.json'):(new/name).write_text('{}')
        cfg=dict(legacy_full_cache=str(old),train_noisy_cache_v33=str(new),train_protocol=str(protocol),
                 preparation_fingerprints={str(p):sha256(p) for p in new.iterdir()})
        for name in ('baseline','warm_checkpoint','ssl_path','train_data_path','dev_data_path','dev_noisy_cache','dev_heldout_cache'):
            cfg[name]=str(root/name)
        return cfg,wave,unrelated

    def test_only_owned_files_deleted_and_metadata_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,wave,unrelated=self.fixture(root)
            result=retire_replaced_cache(cfg,root,lambda:None,root/'run')
            self.assertEqual(result['deleted_files'],1)
            self.assertFalse(wave.exists());self.assertTrue(unrelated.exists())
            self.assertTrue((Path(cfg['legacy_full_cache'])/'manifest.jsonl').is_file())
            self.assertTrue((root/'run'/'retired_cache_metadata.zip').is_file())

    def test_active_job_or_changed_replacement_prevents_any_delete(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,wave,_=self.fixture(root)
            def busy():raise RuntimeError('active training')
            with self.assertRaisesRegex(RuntimeError,'active'):retire_replaced_cache(cfg,root,busy,root/'run')
            self.assertTrue(wave.exists())
            (Path(cfg['train_noisy_cache_v33'])/'complete.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Replacement'):retire_replaced_cache(cfg,root,lambda:None,root/'run')
            self.assertTrue(wave.exists())

    def test_manifest_escape_and_unrecognized_root_are_never_deleted(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,wave,_=self.fixture(root)
            manifest=Path(cfg['legacy_full_cache'])/'manifest.jsonl'
            manifest.write_text(json.dumps({'audio':'../other.wav'})+'\n')
            with self.assertRaises(ValueError):retire_replaced_cache(cfg,root,lambda:None,root/'run')
            self.assertTrue(wave.exists())
            cfg['legacy_full_cache']=str(root/'unknown')
            self.assertEqual(retire_replaced_cache(cfg,root,lambda:None,root/'run')['status'],'kept_outside_allowlist')

    def test_linked_data_ancestor_outside_project_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();cfg,wave,_=self.fixture(root)
            old=Path(cfg['legacy_full_cache']);external=root.parent/'unrelated-data'/'train'
            resolve=Path.resolve
            def linked(path,*args,**kwargs):
                return external if path==old else resolve(path,*args,**kwargs)
            with patch.object(Path,'resolve',linked):
                result=retire_replaced_cache(cfg,root,lambda:None,root/'run')
            self.assertEqual(result['status'],'kept_outside_project')
            self.assertTrue(wave.exists())


class ExperimentTests(unittest.TestCase):
    def test_supervisor_runs_both_arms_and_exports_complete_safe_comparison(self):
        from .control import quality
        from w2v_v31.test_control import dev
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'exp').mkdir();run=root/'exp'/'v33';run.mkdir()
            checkpoint=root/'fixed.pt';checkpoint.write_bytes(b'protected')
            cfg={'version':'3.3','arm':'both','arms':['control','candidate'],
                 'expected_warm_tag':config.EXPECTED_TAG,'warm_checkpoint':str(checkpoint),
                 'warm_checkpoint_sha256':sha256(checkpoint),'baseline':str(checkpoint),
                 'baseline_sha256':sha256(checkpoint),'preparation_fingerprints':{}}
            atomic_json(run/'config.json',cfg)
            anchor={'tag':'baseline',**quality(dev())};calls=[]
            def execute(folder,arm,saved,smoke):
                calls.append(arm);target=folder/arm;target.mkdir()
                (target/'best_model.pt').write_bytes(b'winner')
                selected=dict(anchor,tag='epoch_1',weighted=anchor['weighted']+(.001 if arm=='control' else .002))
                atomic_json(target/'completed.json',{'selection':{'best_safe':selected,'anchor':anchor}})
                return 0
            with patch.object(workflow,'ROOT',root),patch.object(workflow,'check_environment'), \
                 patch.object(workflow,'ensure_idle'),patch.object(workflow,'run_lock',side_effect=lambda _:nullcontext()), \
                 patch.object(workflow,'_train_arm',side_effect=execute),patch.object(workflow,'phase'),patch.object(workflow,'publish'), \
                 patch('sys.argv',['workflow','--resume',str(run),'--device','cpu','--download-dir',str(root/'download')]):
                workflow.main()
            self.assertEqual(calls,['control','candidate'])
            result=json.loads((run/'comparison.json').read_text())
            self.assertEqual(result['status'],'complete');self.assertEqual(result['selected_arm'],'candidate')
            self.assertFalse(result['practical_success']['success'])
            self.assertAlmostEqual(result['candidate_minus_control_pp']['weighted'],.1)
            self.assertTrue((root/'download'/'v33_report.zip').is_file())
            self.assertEqual(select(run),(run/'candidate'/'best_model.pt').resolve())

    def test_stop_roots_select_only_named_version_and_checkout(self):
        from .stop import owned_roots,descendants
        root=Path('/project').resolve()
        def process(pid,version='v33',cwd=root,ppid=1):
            return dict(pid=pid,args=['python','-u','-m','w2v_'+version+'.workflow'],cwd=cwd,state='S',ppid=ppid)
        rows=[process(10),process(20,'v32'),process(30,cwd=root/'other'),
              dict(pid=11,args=['worker'],cwd=root,state='S',ppid=10)]
        picked=owned_roots(rows,root,'v33')
        self.assertEqual([r['pid'] for r in picked],[10])
        self.assertEqual({r['pid'] for r in descendants(rows,picked)},{10,11})

    def test_each_arm_receives_same_start_and_own_objective_then_resumes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg={'seed':1234,'pair_weight':.02,'warm_checkpoint':'fixed.pt'}
            calls=[]
            def child(command,**kwargs):
                calls.append(command);return Namespace(stdout=io.StringIO('test output\n'),wait=lambda:0)
            with patch.object(workflow.subprocess,'Popen',side_effect=child):
                workflow._train_arm(root,'control',cfg)
                workflow._train_arm(root,'candidate',cfg)
                (root/'candidate'/'last.pt').write_bytes(b'resume boundary')
                workflow._train_arm(root,'candidate',cfg)
            a=json.loads((root/'control'/'config.json').read_text());b=json.loads((root/'candidate'/'config.json').read_text())
            self.assertEqual(a['pair_weight'],0.);self.assertEqual(b['pair_weight'],.02)
            self.assertEqual(a['seed'],b['seed']);self.assertEqual(a['warm_checkpoint'],b['warm_checkpoint'])
            self.assertNotIn('--resume',calls[0]);self.assertIn('--resume',calls[2])

    def test_report_zip_includes_both_arms_but_never_audio_or_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'run';run.mkdir()
            for arm in ('control','candidate'):
                folder=run/arm;folder.mkdir()
                (folder/'scores.jsonl').write_text('{}\n')
                (folder/'best_model.pt').write_bytes(b'no upload')
                (folder/'audio.wav').write_bytes(b'no upload')
            archive=workflow.export_report(run,root/'downloads')
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()),{'control/scores.jsonl','candidate/scores.jsonl'})

    def test_default_export_requires_complete_comparison_and_protected_path(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'candidate').mkdir();p=root/'candidate'/'best_model.pt';p.write_bytes(b'weights')
            report=dict(status='partial',selected_arm='candidate',checkpoint=str(p))
            atomic_json(root/'comparison.json',report)
            with self.assertRaises(ValueError):select(root)
            report['status']='complete';atomic_json(root/'comparison.json',report)
            self.assertEqual(select(root),p.resolve())
            report['checkpoint']=str(root/'other.pt');atomic_json(root/'comparison.json',report)
            with self.assertRaises(ValueError):select(root)

    def test_viewer_keeps_same_named_epoch_from_both_arms(self):
        from v33_console import Events
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);events=Events();printed=[]
            for arm in ('control','candidate'):
                folder=root/arm;folder.mkdir()
                record={'tag':'epoch_1','arm':arm,'dev':{},'decision':{}}
                atomic_json(folder/'epoch_1.json',record)
                (folder/'report.md').write_text('| epoch_1 | result |\n')
                printed+=events.consume('V32_RUN='+str(folder))
                printed+=events.consume('V32_EVENT={"kind":"validation","tag":"epoch_1"}')
            text='\n'.join(printed)
            self.assertEqual(text.count('[Dev]'),2)
            self.assertIn('control',text);self.assertIn('candidate',text)


if __name__=='__main__':unittest.main()
